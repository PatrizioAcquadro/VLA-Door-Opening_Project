"""Tests for W&B tracking integration in the active trainer path."""

from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize


class RecordingTracker:
    """Minimal tracker test double with the ExperimentTracker call surface."""

    def __init__(self, run_id: str = "test-run-id") -> None:
        self.run_id = run_id
        self.configs: list[dict[str, Any]] = []
        self.training_steps: list[dict[str, Any]] = []
        self.metric_logs: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.started = False
        self.finished = False

    def log_config(self, config: dict[str, Any]) -> None:
        self.configs.append(config)

    def start_throughput_tracking(self) -> None:
        self.started = True

    def log_training_step(self, **kwargs: Any) -> None:
        self.training_steps.append(kwargs)

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        self.metric_logs.append({"metrics": metrics, "step": step, "commit": commit})

    def log_checkpoint(self, checkpoint_path: str | Path, aliases: list[str] | None = None) -> None:
        self.checkpoints.append({"path": Path(checkpoint_path), "aliases": aliases or []})

    def get_run_id(self) -> str:
        return self.run_id

    def finish(self) -> None:
        self.finished = True


def _compose_training_cfg(tmp_path: Path, extra_overrides: list[str] | None = None):
    overrides = [
        "model=base",
        "trainer=debug",
        "cluster=local",
        "data.dataset.name=dummy",
        "data.dataset.dummy.train_samples=6",
        "data.dataset.dummy.val_samples=4",
        "data.dataloader.num_workers=0",
        "data.dataloader.persistent_workers=false",
        "model.architecture.hidden_size=32",
        "model.architecture.intermediate_size=64",
        "model.architecture.num_layers=1",
        "model.architecture.num_attention_heads=4",
        "model.architecture.max_seq_length=8",
        "trainer.scheduler.warmup_steps=0",
        "trainer.training.max_steps=2",
        "trainer.training.batch_size_per_device=2",
        "trainer.logging.log_every_n_steps=1",
        "trainer.validation.every_n_steps=2",
        "trainer.validation.num_samples=1",
        "trainer.checkpoint.save_every_n_steps=2",
        "trainer.checkpoint.save_optimizer=false",
        f"paths.root={tmp_path}",
        f"paths.checkpoints={tmp_path / 'checkpoints'}",
        f"paths.logs={tmp_path / 'logs'}",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="config", overrides=overrides)


def test_create_tracker_reads_tracking_run_tags(monkeypatch):
    """Factory should use tracking.run.tags plus env overrides and resume IDs."""
    from tracking import experiment

    created: dict[str, Any] = {}

    class DummyExperimentTracker:
        def __init__(self, **kwargs: Any) -> None:
            created.update(kwargs)

    monkeypatch.setattr(experiment, "ExperimentTracker", DummyExperimentTracker)
    monkeypatch.setenv("WANDB_PROJECT", "env-project")
    monkeypatch.setenv("WANDB_ENTITY", "patrizio")

    cfg = {
        "model": {"name": "vla_dev", "architecture": {"type": "vla"}},
        "data": {"dataset": {"name": "door_episodes"}},
        "cluster": {"name": "local"},
        "tracking": {
            "enabled": True,
            "project": "config-project",
            "entity": None,
            "mode": "offline",
            "log_interval": 7,
            "metrics": {"gpu_stats_interval": 31},
            "run": {
                "name": "manual-run",
                "tags": {
                    "model": "custom-model",
                    "dataset": "custom-data",
                    "objective": "custom-objective",
                    "experiment_group": "foundation",
                },
            },
        },
    }

    experiment.create_tracker(cfg, resume_checkpoint={"wandb_run_id": "resume-123"})

    assert created["project"] == "env-project"
    assert created["entity"] == "patrizio"
    assert created["name"] == "manual-run"
    assert created["resume_id"] == "resume-123"
    assert created["tags"] == {
        "model": "custom-model",
        "dataset": "custom-data",
        "objective": "custom-objective",
        "experiment_group": "foundation",
        "cluster": "local",
    }
    assert created["log_interval"] == 7
    assert created["gpu_stats_interval"] == 31


def test_trainer_logs_metrics_and_persists_wandb_run_id(tmp_path, monkeypatch):
    """Trainer should log through the tracker and persist run IDs in checkpoints."""
    from train import trainer as trainer_module

    created: list[dict[str, Any]] = []

    def fake_create_tracker(
        config: dict[str, Any],
        enabled: bool = True,
        resume_checkpoint: dict[str, Any] | None = None,
    ) -> RecordingTracker:
        tracker = RecordingTracker()
        created.append(
            {
                "tracker": tracker,
                "config": config,
                "enabled": enabled,
                "resume_checkpoint": resume_checkpoint,
            }
        )
        return tracker

    monkeypatch.setattr(trainer_module, "create_tracker", fake_create_tracker)

    cfg = _compose_training_cfg(tmp_path)
    trainer = trainer_module.Trainer(cfg)
    trainer.setup()
    trainer.train()

    tracker = created[0]["tracker"]
    assert created[0]["enabled"] is True
    assert tracker.configs
    assert tracker.started is True
    assert len(tracker.training_steps) == 2
    assert tracker.training_steps[-1]["step"] == 2
    assert tracker.training_steps[-1]["extra_metrics"]["loss/total"] >= 0.0
    assert any("val/loss" in entry["metrics"] for entry in tracker.metric_logs)
    assert any("final" in entry["aliases"] for entry in tracker.checkpoints)
    assert tracker.finished is True

    final_checkpoint = torch.load(tmp_path / "checkpoints" / "final.pt", map_location="cpu")
    assert final_checkpoint["wandb_run_id"] == "test-run-id"

    resume_cfg = _compose_training_cfg(
        tmp_path,
        extra_overrides=[
            f"trainer.checkpoint.resume_from={tmp_path / 'checkpoints' / 'final.pt'}",
        ],
    )
    resumed = trainer_module.Trainer(resume_cfg)
    resumed.setup()

    assert resumed.global_step == 2
    assert created[1]["resume_checkpoint"]["wandb_run_id"] == "test-run-id"
