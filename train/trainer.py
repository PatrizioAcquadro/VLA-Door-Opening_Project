"""Main trainer module with Hydra integration."""

import logging
import os
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tracking import create_tracker

log = logging.getLogger(__name__)


class Trainer:
    """Main trainer class.

    Handles:
    - Model, optimizer, scheduler creation
    - Training loop
    - Checkpointing
    - Logging
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # Set up distributed if available
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0

        if torch.cuda.is_available() and "WORLD_SIZE" in os.environ:
            self._setup_distributed()

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Set seed for reproducibility
        self._set_seed(cfg.experiment.seed)

        # Initialize components
        self.model: Module | None = None
        self.optimizer: Optimizer | None = None
        self.scheduler: LRScheduler | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.tracker: Any | None = None

        # Training state
        self.global_step = 0
        self.epoch = 0

    def _setup_distributed(self) -> None:
        """Initialize distributed training."""
        self.distributed = True
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group("nccl")

        if self.rank == 0:
            log.info(f"Distributed training: world_size={self.world_size}")

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random

        import numpy as np

        random.seed(seed + self.rank)
        np.random.seed(seed + self.rank)
        torch.manual_seed(seed + self.rank)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + self.rank)

    def setup(self) -> None:
        """Set up model, optimizer, data, etc."""
        if self.rank == 0:
            log.info("Setting up training...")

        # Create data loaders first so VLA runs fail fast on missing manifests
        # before trying to load a large VLM backbone.
        self._create_dataloaders()

        # Create model
        from models import count_parameters, format_params, get_model

        self.model = get_model(self.cfg)
        if self.cfg.model.architecture.type not in {"vlm", "vla"}:
            self.model = self.model.to(self.device)

        num_params = count_parameters(self.model)
        if self.rank == 0:
            log.info(f"Model parameters: {format_params(num_params)}")

        # Wrap for distributed
        if self.distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = DDP(self.model, device_ids=[self.local_rank])

        # Create optimizer
        self._create_optimizer()

        # Create scheduler
        self._create_scheduler()

        # Resume from checkpoint if specified
        resume_checkpoint = None
        if self.cfg.trainer.checkpoint.resume_from:
            resume_checkpoint = self._load_checkpoint(self.cfg.trainer.checkpoint.resume_from)

        self._create_tracker(resume_checkpoint=resume_checkpoint)

    def _create_tracker(self, resume_checkpoint: dict[str, Any] | None = None) -> None:
        """Initialize experiment tracking after model/checkpoint setup."""
        resolved_config = self._config_container()
        self.tracker = create_tracker(
            resolved_config,
            enabled=self.rank == 0,
            resume_checkpoint=resume_checkpoint,
        )
        if self.tracker and self.cfg.tracking.artifacts.save_config:
            self.tracker.log_config(resolved_config)

    def _config_container(self) -> dict[str, Any]:
        """Convert Hydra config to a plain dict, falling back outside Hydra runtime."""
        try:
            config = OmegaConf.to_container(self.cfg, resolve=True)
        except Exception:
            config = OmegaConf.to_container(self.cfg, resolve=False)
        return dict(config)

    def _create_optimizer(self) -> None:
        """Create optimizer from config."""
        assert self.model is not None, "Model must be created before optimizer."

        opt_cfg = self.cfg.trainer.optimizer

        if opt_cfg.name.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
                eps=opt_cfg.eps,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

    def _create_scheduler(self) -> None:
        """Create learning rate scheduler."""
        assert self.optimizer is not None, "Optimizer must be created before scheduler."

        sched_cfg = self.cfg.trainer.scheduler
        train_cfg = self.cfg.trainer.training

        if sched_cfg.name.lower() == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg.max_steps - sched_cfg.warmup_steps,
                eta_min=self.cfg.trainer.optimizer.lr * sched_cfg.min_lr_ratio,
            )
        # TODO: Add warmup wrapper

    def _create_dataloaders(self) -> None:
        """Create train and validation data loaders."""
        from data import create_dataloader
        from data.dataset import DummyDataset, SimulationDataset

        data_cfg = self.cfg.data
        dataset_name = data_cfg.dataset.get("name", "dummy")

        train_dataset: Dataset
        val_dataset: Dataset

        if self.cfg.model.architecture.type == "vla" and dataset_name == "dummy":
            raise ValueError(
                'dataset.name="dummy" is only valid for transformer smoke tests. '
                "VLA training requires manifest-backed door episodes."
            )

        if dataset_name == "dummy":
            dummy_cfg = data_cfg.dataset.get("dummy", {})
            train_dataset = DummyDataset(
                num_samples=dummy_cfg.get("train_samples", 10000),
                seq_length=self.cfg.model.architecture.max_seq_length,
                state_dim=dummy_cfg.get("state_dim", 256),
            )
            val_dataset = DummyDataset(
                num_samples=dummy_cfg.get("val_samples", 500),
                seq_length=self.cfg.model.architecture.max_seq_length,
                state_dim=dummy_cfg.get("state_dim", 256),
            )
        elif dataset_name in {"door_episodes", "simulation"}:
            train_dataset = SimulationDataset(
                data_path=data_cfg.dataset.path,
                max_length=self._model_max_seq_length(),
                split="train",
            )
            val_dataset = SimulationDataset(
                data_path=data_cfg.dataset.path,
                max_length=self._model_max_seq_length(),
                split="val",
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.train_loader = create_dataloader(
            train_dataset,
            self.cfg,
            is_train=True,
            distributed=self.distributed,
            world_size=self.world_size,
            rank=self.rank,
        )

        self.val_loader = create_dataloader(
            val_dataset,
            self.cfg,
            is_train=False,
            distributed=self.distributed,
            world_size=self.world_size,
            rank=self.rank,
        )

    def _model_max_seq_length(self) -> int:
        """Return the active model context length across model families."""
        arch = self.cfg.model.architecture
        if hasattr(arch, "max_seq_length"):
            return int(arch.max_seq_length)
        if hasattr(self.cfg.model, "vlm") and hasattr(self.cfg.model.vlm, "max_seq_length"):
            return int(self.cfg.model.vlm.max_seq_length)
        return 1024

    def _training_step(self, batch: dict[str, object]) -> tuple[torch.Tensor, dict[str, float]]:
        """Run one forward/loss step for either VLA or transformer models."""
        assert self.model is not None, "Model not initialized. Call setup() first."

        model_for_loss = self.model.module if self.distributed else self.model

        if self.cfg.model.architecture.type == "vla":
            outputs = model_for_loss(batch)  # type: ignore[operator]
            loss = outputs["total_loss"]
            metrics = {
                "loss/total": float(loss.detach().item()),
                "loss/text": float(outputs["text_loss"].detach().item()),
                "loss/action": float(outputs["action_loss"].detach().item()),
            }
            for key, value in outputs.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    metrics[f"loss/{key}"] = float(value)
            return loss, metrics

        outputs = self.model(batch["input_ids"], batch["attention_mask"])  # type: ignore[arg-type]
        loss = model_for_loss.compute_loss(  # type: ignore[union-attr, operator]
            outputs["logits"],
            batch["labels"],
            batch["attention_mask"],
        )
        return loss, {"loss/total": float(loss.detach().item())}

    def _move_batch_to_device(self, batch: dict[str, object]) -> dict[str, object]:
        """Move tensor values in a collated batch to the trainer device."""
        moved: dict[str, object] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _infer_batch_size(self, batch: dict[str, object]) -> int:
        """Infer batch size from the first batched tensor."""
        for key in ("input_ids", "robot_states", "action_chunks"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
        return int(self.cfg.trainer.training.batch_size_per_device)

    def _log_training_metrics(
        self,
        loss: torch.Tensor,
        step_metrics: dict[str, float],
        batch: dict[str, object],
    ) -> None:
        """Send standard training metrics to the active experiment tracker."""
        if self.tracker is None:
            return

        model_for_metrics = self.model.module if self.distributed else self.model
        self.tracker.log_training_step(
            loss=float(loss.detach().item()),
            step=self.global_step,
            batch_size=self._infer_batch_size(batch),
            optimizer=self.optimizer,
            model=model_for_metrics,
            loss_ar=step_metrics.get("loss/text"),
            loss_fm=step_metrics.get("loss/action"),
            extra_metrics=step_metrics,
        )

    def train(self) -> None:
        """Main training loop."""
        assert self.model is not None, "Model not initialized. Call setup() first."
        assert self.optimizer is not None, "Optimizer not initialized. Call setup() first."
        assert self.train_loader is not None, "Train loader not initialized. Call setup() first."

        if self.rank == 0:
            log.info("Starting training...")

        train_cfg = self.cfg.trainer.training
        log_cfg = self.cfg.trainer.logging
        ckpt_cfg = self.cfg.trainer.checkpoint

        self.model.train()

        # Progress bar (only rank 0)
        pbar = None
        if self.rank == 0:
            pbar = tqdm(total=train_cfg.max_steps, desc="Training")
            pbar.update(self.global_step)

        if self.tracker is not None:
            self.tracker.start_throughput_tracking()

        try:
            while self.global_step < train_cfg.max_steps:
                for batch in self.train_loader:
                    # Move batch to device
                    batch = self._move_batch_to_device(batch)

                    # Forward pass and loss
                    loss, step_metrics = self._training_step(batch)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    if self.cfg.trainer.gradient.max_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.cfg.trainer.gradient.max_norm,
                        )

                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()

                    self.global_step += 1
                    self._log_training_metrics(loss, step_metrics, batch)

                    # Logging
                    if self.global_step % log_cfg.log_every_n_steps == 0:
                        if self.rank == 0:
                            lr = self.optimizer.param_groups[0]["lr"]
                            metric_text = ", ".join(
                                f"{key}={value:.4f}" for key, value in step_metrics.items()
                            )
                            log.info(f"Step {self.global_step}: {metric_text}, lr={lr:.2e}")

                    # Update progress bar
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(loss=f"{loss.item():.4f}")

                    # Checkpointing
                    if self.global_step % ckpt_cfg.save_every_n_steps == 0:
                        self._save_checkpoint()

                    # Validation
                    if self.global_step % self.cfg.trainer.validation.every_n_steps == 0:
                        self._validate()
                        self.model.train()

                    if self.global_step >= train_cfg.max_steps:
                        break

            # Final checkpoint
            self._save_checkpoint(final=True)

            if self.rank == 0:
                log.info("Training complete!")
        finally:
            if pbar:
                pbar.close()
            if self.tracker is not None:
                self.tracker.finish()

    def _validate(self) -> float:
        """Run validation."""
        assert self.model is not None, "Model not initialized. Call setup() first."
        assert self.val_loader is not None, "Val loader not initialized. Call setup() first."

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)
                loss, _ = self._training_step(batch)

                total_loss += loss.item()
                num_batches += 1

                if num_batches >= self.cfg.trainer.validation.num_samples:
                    break

        avg_loss = total_loss / max(num_batches, 1)

        if self.rank == 0:
            log.info(f"Validation loss: {avg_loss:.4f}")
            if self.tracker is not None:
                self.tracker.log_metrics({"val/loss": avg_loss}, step=self.global_step)

        return avg_loss

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save training checkpoint."""
        if self.rank != 0:
            return

        assert self.model is not None, "Model not initialized."
        assert self.optimizer is not None, "Optimizer not initialized."

        ckpt_dir = Path(self.cfg.paths.checkpoints)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if final:
            ckpt_path = ckpt_dir / "final.pt"
        else:
            ckpt_path = ckpt_dir / f"step_{self.global_step}.pt"

        model_state = (
            self.model.module.state_dict()  # type: ignore[union-attr]
            if self.distributed
            else self.model.state_dict()
        )

        checkpoint = {
            "step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": model_state,
            "config": self._config_container(),
        }
        if self.tracker is not None:
            wandb_run_id = self.tracker.get_run_id()
            if wandb_run_id:
                checkpoint["wandb_run_id"] = wandb_run_id

        if self.cfg.trainer.checkpoint.save_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, ckpt_path)
        log.info(f"Saved checkpoint: {ckpt_path}")

        if self.tracker is not None and self.cfg.tracking.artifacts.save_checkpoints:
            aliases = list(self.cfg.tracking.artifacts.checkpoint_aliases)
            if final:
                aliases = ["final", *aliases]
            self.tracker.log_checkpoint(ckpt_path, aliases=aliases)

    def _load_checkpoint(self, path: str) -> dict[str, Any] | None:
        """Load checkpoint to resume training."""
        assert self.model is not None, "Model not initialized."
        assert self.optimizer is not None, "Optimizer not initialized."

        if path == "latest":
            # Find latest checkpoint
            ckpt_dir = Path(self.cfg.paths.checkpoints)
            checkpoints = list(ckpt_dir.glob("step_*.pt"))
            if not checkpoints:
                log.warning("No checkpoints found, starting from scratch")
                return None
            path = str(max(checkpoints, key=lambda p: int(p.stem.split("_")[1])))

        if self.rank == 0:
            log.info(f"Loading checkpoint: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        model = self.model.module if self.distributed else self.model
        model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[union-attr]

        self.global_step = checkpoint["step"]
        self.epoch = checkpoint.get("epoch", 0)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training."""
    # Print config
    if int(os.environ.get("RANK", 0)) == 0:
        log.info("Configuration:")
        log.info(OmegaConf.to_yaml(cfg))

    # Create trainer and run
    trainer = Trainer(cfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
