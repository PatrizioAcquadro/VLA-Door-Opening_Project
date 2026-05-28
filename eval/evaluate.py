"""Evaluation entry points for VLA door-opening runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module
from tqdm import tqdm

from sim.door_metrics import DoorMetricConfig, DoorMetricsTracker, flatten_door_metrics

log = logging.getLogger(__name__)


class Evaluator:
    """Model and rollout evaluator for the active door-opening task."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model: Module | None = None

    def setup(self, checkpoint_path: str) -> None:
        """Load a model from checkpoint."""
        log.info("Loading model from: %s", checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        saved_cfg = OmegaConf.create(checkpoint["config"])

        from models import get_model

        self.model = get_model(saved_cfg)
        if saved_cfg.model.architecture.type not in {"vlm", "vla"}:
            self.model = self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.cfg = saved_cfg

    def evaluate(self, test_loader: Any) -> dict[str, float]:
        """Run model loss evaluation on a provided test dataloader."""
        assert self.model is not None, "Model not loaded. Call setup() first."

        total_loss = 0.0
        total_text_loss = 0.0
        total_action_loss = 0.0
        num_batches = 0
        model_type = self.cfg.model.architecture.type

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = _move_batch_to_device(batch, self.device)

                if model_type == "vla":
                    outputs = self.model(batch)  # type: ignore[operator]
                    loss = outputs["total_loss"]
                    total_text_loss += float(outputs["text_loss"].item())
                    total_action_loss += float(outputs["action_loss"].item())
                else:
                    outputs = self.model(  # type: ignore[misc]
                        batch["input_ids"], batch.get("attention_mask")
                    )
                    loss = self.model.compute_loss(  # type: ignore[union-attr, operator]
                        outputs["logits"],
                        batch["labels"],
                        batch.get("attention_mask"),
                    )

                total_loss += float(loss.item())
                num_batches += 1

        metrics = {
            "loss": total_loss / max(num_batches, 1),
            "num_batches": float(num_batches),
        }
        if model_type == "vla":
            metrics["text_loss"] = total_text_loss / max(num_batches, 1)
            metrics["action_loss"] = total_action_loss / max(num_batches, 1)
        return metrics


def evaluate_door_rollout(
    rollout_path: str | Path, cfg: DictConfig | None = None
) -> dict[str, float]:
    """Evaluate door task metrics from a saved rollout JSON file.

    Expected JSON shape:
        ``{"steps": [{"door_angle": ..., "handle_touch": ...}, ...]}``
    A bare list of step dicts is also accepted.
    """
    path = Path(rollout_path)
    payload = json.loads(path.read_text())
    steps = payload.get("steps", payload) if isinstance(payload, dict) else payload
    if not isinstance(steps, list):
        raise ValueError("Rollout JSON must be a list or contain a 'steps' list")

    tracker = DoorMetricsTracker(DoorMetricConfig.from_cfg(cfg.sim if cfg else None))
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Each rollout step must be a dict")
        tracker.update_values(
            door_angle=float(step.get("door_angle", 0.0)),
            door_angular_velocity=float(step.get("door_angular_velocity", 0.0)),
            handle_touch=float(step.get("handle_touch", 0.0)),
            force_limit_violation=bool(step.get("force_limit_violation", False)),
            recovery_success=bool(step.get("recovery_success", False)),
        )

    return flatten_door_metrics(tracker.finalize())


def main() -> None:
    """CLI evaluation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate VLA door-opening outputs")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--rollout", type=str, default=None)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    if args.rollout is None and args.checkpoint is None:
        raise SystemExit("Provide --rollout for door metrics or --checkpoint for model setup.")

    cfg: DictConfig | None = None
    results: dict[str, float] = {}

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        cfg = OmegaConf.create(checkpoint["config"])
        results["checkpoint_loaded"] = 1.0

    if args.rollout:
        results.update(evaluate_door_rollout(args.rollout, cfg))

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved results to {output_path}")


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


if __name__ == "__main__":
    main()
