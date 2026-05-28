"""Evaluation module."""

import logging

import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module
from tqdm import tqdm

log = logging.getLogger(__name__)


class Evaluator:
    """Model evaluator.

    Handles:
    - Loading model from checkpoint
    - Running evaluation metrics
    - Saving results
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model: Module | None = None

    def setup(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        log.info(f"Loading model from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model from saved config
        saved_cfg = OmegaConf.create(checkpoint["config"])

        from models import get_model

        self.model = get_model(saved_cfg).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def evaluate(self, test_loader) -> dict[str, float]:
        """Run evaluation on test set."""
        assert self.model is not None, "Model not loaded. Call setup() first."

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(batch["input_ids"], batch["attention_mask"])
                loss = self.model.compute_loss(  # type: ignore[union-attr, operator]
                    outputs["logits"],
                    batch["labels"],
                    batch["attention_mask"],
                )

                total_loss += loss.item()
                num_batches += 1

        metrics = {
            "loss": total_loss / max(num_batches, 1),
            "num_samples": num_batches * test_loader.batch_size,
        }

        return metrics


def main() -> None:
    """Main evaluation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    # TODO: Implement full evaluation pipeline
    print(f"Evaluating checkpoint: {args.checkpoint}")
    print("Evaluation not yet implemented")


if __name__ == "__main__":
    main()
