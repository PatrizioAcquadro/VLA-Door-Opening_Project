"""Dataset classes for simulation data."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    """Dataset for loading simulation sequences.

    Args:
        data_path: Path to processed data directory
        max_length: Maximum sequence length
        split: One of "train", "val", "test"
    """

    def __init__(
        self,
        data_path: str | Path,
        max_length: int = 1024,
        split: str = "train",
    ) -> None:
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.split = split

        # Load data index/manifest
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load data manifest/index."""
        manifest_path = self.data_path / f"{self.split}_manifest.json"

        if not manifest_path.exists():
            # For now, create dummy data for testing
            self._samples = list(range(1000))
        else:
            import json

            with open(manifest_path) as f:
                self._samples = json.load(f)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns:
            Dictionary containing:
                - input_ids: Input token IDs or state vectors
                - labels: Target labels
                - attention_mask: Attention mask
        """
        # TODO: Replace with actual data loading
        # For now, return dummy tensors for testing
        seq_len = min(self.max_length, 512)

        return {
            "input_ids": torch.randn(seq_len, 256),  # [seq_len, state_dim]
            "labels": torch.randn(seq_len, 256),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
        }


class DummyDataset(Dataset):
    """Dummy dataset for smoke testing.

    Generates random data without needing actual files.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 512,
        state_dim: int = 256,
    ) -> None:
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.state_dim = state_dim

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.randn(self.seq_length, self.state_dim),
            "labels": torch.randn(self.seq_length, self.state_dim),
            "attention_mask": torch.ones(self.seq_length, dtype=torch.long),
        }
