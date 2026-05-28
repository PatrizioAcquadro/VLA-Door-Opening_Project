"""Tests for data loading module."""

import pytest
import torch
from torch.utils.data import DataLoader


class TestDummyDataset:
    """Test DummyDataset for smoke testing."""

    def test_dataset_length(self):
        """Test dataset returns correct length."""
        from data.dataset import DummyDataset

        dataset = DummyDataset(num_samples=100)
        assert len(dataset) == 100

    def test_dataset_getitem(self):
        """Test dataset returns correct item structure."""
        from data.dataset import DummyDataset

        dataset = DummyDataset(
            num_samples=10,
            seq_length=128,
            state_dim=64,
        )

        item = dataset[0]

        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item

        assert item["input_ids"].shape == (128, 64)
        assert item["labels"].shape == (128, 64)
        assert item["attention_mask"].shape == (128,)

    def test_dataset_iteration(self):
        """Test dataset works with DataLoader."""
        from data.dataset import DummyDataset

        dataset = DummyDataset(num_samples=100)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        batch = next(iter(loader))

        assert batch["input_ids"].shape[0] == 8


class TestSimulationDataset:
    """Test SimulationDataset."""

    def test_missing_manifest_raises(self, tmp_path):
        """Door episode data must fail fast when manifests are missing."""
        from data.dataset import SimulationDataset

        with pytest.raises(FileNotFoundError):
            SimulationDataset(
                data_path=tmp_path,
                max_length=512,
                split="train",
            )

    def test_manifest_backed_sample(self, tmp_path):
        """Dataset loads a real VLA sample from a manifest entry."""
        from data.dataset import SimulationDataset

        sample = {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "robot_states": torch.zeros(1, 52),
            "action_chunks": torch.zeros(1, 16, 17),
            "chunk_masks": torch.ones(1, 16),
            "text_labels": torch.tensor([1, 2, 3, 4]),
        }
        sample_path = tmp_path / "sample_000.pt"
        torch.save(sample, sample_path)
        (tmp_path / "train_manifest.json").write_text('[{"path": "sample_000.pt"}]')

        dataset = SimulationDataset(data_path=tmp_path, max_length=3, split="train")
        item = dataset[0]

        assert len(dataset) == 1
        assert set(sample).issubset(item.keys())
        assert item["input_ids"].shape == (3,)
        assert item["attention_mask"].shape == (20,)
        assert item["robot_states"].shape[-1] == 52
        assert item["action_chunks"].shape[-1] == 17
