"""Tests for data loading module."""

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

    def test_dataset_creation(self, tmp_path):
        """Test dataset can be created."""
        from data.dataset import SimulationDataset

        # Should work even without data files (for testing)
        dataset = SimulationDataset(
            data_path=tmp_path,
            max_length=512,
            split="train",
        )

        assert len(dataset) > 0


class TestDataLoader:
    """Test dataloader creation."""

    def test_get_dummy_dataloader(self):
        """Test dummy dataloader factory."""
        from data.loader import get_dummy_dataloader

        loader = get_dummy_dataloader(
            batch_size=4,
            seq_length=128,
            state_dim=32,
            num_samples=100,
        )

        batch = next(iter(loader))

        assert batch["input_ids"].shape == (4, 128, 32)
