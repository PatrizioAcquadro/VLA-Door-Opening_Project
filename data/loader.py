"""DataLoader creation utilities."""

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def create_dataloader(
    dataset: Dataset,
    cfg: DictConfig,
    is_train: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Create a DataLoader with appropriate settings.

    Args:
        dataset: PyTorch dataset
        cfg: Configuration (expects cfg.data.dataloader section)
        is_train: Whether this is for training (affects shuffling)
        distributed: Whether using distributed training
        world_size: Number of distributed processes
        rank: Current process rank

    Returns:
        Configured DataLoader
    """
    dl_cfg = cfg.data.dataloader

    # Sampler for distributed training
    sampler: DistributedSampler | None = None
    shuffle = dl_cfg.shuffle_train if is_train else dl_cfg.shuffle_val

    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=cfg.trainer.training.batch_size_per_device,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=dl_cfg.num_workers,
        pin_memory=dl_cfg.pin_memory,
        drop_last=dl_cfg.drop_last,
        prefetch_factor=dl_cfg.prefetch_factor if dl_cfg.num_workers > 0 else None,
        persistent_workers=dl_cfg.persistent_workers if dl_cfg.num_workers > 0 else False,
    )


def get_dummy_dataloader(
    batch_size: int = 8,
    seq_length: int = 512,
    state_dim: int = 256,
    num_samples: int = 1000,
) -> DataLoader:
    """Create a dummy dataloader for testing.

    No config required - useful for smoke tests.
    """
    from data.dataset import DummyDataset

    dataset = DummyDataset(
        num_samples=num_samples,
        seq_length=seq_length,
        state_dim=state_dim,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
