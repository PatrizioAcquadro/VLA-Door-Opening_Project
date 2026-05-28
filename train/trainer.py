"""Main trainer module with Hydra integration."""

import logging
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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

        # Create model
        from models import count_parameters, format_params, get_model

        self.model = get_model(self.cfg).to(self.device)

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

        # Create data loaders
        self._create_dataloaders()

        # Resume from checkpoint if specified
        if self.cfg.trainer.checkpoint.resume_from:
            self._load_checkpoint(self.cfg.trainer.checkpoint.resume_from)

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
        else:
            train_dataset = SimulationDataset(
                data_path=data_cfg.dataset.path,
                max_length=self.cfg.model.architecture.max_seq_length,
                split="train",
            )
            val_dataset = SimulationDataset(
                data_path=data_cfg.dataset.path,
                max_length=self.cfg.model.architecture.max_seq_length,
                split="val",
            )

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

        while self.global_step < train_cfg.max_steps:
            for batch in self.train_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(batch["input_ids"], batch["attention_mask"])

                # Compute loss
                model_for_loss = self.model.module if self.distributed else self.model
                loss = model_for_loss.compute_loss(  # type: ignore[union-attr, operator]
                    outputs["logits"],
                    batch["labels"],
                    batch["attention_mask"],
                )

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

                # Logging
                if self.global_step % log_cfg.log_every_n_steps == 0:
                    if self.rank == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        log.info(f"Step {self.global_step}: loss={loss.item():.4f}, lr={lr:.2e}")

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

        if pbar:
            pbar.close()

        # Final checkpoint
        self._save_checkpoint(final=True)

        if self.rank == 0:
            log.info("Training complete!")

    def _validate(self) -> float:
        """Run validation."""
        assert self.model is not None, "Model not initialized. Call setup() first."
        assert self.val_loader is not None, "Val loader not initialized. Call setup() first."

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(batch["input_ids"], batch["attention_mask"])

                model_for_loss = self.model.module if self.distributed else self.model
                loss = model_for_loss.compute_loss(  # type: ignore[union-attr, operator]
                    outputs["logits"],
                    batch["labels"],
                    batch["attention_mask"],
                )

                total_loss += loss.item()
                num_batches += 1

                if num_batches >= self.cfg.trainer.validation.num_samples:
                    break

        avg_loss = total_loss / max(num_batches, 1)

        if self.rank == 0:
            log.info(f"Validation loss: {avg_loss:.4f}")

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
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }

        if self.cfg.trainer.checkpoint.save_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, ckpt_path)
        log.info(f"Saved checkpoint: {ckpt_path}")

    def _load_checkpoint(self, path: str) -> None:
        """Load checkpoint to resume training."""
        assert self.model is not None, "Model not initialized."
        assert self.optimizer is not None, "Optimizer not initialized."

        if path == "latest":
            # Find latest checkpoint
            ckpt_dir = Path(self.cfg.paths.checkpoints)
            checkpoints = list(ckpt_dir.glob("step_*.pt"))
            if not checkpoints:
                log.warning("No checkpoints found, starting from scratch")
                return
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
