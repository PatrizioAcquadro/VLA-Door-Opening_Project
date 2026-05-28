"""
Main ExperimentTracker class for W&B integration.

Provides a unified interface for experiment tracking with:
- Online mode with automatic offline fallback
- Distributed training support (rank 0 only logging)
- Metric logging with configurable intervals
- Artifact management (configs, checkpoints)
- Resume support via run IDs
"""

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

from .gpu_monitor import GPUMonitor, get_all_gpu_stats
from .metadata import get_metadata, is_main_process
from .metrics import (
    ThroughputTracker,
    compute_grad_norm,
    extract_loss_components,
    filter_metrics,
    get_amp_scale,
    get_learning_rate,
)
from .naming import generate_run_name, generate_tags_dict, get_run_group


class ExperimentTracker:
    """
    Unified experiment tracking interface using Weights & Biases.

    Features:
    - Only initializes on rank 0 in distributed settings
    - Attempts online mode first, falls back to offline if network unavailable
    - Handles metric logging with optional GPU stats
    - Manages config and checkpoint artifacts
    - Supports run resumption via stored run IDs
    """

    def __init__(
        self,
        project: str = "vla-lego",
        config: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        name: str | None = None,
        group: str | None = None,
        resume_id: str | None = None,
        mode: str = "online",
        entity: str | None = None,
        log_interval: int = 1,
        gpu_stats_interval: int = 50,
        enabled: bool = True,
        dir: str | None = None,
    ):
        """
        Initialize the experiment tracker.

        Args:
            project: W&B project name.
            config: Training configuration dictionary.
            tags: Dictionary of tags (model, dataset, objective, etc.).
            name: Run name. If None, auto-generated.
            group: W&B group for organizing related runs.
            resume_id: W&B run ID to resume from.
            mode: "online" (with fallback), "offline", or "disabled".
            entity: W&B entity (username or team).
            log_interval: Log metrics every N steps.
            gpu_stats_interval: Log GPU stats every N steps.
            enabled: Whether tracking is enabled.
            dir: Directory for W&B files.
        """
        self.project = project
        self.config = config or {}
        self.tags = tags or {}
        self.entity = entity
        self.log_interval = log_interval
        self.gpu_stats_interval = gpu_stats_interval
        self.enabled = enabled
        self._offline_mode = False
        self._run = None
        self._run_id: str | None = None
        self._step_counter = 0

        # Check if we should be active
        self._is_main = is_main_process()
        self._active = enabled and self._is_main

        # Initialize trackers
        self._throughput = ThroughputTracker(world_size=self._get_world_size())
        self._gpu_monitor = GPUMonitor() if self._active else None

        if not self._active:
            return

        # Check for disabled mode via environment
        env_mode = os.environ.get("WANDB_MODE", "").lower()
        if env_mode == "disabled" or mode == "disabled":
            self.enabled = False
            self._active = False
            return

        # Determine run name
        if name is None:
            name = generate_run_name(
                model=self.tags.get("model"),
                objective=self.tags.get("objective"),
                dataset=self.tags.get("dataset"),
            )

        # Determine group
        if group is None:
            group = get_run_group(
                experiment_group=self.tags.get("experiment_group"),
                model=self.tags.get("model"),
                objective=self.tags.get("objective"),
            )

        # Generate tags list
        tags_list = self._generate_tags_list()

        # Set up directory
        if dir is None:
            dir = os.environ.get("WANDB_DIR", "./wandb")

        # Initialize W&B
        self._init_wandb(
            project=project,
            name=name,
            group=group,
            tags=tags_list,
            resume_id=resume_id,
            mode=mode,
            dir=dir,
        )

    def _init_wandb(
        self,
        project: str,
        name: str,
        group: str,
        tags: list[str],
        resume_id: str | None,
        mode: str,
        dir: str,
    ) -> None:
        """Initialize W&B with fallback logic."""
        try:
            import wandb
        except ImportError:
            warnings.warn(
                "wandb not installed. Experiment tracking disabled. "
                "Install with: pip install wandb",
                stacklevel=2,
            )
            self._active = False
            return

        # Collect metadata
        metadata = get_metadata(
            config=self.config,
            seeds=self.config.get("seeds"),
        )

        # Merge config with metadata
        full_config = {
            **self.config,
            "tags": generate_tags_dict(**self.tags),
            "metadata": metadata,
        }

        # Determine resume settings
        resume = None
        run_id = resume_id
        if resume_id:
            resume = "must"
        elif os.environ.get("WANDB_RESUME"):
            resume = os.environ.get("WANDB_RESUME")

        # Try online mode first with timeout
        init_kwargs = {
            "project": project,
            "entity": self.entity,
            "name": name,
            "group": group,
            "tags": tags,
            "config": full_config,
            "dir": dir,
            "resume": resume,
            "id": run_id,
            "reinit": True,
        }

        try:
            if mode == "offline":
                # Explicit offline mode
                self._run = wandb.init(mode="offline", **init_kwargs)
                self._offline_mode = True
                print("[ExperimentTracker] Running in offline mode.")
            else:
                # Try online first
                try:
                    # Set a timeout for network operations
                    os.environ.setdefault("WANDB_HTTP_TIMEOUT", "30")
                    self._run = wandb.init(mode="online", **init_kwargs)
                    self._offline_mode = False
                    print(f"[ExperimentTracker] Online: {self._run.get_url()}")
                except (wandb.errors.CommError, Exception) as e:
                    # Fallback to offline mode
                    warnings.warn(
                        f"W&B online mode failed ({e}). Falling back to offline mode.",
                        stacklevel=2,
                    )
                    self._run = wandb.init(mode="offline", **init_kwargs)
                    self._offline_mode = True
                    print("[ExperimentTracker] Offline mode. Run 'wandb sync' to upload later.")

            self._run_id = self._run.id

        except Exception as e:
            warnings.warn(f"W&B initialization failed: {e}. Tracking disabled.", stacklevel=2)
            self._active = False

    def _generate_tags_list(self) -> list[str]:
        """Generate W&B tags list from tags dictionary."""
        tags_list = []
        for key, value in self.tags.items():
            if value is not None:
                tags_list.append(f"{key}:{value}")
        return tags_list

    def _get_world_size(self) -> int:
        """Get distributed world size."""
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                return dist.get_world_size()
        except ImportError:
            pass
        return 1

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics to log.
            step: Global step number. If None, uses internal counter.
            commit: Whether to commit the log (vs. accumulate).
        """
        if not self._active:
            return

        if step is None:
            step = self._step_counter

        # Filter metrics to valid values
        filtered = filter_metrics(metrics)

        if filtered and self._run is not None:
            try:
                import wandb

                wandb.log(filtered, step=step, commit=commit)
            except Exception as e:
                warnings.warn(f"Failed to log metrics: {e}", stacklevel=2)

    def log_training_step(
        self,
        loss: float | dict[str, Any],
        step: int,
        batch_size: int = 0,
        optimizer: Any | None = None,
        model: Any | None = None,
        scaler: Any | None = None,
        loss_ar: float | None = None,
        loss_fm: float | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a complete training step with all standard metrics.

        Args:
            loss: Total loss or dictionary of losses.
            step: Global step number.
            batch_size: Batch size for throughput calculation.
            optimizer: Optimizer for learning rate extraction.
            model: Model for gradient norm computation.
            scaler: GradScaler for AMP scale value.
            loss_ar: Explicit AR loss value.
            loss_fm: Explicit FM loss value.
            extra_metrics: Additional metrics to log.
        """
        self._step_counter = step

        if not self._active:
            return

        # Track throughput
        self._throughput.step(batch_size=batch_size)

        # Check if we should log this step
        should_log_metrics = step % self.log_interval == 0
        should_log_gpu = step % self.gpu_stats_interval == 0

        if not should_log_metrics and not should_log_gpu:
            return

        metrics = {}

        if should_log_metrics:
            # Loss components
            loss_metrics = extract_loss_components(loss, loss_ar=loss_ar, loss_fm=loss_fm)
            metrics.update(loss_metrics)

            # Learning rate
            if optimizer is not None:
                metrics["train/lr"] = get_learning_rate(optimizer)

            # Gradient norm
            if model is not None:
                metrics["train/grad_norm"] = compute_grad_norm(model)

            # AMP scaler
            if scaler is not None:
                amp_scale = get_amp_scale(scaler)
                if amp_scale is not None:
                    metrics["train/amp_scale"] = amp_scale

            # Throughput
            throughput = self._throughput.get_interval_throughput()
            metrics.update(throughput)

            # Extra metrics
            if extra_metrics:
                metrics.update(extra_metrics)

        if should_log_gpu:
            # GPU stats
            gpu_stats = get_all_gpu_stats()
            metrics.update(gpu_stats)

        self.log_metrics(metrics, step=step)

    def log_config(self, config: dict[str, Any] | None = None) -> None:
        """
        Save configuration as a W&B artifact.

        Args:
            config: Configuration dictionary. If None, uses init config.
        """
        if not self._active or self._run is None:
            return

        if config is None:
            config = self.config

        try:
            import wandb
        except ImportError as e:
            warnings.warn(f"wandb not installed: {e}. Cannot log config artifact.", stacklevel=2)
            return

        try:
            import yaml
        except ImportError as e:
            warnings.warn(f"pyyaml not installed: {e}. Cannot log config artifact.", stacklevel=2)
            return

        try:
            # Create artifact
            artifact = wandb.Artifact(
                name=f"config-{self._run_id}",
                type="config",
                description="Training configuration",
            )

            # Write config to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(config, f, default_flow_style=False)
                temp_path = f.name

            artifact.add_file(temp_path, name="config.yaml")
            self._run.log_artifact(artifact)

            # Clean up
            os.unlink(temp_path)

            print("[ExperimentTracker] Config artifact saved.")

        except Exception as e:
            warnings.warn(f"Failed to log config artifact: {e}", stacklevel=2)

    def _has_symlink_loop(self, path: Path, visited: set | None = None) -> bool:
        """
        Check if path contains self-referential symlinks.

        Args:
            path: Path to check.
            visited: Set of already visited resolved paths.

        Returns:
            True if a symlink loop is detected, False otherwise.
        """
        if visited is None:
            visited = set()

        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            # Broken or recursive symlink
            return True

        if resolved in visited:
            return True
        visited.add(resolved)

        if path.is_dir():
            try:
                for child in path.iterdir():
                    if child.is_symlink():
                        try:
                            target = child.resolve()
                            # Check if symlink points back to parent or already visited
                            if target == resolved or target in visited:
                                return True
                        except (OSError, RuntimeError):
                            # Broken or recursive symlink
                            return True
            except PermissionError:
                pass

        return False

    def _calculate_dir_size(self, path: Path, follow_symlinks: bool = False) -> int:
        """
        Calculate total size of directory in bytes.

        Args:
            path: Path to file or directory.
            follow_symlinks: Whether to follow symlinks (default False for safety).

        Returns:
            Total size in bytes.
        """
        total = 0

        if path.is_file():
            try:
                return path.stat().st_size
            except OSError:
                return 0

        if not path.is_dir():
            return 0

        try:
            for item in path.rglob("*"):
                if item.is_file():
                    # Skip symlinks unless follow_symlinks is True
                    if item.is_symlink() and not follow_symlinks:
                        continue
                    try:
                        total += item.stat().st_size
                    except OSError:
                        pass
        except (OSError, RuntimeError):
            # Handle permission errors or too deep recursion
            pass

        return total

    def log_checkpoint(
        self,
        checkpoint_path: str | Path,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        """
        Log a checkpoint as a W&B artifact.

        By default, checkpoint artifacts are NOT uploaded to minimize bandwidth
        usage on HPC clusters. Set WANDB_LOG_MODEL=1 to enable uploads.

        Args:
            checkpoint_path: Path to checkpoint directory or file.
            aliases: List of aliases (e.g., ["latest", "best"]).
            metadata: Additional metadata to attach.
            force: Bypass WANDB_LOG_MODEL check and upload anyway.
        """
        if not self._active or self._run is None:
            return

        # Check if checkpoint logging is enabled via environment variable
        # Default: WANDB_LOG_MODEL=0 (disabled) to minimize HPC bandwidth usage
        log_model = os.environ.get("WANDB_LOG_MODEL", "0") == "1"

        if not log_model and not force:
            # Print warning once per tracker instance
            if not hasattr(self, "_checkpoint_warning_shown"):
                print(
                    "[ExperimentTracker] Checkpoint artifact logging disabled "
                    "(set WANDB_LOG_MODEL=1 to enable). Metrics and config still logged."
                )
                self._checkpoint_warning_shown = True
            return

        aliases = aliases or ["latest"]
        checkpoint_path = Path(checkpoint_path).resolve()

        # Safety: check for symlink loops before processing
        if self._has_symlink_loop(checkpoint_path):
            warnings.warn(
                f"Symlink loop detected in checkpoint path, skipping artifact: {checkpoint_path}",
                stacklevel=2,
            )
            return

        # Calculate and display upload size (without following symlinks)
        total_size = self._calculate_dir_size(checkpoint_path, follow_symlinks=False)
        size_gb = total_size / (1024**3)

        print(f"[ExperimentTracker] Checkpoint artifact: {size_gb:.2f} GB")

        if size_gb > 5.0:
            print(
                f"  WARNING: Large upload ({size_gb:.1f} GB). "
                "Consider consolidating with zero_to_fp32.py first."
            )

        try:
            import wandb

            artifact = wandb.Artifact(
                name=f"checkpoint-{self._run.name}",
                type="model",
                description="Model checkpoint",
                metadata=metadata or {},
            )

            if checkpoint_path.is_dir():
                # Add directory without following symlinks
                # Note: wandb.Artifact.add_dir doesn't have follow_symlinks param,
                # but our symlink check above should catch problematic cases
                artifact.add_dir(str(checkpoint_path))
            else:
                artifact.add_file(str(checkpoint_path))

            self._run.log_artifact(artifact, aliases=aliases)

            print(f"[ExperimentTracker] Checkpoint artifact saved: {aliases}")

        except Exception as e:
            warnings.warn(f"Failed to log checkpoint artifact: {e}", stacklevel=2)

    def get_run_id(self) -> str | None:
        """
        Get the W&B run ID for checkpoint storage.

        Returns:
            W&B run ID string, or None if not active.
        """
        return self._run_id

    def get_run_url(self) -> str | None:
        """
        Get the W&B run URL.

        Returns:
            URL string, or None if not active or offline.
        """
        if self._run is not None and not self._offline_mode:
            try:
                return self._run.get_url()
            except Exception:
                pass
        return None

    def is_active(self) -> bool:
        """Check if tracking is active."""
        return self._active

    def is_offline(self) -> bool:
        """Check if running in offline mode."""
        return self._offline_mode

    def start_throughput_tracking(self) -> None:
        """Start or reset throughput tracking."""
        self._throughput.start()

    def finish(self) -> None:
        """
        Finish the experiment run and sync data.

        Prints instructions for offline sync if needed.
        """
        if not self._active or self._run is None:
            return

        try:
            self._run.finish()

            if self._offline_mode:
                print(
                    "\n[ExperimentTracker] Run completed in offline mode.\n"
                    "To sync your data, run from a node with internet access:\n"
                    f"  wandb sync {os.path.join(os.getcwd(), 'wandb')}\n"
                )
            else:
                print(f"\n[ExperimentTracker] Run completed: {self.get_run_url()}\n")

        except Exception as e:
            warnings.warn(f"Error finishing W&B run: {e}", stacklevel=2)

    def __enter__(self) -> "ExperimentTracker":
        """Context manager entry."""
        self.start_throughput_tracking()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.finish()


def create_tracker(
    config: dict[str, Any],
    enabled: bool = True,
    resume_checkpoint: dict[str, Any] | None = None,
) -> ExperimentTracker:
    """
    Factory function to create an ExperimentTracker from config.

    Args:
        config: Full training configuration.
        enabled: Whether tracking is enabled.
        resume_checkpoint: Checkpoint dict for resume (contains wandb_run_id).

    Returns:
        Configured ExperimentTracker instance.
    """
    # Extract tracking config
    tracking_config = config.get("tracking", {})

    # Get resume ID from checkpoint if available
    resume_id = None
    if resume_checkpoint:
        resume_id = resume_checkpoint.get("wandb_run_id")

    return ExperimentTracker(
        project=tracking_config.get("project", "vla-lego"),
        config=config,
        tags={
            "model": tracking_config.get("tags", {}).get("model"),
            "dataset": tracking_config.get("tags", {}).get("dataset"),
            "objective": tracking_config.get("tags", {}).get("objective"),
            "experiment_group": tracking_config.get("tags", {}).get("experiment_group"),
        },
        name=tracking_config.get("run", {}).get("name"),
        resume_id=resume_id,
        mode=tracking_config.get("mode", "online"),
        entity=tracking_config.get("entity"),
        log_interval=tracking_config.get("log_interval", 10),
        gpu_stats_interval=tracking_config.get("metrics", {}).get("gpu_stats_interval", 50),
        enabled=enabled and tracking_config.get("enabled", True),
    )
