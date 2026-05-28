"""
Experiment tracking module for VLA-LEGO training.

This module provides comprehensive experiment tracking using Weights & Biases,
with support for distributed training, automatic offline fallback, and
standardized metric/artifact management.

Quick Start:
    from tracking import ExperimentTracker

    # Initialize tracker (only rank 0 logs)
    tracker = ExperimentTracker(
        project="vla-lego",
        config=config,
        tags={"model": "eo1", "dataset": "lego", "objective": "ar+fm"},
    )

    # Log training steps
    for step in range(num_steps):
        loss = train_step(...)
        tracker.log_training_step(
            loss=loss,
            step=step,
            batch_size=batch_size,
            optimizer=optimizer,
            model=model,
        )

    # Save checkpoint with run ID for resume
    client_state = {"wandb_run_id": tracker.get_run_id()}
    model_engine.save_checkpoint(ckpt_dir, client_state=client_state)
    tracker.log_checkpoint(ckpt_dir)

    # Finish run
    tracker.finish()

Environment Variables:
    WANDB_MODE: Set to "disabled" to disable tracking entirely.
    WANDB_DIR: Directory for W&B files (default: ./wandb).
    WANDB_ENTITY: Default W&B entity (username or team).

See tracking/README.md for full documentation.
"""

from .experiment import ExperimentTracker, create_tracker
from .gpu_monitor import (
    GPUMonitor,
    get_all_gpu_stats,
    get_gpu_count,
    get_gpu_memory_stats,
    get_gpu_utilization,
    reset_peak_memory_stats,
)
from .metadata import (
    get_distributed_info,
    get_environment_info,
    get_git_info,
    get_metadata,
    get_slurm_info,
    is_main_process,
    set_seeds,
)
from .metrics import (
    ThroughputTracker,
    aggregate_metrics,
    compute_grad_norm,
    extract_loss_components,
    filter_metrics,
    get_amp_scale,
    get_learning_rate,
)
from .naming import (
    generate_run_name,
    generate_tags,
    generate_tags_dict,
    get_run_group,
    validate_tags,
)

__all__ = [
    # Main tracker
    "ExperimentTracker",
    "create_tracker",
    # Metadata
    "get_metadata",
    "get_git_info",
    "get_environment_info",
    "get_slurm_info",
    "get_distributed_info",
    "set_seeds",
    "is_main_process",
    # Naming
    "generate_run_name",
    "generate_tags",
    "generate_tags_dict",
    "get_run_group",
    "validate_tags",
    # Metrics
    "compute_grad_norm",
    "get_learning_rate",
    "get_amp_scale",
    "extract_loss_components",
    "ThroughputTracker",
    "filter_metrics",
    "aggregate_metrics",
    # GPU monitoring
    "get_gpu_count",
    "get_gpu_memory_stats",
    "get_gpu_utilization",
    "get_all_gpu_stats",
    "reset_peak_memory_stats",
    "GPUMonitor",
]

__version__ = "0.1.0"
