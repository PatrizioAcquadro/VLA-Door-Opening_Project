# Experiment Tracking Module

End-to-end experiment tracking for VLA-LEGO training runs using Weights & Biases (W&B).

## Features

- **Distributed-safe**: Only rank 0 logs metrics (no duplicates)
- **Online with offline fallback**: Attempts real-time sync, falls back to local logging
- **Standard metrics**: Loss, learning rate, gradient norm, AMP scale, GPU stats, throughput
- **Reproducibility metadata**: Git info, seeds, config, SLURM job details
- **Artifacts**: Config files and checkpoints with versioning
- **Resume support**: W&B run ID stored in checkpoints

## Quick Start

```python
from tracking import ExperimentTracker

# Initialize tracker (only rank 0 logs)
tracker = ExperimentTracker(
    project="vla-lego",
    config=config,
    tags={
        "model": "eo1",
        "dataset": "lego",
        "objective": "ar+fm",
        "experiment_group": "baseline",
    },
)

# Log config artifact at start
tracker.log_config(config)

# Training loop
for step in range(num_steps):
    loss = model_engine(x)
    model_engine.backward(loss)
    model_engine.step()

    # Log training step with all metrics
    tracker.log_training_step(
        loss=loss.item(),
        step=step,
        batch_size=batch_size,
        optimizer=optimizer,
        model=model_engine.module,
    )

    # Save checkpoint with W&B run ID for resume
    if step % save_interval == 0:
        client_state = {"wandb_run_id": tracker.get_run_id()}
        model_engine.save_checkpoint(ckpt_dir, client_state=client_state)
        tracker.log_checkpoint(ckpt_dir, aliases=["latest"])

# Finish run
tracker.finish()
```

## Installation

The tracking module requires `wandb` which is already installed in the `vla_lego_env` conda environment:

```bash
source /scratch/gilbreth/$USER/vla-lego/activate_env.sh
```

To log in to W&B (one-time setup):

```bash
wandb login
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_MODE` | `online` | Set to `disabled` to disable tracking |
| `WANDB_LOG_MODEL` | `0` | Set to `1` to enable checkpoint artifact uploads |
| `WANDB_DIR` | `./wandb` | Directory for W&B files |
| `WANDB_ENTITY` | - | Default W&B entity (username/team) |
| `WANDB_PROJECT` | `vla-lego` | Default project name |

### Programmatic Configuration

```python
tracker = ExperimentTracker(
    project="vla-lego",           # W&B project name
    config=config,                 # Training config dict
    tags={...},                    # Tags for filtering
    name="custom-run-name",        # Optional custom run name
    mode="online",                 # "online", "offline", or "disabled"
    entity="my-team",              # W&B entity
    log_interval=10,               # Log metrics every N steps
    gpu_stats_interval=50,         # GPU stats every N steps
)
```

## Metrics Reference

### Training Metrics (logged at `log_interval`)

| Metric | Key | Description |
|--------|-----|-------------|
| Total Loss | `loss/total` | Combined training loss |
| AR Loss | `loss/ar` | Autoregressive component (if available) |
| FM Loss | `loss/fm` | Flow matching component (if available) |
| Learning Rate | `train/lr` | Current learning rate |
| Gradient Norm | `train/grad_norm` | L2 norm of gradients |
| AMP Scale | `train/amp_scale` | GradScaler value (if mixed precision) |

### Throughput Metrics (logged at `log_interval`)

| Metric | Key | Description |
|--------|-----|-------------|
| Steps/sec | `perf/steps_per_sec` | Training steps per second |
| Samples/sec | `perf/samples_per_sec` | Samples processed per second (global) |

### GPU Metrics (logged at `gpu_stats_interval`)

| Metric | Key | Description |
|--------|-----|-------------|
| GPU Utilization | `gpu/utilization` | GPU compute utilization (%) |
| Memory Used | `gpu/memory_used_gb` | Current GPU memory (GB) |
| Memory Peak | `gpu/memory_peak_gb` | Peak GPU memory (GB) |
| Temperature | `gpu/temperature` | GPU temperature (°C) |

### Metadata (logged once at start)

- Git commit, branch, dirty state
- Random seeds (Python, NumPy, PyTorch, CUDA)
- Full training configuration
- SLURM job ID, node list, partition
- PyTorch/CUDA/DeepSpeed versions

## Run Naming Convention

Auto-generated run names follow this format:

```
{model}_{objective}_{dataset}_{YYYYMMDD}_{HHMMSS}_{git_short}
```

Example: `eo1_ar+fm_lego_20260123_143052_a1b2c3`

### Required Tags

| Tag | Description | Examples |
|-----|-------------|----------|
| `model` | Model architecture | `eo1`, `vla-base` |
| `dataset` | Training dataset | `lego`, `lego-bimanual` |
| `objective` | Training objective | `ar`, `fm`, `ar+fm` |
| `experiment_group` | Experiment group | `baseline`, `ablation-lr` |

## Artifacts

### Config Artifact

Saved automatically when `tracker.log_config(config)` is called:
- Type: `config`
- Format: YAML file
- Contains: Full resolved training configuration

### Checkpoint Artifacts

Saved when `tracker.log_checkpoint(path, aliases=["latest"])` is called:
- Type: `model`
- Contains: Model weights, optimizer state, step
- Aliases: `latest`, `best`, `stepN`

## Checkpoint Artifact Policy

By default, checkpoint artifacts are **NOT uploaded** to W&B to minimize bandwidth usage on HPC clusters. DeepSpeed ZeRO checkpoints can be 10+ GB each.

### Default Behavior (WANDB_LOG_MODEL=0)

- Metrics and config are logged normally
- `tracker.log_checkpoint()` prints a one-time info message but does not upload
- W&B sync remains fast (~50 KB for offline runs)

### Enable Checkpoint Uploads (WANDB_LOG_MODEL=1)

```bash
export WANDB_LOG_MODEL=1
```

When enabled:
- Upload size is displayed before upload begins
- Large uploads (>5 GB) trigger a warning
- Symlink loops are detected and skipped for safety

### Best Practice for Final Models

Instead of uploading full DeepSpeed ZeRO checkpoints (which include optimizer states and shards), consolidate to a single FP32 file first:

```bash
# Consolidate ZeRO checkpoint to single file
python zero_to_fp32.py checkpoints/step10000 model_final.pt

# Then upload the consolidated model
tracker.log_checkpoint("model_final.pt", aliases=["final"], force=True)
```

### Force Upload for Specific Checkpoints

To bypass the `WANDB_LOG_MODEL` check for a specific checkpoint:

```python
# Force upload regardless of WANDB_LOG_MODEL setting
tracker.log_checkpoint(ckpt_path, aliases=["best"], force=True)
```

## Resume Support

To resume a run and continue logging to the same W&B run:

```python
# Load checkpoint
_, client_state = model_engine.load_checkpoint(ckpt_dir)
resume_id = client_state.get("wandb_run_id")

# Resume tracker with same run ID
tracker = ExperimentTracker(
    project="vla-lego",
    config=config,
    resume_id=resume_id,  # Continues existing run
)
```

## Offline Mode

If the cluster doesn't have internet access, tracking automatically falls back to offline mode:

```
[ExperimentTracker] Offline mode. Run 'wandb sync' to upload later.
```

To sync offline runs later (from a node with internet):

```bash
wandb sync ./wandb/offline-run-*
```

To force offline mode:

```bash
export WANDB_MODE=offline
```

## Disable Tracking

To disable tracking entirely:

```bash
export WANDB_MODE=disabled
```

Or programmatically:

```python
tracker = ExperimentTracker(enabled=False, ...)
```

## View Results

After runs complete:

```
https://wandb.ai/{entity}/vla-lego
```

Filter by tags:
- `model:eo1`
- `dataset:lego`
- `experiment_group:baseline`

## Cluster Usage

### Submit Tracking Demo

```bash
cd gilbreth_phase0
sbatch job_templates/05_smoke_tracking.sh
```

### Monitor Job

```bash
squeue -u $USER
cat logs/smoke_tracking_*.out
```

### Check W&B Run

After job completes, check the W&B dashboard or look for the URL in the logs:

```
View run at: https://wandb.ai/...
```

## Troubleshooting

### "wandb not installed"

```bash
pip install wandb
wandb login
```

### "Network unavailable, logging offline"

This is expected on some cluster nodes. Runs sync automatically when `tracker.finish()` is called, or manually sync later:

```bash
wandb sync ./wandb/
```

### No metrics appearing

1. Check that tracking is enabled: `WANDB_MODE != disabled`
2. Verify you're on rank 0 (only rank 0 logs)
3. Check W&B dashboard filters

### Duplicate runs on resume

Ensure you pass `resume_id` from the checkpoint:

```python
resume_id = checkpoint.get("wandb_run_id")
tracker = ExperimentTracker(resume_id=resume_id, ...)
```

## API Reference

### ExperimentTracker

```python
class ExperimentTracker:
    def __init__(self, project, config, tags, name=None, resume_id=None, ...)
    def log_metrics(self, metrics: dict, step: int)
    def log_training_step(self, loss, step, batch_size, optimizer, model, ...)
    def log_config(self, config: dict)
    def log_checkpoint(self, path: str, aliases: list, force: bool = False)
    def get_run_id(self) -> str
    def get_run_url(self) -> str
    def is_active(self) -> bool
    def is_offline(self) -> bool
    def finish(self)
```

### Utility Functions

```python
from tracking import (
    compute_grad_norm,      # Compute gradient L2 norm
    get_learning_rate,      # Get LR from optimizer
    get_amp_scale,          # Get GradScaler value
    get_all_gpu_stats,      # Get GPU memory/utilization
    set_seeds,              # Set random seeds
    get_metadata,           # Get reproducibility metadata
    is_main_process,        # Check if rank 0
)
```
