#!/bin/bash
#===============================================================================
# JOB TEMPLATE 5: DeepSpeed ZeRO-1 with Experiment Tracking
# Purpose: Verify experiment tracking integration with DeepSpeed training
# Milestone: "Training run visible in W&B with all metrics, metadata, artifacts"
#===============================================================================

#SBATCH --job-name=smoke_tracking
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --account=euge
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=100G
#SBATCH --time=00:30:00

#===============================================================================
# SETUP
#===============================================================================
set -e

echo "========================================"
echo "SMOKE TEST: DeepSpeed + Experiment Tracking"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "Start: $(date)"
echo "========================================"

cd $SLURM_SUBMIT_DIR
source /scratch/gilbreth/$(whoami)/vla-lego/activate_env.sh 2>/dev/null || {
    module purge
    module load external
    module load cuda/12.1.1 anaconda/2024.10-py312
    conda activate vla_lego_env 2>/dev/null || source activate vla_lego_env
}

mkdir -p logs checkpoints

# Environment
export NCCL_DEBUG=INFO
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# CRITICAL: Single-node NCCL settings (IB/network transport doesn't work for intra-node)
export NCCL_IB_DISABLE=1    # Disable InfiniBand
export NCCL_NET_DISABLE=1   # Disable network transport
export NCCL_P2P_LEVEL=NVL   # Use NVLink/P2P for GPU-to-GPU
export NCCL_SHM_DISABLE=0   # Enable shared memory fallback

# GLOO settings for rendezvous
export GLOO_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected GPUs: $NUM_GPUS"

# Add tracking module to Python path
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

#===============================================================================
# VERIFY IMPORTS
#===============================================================================
echo ""
echo "=== Verifying Imports ==="
python -c "
import deepspeed
print(f'DeepSpeed: {deepspeed.__version__}')
import wandb
print(f'W&B: {wandb.__version__}')
from tracking import ExperimentTracker
print('Tracking module: OK')
"

#===============================================================================
# DEEPSPEED CONFIG
#===============================================================================
echo ""
echo "=== Creating DeepSpeed Config ==="

cat << 'DSCONFIG' > /tmp/ds_config.json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 1,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 500000000,
        "allgather_bucket_size": 500000000,
        "overlap_comm": true,
        "contiguous_gradients": true
    },

    "bf16": {
        "enabled": true
    },

    "gradient_clipping": 1.0,

    "steps_per_print": 25,

    "wall_clock_breakdown": false
}
DSCONFIG

cat /tmp/ds_config.json

#===============================================================================
# TRAINING WITH TRACKING
#===============================================================================
echo ""
echo "=== DeepSpeed + Tracking Training Test ==="

cat << 'PYEOF' > /tmp/tracking_test.py
import os
import sys
import torch
import torch.nn as nn
import deepspeed
import time
import argparse

# Add tracking module to path
sys.path.insert(0, os.environ.get("SLURM_SUBMIT_DIR", "."))
from tracking import (
    ExperimentTracker,
    compute_grad_norm,
    set_seeds,
    get_metadata,
)


class DummyModel(nn.Module):
    """Larger model to benefit from ZeRO-1 optimizer partitioning."""
    def __init__(self, hidden_size=4096, num_layers=6):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Set seeds for reproducibility
    seed = 42
    seeds = set_seeds(seed)

    # Initialize DeepSpeed
    deepspeed.init_distributed()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Using DeepSpeed ZeRO Stage 1")

    print(f"[Rank {rank}] GPU {local_rank}: {torch.cuda.get_device_name(device)}")

    # Training configuration
    config = {
        "model": {
            "hidden_size": 4096,
            "num_layers": 6,
        },
        "training": {
            "batch_size": 32,
            "num_steps": 100,
            "seed": seed,
        },
        "deepspeed": {
            "stage": 1,
            "bf16": True,
        },
        "seeds": seeds,
        "tracking": {
            "project": "vla-lego",
            "tags": {
                "model": "dummy-mlp",
                "dataset": "synthetic",
                "objective": "mse",
                "experiment_group": "phase0-smoke",
            },
        },
    }

    # Initialize experiment tracker (only rank 0 logs)
    tracker = ExperimentTracker(
        project="vla-lego",
        config=config,
        tags={
            "model": "dummy-mlp",
            "dataset": "synthetic",
            "objective": "mse",
            "experiment_group": "phase0-smoke",
        },
        log_interval=5,  # Log every 5 steps for demo
        gpu_stats_interval=25,  # GPU stats every 25 steps
    )

    # Log config artifact
    tracker.log_config(config)

    # Create model
    model = DummyModel(hidden_size=4096, num_layers=6)
    num_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(f"Model parameters: {num_params:,} ({num_params * 4 / 1e9:.2f} GB in fp32)")

    # DeepSpeed initialization
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    if rank == 0:
        print(f"DeepSpeed engine initialized")
        print(f"  ZeRO stage: {model_engine.zero_optimization_stage()}")
        print(f"  BF16 enabled: {model_engine.bfloat16_enabled()}")

    criterion = nn.MSELoss()

    # Training loop
    if rank == 0:
        print("\nRunning 100 training steps with tracking...")

    tracker.start_throughput_tracking()
    start = time.time()

    for step in range(100):
        # Create input in BF16 to match model dtype
        x = torch.randn(32, 4096, device=device, dtype=torch.bfloat16)
        y = torch.randn(32, 4096, device=device, dtype=torch.bfloat16)

        output = model_engine(x)
        loss = criterion(output, y)

        model_engine.backward(loss)
        model_engine.step()

        # Log training step with all metrics
        tracker.log_training_step(
            loss=loss.item(),
            step=step,
            batch_size=32,
            optimizer=optimizer,
            model=model_engine.module,
        )

        if (step + 1) % 25 == 0 and rank == 0:
            print(f"  Step {step+1}/100 | Loss: {loss.item():.4f}")

    elapsed = time.time() - start
    max_mem = torch.cuda.max_memory_allocated(device) / 1e9

    if rank == 0:
        print(f"\nCompleted in {elapsed:.2f}s ({100/elapsed:.1f} steps/sec)")
        print(f"Peak GPU memory per device: {max_mem:.2f} GB")

    # Save DeepSpeed checkpoint with W&B run ID for resume support
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_dir = "checkpoints/tracking_test"

    # Store W&B run ID in client state for resume
    client_state = {
        "step": 100,
        "wandb_run_id": tracker.get_run_id(),
    }
    model_engine.save_checkpoint(ckpt_dir, tag="step100", client_state=client_state)

    if rank == 0:
        print(f"Checkpoint saved: {ckpt_dir}")

    # Log checkpoint as artifact
    # NOTE: Checkpoint artifact uploads are disabled by default (WANDB_LOG_MODEL=0)
    # to minimize HPC bandwidth usage. Set WANDB_LOG_MODEL=1 to enable.
    # For production, prefer consolidating with zero_to_fp32.py before uploading.
    tracker.log_checkpoint(ckpt_dir, aliases=["latest", "step100"])

    # Verify checkpoint reload
    _, loaded_client_state = model_engine.load_checkpoint(ckpt_dir, tag="step100")
    if rank == 0:
        print(f"Checkpoint reload: SUCCESS")
        print(f"  Restored step: {loaded_client_state.get('step')}")
        print(f"  W&B run ID: {loaded_client_state.get('wandb_run_id')}")

    # Finish tracking
    tracker.finish()

    if rank == 0:
        print("\n" + "="*50)
        print("SMOKE TEST TRACKING: PASSED")
        print("="*50)

        if tracker.get_run_url():
            print(f"\nView run at: {tracker.get_run_url()}")
        else:
            print("\nRun in offline mode. Sync with: wandb sync ./wandb")


if __name__ == "__main__":
    main()
PYEOF

# Launch with DeepSpeed
echo "Launching DeepSpeed with $NUM_GPUS GPUs..."
deepspeed --num_gpus=$NUM_GPUS /tmp/tracking_test.py \
    --deepspeed_config /tmp/ds_config.json

echo ""
echo "========================================"
echo "JOB COMPLETE: $(date)"
echo "========================================"
