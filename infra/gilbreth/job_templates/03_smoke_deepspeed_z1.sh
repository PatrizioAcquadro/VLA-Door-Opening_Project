#!/bin/bash
#===============================================================================
# JOB TEMPLATE 3: DeepSpeed ZeRO-1 Test (2×A100)
# Purpose: Verify DeepSpeed ZeRO-1 works correctly
# Milestone: "DeepSpeed ZeRO-1 run completes, logs metrics, checkpoint reloads"
#===============================================================================

#SBATCH --job-name=smoke_deepspeed_z1
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
echo "SMOKE TEST: DeepSpeed ZeRO-1"
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

#===============================================================================
# VERIFY DEEPSPEED
#===============================================================================
echo ""
echo "=== DeepSpeed Version ==="
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
ds_report 2>&1 | head -30 || echo "[ds_report not available]"

#===============================================================================
# DEEPSPEED CONFIG
#===============================================================================
echo ""
echo "=== Creating DeepSpeed Config ==="

# Create DeepSpeed ZeRO-1 config
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
# DEEPSPEED TRAINING TEST
#===============================================================================
echo ""
echo "=== DeepSpeed ZeRO-1 Training Test ==="

cat << 'PYEOF' > /tmp/deepspeed_test.py
import os
import torch
import torch.nn as nn
import deepspeed
import time
import argparse

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
        print(f"  FP16 enabled: {model_engine.fp16_enabled()}")

    criterion = nn.MSELoss()

    # Training loop
    if rank == 0:
        print("\nRunning 100 DeepSpeed training steps...")

    start = time.time()

    for step in range(100):
        # Create input in BF16 to match model dtype
        x = torch.randn(32, 4096, device=device, dtype=torch.bfloat16)
        y = torch.randn(32, 4096, device=device, dtype=torch.bfloat16)

        output = model_engine(x)
        loss = criterion(output, y)

        model_engine.backward(loss)
        model_engine.step()

        if (step + 1) % 25 == 0 and rank == 0:
            print(f"  Step {step+1}/100 | Loss: {loss.item():.4f}")

    elapsed = time.time() - start
    max_mem = torch.cuda.max_memory_allocated(device) / 1e9

    if rank == 0:
        print(f"\nCompleted in {elapsed:.2f}s ({100/elapsed:.1f} steps/sec)")
        print(f"Peak GPU memory per device: {max_mem:.2f} GB")

    # Save DeepSpeed checkpoint (collective operation - all ranks must call)
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_dir = "checkpoints/deepspeed_z1"
    model_engine.save_checkpoint(ckpt_dir, tag="step100")
    if rank == 0:
        print(f"DeepSpeed checkpoint saved: {ckpt_dir}")

    # Verify we can load it (collective operation - all ranks must call)
    _, client_state = model_engine.load_checkpoint(ckpt_dir, tag="step100")
    if rank == 0:
        print("Checkpoint reload: SUCCESS ✓")

        print("\n" + "="*50)
        print("SMOKE TEST DEEPSPEED ZeRO-1: PASSED ✓")
        print("="*50)

if __name__ == "__main__":
    main()
PYEOF

# Launch with DeepSpeed
echo "Launching DeepSpeed with $NUM_GPUS GPUs..."
deepspeed --num_gpus=$NUM_GPUS /tmp/deepspeed_test.py \
    --deepspeed_config /tmp/ds_config.json

echo ""
echo "========================================"
echo "JOB COMPLETE: $(date)"
echo "========================================"
