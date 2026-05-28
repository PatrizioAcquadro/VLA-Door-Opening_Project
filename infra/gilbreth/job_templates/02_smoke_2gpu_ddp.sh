#!/bin/bash
#===============================================================================
# JOB TEMPLATE 2: Multi-GPU Single Node (2×A100)
# Purpose: Verify DDP works on single node
# Milestone: "multi-GPU run completes N steps, produces checkpoint"
#===============================================================================

#SBATCH --job-name=smoke_2gpu_ddp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --account=euge
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=00:15:00

#===============================================================================
# SETUP
#===============================================================================
set -e

echo "========================================"
echo "SMOKE TEST: 2 GPU DDP (Single Node)"
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

# NCCL settings for single-node
export NCCL_DEBUG=INFO
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# For single-node multi-GPU, force P2P communication (disable network transports)
export NCCL_P2P_LEVEL=NVL  # Prefer NVLink/P2P
export NCCL_SHM_DISABLE=0  # Enable shared memory
export NCCL_IB_DISABLE=1   # Disable InfiniBand - not needed for single node
export NCCL_NET_DISABLE=1  # Disable network transport - use P2P only

# Set GLOO for rendezvous (fallback)
export GLOO_SOCKET_IFNAME=lo  # Use loopback for gloo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected GPUs: $NUM_GPUS"

#===============================================================================
# GPU INFO
#===============================================================================
echo ""
echo "=== nvidia-smi ==="
nvidia-smi

echo ""
echo "=== GPU Topology ==="
nvidia-smi topo -m

#===============================================================================
# DDP TRAINING TEST
#===============================================================================
echo ""
echo "=== DDP Training Test ==="

# Create training script
cat << 'PYEOF' > /tmp/ddp_test.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time

class DummyModel(nn.Module):
    def __init__(self, hidden_size=2048):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x):
        return self.layers(x)

def main():
    # Initialize distributed with timeout
    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(minutes=2))

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Backend: {dist.get_backend()}")

    print(f"[Rank {rank}] Using GPU {local_rank}: {torch.cuda.get_device_name(device)}", flush=True)

    # CUDA synchronization
    torch.cuda.synchronize()

    # Synchronize before model creation
    print(f"[Rank {rank}] Barrier 1 start", flush=True)
    dist.barrier()
    print(f"[Rank {rank}] Barrier 1 done", flush=True)

    # Create model with DDP
    model = DummyModel(hidden_size=2048).to(device)
    torch.cuda.synchronize()
    print(f"[Rank {rank}] Model created on device", flush=True)

    # Synchronize before DDP wrapper
    print(f"[Rank {rank}] Barrier 2 start", flush=True)
    dist.barrier()
    print(f"[Rank {rank}] Barrier 2 done", flush=True)

    print(f"[Rank {rank}] Creating DDP wrapper...", flush=True)
    model = DDP(model, device_ids=[local_rank])
    print(f"[Rank {rank}] DDP wrapper created", flush=True)

    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model parameters: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop

    start = time.time()

    for step in range(100):
        x = torch.randn(64, 2048, device=device)
        y = torch.randn(64, 2048, device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if (step + 1) % 25 == 0 and rank == 0:
            print(f"  Step {step+1}/100 | Loss: {loss.item():.4f}")

    elapsed = time.time() - start
    max_mem = torch.cuda.max_memory_allocated(device) / 1e9

    # Synchronize before checkpoint
    dist.barrier()

    if rank == 0:
        print(f"\nCompleted in {elapsed:.2f}s ({100/elapsed:.1f} steps/sec)")
        print(f"Peak GPU memory per device: {max_mem:.2f} GB")

        # Save checkpoint (only rank 0)
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = "checkpoints/smoke_2gpu_ddp.pt"
        checkpoint = {
            "step": 100,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "world_size": world_size,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        # Verify reload
        loaded = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(loaded["model_state_dict"])
        print("Checkpoint reload: SUCCESS ✓")

        print("\n" + "="*50)
        print("SMOKE TEST 2 GPU DDP: PASSED ✓")
        print("="*50)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PYEOF

# Debug info
echo "Python: $(which python) ($(python -V 2>&1))"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# Verify Python syntax
echo "=== Verifying Python syntax ==="
python -m py_compile /tmp/ddp_test.py && echo "Syntax OK"

# Set additional debug flags
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1

# Launch with torchrun
echo ""
echo "=== Launching torchrun with $NUM_GPUS GPUs ==="
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    /tmp/ddp_test.py

echo ""
echo "========================================"
echo "JOB COMPLETE: $(date)"
echo "========================================"
