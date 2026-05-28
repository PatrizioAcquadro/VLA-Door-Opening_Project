#!/bin/bash
#===============================================================================
# JOB TEMPLATE 4: Multi-Node DeepSpeed ZeRO-1 (8×A100 / 4 Nodes)
# Purpose: Verify multi-node DeepSpeed training at scale
# Milestone: "8-GPU DeepSpeed run completes, logs metrics, checkpoint reloads"
#===============================================================================

#SBATCH --job-name=smoke_ds_8gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --account=euge
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --exclusive

#===============================================================================
# SETUP
#===============================================================================
set -e

echo "========================================"
echo "SMOKE TEST: Multi-Node DeepSpeed ZeRO-1"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Start: $(date)"
echo "========================================"

cd $SLURM_SUBMIT_DIR
source /scratch/gilbreth/$(whoami)/vla-lego/activate_env.sh 2>/dev/null || {
    module purge
    module load external
    module load cuda/12.1.1 anaconda/2024.10-py312
    conda activate vla_lego_env 2>/dev/null || source activate vla_lego_env
}

# Unset single-node network config from activate_env.sh (interface names vary by node)
unset NCCL_SOCKET_IFNAME
unset GLOO_SOCKET_IFNAME

mkdir -p logs checkpoints

#-------------------------------------------------------------------------------
# MULTI-NODE NETWORKING
#-------------------------------------------------------------------------------
# Get master node
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo ""
echo "=== Multi-Node Configuration ==="
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "All nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"

# NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0     # Enable InfiniBand
# DO NOT set NCCL_SOCKET_IFNAME - let NCCL auto-detect (names vary: ibp65s0, ibp161s0, etc.)
export NCCL_TIMEOUT=600
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_LEVEL=NVL    # Use NVLink for intra-node GPU-to-GPU
export NCCL_SHM_DISABLE=0    # Enable shared memory

GPUS_PER_NODE=2
NNODES=$SLURM_JOB_NUM_NODES
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "Total GPUs: $WORLD_SIZE"

#-------------------------------------------------------------------------------
# VERIFY ALL NODES
#-------------------------------------------------------------------------------
echo ""
echo "=== Verifying Nodes ==="
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c '
    echo "$(hostname): $(nvidia-smi -L | wc -l) GPUs"
'

#-------------------------------------------------------------------------------
# DEEPSPEED CONFIG
#-------------------------------------------------------------------------------
# Use home directory (more reliable than scratch for NFS)
SHARED_DIR=$HOME/vla_lego_multinode
mkdir -p $SHARED_DIR

echo ""
echo "=== DeepSpeed Config ==="

cat << DSCONFIG > $SHARED_DIR/ds_config_multinode.json
{
    "train_batch_size": $((32 * WORLD_SIZE)),
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

    "wall_clock_breakdown": false,

    "comms_logger": {
        "enabled": false
    }
}
DSCONFIG

cat $SHARED_DIR/ds_config_multinode.json

#-------------------------------------------------------------------------------
# TRAINING SCRIPT
#-------------------------------------------------------------------------------
cat << 'PYEOF' > $SHARED_DIR/multinode_ds_test.py
import os
import socket
import torch
import torch.nn as nn
import deepspeed
import time
import argparse

class DummyModel(nn.Module):
    """Larger model to simulate real training."""
    def __init__(self, hidden_size=4096, num_layers=8):
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

    # Initialize DeepSpeed distributed
    deepspeed.init_distributed()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    hostname = socket.gethostname()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    print(f"[Rank {rank}/{world_size}] Node: {hostname}, GPU: {local_rank}", flush=True)

    print(f"[Rank {rank}] Entering barrier...", flush=True)
    torch.distributed.barrier()
    print(f"[Rank {rank}] Barrier passed!", flush=True)

    if rank == 0:
        print(f"\n{'='*50}")
        print(f"World size: {world_size} GPUs")
        print(f"Backend: {torch.distributed.get_backend()}")
        print(f"DeepSpeed ZeRO Stage 1")
        print(f"{'='*50}\n")

    # Create model
    model = DummyModel(hidden_size=4096, num_layers=8)
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
        print(f"DeepSpeed initialized")
        print(f"  ZeRO stage: {model_engine.zero_optimization_stage()}")

    criterion = nn.MSELoss()

    # Training loop
    if rank == 0:
        print("\nRunning 100 multi-node training steps...")

    start = time.time()

    for step in range(100):
        x = torch.randn(32, 4096, device=device, dtype=torch.bfloat16)
        y = torch.randn(32, 4096, device=device, dtype=torch.bfloat16)

        output = model_engine(x)
        loss = criterion(output, y)

        model_engine.backward(loss)
        model_engine.step()

        # All-reduce loss for logging
        if (step + 1) % 25 == 0:
            loss_tensor = loss.detach().clone()
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
            if rank == 0:
                print(f"  Step {step+1}/100 | Avg Loss: {loss_tensor.item():.4f}")

    elapsed = time.time() - start
    max_mem = torch.cuda.max_memory_allocated(device) / 1e9

    # Gather memory stats from all ranks
    mem_tensor = torch.tensor([max_mem], device=device)
    torch.distributed.all_reduce(mem_tensor, op=torch.distributed.ReduceOp.MAX)

    if rank == 0:
        print(f"\nCompleted in {elapsed:.2f}s ({100/elapsed:.1f} steps/sec)")
        print(f"Effective throughput: {100 * world_size / elapsed:.1f} samples/sec (world)")
        print(f"Max GPU memory (across all): {mem_tensor.item():.2f} GB")

    # Save checkpoint (collective operation - ALL ranks must call)
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_dir = "checkpoints/deepspeed_multinode"
    model_engine.save_checkpoint(ckpt_dir, tag="step100")
    if rank == 0:
        print(f"Checkpoint saved: {ckpt_dir}")

    # Verify reload (collective operation - ALL ranks must call)
    _, client_state = model_engine.load_checkpoint(ckpt_dir, tag="step100")
    if rank == 0:
        print("Checkpoint reload: SUCCESS ✓")

        print("\n" + "="*50)
        print(f"MULTI-NODE DEEPSPEED ({world_size} GPUs): PASSED ✓")
        print("="*50)

    torch.distributed.barrier()

if __name__ == "__main__":
    main()
PYEOF

#-------------------------------------------------------------------------------
# LAUNCH (using srun + torchrun for SLURM compatibility)
#-------------------------------------------------------------------------------
echo ""
echo "=== Launching Multi-Node DeepSpeed with srun ==="

# Export variables for srun
export GPUS_PER_NODE
export NNODES
export WORLD_SIZE
export SHARED_DIR

# Use srun to launch torchrun on each node
# torchrun spawns the per-GPU processes on each node
# --export=ALL propagates all environment variables including NCCL settings
srun --ntasks=$NNODES --ntasks-per-node=1 --export=ALL bash -c '
    # Set node rank from SLURM
    export NODE_RANK=$SLURM_NODEID

    # Re-export NCCL settings explicitly for torchrun subprocesses
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0        # Enable IB
    unset NCCL_SOCKET_IFNAME        # Let NCCL auto-detect (interface names vary by node)
    export NCCL_P2P_LEVEL=NVL
    export NCCL_SHM_DISABLE=0
    export NCCL_TIMEOUT=600
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    echo "$(hostname): Launching torchrun with NODE_RANK=$NODE_RANK"

    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $SHARED_DIR/multinode_ds_test.py \
        --deepspeed_config $SHARED_DIR/ds_config_multinode.json
'

echo ""
echo "========================================"
echo "JOB COMPLETE: $(date)"
echo "========================================"
