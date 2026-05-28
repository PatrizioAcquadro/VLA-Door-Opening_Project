# Gilbreth HPC Infrastructure

This directory contains cluster-specific infrastructure for running VLA-LEGO on Purdue's Gilbreth HPC cluster.

## Directory Structure

```
infra/gilbreth/
├── job_templates/          # SLURM batch job scripts
│   ├── 01_smoke_1gpu.sh    # Single GPU smoke test
│   ├── 02_smoke_2gpu_ddp.sh # 2 GPU DDP test
│   ├── 03_smoke_deepspeed_z1.sh # DeepSpeed ZeRO-1 test
│   ├── 04_smoke_8gpu_deepspeed.sh # Multi-node 8 GPU test
│   └── 05_smoke_tracking.sh # DeepSpeed + W&B tracking
├── scripts/                # Setup and utility scripts
│   ├── 00_discovery.sh     # Cluster resource discovery
│   ├── 01_setup_env.sh     # Environment setup
│   └── verify_checkpoints.sh # Checkpoint verification
└── README.md               # This file
```

## Quick Start

### 1. Set Environment Variables

Add to your `~/.bashrc`:

```bash
export VLA_PROJECT_ROOT=$HOME/VLA-LEGO_Project
export VLA_SCRATCH_ROOT=/scratch/gilbreth/$USER/vla-lego
```

### 2. Run Discovery Script

```bash
bash infra/gilbreth/scripts/00_discovery.sh 2>&1 | tee discovery_output.txt
```

### 3. Setup Environment

```bash
bash infra/gilbreth/scripts/01_setup_env.sh
```

### 4. Submit Jobs

```bash
cd $VLA_PROJECT_ROOT
sbatch infra/gilbreth/job_templates/01_smoke_1gpu.sh
```

## Job Templates

| Template | GPUs | Purpose |
|----------|------|---------|
| `01_smoke_1gpu.sh` | 1 | Single GPU training verification |
| `02_smoke_2gpu_ddp.sh` | 2 | PyTorch DDP on single node |
| `03_smoke_deepspeed_z1.sh` | 2 | DeepSpeed ZeRO-1 on single node |
| `04_smoke_8gpu_deepspeed.sh` | 8 | Multi-node (4 nodes x 2 GPUs) |
| `05_smoke_tracking.sh` | 2 | DeepSpeed + W&B integration |

## NCCL Configuration

### Single-Node (Tests 01-03)
```bash
export NCCL_IB_DISABLE=1      # Disable InfiniBand
export NCCL_NET_DISABLE=1     # Disable network transport
export NCCL_P2P_LEVEL=NVL     # Use NVLink/P2P
export NCCL_SHM_DISABLE=0     # Enable shared memory
```

### Multi-Node (Test 04+)
```bash
export NCCL_IB_DISABLE=0      # Enable InfiniBand
unset NCCL_SOCKET_IFNAME      # Let NCCL auto-detect (CRITICAL!)
export NCCL_P2P_LEVEL=NVL     # NVLink for intra-node
export NCCL_TIMEOUT=600       # 10 minute timeout
```

**Important**: Never hardcode `NCCL_SOCKET_IFNAME` - interface names vary by node.

## Documentation

For comprehensive cluster documentation, see:
- [Gilbreth Training Guide](../docs/cluster/gilbreth-training-guide.md)
