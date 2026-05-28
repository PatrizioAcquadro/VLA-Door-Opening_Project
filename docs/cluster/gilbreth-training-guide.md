# Gilbreth HPC Training Guide

**Project**: VLA-LEGO (EO-1 for Bimanual LEGO Assembly)
**Cluster**: Purdue Gilbreth HPC
**Last Updated**: January 23, 2026
**Status**: All Phase 0.1 smoke tests PASSED

---

## Table of Contents

1. [Cluster Overview](#1-cluster-overview)
   - [1.1 Hardware Specifications](#11-hardware-specifications)
   - [1.2 Partition](#12-partition-a100-80gb)
   - [1.3 Network Interfaces](#13-network-interfaces)
   - [1.4 Node Interconnect](#14-node-interconnect)
   - [1.5 Shared Filesystems](#15-shared-filesystems)
   - [1.6 Storage Paths](#16-storage-paths)
2. [Environment Setup](#2-environment-setup)
3. [SLURM Configuration](#3-slurm-configuration)
4. [NCCL Configuration](#4-nccl-configuration)
   - [4.1 Single-Node Multi-GPU](#41-single-node-multi-gpu-tests-02-03)
   - [4.2 Multi-Node](#42-multi-node-test-04)
   - [4.3 Multi-Node Launcher](#43-multi-node-launcher-critical)
5. [Test 01: Single GPU Smoke Test](#5-test-01-single-gpu-smoke-test)
6. [Test 02: 2 GPU DDP](#6-test-02-2-gpu-ddp)
7. [Test 03: DeepSpeed ZeRO-1](#7-test-03-deepspeed-zero-1)
8. [Test 04: Multi-Node 8 GPU DeepSpeed](#8-test-04-multi-node-8-gpu-deepspeed)
   - [8.1 Purpose](#81-purpose)
   - [8.2 Job Configuration](#82-job-configuration)
   - [8.3 Critical NCCL Configuration](#83-critical-nccl-configuration-for-multi-node)
   - [8.4 Single-Node vs Multi-Node Comparison](#84-comparison-single-node-vs-multi-node-nccl)
   - [8.5 Launch Pattern (srun + torchrun)](#85-multi-node-launch-pattern-srun--torchrun)
   - [8.6 DeepSpeed Configuration](#86-deepspeed-configuration-for-multi-node)
   - [8.7 Test Model](#87-test-model)
   - [8.8 Training Script Patterns](#88-training-script-key-patterns)
   - [8.9 Verified Results](#89-verified-results-job-10213927)
   - [8.10 Log Files](#810-log-files)
   - [8.11 Success Criteria](#811-success-criteria)
   - [8.12 Key Learnings and Pitfalls](#812-key-learnings-and-pitfalls)
   - [8.13 Best Practices](#813-best-practices-for-multi-node-training-on-gilbreth)
   - [8.14 Debugging Multi-Node](#814-debugging-multi-node-issues)
9. [Troubleshooting Reference](#9-troubleshooting-reference)

---

## 1. Cluster Overview

### 1.1 Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **GPU Model** | NVIDIA A100 80GB PCIe |
| **GPU Memory** | 81,920 MiB (85.1 GB usable) |
| **GPUs per Node** | 2 |
| **Driver Version** | 565.57.01 |
| **CUDA Capability** | 12.7 (driver), 12.1 (runtime) |
| **Interconnect** | InfiniBand (100Gbps) |
| **Persistence Mode** | Enabled |

### 1.2 Partition: `a100-80gb`

The primary partition for A100 80GB GPUs. Nodes follow naming convention `gilbreth-kXXX` or `gilbreth-iXXX`.

### 1.3 Network Interfaces

**Important**: Interface names vary by node:
- `ibp161s0` - InfiniBand (e.g., gilbreth-k016, gilbreth-i004, 172.18.36.x/23)
- `ibp65s0` - InfiniBand (e.g., gilbreth-k023, gilbreth-k024, gilbreth-k025)
- `eno1` - Ethernet (10Gbps, available on all nodes)

This variation is **critical** for multi-node training configuration (see [Section 4](#4-nccl-configuration)). Never hardcode interface names.

### 1.4 Node Interconnect

| Type | Bandwidth | Use Case |
|------|-----------|----------|
| InfiniBand | ~100 Gbps | Cross-node GPU communication (NCCL) |
| NVLink | 600 GB/s | Intra-node GPU-to-GPU (P2P) |
| Ethernet | 10 Gbps | Fallback (not recommended for training) |

**Important Cluster Restrictions**:
- **No SSH between compute nodes**: SSH-based launchers (pdsh, DeepSpeed hostfile) do not work
- **Slurm srun required**: Use `srun` with `torchrun` for multi-node training

### 1.5 Shared Filesystems

| Path | Shared Across Nodes | Recommended Use |
|------|---------------------|-----------------|
| `/home/$USER/` | ✅ Yes (NFS) | Multi-node scripts, configs |
| `/scratch/gilbreth/$USER/` | ✅ Yes (parallel FS) | Large data, checkpoints |
| `/tmp/` | ❌ No (local) | **Never use for multi-node** |

**Multi-node Critical**: Scripts and config files must reside on a shared filesystem (`$HOME` or `/scratch`) - files in `/tmp/` are node-local and invisible to other nodes.

### 1.6 Storage Paths

| Path | Purpose | Quota |
|------|---------|-------|
| `/home/$USER/` | Home directory, NFS-shared configs | Limited |
| `/scratch/gilbreth/$USER/` | Project workspace | Large |
| `/scratch/gilbreth/$USER/vla-lego/` | Project root | - |
| `/scratch/gilbreth/$USER/vla-lego/checkpoints/` | Model checkpoints | - |
| `/scratch/gilbreth/$USER/vla-lego/logs/` | Training logs | - |
| `/scratch/gilbreth/$USER/vla-lego/cache/huggingface/` | HuggingFace cache | - |
| `$HOME/vla_lego_multinode/` | Multi-node scripts/configs | - |

---

## 2. Environment Setup

### 2.1 Required Modules

```bash
module purge
module load external
module load cuda/12.1.1
module load anaconda/2024.10-py312
```

**Critical**: Always load `external` module first before loading `cuda` or `anaconda`.

### 2.2 Conda Environment

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10.19 | Runtime |
| PyTorch | 2.2.0+cu121 | Deep learning framework |
| CUDA (PyTorch) | 12.1 | GPU acceleration |
| cuDNN | 8902 (8.9.2.26) | DNN primitives |
| DeepSpeed | 0.14.0 | Distributed training |
| Accelerate | 1.12.0 | HuggingFace accelerator |
| NCCL | 2.19.3 | Multi-GPU communication |

### 2.3 Environment Activation

**Quick activation** (after initial setup):
```bash
source /scratch/gilbreth/$USER/vla-lego/activate_env.sh
```

**Manual activation**:
```bash
module purge
module load external
module load cuda/12.1.1 anaconda/2024.10-py312
conda activate vla_lego_env
```

### 2.4 Verification Commands

```bash
# Check Python
which python
python --version

# Check PyTorch and CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Check DeepSpeed
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
```

---

## 3. SLURM Configuration

### 3.1 Account and QOS Settings

| Setting | Value | Notes |
|---------|-------|-------|
| **Account** | `euge` | Change if using different allocation |
| **Partition** | `a100-80gb` | A100 80GB GPUs |
| **QOS (1 GPU)** | `standby` | Lower priority, shorter queue |
| **QOS (multi-GPU)** | `normal` | Requires allocation |

### 3.2 Resource Requests by Test

| Test | Nodes | GPUs | CPUs | Memory | Time |
|------|-------|------|------|--------|------|
| 01_smoke_1gpu | 1 | 1 | 8 | 50G | 30min |
| 02_smoke_2gpu_ddp | 1 | 2 | 16 | 100G | 30min |
| 03_smoke_deepspeed_z1 | 1 | 2 | 16 | 100G | 30min |
| 04_smoke_8gpu_deepspeed | 4 | 8 (2/node) | 64 | 200G | 1hr |

### 3.3 Job Submission

```bash
cd /home/$USER/phase0_setup/gilbreth_phase0
sbatch job_templates/01_smoke_1gpu.sh
```

### 3.4 Job Monitoring

```bash
# Check job status
squeue -u $USER

# Check job status with reason
squeue -u $USER -o "%i %t %M %r"

# Check partition queue
squeue -p a100-80gb -o "%P %u %t %D %M" | head -15

# View job output (while running or after)
cat /scratch/gilbreth/$USER/vla-lego/logs/smoke_*_JOBID.out

# Check for errors
cat /scratch/gilbreth/$USER/vla-lego/logs/smoke_*_JOBID.err
```

### 3.5 Common Job States

| State | Code | Meaning |
|-------|------|---------|
| Pending | PD | Waiting for resources |
| Running | R | Currently executing |
| Completed | CD | Finished successfully |
| Failed | F | Exited with error |

**Pending Reasons**:
- `Priority` - Waiting in queue (normal for `standby` QOS)
- `Resources` - Waiting for requested resources to become available

---

## 4. NCCL Configuration

### 4.1 Single-Node Multi-GPU (Tests 02, 03)

**Critical Issue Discovered**: On Gilbreth, NCCL's InfiniBand/network transport does NOT work for intra-node GPU communication. Collective operations (barriers, allreduce, allgather) will timeout even though NCCL initialization succeeds.

**Required Environment Variables**:
```bash
export NCCL_IB_DISABLE=1      # CRITICAL: Disable InfiniBand
export NCCL_NET_DISABLE=1     # CRITICAL: Disable network transport
export NCCL_P2P_LEVEL=NVL     # Use NVLink/P2P for GPU-to-GPU
export NCCL_SHM_DISABLE=0     # Enable shared memory fallback
export GLOO_SOCKET_IFNAME=lo  # Use loopback for GLOO rendezvous
```

**Working Communication Method**: `SHM/direct/direct` (shared memory)

### 4.2 Multi-Node (Test 04)

**Critical**: Do NOT set `NCCL_SOCKET_IFNAME` - interface names vary by node. Let NCCL auto-detect.

```bash
export NCCL_IB_DISABLE=0      # Enable InfiniBand for cross-node
# DO NOT set NCCL_SOCKET_IFNAME - let NCCL auto-detect
export NCCL_P2P_LEVEL=NVL     # Use NVLink for intra-node
export NCCL_SHM_DISABLE=0     # Enable shared memory
export NCCL_TIMEOUT=600       # 10 minute timeout
```

**Working Communication Method**:
- Inter-node: `NET/IB/0` (InfiniBand)
- Intra-node: `P2P/CUMEM`

### 4.3 Multi-Node Launcher (Critical)

**Issue**: DeepSpeed's pdsh-based launcher (`deepspeed --hostfile`) requires SSH between nodes. Gilbreth does **not** allow SSH between compute nodes - attempting to use pdsh results in:
```
***** Use of Purdue BoilerKey or SSH keys is Required ******
```

**Solution**: Use `srun` to launch `torchrun` on each node. This is the **only** working multi-node launch method on Gilbreth.

```bash
# Get master node from SLURM
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR MASTER_PORT

# Export variables for srun subshells
export GPUS_PER_NODE=2
export NNODES=$SLURM_JOB_NUM_NODES

# Launch using srun + torchrun
srun --ntasks=$NNODES --ntasks-per-node=1 --export=ALL bash -c '
    # Set node rank from SLURM (CRITICAL - each node needs unique rank)
    export NODE_RANK=$SLURM_NODEID

    # Re-export NCCL settings (environment may not propagate fully)
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    unset NCCL_SOCKET_IFNAME    # Let NCCL auto-detect
    export NCCL_P2P_LEVEL=NVL
    export NCCL_SHM_DISABLE=0
    export NCCL_TIMEOUT=600
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        /path/to/training_script.py \
        --deepspeed_config /path/to/ds_config.json
'
```

**Critical Notes**:
1. **`--export=ALL`**: Required to propagate environment variables to srun subprocesses
2. **Re-export NCCL vars**: Even with `--export=ALL`, re-export inside the bash command for reliability
3. **`$SLURM_NODEID`**: Provides unique node rank (0, 1, 2, 3...) for each node
4. **Script paths**: Must be on shared filesystem (`$HOME` or `/scratch`), not `/tmp/`
5. **`unset NCCL_SOCKET_IFNAME`**: Never hardcode - interface names vary by node

---

## 5. Test 01: Single GPU Smoke Test

### 5.1 Purpose

Verify basic CUDA, PyTorch, and training functionality on a single GPU:
- CUDA availability and GPU detection
- Forward and backward pass
- Checkpoint save and reload

### 5.2 Job Configuration

```bash
#SBATCH --job-name=smoke_1gpu
#SBATCH --account=euge
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=00:30:00
```

### 5.3 Test Model

A simple 3-layer MLP for testing:

```python
class DummyModel(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),    # 1024 -> 4096
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size * 4), # 4096 -> 4096
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),    # 4096 -> 1024
        )
```

| Property | Value |
|----------|-------|
| Parameters | 25,175,040 (~25M) |
| Hidden Size | 1024 |
| Batch Size | 32 |
| Optimizer | AdamW (lr=1e-4) |
| Loss | MSELoss |

### 5.4 Test Execution

```bash
cd /home/$USER/phase0_setup/gilbreth_phase0
sbatch job_templates/01_smoke_1gpu.sh
```

### 5.5 Verified Results (Job 10212764)

**Execution Details**:
| Metric | Value |
|--------|-------|
| Job ID | 10212764 |
| Node | gilbreth-k016 |
| Date | January 22, 2026 |
| Start Time | 19:37:13 EST |
| End Time | 19:49:26 EST |

**Environment Verification**:
| Component | Verified Value |
|-----------|----------------|
| Python | 3.10.19 |
| Python Path | `/home/pacquadr/.conda/envs/2024.10-py312/vla_lego_env/bin/python` |
| PyTorch | 2.2.0+cu121 |
| CUDA Available | True |
| CUDA Version | 12.1 |
| cuDNN Version | 8902 |
| GPU | NVIDIA A100 80GB PCIe |
| GPU Memory | 85.1 GB |
| Driver | 565.57.01 |

**Training Results**:
| Metric | Value |
|--------|-------|
| Training Steps | 100 |
| Training Time | 1.05 seconds |
| Throughput | 95.0 steps/sec |
| Peak GPU Memory | 0.52 GB |
| Final Loss | 1.0137 |

**Loss Progression**:
```
Step 25/100  | Loss: 1.0087
Step 50/100  | Loss: 0.9850
Step 75/100  | Loss: 0.9944
Step 100/100 | Loss: 1.0137
```

**Checkpoint Verification**:
- Saved to: `checkpoints/smoke_1gpu.pt`
- Reload: SUCCESS

### 5.6 Log Files

- **Output**: `/scratch/gilbreth/$USER/vla-lego/logs/smoke_1gpu_10212764.out`
- **Errors**: `/scratch/gilbreth/$USER/vla-lego/logs/smoke_1gpu_10212764.err`

**Note**: The `.err` file contains only harmless pynvml deprecation warnings:
```
FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.
```

### 5.7 Success Criteria

- [x] CUDA available and GPU detected
- [x] PyTorch version matches expected (2.2.0+cu121)
- [x] Forward pass completes without error
- [x] Backward pass completes without error
- [x] 100 training steps complete
- [x] Checkpoint saves successfully
- [x] Checkpoint reloads successfully
- [x] Final status: **PASSED**

### 5.8 Key Learnings

1. **Environment Activation**: The job script includes a fallback mechanism if `activate_env.sh` fails
2. **Module Loading Order**: Always load `external` before `cuda` and `anaconda`
3. **Queue Wait Times**: With `standby` QOS, expect variable queue times depending on cluster load
4. **pynvml Warning**: Safe to ignore - does not affect functionality

---

## 6. Test 02: 2 GPU DDP

### 6.1 Purpose

Verify PyTorch DistributedDataParallel (DDP) works correctly on a single node with 2 GPUs:
- Multi-GPU initialization and communication
- NCCL backend configuration for intra-node training
- Gradient synchronization across GPUs
- Checkpoint save and reload with DDP wrapper

### 6.2 Job Configuration

```bash
#SBATCH --job-name=smoke_2gpu_ddp
#SBATCH --account=euge
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal              # Requires allocation (not standby)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2              # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --time=00:15:00
```

**Note**: Uses `normal` QOS instead of `standby` because multi-GPU jobs typically require allocation.

### 6.3 Critical NCCL Configuration

**This is the most important section for 2 GPU training on Gilbreth.**

On Gilbreth, NCCL's default InfiniBand/network transport **does NOT work** for intra-node GPU communication. The collective operations (barriers, allreduce, allgather) will timeout even though NCCL initialization shows "Init COMPLETE".

**Required Environment Variables** (must be set before launching training):

```bash
# CRITICAL: Disable network transports for single-node
export NCCL_IB_DISABLE=1      # Disable InfiniBand (causes timeouts on single-node)
export NCCL_NET_DISABLE=1     # Disable network transport entirely
export NCCL_P2P_LEVEL=NVL     # Prefer NVLink/P2P for GPU-to-GPU communication
export NCCL_SHM_DISABLE=0     # Enable shared memory as communication fallback

# GLOO settings for rendezvous
export GLOO_SOCKET_IFNAME=lo  # Use loopback interface for GLOO
export MASTER_ADDR=127.0.0.1  # Localhost for single-node
export MASTER_PORT=29500      # Any available port

# Debugging (optional, but helpful)
export NCCL_DEBUG=INFO
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

**Symptoms of Incorrect Configuration**:
- NCCL logs show `Init COMPLETE` but training hangs
- Timeout errors after 10 minutes (default NCCL timeout)
- Error: `DDP expects same model across all ranks, but Rank X has Y params, while rank Z has inconsistent 0 params`

**Correct NCCL Log Output** (what you should see):
```
NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
NCCL INFO Connected all rings
NCCL INFO Connected all trees
```

**Incorrect NCCL Log Output** (causes hangs):
```
NCCL INFO Using network IB
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM
```

### 6.4 Test Model

A larger 5-layer MLP to test gradient synchronization:

```python
class DummyModel(nn.Module):
    def __init__(self, hidden_size=2048):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),    # 2048 -> 8192
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size * 4), # 8192 -> 8192
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),    # 8192 -> 2048
        )
```

| Property | Value |
|----------|-------|
| Parameters | 100,681,728 (~100M) |
| Hidden Size | 2048 |
| Batch Size per GPU | 64 |
| Global Batch Size | 128 (64 × 2 GPUs) |
| Optimizer | AdamW (lr=1e-4) |
| Loss | MSELoss |

### 6.5 Training Script Structure

The training script follows this pattern for DDP:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

def main():
    # 1. Initialize distributed with explicit timeout
    dist.init_process_group("nccl", timeout=timedelta(minutes=2))

    # 2. Get rank information
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    # 3. Set device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 4. Synchronize before model creation (recommended)
    torch.cuda.synchronize()
    dist.barrier()

    # 5. Create model and wrap with DDP
    model = DummyModel().to(device)
    torch.cuda.synchronize()
    dist.barrier()
    model = DDP(model, device_ids=[local_rank])

    # 6. Training loop with gradient sync (automatic with DDP)
    for step in range(num_steps):
        output = model(x)
        loss = criterion(output, y)
        loss.backward()      # Gradients synchronized automatically
        optimizer.step()

    # 7. Synchronize before checkpoint
    dist.barrier()

    # 8. Save checkpoint (rank 0 only for standard PyTorch checkpoints)
    if rank == 0:
        torch.save({
            "model_state_dict": model.module.state_dict(),  # Note: .module for DDP
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

    # 9. Cleanup
    dist.destroy_process_group()
```

**Key Points**:
- Use `model.module.state_dict()` to save DDP-wrapped model weights
- Add `dist.barrier()` before critical sections to ensure synchronization
- Use `flush=True` on print statements for debugging
- Set explicit timeout in `init_process_group()` for faster failure detection

### 6.6 Launch Command

Use `torchrun` with `--standalone` flag for single-node:

```bash
torchrun \
    --standalone \
    --nproc_per_node=2 \
    training_script.py
```

**Parameters**:
- `--standalone`: Single-node mode with built-in rendezvous
- `--nproc_per_node`: Number of processes (GPUs) per node

**Do NOT use**:
- `--nnodes` and `--node_rank` (not needed for standalone)
- `--master_addr`/`--master_port` with `--standalone` (handled automatically)

### 6.7 Verified Results (Job 10213612)

**Execution Details**:
| Metric | Value |
|--------|-------|
| Job ID | 10213612 |
| Node | gilbreth-k032 |
| Date | January 22, 2026 |
| Total Duration | 32 seconds |
| Training Time | 4.21 seconds |

**Distributed Configuration**:
| Component | Value |
|-----------|-------|
| World Size | 2 |
| Backend | nccl |
| Communication | SHM/direct/direct |
| GPUs | 2× NVIDIA A100 80GB PCIe |

**Training Results**:
| Metric | Value |
|--------|-------|
| Training Steps | 100 |
| Throughput | 23.7 steps/sec |
| Samples/sec | ~3,033 (64 batch × 2 GPUs × 23.7) |
| Peak GPU Memory | 2.44 GB per device |

**Loss Progression**:
```
Step 25/100  | Loss: 1.0058
Step 50/100  | Loss: 0.9993
Step 75/100  | Loss: 0.9987
Step 100/100 | Loss: 1.0099
```

**Checkpoint Verification**:
- Saved to: `checkpoints/smoke_2gpu_ddp.pt`
- Reload: SUCCESS

**NCCL Communication (from logs)**:
```
NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
NCCL INFO Connected all rings
NCCL INFO Connected all trees
NCCL INFO comm 0x... rank 0 nranks 2 - Init COMPLETE
NCCL INFO comm 0x... rank 1 nranks 2 - Init COMPLETE
```

### 6.8 Log Files

- **Output**: `/scratch/gilbreth/$USER/vla-lego/logs/smoke_2gpu_ddp_10213612.out`
- **Errors**: `/scratch/gilbreth/$USER/vla-lego/logs/smoke_2gpu_ddp_10213612.err`

**Note**: The `.err` file contains only harmless pynvml deprecation warnings.

### 6.9 Success Criteria

- [x] Distributed initialization completes (world_size=2)
- [x] NCCL uses SHM/direct (not network transport)
- [x] Both GPUs detected and utilized
- [x] Barriers complete without timeout
- [x] DDP wrapper created successfully
- [x] Forward pass completes on both ranks
- [x] Backward pass with gradient sync completes
- [x] 100 training steps complete
- [x] Checkpoint saves successfully
- [x] Checkpoint reloads successfully
- [x] Final status: **PASSED**

### 6.10 Key Learnings

1. **NCCL Network Transport Failure**: Gilbreth's InfiniBand configuration does not work for intra-node NCCL communication. This was discovered after multiple timeout failures. Always disable IB and network transport for single-node jobs.

2. **Debugging with Barriers**: Adding `dist.barrier()` calls with print statements (using `flush=True`) helped isolate exactly where the communication hung.

3. **Shorter Timeout for Debugging**: Setting `timeout=timedelta(minutes=2)` in `init_process_group()` allows faster failure detection (default is 10 minutes).

4. **CUDA Synchronization**: Adding `torch.cuda.synchronize()` before barriers ensures GPU operations complete before collective communication.

5. **DDP Model Access**: Use `model.module.state_dict()` to access the underlying model weights, not `model.state_dict()`.

### 6.11 Best Practices for 2 GPU Training on Gilbreth

#### Environment Setup Template

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=a100-80gb

# Activate environment
source /scratch/gilbreth/$USER/vla-lego/activate_env.sh

# CRITICAL: NCCL settings for single-node
export NCCL_IB_DISABLE=1
export NCCL_NET_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export GLOO_SOCKET_IFNAME=lo

# Optional debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Launch training
torchrun --standalone --nproc_per_node=2 train.py
```

#### Python Template

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

def setup_distributed():
    """Initialize distributed training with proper error handling."""
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    return rank, local_rank, world_size, device

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def main():
    rank, local_rank, world_size, device = setup_distributed()

    try:
        # Synchronize before model creation
        dist.barrier()

        model = YourModel().to(device)
        model = DDP(model, device_ids=[local_rank])

        # ... training loop ...

        # Synchronize before checkpoint
        dist.barrier()

        if rank == 0:
            torch.save(model.module.state_dict(), "checkpoint.pt")
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
```

#### Common Pitfalls to Avoid

| Pitfall | Solution |
|---------|----------|
| Using InfiniBand for single-node | Set `NCCL_IB_DISABLE=1` |
| Hangs at first barrier | Set `NCCL_NET_DISABLE=1` |
| Saving DDP model incorrectly | Use `model.module.state_dict()` |
| Missing output during hangs | Use `flush=True` on prints |
| 10-minute timeout wastes time | Set shorter timeout in `init_process_group()` |
| Rank desync before checkpoint | Add `dist.barrier()` before save |

---

## 7. Test 03: DeepSpeed ZeRO-1

### 7.1 Purpose

Verify DeepSpeed with ZeRO Stage 1 optimization works correctly on Gilbreth:
- DeepSpeed initialization with ZeRO-1 optimizer state partitioning
- BF16 mixed precision training on A100 GPUs
- NCCL communication with DeepSpeed (same settings as DDP)
- DeepSpeed checkpoint save and reload (collective operations)
- Memory efficiency from optimizer state sharding

**ZeRO Stage 1** partitions optimizer states across GPUs, reducing memory per GPU while keeping full model replicas. This is the first step toward larger model training.

### 7.2 Job Configuration

```bash
#SBATCH --job-name=smoke_deepspeed_z1
#SBATCH --account=euge
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=00:30:00
```

### 7.3 Critical NCCL Configuration

**DeepSpeed uses NCCL under the hood** - the same NCCL settings from Test 02 apply:

```bash
# CRITICAL: Disable network transports for single-node
export NCCL_IB_DISABLE=1      # Disable InfiniBand
export NCCL_NET_DISABLE=1     # Disable network transport
export NCCL_P2P_LEVEL=NVL     # Prefer NVLink/P2P
export NCCL_SHM_DISABLE=0     # Enable shared memory

# GLOO settings for rendezvous
export GLOO_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Debugging (optional)
export NCCL_DEBUG=INFO
```

**Important**: These variables MUST be set BEFORE launching the DeepSpeed command.

### 7.4 DeepSpeed Configuration (JSON)

DeepSpeed requires a JSON configuration file. For ZeRO-1, the critical requirement is **including an optimizer configuration**:

```json
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
```

**Critical Configuration Notes**:

| Setting | Requirement | Consequence if Missing |
|---------|-------------|------------------------|
| `optimizer` section | **REQUIRED** for ZeRO-1+ | `AssertionError: zero stage 1 requires an optimizer` |
| `bf16.enabled` | Recommended for A100 | Falls back to FP32 (slower) |
| `train_batch_size` | Must equal `micro_batch × gpus × grad_accum` | Training errors |

### 7.5 Critical Issues Discovered and Solutions

During testing, three critical issues were discovered that caused failures:

#### Issue 1: Missing Optimizer Configuration

**Error**:
```
AssertionError: zero stage 1 requires an optimizer
```

**Cause**: ZeRO Stage 1 partitions optimizer states across GPUs. Without an optimizer definition in the JSON config, DeepSpeed cannot create the optimizer internally.

**Solution**: Always include the `optimizer` section in the DeepSpeed config JSON for ZeRO Stage 1+:
```json
"optimizer": {
    "type": "AdamW",
    "params": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
    }
}
```

#### Issue 2: BF16 Input Dtype Mismatch

**Error**:
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

**Cause**: When BF16 is enabled, the model weights are converted to `bfloat16`, but input tensors created with `torch.randn()` default to `float32`.

**Solution**: Create input tensors with explicit BF16 dtype:
```python
# WRONG - creates float32 tensors
x = torch.randn(batch_size, hidden_size, device=device)

# CORRECT - matches model dtype when BF16 enabled
x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
```

#### Issue 3: Checkpoint Save/Load as Collective Operations

**Error**: Checkpoint directory created but empty, or process hangs during checkpoint operations.

**Cause**: `model_engine.save_checkpoint()` and `model_engine.load_checkpoint()` are **collective operations** - ALL ranks must call them, not just rank 0.

**Solution**: Move checkpoint operations OUTSIDE of `if rank == 0` blocks:
```python
# WRONG - only rank 0 calls checkpoint operations
if rank == 0:
    model_engine.save_checkpoint(ckpt_dir, tag="step100")
    model_engine.load_checkpoint(ckpt_dir, tag="step100")

# CORRECT - all ranks call checkpoint operations
model_engine.save_checkpoint(ckpt_dir, tag="step100")
if rank == 0:
    print(f"Checkpoint saved: {ckpt_dir}")

model_engine.load_checkpoint(ckpt_dir, tag="step100")
if rank == 0:
    print("Checkpoint reload: SUCCESS")
```

### 7.6 Test Model

A larger 6-layer transformer-style MLP (~805M parameters) to benefit from ZeRO-1 optimizer partitioning:

```python
class DummyModel(nn.Module):
    """Larger model to benefit from ZeRO-1 optimizer partitioning."""
    def __init__(self, hidden_size=4096, num_layers=6):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size * 4),  # 4096 -> 16384
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),  # 16384 -> 4096
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
```

| Property | Value |
|----------|-------|
| Parameters | 805,429,248 (~805M) |
| Model Size (FP32) | 3.22 GB |
| Model Size (BF16) | 1.61 GB |
| Hidden Size | 4096 |
| Batch Size per GPU | 32 |
| Global Batch Size | 64 (32 × 2 GPUs) |
| Optimizer | AdamW (DeepSpeed FusedAdam) |
| Loss | MSELoss |

### 7.7 DeepSpeed Training Script Structure

The training script follows this pattern for DeepSpeed:

```python
import os
import torch
import torch.nn as nn
import deepspeed
import argparse

def main():
    # 1. Parse arguments (DeepSpeed adds its own args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 2. Initialize DeepSpeed distributed
    deepspeed.init_distributed()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 3. Create model (DO NOT move to device - DeepSpeed handles this)
    model = DummyModel(hidden_size=4096, num_layers=6)

    # 4. Initialize DeepSpeed engine (handles optimizer, distributed wrapper)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    # 5. Training loop with DeepSpeed API
    criterion = nn.MSELoss()

    for step in range(num_steps):
        # Create input in BF16 to match model dtype
        x = torch.randn(32, 4096, device=device, dtype=torch.bfloat16)
        y = torch.randn(32, 4096, device=device, dtype=torch.bfloat16)

        # Forward pass through DeepSpeed engine
        output = model_engine(x)
        loss = criterion(output, y)

        # Backward pass (DeepSpeed handles gradient sync)
        model_engine.backward(loss)

        # Optimizer step (DeepSpeed handles ZeRO communication)
        model_engine.step()

    # 6. Save checkpoint (COLLECTIVE operation - all ranks must call)
    ckpt_dir = "checkpoints/deepspeed_z1"
    model_engine.save_checkpoint(ckpt_dir, tag="final")

    # 7. Verify checkpoint reload (COLLECTIVE operation)
    model_engine.load_checkpoint(ckpt_dir, tag="final")

    if rank == 0:
        print("SMOKE TEST DEEPSPEED ZeRO-1: PASSED")

if __name__ == "__main__":
    main()
```

**Key Differences from PyTorch DDP**:

| Aspect | PyTorch DDP | DeepSpeed ZeRO-1 |
|--------|-------------|------------------|
| Initialization | `dist.init_process_group()` | `deepspeed.init_distributed()` |
| Model wrapper | `DDP(model, device_ids=[...])` | `deepspeed.initialize(model=model)` |
| Optimizer | Created manually | Defined in JSON config |
| Backward | `loss.backward()` | `model_engine.backward(loss)` |
| Optimizer step | `optimizer.step()` | `model_engine.step()` |
| Gradient zeroing | `optimizer.zero_grad()` | Automatic |
| Checkpoint save | `torch.save()` (rank 0 only) | `model_engine.save_checkpoint()` (all ranks) |
| Model access | `model.module` | `model_engine.module` |

### 7.8 Launch Command

Use the `deepspeed` launcher for single-node training:

```bash
deepspeed --num_gpus=2 /path/to/training_script.py \
    --deepspeed_config /path/to/ds_config.json
```

**Parameters**:
- `--num_gpus`: Number of GPUs to use (autodetected if not specified)
- `--deepspeed_config`: Path to DeepSpeed JSON configuration

**Do NOT use** for single-node:
- `--hostfile` (requires SSH between nodes, not available on Gilbreth)
- `--master_addr`/`--master_port` (handled automatically)

### 7.9 Verified Results (Job 10213866)

**Execution Details**:
| Metric | Value |
|--------|-------|
| Job ID | 10213866 |
| Node | gilbreth-i004 |
| Date | January 23, 2026 |
| Total Training Time | 23.14 seconds |
| JIT Compilation Time | ~22 seconds (FusedAdam kernel, first run only) |

**Distributed Configuration**:
| Component | Value |
|-----------|-------|
| World Size | 2 |
| Backend | nccl (via DeepSpeed) |
| Communication | SHM/direct/direct |
| ZeRO Stage | 1 |
| Precision | BF16 |

**Training Results**:
| Metric | Value |
|--------|-------|
| Training Steps | 100 |
| Throughput | 4.3 steps/sec |
| Samples/sec | ~275 (32 batch × 2 GPUs × 4.3) |
| Peak GPU Memory | 9.68 GB per device |
| Model Memory (BF16) | ~6.02 GB |
| Optimizer Memory (partitioned) | ~3 GB per GPU |

**Loss Progression**:
```
Step 25/100  | Loss: 1.0078
Step 50/100  | Loss: 1.0000
Step 75/100  | Loss: 1.0156
Step 100/100 | Loss: 1.0078
```

**Checkpoint Files Created**:
```
checkpoints/deepspeed_z1/step100/
├── mp_rank_00_model_states.pt           (1.61 GB - model weights)
├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt  (4.83 GB - rank 0 optimizer)
├── bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt  (4.83 GB - rank 1 optimizer)
└── zero_to_fp32.py                      (recovery script)
```

**DeepSpeed Engine Configuration (from logs)**:
```
DeepSpeed Basic Optimizer = FusedAdam
Creating torch.bfloat16 ZeRO stage 1 optimizer
Reduce bucket size 500000000
Allgather bucket size 500000000
CPU Offload: False
```

### 7.10 Log Files

- **Output**: `/scratch/gilbreth/$USER/vla-lego/logs/smoke_deepspeed_z1_10213866.out`
- **Errors**: `/scratch/gilbreth/$USER/vla-lego/logs/smoke_deepspeed_z1_10213866.err`

**Expected Warning Messages (Safe to Ignore)**:
```
FutureWarning: The pynvml package is deprecated
UserWarning: Positional args are being deprecated, use kwargs instead
WARNING: Unable to find hostfile, will proceed with training with local resources only
```

### 7.11 Success Criteria

- [x] DeepSpeed initialization completes (world_size=2)
- [x] NCCL uses SHM/direct (not network transport)
- [x] ZeRO Stage 1 optimizer created with state partitioning
- [x] BF16 precision enabled and working
- [x] FusedAdam optimizer JIT compiled
- [x] Forward pass completes on both ranks
- [x] Backward pass with gradient sync completes
- [x] 100 training steps complete
- [x] Checkpoint saves successfully (all optimizer states)
- [x] Final status: **PASSED**

### 7.12 Key Learnings

1. **Optimizer Config is Mandatory**: Unlike PyTorch DDP where you create the optimizer manually, ZeRO Stage 1+ requires the optimizer to be defined in the DeepSpeed JSON config. This allows DeepSpeed to partition optimizer states correctly.

2. **BF16 Dtype Consistency**: When BF16 is enabled, model weights are automatically converted to `bfloat16`. Input tensors must be created with matching dtype or you'll get dtype mismatch errors.

3. **Collective Checkpoint Operations**: DeepSpeed's `save_checkpoint()` and `load_checkpoint()` are collective operations involving all ranks (for gathering optimizer states). Calling them only on rank 0 causes deadlock.

4. **First-Run JIT Compilation**: The first training run includes ~22 seconds of JIT compilation for FusedAdam CUDA kernels. Subsequent runs (or precompiled kernels) will be faster.

5. **Same NCCL Settings as DDP**: DeepSpeed uses NCCL for communication. The single-node NCCL settings (disabling IB/network transport) apply equally to DeepSpeed.

6. **Memory Efficiency**: With ZeRO-1, each GPU stores only its partition of optimizer states (~3 GB per GPU instead of ~6 GB total), enabling larger models or batch sizes.

### 7.13 Best Practices for DeepSpeed ZeRO-1 on Gilbreth

#### Environment Setup Template (Bash)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --time=00:30:00

# Activate environment
source /scratch/gilbreth/$USER/vla-lego/activate_env.sh

# CRITICAL: NCCL settings for single-node (same as DDP)
export NCCL_IB_DISABLE=1
export NCCL_NET_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export GLOO_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Optional debugging
export NCCL_DEBUG=INFO

# Create DeepSpeed config
cat << 'EOF' > /tmp/ds_config.json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-4, "weight_decay": 0.01}
    },
    "zero_optimization": {"stage": 1},
    "bf16": {"enabled": true},
    "gradient_clipping": 1.0
}
EOF

# Launch training
deepspeed --num_gpus=2 train.py --deepspeed_config /tmp/ds_config.json
```

#### Python Training Template

```python
import os
import torch
import torch.nn as nn
import deepspeed
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Initialize distributed
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Create model (don't move to device)
    model = YourModel()

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    # Determine dtype from config
    dtype = torch.bfloat16 if model_engine.bfloat16_enabled() else torch.float32

    # Training loop
    for step in range(num_steps):
        x = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
        y = get_labels(...)

        output = model_engine(x)
        loss = criterion(output, y)
        model_engine.backward(loss)
        model_engine.step()

    # Checkpoint (COLLECTIVE - all ranks must call)
    model_engine.save_checkpoint("checkpoints/", tag="final")

    # Verify reload (COLLECTIVE)
    model_engine.load_checkpoint("checkpoints/", tag="final")

if __name__ == "__main__":
    main()
```

#### Common Pitfalls to Avoid

| Pitfall | Error/Symptom | Solution |
|---------|---------------|----------|
| Missing optimizer in JSON | `AssertionError: zero stage 1 requires an optimizer` | Add `optimizer` section to JSON config |
| BF16 input dtype mismatch | `RuntimeError: mat1 and mat2 must have the same dtype` | Use `dtype=torch.bfloat16` for inputs |
| Rank-0 only checkpoint | Hangs or empty checkpoint dir | Call `save_checkpoint()`/`load_checkpoint()` from ALL ranks |
| Using InfiniBand for single-node | Timeout after NCCL init | Set `NCCL_IB_DISABLE=1` and `NCCL_NET_DISABLE=1` |
| Moving model to device before DeepSpeed | Memory issues or errors | Let DeepSpeed handle device placement |
| Using `--hostfile` on Gilbreth | SSH connection refused | Use `--num_gpus` flag instead |

#### DeepSpeed vs PyTorch DDP Decision Guide

| Use DeepSpeed ZeRO-1 When | Use PyTorch DDP When |
|---------------------------|----------------------|
| Model optimizer states exceed single GPU memory | Optimizer states fit in GPU memory |
| Training models >1B parameters | Training models <1B parameters |
| Need gradient checkpointing integration | Simple gradient sync is sufficient |
| Planning to scale to ZeRO-2/3 later | No plans for ZeRO optimization |
| Using HuggingFace Trainer/Accelerate | Using custom training loops |

---

## 8. Test 04: Multi-Node 8 GPU DeepSpeed

### 8.1 Purpose

Verify multi-node distributed training with DeepSpeed ZeRO-1 across 4 nodes with 8 GPUs total:
- Cross-node NCCL communication via InfiniBand
- Intra-node GPU-to-GPU communication via NVLink/P2P
- Multi-node DeepSpeed initialization and training
- Checkpoint save/reload across distributed ranks
- Proper launcher configuration without SSH

**This test validates the full distributed training stack required for large-scale VLA model training.**

### 8.2 Job Configuration

```bash
#SBATCH --job-name=smoke_ds_8gpu
#SBATCH --account=euge
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal              # Multi-node requires allocation

#SBATCH --nodes=4                 # 4 nodes
#SBATCH --ntasks-per-node=1       # 1 srun task per node (torchrun handles GPUs)
#SBATCH --cpus-per-task=32        # CPUs per node
#SBATCH --gpus-per-node=2         # 2 GPUs per node = 8 total
#SBATCH --mem=200G                # Memory per node
#SBATCH --time=01:00:00
#SBATCH --exclusive               # Exclusive node access
```

**Key Configuration Points**:
- `--ntasks-per-node=1`: srun launches ONE task per node; `torchrun` spawns GPU processes
- `--exclusive`: Ensures no resource contention from other jobs
- `--gpus-per-node=2`: Each A100 node has 2 GPUs

### 8.3 Critical NCCL Configuration for Multi-Node

**The single most important section for multi-node training on Gilbreth.**

Unlike single-node training (which disables network transport), multi-node training **requires** InfiniBand for cross-node communication.

```bash
# Multi-node NCCL settings
export NCCL_DEBUG=INFO              # Debug logging
export NCCL_IB_DISABLE=0            # ENABLE InfiniBand (opposite of single-node!)
export NCCL_TIMEOUT=600             # 10-minute timeout
export NCCL_P2P_LEVEL=NVL           # NVLink for intra-node GPU-to-GPU
export NCCL_SHM_DISABLE=0           # Enable shared memory
export CUDA_DEVICE_ORDER=PCI_BUS_ID # Consistent GPU ordering

# CRITICAL: Do NOT set these - let NCCL auto-detect
unset NCCL_SOCKET_IFNAME
unset GLOO_SOCKET_IFNAME
```

**Why NOT set `NCCL_SOCKET_IFNAME`?**

InfiniBand interface names **vary by node** on Gilbreth:

| Node | Interface |
|------|-----------|
| gilbreth-k023 | `ibp65s0` |
| gilbreth-k024 | `ibp65s0` |
| gilbreth-k025 | `ibp65s0` |
| gilbreth-i004 | `ibp161s0` |

Hardcoding any interface name causes failures on nodes with different names:
```
Bootstrap : no socket interface found
```

**Solution**: Let NCCL auto-detect the InfiniBand interface on each node.

### 8.4 Comparison: Single-Node vs Multi-Node NCCL

| Setting | Single-Node (Tests 02, 03) | Multi-Node (Test 04) |
|---------|---------------------------|----------------------|
| `NCCL_IB_DISABLE` | `1` (disable) | `0` (enable) |
| `NCCL_NET_DISABLE` | `1` (disable) | NOT SET |
| `NCCL_SOCKET_IFNAME` | `lo` or unset | **MUST be unset** |
| `GLOO_SOCKET_IFNAME` | `lo` | **MUST be unset** |
| `NCCL_P2P_LEVEL` | `NVL` | `NVL` |
| `NCCL_SHM_DISABLE` | `0` | `0` |
| Communication | `SHM/direct/direct` | `NET/IB/0` (inter-node) + `P2P/CUMEM` (intra-node) |

### 8.5 Multi-Node Launch Pattern (srun + torchrun)

DeepSpeed's native launcher (`deepspeed --hostfile`) uses pdsh/SSH which is **not available** on Gilbreth. The working pattern is:

```bash
#!/bin/bash

# === MASTER NODE DISCOVERY ===
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

# === NCCL CONFIGURATION ===
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
unset NCCL_SOCKET_IFNAME
unset GLOO_SOCKET_IFNAME
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_TIMEOUT=600
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# === WORLD SIZE CALCULATION ===
GPUS_PER_NODE=2
NNODES=$SLURM_JOB_NUM_NODES
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

export GPUS_PER_NODE NNODES WORLD_SIZE

# === SHARED DIRECTORY (NFS-accessible) ===
SHARED_DIR=$HOME/vla_lego_multinode
mkdir -p $SHARED_DIR
export SHARED_DIR

# === LAUNCH ===
srun --ntasks=$NNODES --ntasks-per-node=1 --export=ALL bash -c '
    # Node-specific setup
    export NODE_RANK=$SLURM_NODEID

    # Re-export NCCL (critical for torchrun subprocesses)
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    unset NCCL_SOCKET_IFNAME
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
        $SHARED_DIR/training_script.py \
        --deepspeed_config $SHARED_DIR/ds_config.json
'
```

**Critical Elements**:

| Element | Purpose |
|---------|---------|
| `scontrol show hostnames` | Gets SLURM nodelist as hostnames |
| `--export=ALL` | Propagates ALL env vars to srun subshells |
| `$SLURM_NODEID` | Unique node rank (0, 1, 2, 3) per node |
| Re-export NCCL vars | Environment may not propagate fully through torchrun |
| `$SHARED_DIR` | Must be NFS-shared, NOT `/tmp/` |

### 8.6 DeepSpeed Configuration for Multi-Node

```json
{
    "train_batch_size": 256,
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
```

**Key Configuration Notes**:
- `train_batch_size`: Must equal `micro_batch × gpus × grad_accum` (32 × 8 × 1 = 256)
- `optimizer` section: **Required** for ZeRO Stage 1+
- `bf16.enabled`: Use BF16 on A100 for efficiency
- `reduce_bucket_size`/`allgather_bucket_size`: 500MB buckets for communication

### 8.7 Test Model

An 8-layer transformer-style MLP (~1B parameters) to test multi-node scaling:

```python
class DummyModel(nn.Module):
    """Larger model to simulate real training."""
    def __init__(self, hidden_size=4096, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size * 4),  # 4096 -> 16384
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),  # 16384 -> 4096
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
```

| Property | Value |
|----------|-------|
| Parameters | 1,073,905,664 (~1.07B) |
| Model Size (FP32) | 4.30 GB |
| Model Size (BF16) | 2.15 GB |
| Hidden Size | 4096 |
| Layers | 8 |
| Batch Size per GPU | 32 |
| Global Batch Size | 256 (32 × 8 GPUs) |

### 8.8 Training Script Key Patterns

```python
import os
import socket
import torch
import torch.nn as nn
import deepspeed
import argparse

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

    # Debug: Print rank info with flush
    print(f"[Rank {rank}/{world_size}] Node: {hostname}, GPU: {local_rank}", flush=True)

    # Synchronization barrier
    print(f"[Rank {rank}] Entering barrier...", flush=True)
    torch.distributed.barrier()
    print(f"[Rank {rank}] Barrier passed!", flush=True)

    # Create model
    model = DummyModel(hidden_size=4096, num_layers=8)

    # DeepSpeed initialization
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    criterion = nn.MSELoss()

    # Training loop
    for step in range(100):
        # CRITICAL: Match input dtype to model (BF16)
        x = torch.randn(32, 4096, device=device, dtype=torch.bfloat16)
        y = torch.randn(32, 4096, device=device, dtype=torch.bfloat16)

        output = model_engine(x)
        loss = criterion(output, y)

        model_engine.backward(loss)
        model_engine.step()

        # Aggregate loss across all ranks for logging
        if (step + 1) % 25 == 0:
            loss_tensor = loss.detach().clone()
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
            if rank == 0:
                print(f"  Step {step+1}/100 | Avg Loss: {loss_tensor.item():.4f}")

    # CRITICAL: Checkpoint is COLLECTIVE - ALL ranks must call
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_dir = "checkpoints/deepspeed_multinode"
    model_engine.save_checkpoint(ckpt_dir, tag="step100")
    if rank == 0:
        print(f"Checkpoint saved: {ckpt_dir}")

    # CRITICAL: Load is also COLLECTIVE
    _, client_state = model_engine.load_checkpoint(ckpt_dir, tag="step100")
    if rank == 0:
        print("Checkpoint reload: SUCCESS ✓")
        print(f"\nMULTI-NODE DEEPSPEED ({world_size} GPUs): PASSED ✓")

    torch.distributed.barrier()

if __name__ == "__main__":
    main()
```

**Critical Code Patterns**:

| Pattern | Reason |
|---------|--------|
| `flush=True` on prints | Ensures output visible before potential hangs |
| Barrier after init | Confirms all ranks initialized before training |
| `dtype=torch.bfloat16` | Matches model dtype when BF16 enabled |
| Checkpoint outside `if rank == 0` | Save/load are collective operations |
| `all_reduce` for loss | Aggregates loss across all nodes for logging |

### 8.9 Verified Results (Job 10213927)

**Execution Details**:

| Metric | Value |
|--------|-------|
| Job ID | 10213927 |
| Date | January 23, 2026 |
| Nodes | gilbreth-i004, gilbreth-k023, gilbreth-k024, gilbreth-k025 |
| Total GPUs | 8 (4 nodes × 2 GPUs) |
| Total Duration | 74.6 seconds |
| Training Steps | 100 |

**Distributed Configuration**:

| Component | Value |
|-----------|-------|
| World Size | 8 |
| Backend | NCCL (via DeepSpeed) |
| Inter-node Comm | `NET/IB/0` (InfiniBand) |
| Intra-node Comm | `P2P/CUMEM` (NVLink) |
| ZeRO Stage | 1 |
| Precision | BF16 |

**Training Results**:

| Metric | Value |
|--------|-------|
| Throughput | 1.3 steps/sec |
| Samples/sec (world) | 10.7 |
| Peak GPU Memory | 7.12 GB |
| Model Parameters | 1,073,905,664 (~1.07B) |

**Loss Progression**:
```
Step 25/100  | Avg Loss: 1.0000
Step 50/100  | Avg Loss: 1.0000
Step 75/100  | Avg Loss: 1.0000
Step 100/100 | Avg Loss: 1.0000
```

**Checkpoint Files Created**:
```
checkpoints/deepspeed_multinode/step100/
├── mp_rank_00_model_states.pt               (model weights)
├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
├── bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt
├── bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt
├── bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt
├── bf16_zero_pp_rank_4_mp_rank_00_optim_states.pt
├── bf16_zero_pp_rank_5_mp_rank_00_optim_states.pt
├── bf16_zero_pp_rank_6_mp_rank_00_optim_states.pt
├── bf16_zero_pp_rank_7_mp_rank_00_optim_states.pt
└── zero_to_fp32.py
```

**NCCL Communication (from logs)**:
```
gilbreth-i004: NCCL INFO NET/IB : Using [0]ibp161s0:1/RoCE/DETH
gilbreth-k023: NCCL INFO NET/IB : Using [0]ibp65s0:1/RoCE/DETH
gilbreth-k024: NCCL INFO NET/IB : Using [0]ibp65s0:1/RoCE/DETH
gilbreth-k025: NCCL INFO NET/IB : Using [0]ibp65s0:1/RoCE/DETH
NCCL INFO Channel 00 : 0[0] -> 4[0] via P2P/CUMEM
NCCL INFO Channel 00 : 0[0] -> 2[0] via NET/IB/0
```

### 8.10 Log Files

- **Output**: `/scratch/gilbreth/$USER/vla-lego/logs/smoke_ds_8gpu_10213927.out`
- **Errors**: `/scratch/gilbreth/$USER/vla-lego/logs/smoke_ds_8gpu_10213927.err`

**Expected Warning Messages (Safe to Ignore)**:
```
FutureWarning: The pynvml package is deprecated
Use of Purdue BoilerKey or SSH keys is Required  # From pdsh attempt (if any)
```

### 8.11 Success Criteria

- [x] All 4 nodes allocated (8 GPUs total)
- [x] NCCL uses InfiniBand for cross-node (`NET/IB/0`)
- [x] NCCL uses P2P/NVLink for intra-node (`P2P/CUMEM`)
- [x] All 8 ranks initialize and pass barrier
- [x] DeepSpeed ZeRO-1 engine created
- [x] BF16 precision enabled
- [x] 100 training steps complete
- [x] Loss aggregated correctly across ranks
- [x] Checkpoint saves successfully (8 optimizer state files)
- [x] Checkpoint reloads successfully
- [x] Final status: **PASSED ✓**

### 8.12 Key Learnings and Pitfalls

#### Failure Mode 1: Scripts in /tmp/

**Symptom**:
```
can't open file '/tmp/multinode_ds_test.py': [Errno 2] No such file or directory
```

**Cause**: Scripts written to `/tmp/` are node-local; other nodes cannot access them.

**Fix**: Use shared filesystem (`$HOME` or `/scratch/gilbreth/`):
```bash
SHARED_DIR=$HOME/vla_lego_multinode
mkdir -p $SHARED_DIR
cat << 'PYEOF' > $SHARED_DIR/training_script.py
...
PYEOF
```

#### Failure Mode 2: Hardcoded NCCL_SOCKET_IFNAME

**Symptom**:
```
Bootstrap : no socket interface found
```

**Cause**: InfiniBand interface names vary by node (`ibp65s0` vs `ibp161s0`).

**Fix**: Never set `NCCL_SOCKET_IFNAME` for multi-node. Let NCCL auto-detect:
```bash
unset NCCL_SOCKET_IFNAME
unset GLOO_SOCKET_IFNAME
```

#### Failure Mode 3: SSH-Based Launchers

**Symptom**:
```
***** Use of Purdue BoilerKey or SSH keys is Required ******
pdsh@node: ssh exited with exit code 2
```

**Cause**: DeepSpeed's native launcher uses pdsh/SSH which is blocked on Gilbreth.

**Fix**: Use `srun` + `torchrun` instead of `deepspeed --hostfile`:
```bash
srun --ntasks=$NNODES --ntasks-per-node=1 --export=ALL bash -c '
    torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE ...
'
```

#### Failure Mode 4: Environment Not Propagating

**Symptom**: NCCL hangs after initialization, or uses wrong transport.

**Cause**: Environment variables may not fully propagate through srun → torchrun → Python.

**Fix**: Re-export NCCL variables inside the srun bash command:
```bash
srun ... bash -c '
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    unset NCCL_SOCKET_IFNAME
    ...
    torchrun ...
'
```

#### Failure Mode 5: Checkpoint Operations Only on Rank 0

**Symptom**: Checkpoint directory created but empty, or process hangs.

**Cause**: `save_checkpoint()` and `load_checkpoint()` are collective operations.

**Fix**: Call from ALL ranks, not just rank 0:
```python
# WRONG
if rank == 0:
    model_engine.save_checkpoint(...)

# CORRECT
model_engine.save_checkpoint(...)  # All ranks
if rank == 0:
    print("Checkpoint saved")       # Only rank 0 logs
```

### 8.13 Best Practices for Multi-Node Training on Gilbreth

#### Complete Job Template

```bash
#!/bin/bash
#SBATCH --job-name=multinode_train
#SBATCH --account=euge
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node List: $SLURM_JOB_NODELIST"

# === ENVIRONMENT ===
cd $SLURM_SUBMIT_DIR
source /scratch/gilbreth/$(whoami)/vla-lego/activate_env.sh 2>/dev/null || {
    module purge
    module load external cuda/12.1.1 anaconda/2024.10-py312
    conda activate vla_lego_env
}

# Remove single-node interface config from activate_env.sh
unset NCCL_SOCKET_IFNAME
unset GLOO_SOCKET_IFNAME

mkdir -p logs checkpoints

# === MASTER NODE ===
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "Master: $MASTER_ADDR:$MASTER_PORT"

# === NCCL FOR MULTI-NODE ===
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0        # Enable InfiniBand
export NCCL_TIMEOUT=600
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# === WORLD SIZE ===
GPUS_PER_NODE=2
NNODES=$SLURM_JOB_NUM_NODES
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export GPUS_PER_NODE NNODES WORLD_SIZE

echo "World Size: $WORLD_SIZE GPUs"

# === SHARED CONFIG DIRECTORY ===
SHARED_DIR=$HOME/vla_lego_multinode
mkdir -p $SHARED_DIR
export SHARED_DIR

# === CREATE DEEPSPEED CONFIG ===
cat << DSCONFIG > $SHARED_DIR/ds_config.json
{
    "train_batch_size": $((32 * WORLD_SIZE)),
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}
    },
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 500000000,
        "allgather_bucket_size": 500000000,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "bf16": {"enabled": true},
    "gradient_clipping": 1.0
}
DSCONFIG

# === LAUNCH ===
srun --ntasks=$NNODES --ntasks-per-node=1 --export=ALL bash -c '
    export NODE_RANK=$SLURM_NODEID

    # Re-export for torchrun subprocesses
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    unset NCCL_SOCKET_IFNAME
    export NCCL_P2P_LEVEL=NVL
    export NCCL_SHM_DISABLE=0
    export NCCL_TIMEOUT=600
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    echo "$(hostname): NODE_RANK=$NODE_RANK"

    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $SHARED_DIR/train.py \
        --deepspeed_config $SHARED_DIR/ds_config.json
'

echo "JOB COMPLETE: $(date)"
```

#### Checklist Before Submission

- [ ] Scripts on shared filesystem (`$HOME` or `/scratch`), not `/tmp/`
- [ ] `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` unset
- [ ] `NCCL_IB_DISABLE=0` (enable InfiniBand)
- [ ] Using `srun + torchrun`, not `deepspeed --hostfile`
- [ ] `--export=ALL` on srun command
- [ ] NCCL vars re-exported inside srun bash command
- [ ] Checkpoint operations called by ALL ranks
- [ ] Input tensors have correct dtype (BF16 if enabled)
- [ ] DeepSpeed JSON includes `optimizer` section

#### Scaling Recommendations

| Nodes | GPUs | Global Batch | Notes |
|-------|------|--------------|-------|
| 4 | 8 | 256 | Tested and verified |
| 8 | 16 | 512 | Scale linearly |
| 16 | 32 | 1024 | Consider ZeRO-2 |
| 32+ | 64+ | 2048+ | Consider ZeRO-3 |

**Scaling batch size**: Maintain same per-GPU batch size (32) and increase global batch proportionally.

### 8.14 Debugging Multi-Node Issues

#### Step 1: Verify Node Allocation
```bash
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c '
    echo "$(hostname): $(nvidia-smi -L | wc -l) GPUs"
'
```

#### Step 2: Check Network Interfaces
```bash
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c '
    echo "$(hostname): $(ip -o -4 addr show | grep ib | awk "{print \$2}")"
'
```

#### Step 3: Test NCCL Communication
```bash
# Set NCCL_DEBUG=TRACE for detailed communication logs
export NCCL_DEBUG=TRACE
```

#### Step 4: Check for Hangs
Add barriers with print statements:
```python
print(f"[Rank {rank}] Before barrier", flush=True)
torch.distributed.barrier()
print(f"[Rank {rank}] After barrier", flush=True)
```

#### Step 5: Monitor GPU Utilization
```bash
# On each node
watch -n 1 nvidia-smi
```

---

## 9. Troubleshooting Reference

### 9.1 Known Warnings (Safe to Ignore)

| Warning | Source | Impact |
|---------|--------|--------|
| `FutureWarning: The pynvml package is deprecated` | PyTorch | None |

### 9.2 Common Issues and Fixes

#### Invalid Account Error
```bash
# Fix: Replace account name in all job templates
sed -i 's/euge/YOUR_ACCOUNT/g' job_templates/*.sh
```

#### NCCL Timeout on Single-Node Multi-GPU
```bash
# Fix: Disable network transports, use P2P/SHM
export NCCL_IB_DISABLE=1
export NCCL_NET_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
```

#### NCCL Timeout on Multi-Node
```bash
# Fix: Do NOT set NCCL_SOCKET_IFNAME (interface names vary)
unset NCCL_SOCKET_IFNAME
unset GLOO_SOCKET_IFNAME
export NCCL_IB_DISABLE=0   # Enable InfiniBand
```

#### Multi-Node Scripts Not Found
```bash
# Symptom: "can't open file '/tmp/script.py': No such file or directory"
# Cause: /tmp/ is node-local, not shared

# Fix: Use shared filesystem
SHARED_DIR=$HOME/vla_lego_multinode
mkdir -p $SHARED_DIR
# Write scripts to $SHARED_DIR, not /tmp/
```

#### SSH/pdsh Failures on Multi-Node
```bash
# Symptom: "Use of Purdue BoilerKey or SSH keys is Required"
# Cause: DeepSpeed's native launcher uses SSH which is blocked

# Fix: Use srun + torchrun instead of deepspeed --hostfile
srun --ntasks=$NNODES --ntasks-per-node=1 --export=ALL bash -c '
    torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE ...
'
```

#### NCCL Bootstrap Interface Error
```bash
# Symptom: "Bootstrap : no socket interface found"
# Cause: Hardcoded interface name doesn't exist on some nodes

# Fix: Let NCCL auto-detect interfaces
unset NCCL_SOCKET_IFNAME
# Never hardcode: ibp65s0, ibp161s0, etc. (vary by node)
```

#### DeepSpeed Import Error
```bash
pip uninstall deepspeed -y && pip install deepspeed==0.14.0
```

#### Module Not Found
```bash
# Fix: Ensure external module is loaded first
module purge
module load external
module load cuda/12.1.1 anaconda/2024.10-py312
```

#### NFS Stale Handle
The scratch filesystem may show "Cannot send after transport endpoint shutdown". Workarounds:
- Wait a few minutes and retry
- Use `sacct -j JOBID` to check job status
- Try from a fresh shell session
- Use `$HOME` instead of `/scratch` for small files (configs, scripts)

### 9.3 Debugging Distributed Training

1. Add `dist.barrier()` with print statements to isolate hangs
2. Use `flush=True` on all prints for immediate output
3. Set `NCCL_DEBUG=INFO` for NCCL details
4. Set `TORCH_DISTRIBUTED_DEBUG=DETAIL` for PyTorch details
5. Check for "Init COMPLETE" in logs (means NCCL initialized)
6. If init succeeds but operations hang, check transport method

### 9.4 Interpreting NCCL Logs

| Log Message | Meaning |
|-------------|---------|
| `via P2P/CUMEM` | Good - using GPU direct memory |
| `via SHM/direct/direct` | Good - using shared memory |
| `via NET/...` | Network transport (may cause issues on single-node) |
| `Init COMPLETE` | NCCL initialized successfully |
| `Abort COMPLETE` | Normal cleanup after `dist.destroy_process_group()` |

---

## Appendix A: File Locations

| File | Path |
|------|------|
| Job Templates | `gilbreth_phase0/job_templates/` |
| Setup Scripts | `gilbreth_phase0/scripts/` |
| DeepSpeed Configs | `gilbreth_phase0/configs/` |
| Logs (symlink) | `gilbreth_phase0/logs/` -> `/scratch/gilbreth/$USER/vla-lego/logs/` |
| Checkpoints (symlink) | `gilbreth_phase0/checkpoints/` -> `/scratch/gilbreth/$USER/vla-lego/checkpoints/` |
| Activation Script | `/scratch/gilbreth/$USER/vla-lego/activate_env.sh` |

---

## Appendix B: DeepSpeed ZeRO-1 Configuration

```json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",

    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    },

    "bf16": {
        "enabled": true
    },

    "gradient_clipping": 1.0,
    "zero_allow_untested_optimizer": true
}
```

**Important Notes**:
- BF16 is enabled by default (optimal for A100)
- Input tensors must match model dtype when BF16 enabled
- Checkpoint save/load are collective operations - ALL ranks must call them
- ZeRO stage 1+ requires optimizer config in DeepSpeed JSON

---

*Document maintained as part of VLA-LEGO Phase 0.1 verification.*
