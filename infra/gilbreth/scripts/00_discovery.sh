#!/bin/bash
#===============================================================================
# Gilbreth Discovery Script - Complete Phase 0.1
# Purpose: Gather ALL info needed for Phase 0.1 setup
# Usage: bash 00_discovery.sh 2>&1 | tee discovery_output.txt
#===============================================================================

set -e

echo "=============================================="
echo "GILBRETH PHASE 0.1 DISCOVERY REPORT"
echo "Generated: $(date)"
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo "=============================================="

#-------------------------------------------------------------------------------
# SECTION 1: Account & Queue Access
#-------------------------------------------------------------------------------
echo ""
echo "=== 1. YOUR ACCOUNTS & GPU ALLOCATION ==="
echo "----------------------------------------------"
echo "Command: slist"
slist 2>&1 || echo "[slist not available]"

echo ""
echo "Command: sacctmgr show user $(whoami) withassoc"
sacctmgr show user $(whoami) withassoc format=Account%15,Partition%15,QOS%30,MaxNodes,MaxSubmit,GrpTRES%30 2>&1

#-------------------------------------------------------------------------------
# SECTION 2: Partition Access
#-------------------------------------------------------------------------------
echo ""
echo "=== 2. PARTITION STATUS ==="
echo "----------------------------------------------"

for partition in a100-80gb a100-40gb a30 v100 training; do
    echo ""
    echo "Partition: $partition"
    sinfo -p $partition -o "%P %a %l %D %c %m %G" 2>&1 | head -5 || echo "  [Not accessible]"
done

#-------------------------------------------------------------------------------
# SECTION 3: QOS Time Limits
#-------------------------------------------------------------------------------
echo ""
echo "=== 3. QOS TIME LIMITS & POLICIES ==="
echo "----------------------------------------------"
sacctmgr show qos format=Name%15,MaxWall,MaxTRESPU%40,Priority,MaxJobsPU 2>&1

#-------------------------------------------------------------------------------
# SECTION 4: Available CUDA Modules (CRITICAL for version pinning)
#-------------------------------------------------------------------------------
echo ""
echo "=== 4. AVAILABLE CUDA VERSIONS ==="
echo "----------------------------------------------"
module avail cuda 2>&1 | grep -E "cuda" || echo "[No CUDA modules found]"

#-------------------------------------------------------------------------------
# SECTION 5: Available Python/Anaconda Modules
#-------------------------------------------------------------------------------
echo ""
echo "=== 5. AVAILABLE PYTHON/ANACONDA ==="
echo "----------------------------------------------"
module avail anaconda 2>&1 | grep -E "anaconda" | head -10
echo ""
module avail python 2>&1 | grep -E "python" | head -10

#-------------------------------------------------------------------------------
# SECTION 6: Available Deep Learning Modules
#-------------------------------------------------------------------------------
echo ""
echo "=== 6. AVAILABLE ML MODULES ==="
echo "----------------------------------------------"
echo "PyTorch:"
module avail pytorch 2>&1 | grep -E "pytorch" | head -5
echo ""
echo "cuDNN:"
module avail cudnn 2>&1 | grep -E "cudnn" | head -5
echo ""
echo "NCCL:"
module avail nccl 2>&1 | grep -E "nccl" | head -5

#-------------------------------------------------------------------------------
# SECTION 7: GPU Driver Version (from front-end)
#-------------------------------------------------------------------------------
echo ""
echo "=== 7. NVIDIA DRIVER (front-end) ==="
echo "----------------------------------------------"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "[No GPU on front-end]"

#-------------------------------------------------------------------------------
# SECTION 8: Storage Paths & Quotas
#-------------------------------------------------------------------------------
echo ""
echo "=== 8. STORAGE PATHS & QUOTAS ==="
echo "----------------------------------------------"
echo "HOME: $HOME"
echo "SCRATCH: /scratch/gilbreth/$(whoami)"

echo ""
echo "Quota info:"
myquota 2>&1 || quota -s 2>&1 || echo "[quota command not available]"

echo ""
echo "Home usage:"
du -sh $HOME 2>&1 || echo "[Cannot check home usage]"

echo ""
if [ -d "/scratch/gilbreth/$(whoami)" ]; then
    echo "Scratch exists: YES"
    du -sh /scratch/gilbreth/$(whoami) 2>&1
else
    echo "Scratch exists: NO (will create)"
    mkdir -p /scratch/gilbreth/$(whoami)
    echo "Created: /scratch/gilbreth/$(whoami)"
fi

#-------------------------------------------------------------------------------
# SECTION 9: Network Interfaces (for NCCL multi-node)
#-------------------------------------------------------------------------------
echo ""
echo "=== 9. NETWORK INTERFACES (for NCCL) ==="
echo "----------------------------------------------"
ip -brief addr show 2>&1 | grep -E "(ib|eth|en)" || echo "[Check on compute node]"

echo ""
echo "InfiniBand status:"
ibstat 2>&1 | head -20 || echo "[ibstat not available on front-end]"

#-------------------------------------------------------------------------------
# SECTION 10: Submission Validation Tests
#-------------------------------------------------------------------------------
echo ""
echo "=== 10. SUBMISSION VALIDATION ==="
echo "----------------------------------------------"

# Test 1: Single GPU with euge-k
echo ""
echo "Test: 1 GPU on euge-k + a100-80gb + standby"
sbatch --test-only --account=euge --partition=a100-80gb --qos=standby \
    --nodes=1 --gpus-per-node=1 --mem=50G --time=00:30:00 --wrap="hostname" 2>&1 \
    && echo "  ✓ PASS" || echo "  ✗ FAIL"

# Test 2: 2 GPUs (full node) with euge-k
echo ""
echo "Test: 2 GPUs (1 node) on euge-k + a100-80gb + standby"
sbatch --test-only --account=euge --partition=a100-80gb --qos=standby \
    --nodes=1 --gpus-per-node=2 --mem=100G --time=00:30:00 --wrap="hostname" 2>&1 \
    && echo "  ✓ PASS" || echo "  ✗ FAIL"

# Test 3: Multi-node (4 nodes = 8 GPUs) with normal QOS
echo ""
echo "Test: 8 GPUs (4 nodes) on euge-k + a100-80gb + normal"
sbatch --test-only --account=euge --partition=a100-80gb --qos=normal \
    --nodes=4 --gpus-per-node=2 --mem=200G --time=01:00:00 --wrap="hostname" 2>&1 \
    && echo "  ✓ PASS" || echo "  ✗ FAIL"

# Test 4: Training partition
echo ""
echo "Test: Training partition access"
sbatch --test-only --account=euge --partition=training --qos=training \
    --nodes=1 --gpus-per-node=4 --mem=200G --time=04:00:00 --wrap="hostname" 2>&1 \
    && echo "  ✓ PASS" || echo "  ✗ FAIL"

#-------------------------------------------------------------------------------
# SECTION 11: Current Queue Status
#-------------------------------------------------------------------------------
echo ""
echo "=== 11. CURRENT QUEUE STATUS ==="
echo "----------------------------------------------"
echo "Your jobs:"
squeue -u $(whoami) 2>&1

echo ""
echo "A100-80GB partition utilization:"
squeue -p a100-80gb -o "%P %u %t %D %R" 2>&1 | head -10

#-------------------------------------------------------------------------------
# SUMMARY
#-------------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "DISCOVERY COMPLETE"
echo "=============================================="
echo ""
echo "NEXT: Review output and run environment setup script"
echo "=============================================="
