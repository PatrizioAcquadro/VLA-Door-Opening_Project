#!/bin/bash
#===============================================================================
# Environment Setup Script for Phase 0.1
# Purpose: Create reproducible environment with pinned versions
# Usage: source 01_setup_env.sh
#===============================================================================

set -e

echo "=============================================="
echo "PHASE 0.1 ENVIRONMENT SETUP"
echo "=============================================="

#-------------------------------------------------------------------------------
# 1. Define Standard Paths
#-------------------------------------------------------------------------------
echo ""
echo "=== 1. Setting up standard paths ==="

export PROJECT_NAME="vla-lego"
export USER_NAME=$(whoami)

# Base directories
export PROJECT_ROOT="/scratch/gilbreth/${USER_NAME}/${PROJECT_NAME}"
export HOME_PROJECT="${HOME}/${PROJECT_NAME}"

# Data & output directories (use scratch for large files)
export DATA_DIR="${PROJECT_ROOT}/datasets"
export CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
export LOG_DIR="${PROJECT_ROOT}/logs"
export CACHE_DIR="${PROJECT_ROOT}/cache"
export RUNS_DIR="${PROJECT_ROOT}/runs"
export WANDB_DIR="${PROJECT_ROOT}/wandb"
export RENDERS_DIR="${PROJECT_ROOT}/renders"

# HuggingFace cache (important for transformers/datasets)
export HF_HOME="${CACHE_DIR}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Create all artifact directories on SCRATCH
echo "Creating project directories on SCRATCH..."
mkdir -p "${PROJECT_ROOT}"
mkdir -p "${DATA_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${RUNS_DIR}"
mkdir -p "${WANDB_DIR}"
mkdir -p "${RENDERS_DIR}"
mkdir -p "${HF_HOME}"

echo "  PROJECT_ROOT: ${PROJECT_ROOT}"
echo "  DATA_DIR: ${DATA_DIR}"
echo "  CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "  LOG_DIR: ${LOG_DIR}"
echo "  RUNS_DIR: ${RUNS_DIR}"
echo "  WANDB_DIR: ${WANDB_DIR}"
echo "  RENDERS_DIR: ${RENDERS_DIR}"

#-------------------------------------------------------------------------------
# 1b. Create Repo Symlinks (if repo root provided)
#-------------------------------------------------------------------------------
# This section creates symlinks from the repo root to SCRATCH directories
# Set REPO_ROOT to your VLA-LEGO_Project location to enable this
REPO_ROOT="${REPO_ROOT:-}"

if [ -n "${REPO_ROOT}" ] && [ -d "${REPO_ROOT}" ]; then
    echo ""
    echo "=== 1b. Creating repo-root symlinks ==="

    SYMLINKS=(
        "datasets:${DATA_DIR}"
        "checkpoints:${CHECKPOINT_DIR}"
        "logs:${LOG_DIR}"
        "cache:${CACHE_DIR}"
        "runs:${RUNS_DIR}"
        "wandb:${WANDB_DIR}"
        "renders:${RENDERS_DIR}"
    )

    for entry in "${SYMLINKS[@]}"; do
        name="${entry%%:*}"
        target="${entry##*:}"
        link_path="${REPO_ROOT}/${name}"

        if [ -L "${link_path}" ]; then
            echo "  ${name}: symlink exists -> $(readlink ${link_path})"
        elif [ -e "${link_path}" ]; then
            echo "  WARNING: ${link_path} exists but is not a symlink, skipping"
        else
            ln -s "${target}" "${link_path}"
            echo "  ${name}: created -> ${target}"
        fi
    done
else
    echo ""
    echo "  Note: Set REPO_ROOT to create symlinks in your repo directory"
    echo "  Example: REPO_ROOT=/home/\${USER}/VLA-LEGO_Project source 01_setup_env.sh"
fi

#-------------------------------------------------------------------------------
# 2. Load Modules (Version Pinning)
#-------------------------------------------------------------------------------
echo ""
echo "=== 2. Loading modules with pinned versions ==="

module purge
module load external

# CUDA - Pin to specific version compatible with PyTorch 2.x
# Check 'module avail cuda' and pick the right one
# Common choices: cuda/11.8, cuda/12.1
MODULE_CUDA="cuda/12.1.1"  # ADJUST based on discovery output

# Anaconda
MODULE_ANACONDA="anaconda/2024.10-py312"  # ADJUST based on discovery output

# cuDNN (if separate module exists)
# MODULE_CUDNN="cudnn/8.9"

echo "Loading: ${MODULE_CUDA}"
module load ${MODULE_CUDA} 2>/dev/null || {
    echo "  WARNING: ${MODULE_CUDA} not found, trying alternatives..."
    module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "  CUDA module load failed"
}

echo "Loading: ${MODULE_ANACONDA}"
module load ${MODULE_ANACONDA} 2>/dev/null || {
    echo "  WARNING: ${MODULE_ANACONDA} not found, trying alternatives..."
    module load anaconda 2>/dev/null || echo "  Anaconda module load failed"
}

echo ""
echo "Loaded modules:"
module list 2>&1

#-------------------------------------------------------------------------------
# 3. CUDA Environment Variables
#-------------------------------------------------------------------------------
echo ""
echo "=== 3. Setting CUDA environment variables ==="

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-""}  # Set by SLURM

# Verify CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "  CUDA Version: ${CUDA_VERSION}"
else
    echo "  WARNING: nvcc not found"
fi

#-------------------------------------------------------------------------------
# 4. NCCL Environment Variables (for multi-GPU/multi-node)
#-------------------------------------------------------------------------------
echo ""
echo "=== 4. Setting NCCL environment variables ==="

# NCCL debugging (set to WARN in production)
export NCCL_DEBUG=INFO

# InfiniBand settings (Gilbreth has 100Gbps IB)
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

# Socket interface (check with 'ip a' on compute node)
export NCCL_SOCKET_IFNAME=ibp161s0

# Timeout (increase for large models)
export NCCL_TIMEOUT=1800  # 30 minutes

# For PyTorch distributed
export TORCH_NCCL_BLOCKING_WAIT=1

echo "  NCCL_DEBUG: ${NCCL_DEBUG}"
echo "  NCCL_IB_DISABLE: ${NCCL_IB_DISABLE}"
echo "  NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"

#-------------------------------------------------------------------------------
# 5. Python/Conda Environment
#-------------------------------------------------------------------------------
echo ""
echo "=== 5. Setting up Conda environment ==="

CONDA_ENV_NAME="${PROJECT_NAME}_env"
CONDA_ENV_PATH="${PROJECT_ROOT}/conda_envs/${CONDA_ENV_NAME}"

# Check if environment exists
if conda env list | grep -q "${CONDA_ENV_NAME}"; then
    echo "Conda environment '${CONDA_ENV_NAME}' exists"
    echo "Activating..."
    source activate ${CONDA_ENV_NAME} || conda activate ${CONDA_ENV_NAME}
else
    echo "Creating new conda environment: ${CONDA_ENV_NAME}"
    echo "This will take a few minutes..."

    conda create -y -n ${CONDA_ENV_NAME} python=3.10
    source activate ${CONDA_ENV_NAME} || conda activate ${CONDA_ENV_NAME}

    # Install PyTorch with CUDA support
    echo ""
    echo "Installing PyTorch 2.2 with CUDA 12.1..."
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu121

    # Install DeepSpeed
    echo ""
    echo "Installing DeepSpeed..."
    pip install deepspeed==0.14.0

    # Install other dependencies
    echo ""
    echo "Installing additional packages..."
    pip install \
        numpy>=1.24.0 \
        hydra-core>=1.3.0 \
        omegaconf>=2.3.0 \
        wandb>=0.15.0 \
        tqdm>=4.65.0 \
        einops>=0.6.0 \
        safetensors>=0.3.0 \
        accelerate>=0.25.0 \
        pytest>=7.0.0
fi

#-------------------------------------------------------------------------------
# 6. Verify Installation
#-------------------------------------------------------------------------------
echo ""
echo "=== 6. Verifying installation ==="

echo ""
echo "Python:"
which python
python --version

echo ""
echo "PyTorch:"
python -c "import torch; print(f'  Version: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  CUDA version: {torch.version.cuda}')"

echo ""
echo "DeepSpeed:"
python -c "import deepspeed; print(f'  Version: {deepspeed.__version__}')" 2>/dev/null || echo "  [Not installed or import error]"

echo ""
echo "Accelerate:"
python -c "import accelerate; print(f'  Version: {accelerate.__version__}')" 2>/dev/null || echo "  [Not installed]"

#-------------------------------------------------------------------------------
# 7. Save Environment Info
#-------------------------------------------------------------------------------
echo ""
echo "=== 7. Saving environment info ==="

ENV_FILE="${PROJECT_ROOT}/environment_info.txt"
{
    echo "Environment captured: $(date)"
    echo ""
    echo "=== Modules ==="
    module list 2>&1
    echo ""
    echo "=== Python packages ==="
    pip list
    echo ""
    echo "=== Environment variables ==="
    env | grep -E "(CUDA|NCCL|HF_|PROJECT|TORCH)" | sort
} > ${ENV_FILE}

echo "Saved to: ${ENV_FILE}"

#-------------------------------------------------------------------------------
# 8. Create activation script for future use
#-------------------------------------------------------------------------------
ACTIVATE_SCRIPT="${PROJECT_ROOT}/activate_env.sh"
cat > ${ACTIVATE_SCRIPT} << 'ACTIVATE_EOF'
#!/bin/bash
# Quick activation script - source this to set up environment
# Usage: source /scratch/gilbreth/$USER/vla-lego/activate_env.sh

module purge
module load external
module load cuda/12.1.1 anaconda/2024.10-py312

export PROJECT_ROOT="/scratch/gilbreth/$(whoami)/vla-lego"
export DATA_DIR="${PROJECT_ROOT}/datasets"
export CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
export LOG_DIR="${PROJECT_ROOT}/logs"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ibp161s0

conda activate vla_lego_env

echo "Environment activated: vla_lego_env"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
ACTIVATE_EOF

chmod +x ${ACTIVATE_SCRIPT}
echo ""
echo "Created quick activation script: ${ACTIVATE_SCRIPT}"
echo "Future use: source ${ACTIVATE_SCRIPT}"

#-------------------------------------------------------------------------------
# COMPLETE
#-------------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "ENVIRONMENT SETUP COMPLETE"
echo "=============================================="
echo ""
echo "Key paths:"
echo "  PROJECT_ROOT: ${PROJECT_ROOT}"
echo "  CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "  LOG_DIR: ${LOG_DIR}"
echo ""
echo "To activate in future sessions:"
echo "  source ${ACTIVATE_SCRIPT}"
echo "=============================================="
