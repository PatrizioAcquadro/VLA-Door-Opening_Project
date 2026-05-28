#!/bin/bash
# =============================================================================
# Container Entrypoint for VLA-LEGO (Deps-Only Model)
#
# This script:
# 1. Validates that /workspace is mounted with the project code
# 2. Sets PYTHONPATH so imports work without pip install
# 3. Prints run info (git commit, Python version, GPU) for reproducibility
# 4. Executes the provided command
# =============================================================================
set -e

# -----------------------------------------------------------------------------
# Validate /workspace mount
# -----------------------------------------------------------------------------
if [[ ! -f /workspace/pyproject.toml ]]; then
    echo "ERROR: /workspace not mounted or missing pyproject.toml"
    echo ""
    echo "This is a deps-only container. You must bind-mount your code:"
    echo ""
    echo "  Docker:    docker run -v \$(pwd):/workspace IMAGE COMMAND"
    echo "  Apptainer: apptainer exec -B \$(pwd):/workspace IMAGE COMMAND"
    echo ""
    echo "Or use the wrapper scripts:"
    echo "  ./scripts/docker-run.sh COMMAND"
    echo "  ./scripts/apptainer-run.sh COMMAND"
    exit 1
fi

# -----------------------------------------------------------------------------
# Set PYTHONPATH so imports work without pip install
# -----------------------------------------------------------------------------
export PYTHONPATH=/workspace:${PYTHONPATH:-}

# -----------------------------------------------------------------------------
# Collect and print run info for reproducibility
# -----------------------------------------------------------------------------
RUN_INFO_FILE="/tmp/vla_run_info.txt"
{
    echo "=== VLA-LEGO Container Run ==="
    echo "Timestamp: $(date -Iseconds)"
    echo "Python: $(python --version 2>&1)"

    if [[ -d /workspace/.git ]]; then
        cd /workspace
        GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo 'N/A')
        GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')
        GIT_DIRTY=$(git status --porcelain 2>/dev/null | wc -l)
        echo "Git commit: $GIT_COMMIT"
        echo "Git branch: $GIT_BRANCH"
        echo "Git dirty: $GIT_DIRTY files"
    else
        echo "Git: N/A (not a git repo)"
    fi

    # PyTorch info
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: N/A"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "CUDA: N/A"

    # GPU info
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')
        echo "GPU: $GPU_NAME"
    else
        echo "GPU: N/A (nvidia-smi not found)"
    fi

    echo "=============================="
} | tee "$RUN_INFO_FILE"

# -----------------------------------------------------------------------------
# Execute command from /workspace
# -----------------------------------------------------------------------------
cd /workspace
exec "$@"
