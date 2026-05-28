#!/bin/bash
# =============================================================================
# Docker Wrapper for VLA-LEGO (Deps-Only Container)
#
# This script runs commands inside the VLA-LEGO Docker container with
# the current directory bind-mounted at /workspace.
#
# Usage:
#   ./scripts/docker-run.sh python -m train.trainer --help
#   ./scripts/docker-run.sh pytest tests/ -v
#
# Environment variables:
#   VLA_DOCKER_IMAGE - Override the default image (default: ghcr.io/patrizioacquadro/vla-lego_project:latest)
#   WANDB_API_KEY    - Weights & Biases API key (passed through)
#   WANDB_MODE       - W&B mode: online/offline/disabled (passed through)
#   CUDA_VISIBLE_DEVICES - GPU selection (passed through)
# =============================================================================
set -e

# Default image (can be overridden via environment variable)
IMAGE="${VLA_DOCKER_IMAGE:-ghcr.io/patrizioacquadro/vla-lego_project:latest}"

# Show usage if no command provided
if [[ $# -eq 0 ]]; then
    echo "VLA-LEGO Docker Runner (Deps-Only Container)"
    echo ""
    echo "Usage: $0 COMMAND [ARGS...]"
    echo ""
    echo "Examples:"
    echo "  $0 python -m train.trainer --help"
    echo "  $0 python -m train.trainer trainer=debug cluster=local"
    echo "  $0 pytest tests/ -v"
    echo "  $0 python -c \"import torch; print(torch.cuda.is_available())\""
    echo ""
    echo "Current image: $IMAGE"
    echo "Override with: VLA_DOCKER_IMAGE=your-image $0 COMMAND"
    exit 1
fi

# Check if we're in the project directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "ERROR: pyproject.toml not found in current directory."
    echo "Please run this script from the VLA-LEGO project root."
    exit 1
fi

# Build the docker run command
DOCKER_ARGS=(
    --rm
    -it
    -v "$(pwd):/workspace"
)

# Add GPU support if available
if command -v nvidia-smi &>/dev/null; then
    DOCKER_ARGS+=(--gpus all)
fi

# Pass through relevant environment variables
for VAR in WANDB_API_KEY WANDB_MODE WANDB_PROJECT WANDB_ENTITY CUDA_VISIBLE_DEVICES; do
    if [[ -n "${!VAR:-}" ]]; then
        DOCKER_ARGS+=(-e "$VAR")
    fi
done

# Run the container
exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" "$@"
