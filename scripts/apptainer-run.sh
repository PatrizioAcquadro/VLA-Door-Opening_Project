#!/bin/bash
# =============================================================================
# Apptainer/Singularity Wrapper for VLA-LEGO (Deps-Only Container)
#
# This script runs commands inside the VLA-LEGO Apptainer container with
# the current directory bind-mounted at /workspace. Use this on HPC clusters
# like Gilbreth where Docker is not available.
#
# Usage:
#   ./scripts/apptainer-run.sh python -m train.trainer cluster=gilbreth
#   ./scripts/apptainer-run.sh pytest tests/ -v
#
# Environment variables:
#   VLA_APPTAINER_SIF - Path to the .sif image file (default: vla-lego.sif)
#
# To download the image:
#   apptainer pull vla-lego.sif docker://ghcr.io/patrizioacquadro/vla-lego_project:latest
# =============================================================================
set -e

# Default SIF path (can be overridden via environment variable)
SIF="${VLA_APPTAINER_SIF:-vla-lego.sif}"

# Show usage if no command provided
if [[ $# -eq 0 ]]; then
    echo "VLA-LEGO Apptainer Runner (Deps-Only Container)"
    echo ""
    echo "Usage: $0 COMMAND [ARGS...]"
    echo ""
    echo "Examples:"
    echo "  $0 python -m train.trainer cluster=gilbreth"
    echo "  $0 python -m train.trainer trainer=debug"
    echo "  $0 pytest tests/ -v"
    echo ""
    echo "Current SIF: $SIF"
    echo "Override with: VLA_APPTAINER_SIF=/path/to/image.sif $0 COMMAND"
    echo ""
    echo "To download the image:"
    echo "  apptainer pull vla-lego.sif docker://ghcr.io/patrizioacquadro/vla-lego_project:latest"
    exit 1
fi

# Check if SIF file exists
if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Apptainer image not found: $SIF"
    echo ""
    echo "Download the image first:"
    echo "  apptainer pull vla-lego.sif docker://ghcr.io/patrizioacquadro/vla-lego_project:latest"
    echo ""
    echo "Or set VLA_APPTAINER_SIF to point to your image:"
    echo "  VLA_APPTAINER_SIF=/scratch/user/vla-lego.sif $0 COMMAND"
    exit 1
fi

# Check if we're in the project directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "ERROR: pyproject.toml not found in current directory."
    echo "Please run this script from the VLA-LEGO project root."
    exit 1
fi

# Build apptainer command
APPTAINER_ARGS=(
    exec
    --nv  # Enable NVIDIA GPU support
    -B "$(pwd):/workspace"
)

# Add scratch bindings for HPC (if VLA_SCRATCH_ROOT is set)
if [[ -n "${VLA_SCRATCH_ROOT:-}" ]] && [[ -d "$VLA_SCRATCH_ROOT" ]]; then
    APPTAINER_ARGS+=(-B "$VLA_SCRATCH_ROOT:$VLA_SCRATCH_ROOT")
fi

# Run the container with entrypoint
exec apptainer "${APPTAINER_ARGS[@]}" "$SIF" /usr/local/bin/entrypoint.sh "$@"
