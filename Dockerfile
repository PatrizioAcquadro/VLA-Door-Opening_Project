# syntax=docker/dockerfile:1
# ============================================================================
# Dockerfile for VLA-Door-Opening (Deps-Only Model)
#
# This image contains ONLY dependencies (CUDA, Python, PyTorch, etc.).
# Repository code is NOT baked in - it must be bind-mounted at runtime.
#
# INVARIANTS (must always hold):
# 1. Without bind-mount: `python -c "import train"` MUST fail
# 2. With bind-mount + PYTHONPATH=/workspace: imports MUST work
# 3. PyTorch must be CUDA-enabled (cu121)
#
# Usage:
#   docker run -v $(pwd):/workspace IMAGE python -m train.trainer
#   ./scripts/docker-run.sh python -m train.trainer
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base CUDA image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Indiana/Indianapolis

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    # OpenGL/rendering deps for MuJoCo
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 2: Python environment
# ---------------------------------------------------------------------------
FROM base AS python

# Install Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Stage 3: Install dependencies (NOT the package itself)
# ---------------------------------------------------------------------------
FROM python AS dependencies

# Install PyTorch with CUDA support first (large, changes rarely)
# Using explicit cu121 index to ensure CUDA-enabled build
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install tomli for TOML parsing (Python 3.10 doesn't have tomllib)
RUN pip install --no-cache-dir tomli

# Copy ONLY pyproject.toml to extract dependencies
COPY pyproject.toml /tmp/pyproject.toml

# Extract dependencies from pyproject.toml and install them
# IMPORTANT: We do NOT run `pip install .` which would install the package itself
RUN python << 'EXTRACT_DEPS' && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/pyproject.toml /tmp/requirements.txt
import tomli

with open('/tmp/pyproject.toml', 'rb') as f:
    config = tomli.load(f)

# Get base dependencies
deps = config['project']['dependencies']

# Get train extras (for distributed training)
train_deps = config['project']['optional-dependencies']['train']

# Get sim extras (for MuJoCo simulation)
sim_deps = config['project']['optional-dependencies'].get('sim', [])

# Combine all deps (skip torch* since we installed CUDA version explicitly)
all_deps = []
for dep in deps + train_deps + sim_deps:
    dep_lower = dep.lower()
    if not dep_lower.startswith('torch'):
        all_deps.append(dep)

# Write to requirements file
with open('/tmp/requirements.txt', 'w') as f:
    f.write('\n'.join(all_deps))
    print(f"Extracted {len(all_deps)} dependencies")
EXTRACT_DEPS

# Install pytest for testing inside container
RUN pip install --no-cache-dir pytest>=7.0.0 pytest-cov>=4.0.0

# ---------------------------------------------------------------------------
# Stage 4: Final image with validation
# ---------------------------------------------------------------------------
FROM dependencies AS final

# Set working directory to /workspace (will be bind-mounted)
WORKDIR /workspace

# Copy entrypoint script
COPY scripts/container-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Labels
LABEL org.opencontainers.image.title="VLA-Door-Opening Training (Deps-Only)"
LABEL org.opencontainers.image.description="Dependencies-only image for VLA-Door-Opening. Bind-mount code at /workspace."

# ---------------------------------------------------------------------------
# Build-time validation: Verify deps-only invariants
# ---------------------------------------------------------------------------

# Validate 1: Project modules must NOT be importable (no code baked in)
RUN <<'VALIDATE_DEPS_ONLY'
python -c "
try:
    import train
    raise AssertionError('FAIL: train module should not be importable without bind-mount')
except ModuleNotFoundError:
    print('OK: train module not found (deps-only verified)')

try:
    import models
    raise AssertionError('FAIL: models module should not be importable without bind-mount')
except ModuleNotFoundError:
    print('OK: models module not found (deps-only verified)')
"
VALIDATE_DEPS_ONLY

# Validate 2: PyTorch must be CUDA-enabled (cu121)
RUN <<'VALIDATE_CUDA'
python -c "
import torch
cuda_version = torch.version.cuda
assert cuda_version is not None, f'FAIL: torch.version.cuda is None (CPU-only build?)'
assert cuda_version.startswith('12.'), f'FAIL: expected CUDA 12.x, got {cuda_version}'
print(f'OK: PyTorch {torch.__version__} with CUDA {cuda_version}')
"
VALIDATE_CUDA

# Validate 3: Core dependencies are available
RUN <<'VALIDATE_CORE_DEPS'
python -c "
import numpy; print(f'numpy: {numpy.__version__}')
import hydra; print(f'hydra: {hydra.__version__}')
import wandb; print(f'wandb: {wandb.__version__}')
import einops; print(f'einops: {einops.__version__}')
import accelerate; print(f'accelerate: {accelerate.__version__}')
print('OK: All core dependencies available')
"
VALIDATE_CORE_DEPS

# Entrypoint validates mount and sets up environment
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command (shows help if no command provided)
CMD ["python", "--version"]

# ---------------------------------------------------------------------------
# Healthcheck - verify PyTorch CUDA support (doesn't require GPU)
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.version.cuda is not None" || exit 1
