"""
Reproducibility metadata capture for experiment tracking.

Captures:
- Git information (commit, branch, dirty state)
- Random seeds (Python, NumPy, PyTorch, CUDA)
- Environment info (PyTorch version, CUDA, etc.)
- SLURM job information
"""

import os
import subprocess
import sys
from datetime import datetime
from typing import Any


def get_git_info(repo_path: str | None = None) -> dict[str, Any]:
    """
    Capture git repository information.

    Args:
        repo_path: Path to git repository. If None, uses current directory.

    Returns:
        Dictionary with git commit, branch, and dirty state.
    """
    cwd = repo_path or os.getcwd()
    git_info = {
        "commit": None,
        "commit_short": None,
        "branch": None,
        "dirty": None,
        "remote_url": None,
    }

    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()
            git_info["commit_short"] = result.stdout.strip()[:7]

        # Get branch name
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            # If empty (detached HEAD), try to get tag or commit
            if not branch:
                result = subprocess.run(
                    ["git", "describe", "--tags", "--always"],
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=5,
                )
                branch = result.stdout.strip() if result.returncode == 0 else "detached"
            git_info["branch"] = branch

        # Check if working directory is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["dirty"] = len(result.stdout.strip()) > 0

        # Get remote URL
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["remote_url"] = result.stdout.strip()

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # Git not available or not a git repo - return defaults
        pass

    return git_info


def get_environment_info() -> dict[str, Any]:
    """
    Capture environment information.

    Returns:
        Dictionary with Python, PyTorch, CUDA versions, etc.
    """
    env_info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": sys.platform,
    }

    # PyTorch info
    try:
        import torch

        env_info["pytorch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["cudnn_version"] = torch.backends.cudnn.version()
            env_info["gpu_count"] = torch.cuda.device_count()
            env_info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    # DeepSpeed info
    try:
        import deepspeed

        env_info["deepspeed_version"] = deepspeed.__version__
    except ImportError:
        pass

    # Transformers info
    try:
        import transformers

        env_info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    return env_info


def get_slurm_info() -> dict[str, Any]:
    """
    Capture SLURM job information from environment variables.

    Returns:
        Dictionary with SLURM job details.
    """
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_GPUS_PER_NODE",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_QOS",
        "SLURM_SUBMIT_DIR",
        "SLURMD_NODENAME",
    ]

    slurm_info = {}
    for var in slurm_vars:
        value = os.environ.get(var)
        if value is not None:
            # Convert to snake_case key
            key = var.lower().replace("slurm_", "")
            slurm_info[key] = value

    return slurm_info


def get_distributed_info() -> dict[str, Any]:
    """
    Capture distributed training information.

    Returns:
        Dictionary with world size, rank, etc.
    """
    dist_info = {
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "backend": None,
    }

    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist_info["world_size"] = dist.get_world_size()
            dist_info["rank"] = dist.get_rank()
            dist_info["backend"] = dist.get_backend()
    except ImportError:
        pass

    # Check environment variables for local rank
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        dist_info["local_rank"] = int(local_rank)

    return dist_info


def get_seeds() -> dict[str, int | None]:
    """
    Capture current random seeds if they can be determined.

    Note: This captures the current state, not the initial seeds.
    For reproducibility, seeds should be explicitly set and passed to metadata.

    Returns:
        Dictionary with seed information.
    """
    seeds = {
        "python": None,
        "numpy": None,
        "torch": None,
        "torch_cuda": None,
    }

    # Note: Python's random module doesn't expose current seed
    # NumPy and PyTorch seeds should be set explicitly and passed in

    return seeds


def set_seeds(seed: int, deterministic: bool = False) -> dict[str, int]:
    """
    Set random seeds for reproducibility.

    Args:
        seed: The seed value to use.
        deterministic: If True, set PyTorch to deterministic mode.

    Returns:
        Dictionary with the seeds that were set.
    """
    import random

    random.seed(seed)

    seeds = {"python": seed}

    try:
        import numpy as np

        np.random.seed(seed)
        seeds["numpy"] = seed
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        seeds["torch"] = seed

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeds["torch_cuda"] = seed

            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seeds


def get_metadata(
    config: dict[str, Any] | None = None,
    seeds: dict[str, int] | None = None,
    repo_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Capture all reproducibility metadata.

    Args:
        config: Training configuration dictionary.
        seeds: Dictionary of seeds used (if explicitly set).
        repo_path: Path to git repository.
        extra: Additional metadata to include.

    Returns:
        Complete metadata dictionary.
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git": get_git_info(repo_path),
        "environment": get_environment_info(),
        "slurm": get_slurm_info(),
        "distributed": get_distributed_info(),
        "seeds": seeds or get_seeds(),
    }

    if config is not None:
        metadata["config"] = config

    if extra is not None:
        metadata["extra"] = extra

    return metadata


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0) in distributed training.

    Returns:
        True if this is rank 0 or not in distributed mode.
    """
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank() == 0
    except ImportError:
        pass

    # Check environment variable as fallback
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    return int(rank) == 0
