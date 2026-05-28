"""Collect environment metadata for reproducibility."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def collect_metadata(project_root: Path | None = None) -> dict[str, Any]:
    """Collect environment metadata for reproducibility tracking.

    Args:
        project_root: Path to the project root (for git info). Defaults to cwd.

    Returns:
        Dict with keys: os, python_version, gpu_driver, mujoco_version,
        git_commit, git_dirty, deps_hash.
    """
    meta: dict[str, Any] = {}
    root = project_root or Path.cwd()

    # OS info
    meta["os"] = f"{platform.system()} {platform.release()}"
    meta["python_version"] = sys.version

    # GPU driver
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        meta["gpu_driver"] = (
            result.stdout.strip().split("\n")[0] if result.returncode == 0 else "N/A"
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        meta["gpu_driver"] = "N/A"

    # MuJoCo version
    try:
        import mujoco

        meta["mujoco_version"] = mujoco.__version__
    except ImportError:
        meta["mujoco_version"] = "not installed"

    # Git info
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=5,
        )
        meta["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else "N/A"

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=5,
        )
        meta["git_dirty"] = len(dirty.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        meta["git_commit"] = "N/A"
        meta["git_dirty"] = None

    # Deps hash (hash of pyproject.toml as a proxy for pinned deps)
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        meta["deps_hash"] = hashlib.sha256(pyproject.read_bytes()).hexdigest()[:16]
    else:
        meta["deps_hash"] = "N/A"

    return meta
