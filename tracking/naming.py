"""
Run naming conventions for experiment tracking.

Provides consistent naming for W&B runs and standardized tags.
"""

import re
from datetime import datetime
from typing import Any

from .metadata import get_git_info, get_slurm_info

# Required tags for consistency
REQUIRED_TAGS = ["model", "dataset", "objective"]

# Optional but recommended tags
OPTIONAL_TAGS = ["experiment_group", "cluster", "qos"]


def generate_run_name(
    model: str | None = None,
    objective: str | None = None,
    dataset: str | None = None,
    timestamp: datetime | None = None,
    git_short: str | None = None,
    suffix: str | None = None,
) -> str:
    """
    Generate a consistent run name.

    Format: {model}_{objective}_{dataset}_{YYYYMMDD}_{HHMMSS}_{git_short}

    Args:
        model: Model name (e.g., "eo1", "vla-base").
        objective: Training objective (e.g., "ar", "fm", "ar+fm").
        dataset: Dataset name (e.g., "lego", "lego-bimanual").
        timestamp: Timestamp for the run. If None, uses current time.
        git_short: Short git commit hash. If None, attempts to get from repo.
        suffix: Optional suffix to append.

    Returns:
        Formatted run name string.
    """
    if timestamp is None:
        timestamp = datetime.now()

    if git_short is None:
        git_info = get_git_info()
        git_short = git_info.get("commit_short", "unknown")

    # Build name components
    components = []

    if model:
        components.append(_sanitize_component(model))
    else:
        components.append("model")

    if objective:
        components.append(_sanitize_component(objective))

    if dataset:
        components.append(_sanitize_component(dataset))

    # Add timestamp
    components.append(timestamp.strftime("%Y%m%d"))
    components.append(timestamp.strftime("%H%M%S"))

    # Add git hash
    if git_short:
        components.append(git_short)

    # Add suffix if provided
    if suffix:
        components.append(_sanitize_component(suffix))

    return "_".join(components)


def _sanitize_component(value: str) -> str:
    """
    Sanitize a name component for use in run names.

    Replaces special characters with underscores and converts to lowercase.
    """
    # Replace special characters with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", str(value))
    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized.lower()


def generate_tags(
    model: str | None = None,
    dataset: str | None = None,
    objective: str | None = None,
    experiment_group: str | None = None,
    cluster: str | None = None,
    qos: str | None = None,
    extra_tags: list[str] | None = None,
) -> list[str]:
    """
    Generate a list of tags for a run.

    Args:
        model: Model name/architecture.
        dataset: Dataset name.
        objective: Training objective.
        experiment_group: Group for organizing related experiments.
        cluster: Cluster name.
        qos: SLURM QOS.
        extra_tags: Additional tags to include.

    Returns:
        List of tag strings.
    """
    tags = []

    # Add standard tags
    if model:
        tags.append(f"model:{model}")
    if dataset:
        tags.append(f"dataset:{dataset}")
    if objective:
        tags.append(f"objective:{objective}")
    if experiment_group:
        tags.append(f"group:{experiment_group}")

    # Auto-detect cluster from environment
    if cluster is None:
        slurm_info = get_slurm_info()
        if slurm_info:
            # Try to infer cluster from partition or submit dir
            cluster = "gilbreth"  # Default for this project
    if cluster:
        tags.append(f"cluster:{cluster}")

    # Auto-detect QOS from SLURM
    if qos is None:
        slurm_info = get_slurm_info()
        qos = slurm_info.get("job_qos")
    if qos:
        tags.append(f"qos:{qos}")

    # Add extra tags
    if extra_tags:
        tags.extend(extra_tags)

    return tags


def generate_tags_dict(
    model: str | None = None,
    dataset: str | None = None,
    objective: str | None = None,
    experiment_group: str | None = None,
    cluster: str | None = None,
    qos: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a dictionary of tags for W&B config.

    Args:
        model: Model name/architecture.
        dataset: Dataset name.
        objective: Training objective.
        experiment_group: Group for organizing related experiments.
        cluster: Cluster name.
        qos: SLURM QOS.
        extra: Additional key-value pairs to include.

    Returns:
        Dictionary of tags.
    """
    tags = {}

    if model:
        tags["model"] = model
    if dataset:
        tags["dataset"] = dataset
    if objective:
        tags["objective"] = objective
    if experiment_group:
        tags["experiment_group"] = experiment_group

    # Auto-detect cluster
    if cluster is None:
        cluster = "gilbreth"  # Default for this project
    tags["cluster"] = cluster

    # Auto-detect QOS from SLURM
    if qos is None:
        slurm_info = get_slurm_info()
        qos = slurm_info.get("job_qos")
    if qos:
        tags["qos"] = qos

    # Add SLURM job ID if available
    slurm_info = get_slurm_info()
    if slurm_info.get("job_id"):
        tags["slurm_job_id"] = slurm_info["job_id"]

    # Add extra tags
    if extra:
        tags.update(extra)

    return tags


def validate_tags(
    tags: dict[str, Any],
    require_all: bool = False,
) -> list[str]:
    """
    Validate that required tags are present.

    Args:
        tags: Dictionary of tags to validate.
        require_all: If True, all required tags must be present.

    Returns:
        List of missing required tags (empty if all present).
    """
    missing = []
    for tag in REQUIRED_TAGS:
        if tag not in tags or tags[tag] is None:
            missing.append(tag)

    if require_all and missing:
        raise ValueError(f"Missing required tags: {missing}")

    return missing


def get_run_group(
    experiment_group: str | None = None,
    model: str | None = None,
    objective: str | None = None,
) -> str:
    """
    Generate a W&B group name for organizing related runs.

    Args:
        experiment_group: Explicit group name.
        model: Model name (used if no explicit group).
        objective: Objective (used if no explicit group).

    Returns:
        Group name string.
    """
    if experiment_group:
        return experiment_group

    # Generate from model and objective
    components = []
    if model:
        components.append(model)
    if objective:
        components.append(objective)

    if components:
        return "_".join(components)

    return "default"


def parse_run_name(run_name: str) -> dict[str, str | None]:
    """
    Parse a run name back into its components.

    Args:
        run_name: Run name string in standard format.

    Returns:
        Dictionary with parsed components.
    """
    parts = run_name.split("_")

    result = {
        "model": None,
        "objective": None,
        "dataset": None,
        "date": None,
        "time": None,
        "git_short": None,
    }

    # Try to parse based on expected format
    # Format: {model}_{objective}_{dataset}_{YYYYMMDD}_{HHMMSS}_{git_short}
    if len(parts) >= 6:
        result["model"] = parts[0]
        result["objective"] = parts[1]
        result["dataset"] = parts[2]
        result["date"] = parts[3]
        result["time"] = parts[4]
        result["git_short"] = parts[5]
    elif len(parts) >= 4:
        # Minimal format: model_date_time_git
        result["model"] = parts[0]
        result["date"] = parts[-3] if len(parts) > 3 else None
        result["time"] = parts[-2] if len(parts) > 2 else None
        result["git_short"] = parts[-1]

    return result
