"""
Metric calculation utilities for experiment tracking.

Provides utilities for computing:
- Gradient norms
- Throughput metrics
- Loss component extraction
- AMP scaler values
"""

import time
from collections.abc import Iterator
from typing import Any

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_grad_norm(
    model_or_params: "nn.Module | Iterator[torch.nn.Parameter]",
    norm_type: float = 2.0,
    foreach: bool = True,
) -> float:
    """
    Compute the total gradient norm of model parameters.

    Args:
        model_or_params: Either a model or an iterator of parameters.
        norm_type: Type of norm to compute (default: L2).
        foreach: Use vectorized implementation if available.

    Returns:
        Total gradient norm as a float.
    """
    if not TORCH_AVAILABLE:
        return 0.0

    if isinstance(model_or_params, nn.Module):
        parameters = model_or_params.parameters()
    else:
        parameters = model_or_params

    # Filter parameters with gradients
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.detach())

    if not grads:
        return 0.0

    # Compute norm
    if norm_type == float("inf"):
        total_norm = max(g.abs().max() for g in grads)
    else:
        if foreach and hasattr(torch, "_foreach_norm"):
            # Use vectorized implementation if available (PyTorch 2.0+)
            norms = torch._foreach_norm(grads, norm_type)
            total_norm = torch.stack(norms).norm(norm_type)
        else:
            total_norm = torch.norm(
                torch.stack([torch.norm(g, norm_type) for g in grads]),
                norm_type,
            )

    return total_norm.item()


def get_learning_rate(
    optimizer: "torch.optim.Optimizer",
    param_group: int = 0,
) -> float:
    """
    Get the current learning rate from an optimizer.

    Args:
        optimizer: PyTorch optimizer.
        param_group: Index of parameter group (default: 0).

    Returns:
        Current learning rate.
    """
    if not TORCH_AVAILABLE:
        return 0.0

    if hasattr(optimizer, "param_groups"):
        if len(optimizer.param_groups) > param_group:
            return optimizer.param_groups[param_group].get("lr", 0.0)
    return 0.0


def get_amp_scale(scaler: "torch.cuda.amp.GradScaler | None" = None) -> float | None:
    """
    Get the current AMP scaler value.

    Args:
        scaler: GradScaler instance, or None.

    Returns:
        Current scale value, or None if not using AMP.
    """
    if scaler is None:
        return None

    if not TORCH_AVAILABLE:
        return None

    try:
        scale = scaler.get_scale()
        return float(scale)
    except Exception:
        return None


class ThroughputTracker:
    """
    Tracks throughput metrics (steps/sec, samples/sec, tokens/sec).
    """

    def __init__(self, world_size: int = 1):
        """
        Initialize the throughput tracker.

        Args:
            world_size: Number of distributed processes.
        """
        self.world_size = world_size
        self._start_time: float | None = None
        self._step_count: int = 0
        self._sample_count: int = 0
        self._token_count: int = 0
        self._last_time: float | None = None
        self._last_step: int = 0
        self._last_samples: int = 0
        self._last_tokens: int = 0

    def start(self) -> None:
        """Start or reset the throughput tracker."""
        self._start_time = time.time()
        self._step_count = 0
        self._sample_count = 0
        self._token_count = 0
        self._last_time = self._start_time
        self._last_step = 0
        self._last_samples = 0
        self._last_tokens = 0

    def step(
        self,
        batch_size: int = 0,
        tokens: int = 0,
    ) -> None:
        """
        Record a training step.

        Args:
            batch_size: Number of samples in the batch.
            tokens: Number of tokens processed (optional).
        """
        self._step_count += 1
        self._sample_count += batch_size * self.world_size
        self._token_count += tokens * self.world_size

    def get_throughput(self) -> dict[str, float]:
        """
        Get current throughput metrics.

        Returns:
            Dictionary with throughput values.
        """
        if self._start_time is None:
            return {
                "perf/steps_per_sec": 0.0,
                "perf/samples_per_sec": 0.0,
            }

        current_time = time.time()
        elapsed = current_time - self._start_time

        if elapsed <= 0:
            return {
                "perf/steps_per_sec": 0.0,
                "perf/samples_per_sec": 0.0,
            }

        metrics = {
            "perf/steps_per_sec": self._step_count / elapsed,
            "perf/samples_per_sec": self._sample_count / elapsed,
        }

        if self._token_count > 0:
            metrics["perf/tokens_per_sec"] = self._token_count / elapsed

        return metrics

    def get_interval_throughput(self) -> dict[str, float]:
        """
        Get throughput since last call to this method.

        Returns:
            Dictionary with interval throughput values.
        """
        if self._last_time is None:
            self._last_time = time.time()
            self._last_step = self._step_count
            self._last_samples = self._sample_count
            self._last_tokens = self._token_count
            return self.get_throughput()

        current_time = time.time()
        elapsed = current_time - self._last_time

        if elapsed <= 0:
            return self.get_throughput()

        steps = self._step_count - self._last_step
        samples = self._sample_count - self._last_samples
        tokens = self._token_count - self._last_tokens

        # Update last values
        self._last_time = current_time
        self._last_step = self._step_count
        self._last_samples = self._sample_count
        self._last_tokens = self._token_count

        metrics = {
            "perf/steps_per_sec": steps / elapsed,
            "perf/samples_per_sec": samples / elapsed,
        }

        if tokens > 0:
            metrics["perf/tokens_per_sec"] = tokens / elapsed

        return metrics


def extract_loss_components(
    loss: "torch.Tensor | dict[str, Any] | float",
    loss_ar: "torch.Tensor | float | None" = None,
    loss_fm: "torch.Tensor | float | None" = None,
) -> dict[str, float]:
    """
    Extract loss components into a standardized format.

    Args:
        loss: Total loss value (tensor, dict, or float).
        loss_ar: Autoregressive loss component (optional).
        loss_fm: Flow matching loss component (optional).

    Returns:
        Dictionary with standardized loss keys.
    """
    metrics = {}

    # Handle total loss
    if isinstance(loss, dict):
        # Loss is a dictionary - extract components
        if "total" in loss:
            metrics["loss/total"] = _to_float(loss["total"])
        elif "loss" in loss:
            metrics["loss/total"] = _to_float(loss["loss"])
        else:
            # Sum all values as total
            total = sum(_to_float(v) for v in loss.values())
            metrics["loss/total"] = total

        # Extract named components
        for key, value in loss.items():
            if key not in ("total", "loss"):
                metrics[f"loss/{key}"] = _to_float(value)
    else:
        metrics["loss/total"] = _to_float(loss)

    # Add explicit AR/FM components if provided
    if loss_ar is not None:
        metrics["loss/ar"] = _to_float(loss_ar)
    if loss_fm is not None:
        metrics["loss/fm"] = _to_float(loss_fm)

    return metrics


def _to_float(value: "torch.Tensor | float | int") -> float:
    """Convert a value to float."""
    if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        return value.detach().item()
    return float(value)


def aggregate_metrics(
    metrics_list: list[dict[str, float]],
    method: str = "mean",
) -> dict[str, float]:
    """
    Aggregate a list of metric dictionaries.

    Args:
        metrics_list: List of metric dictionaries.
        method: Aggregation method ("mean", "sum", "last").

    Returns:
        Aggregated metrics dictionary.
    """
    if not metrics_list:
        return {}

    # Collect all keys
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    result = {}
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m and m[key] is not None]
        if not values:
            continue

        if method == "mean":
            result[key] = sum(values) / len(values)
        elif method == "sum":
            result[key] = sum(values)
        elif method == "last":
            result[key] = values[-1]
        else:
            result[key] = sum(values) / len(values)

    return result


def filter_metrics(
    metrics: dict[str, Any],
    include_none: bool = False,
) -> dict[str, float]:
    """
    Filter metrics dictionary to only include valid numeric values.

    Args:
        metrics: Raw metrics dictionary.
        include_none: Whether to include None values.

    Returns:
        Filtered metrics dictionary.
    """
    result = {}
    for key, value in metrics.items():
        if value is None:
            if include_none:
                result[key] = None
            continue

        try:
            result[key] = _to_float(value)
        except (TypeError, ValueError):
            # Skip non-numeric values
            continue

    return result
