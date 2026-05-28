"""
GPU monitoring utilities for experiment tracking.

Captures GPU utilization, memory usage, and temperature metrics.
Uses torch.cuda and optionally pynvml for detailed metrics.
"""

import os
from typing import Any


def get_gpu_count() -> int:
    """
    Get the number of available GPUs.

    Returns:
        Number of GPUs available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 0


def get_gpu_memory_stats(device: int | None = None) -> dict[str, float]:
    """
    Get GPU memory statistics using torch.cuda.

    Args:
        device: GPU device index. If None, uses current device.

    Returns:
        Dictionary with memory stats in GB.
    """
    stats = {
        "memory_allocated_gb": 0.0,
        "memory_reserved_gb": 0.0,
        "memory_peak_gb": 0.0,
        "memory_total_gb": 0.0,
        "memory_free_gb": 0.0,
    }

    try:
        import torch

        if not torch.cuda.is_available():
            return stats

        if device is None:
            device = torch.cuda.current_device()

        # Current allocation
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)

        # Total memory
        total = torch.cuda.get_device_properties(device).total_memory

        stats["memory_allocated_gb"] = allocated / 1e9
        stats["memory_reserved_gb"] = reserved / 1e9
        stats["memory_peak_gb"] = max_allocated / 1e9
        stats["memory_total_gb"] = total / 1e9
        stats["memory_free_gb"] = (total - allocated) / 1e9

    except Exception:
        pass

    return stats


def reset_peak_memory_stats(device: int | None = None) -> None:
    """
    Reset peak memory statistics for a device.

    Args:
        device: GPU device index. If None, uses current device.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def get_gpu_utilization(device: int = 0) -> dict[str, Any]:
    """
    Get GPU utilization metrics using pynvml.

    Args:
        device: GPU device index.

    Returns:
        Dictionary with utilization metrics.
    """
    stats = {
        "utilization_gpu": None,
        "utilization_memory": None,
        "temperature": None,
        "power_draw_w": None,
        "power_limit_w": None,
    }

    try:
        import pynvml

        pynvml.nvmlInit()

        handle = pynvml.nvmlDeviceGetHandleByIndex(device)

        # GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats["utilization_gpu"] = util.gpu
        stats["utilization_memory"] = util.memory

        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            stats["temperature"] = temp
        except pynvml.NVMLError:
            pass

        # Power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            stats["power_draw_w"] = power / 1000.0  # Convert mW to W

            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            stats["power_limit_w"] = power_limit / 1000.0
        except pynvml.NVMLError:
            pass

        pynvml.nvmlShutdown()

    except ImportError:
        # pynvml not available, try nvidia-ml-py
        try:
            from py3nvml import py3nvml as nvml

            nvml.nvmlInit()

            handle = nvml.nvmlDeviceGetHandleByIndex(device)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            stats["utilization_gpu"] = util.gpu
            stats["utilization_memory"] = util.memory

            nvml.nvmlShutdown()
        except (ImportError, Exception):
            pass

    except Exception:
        pass

    return stats


def get_all_gpu_stats(local_rank: int | None = None) -> dict[str, Any]:
    """
    Get comprehensive GPU statistics for the current device.

    Args:
        local_rank: Local GPU rank. If None, uses LOCAL_RANK env var or 0.

    Returns:
        Dictionary with all GPU metrics.
    """
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    stats = {}

    # Memory stats from torch.cuda
    memory_stats = get_gpu_memory_stats(local_rank)
    stats.update(
        {
            "gpu/memory_used_gb": memory_stats["memory_allocated_gb"],
            "gpu/memory_reserved_gb": memory_stats["memory_reserved_gb"],
            "gpu/memory_peak_gb": memory_stats["memory_peak_gb"],
            "gpu/memory_total_gb": memory_stats["memory_total_gb"],
            "gpu/memory_free_gb": memory_stats["memory_free_gb"],
        }
    )

    # Utilization stats from pynvml
    util_stats = get_gpu_utilization(local_rank)
    if util_stats["utilization_gpu"] is not None:
        stats["gpu/utilization"] = util_stats["utilization_gpu"]
    if util_stats["utilization_memory"] is not None:
        stats["gpu/utilization_memory"] = util_stats["utilization_memory"]
    if util_stats["temperature"] is not None:
        stats["gpu/temperature"] = util_stats["temperature"]
    if util_stats["power_draw_w"] is not None:
        stats["gpu/power_draw_w"] = util_stats["power_draw_w"]
    if util_stats["power_limit_w"] is not None:
        stats["gpu/power_limit_w"] = util_stats["power_limit_w"]

    return stats


class GPUMonitor:
    """
    GPU monitoring class for periodic metric collection.

    Tracks GPU metrics over time and provides aggregated statistics.
    """

    def __init__(self, device: int | None = None):
        """
        Initialize the GPU monitor.

        Args:
            device: GPU device index. If None, uses LOCAL_RANK or 0.
        """
        if device is None:
            device = int(os.environ.get("LOCAL_RANK", 0))
        self.device = device
        self._history: list[dict[str, float]] = []

    def sample(self) -> dict[str, Any]:
        """
        Take a sample of current GPU metrics.

        Returns:
            Dictionary with current GPU stats.
        """
        stats = get_all_gpu_stats(self.device)
        self._history.append(stats)
        return stats

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics from all samples.

        Returns:
            Dictionary with min, max, mean for each metric.
        """
        if not self._history:
            return {}

        summary = {}
        keys = self._history[0].keys()

        for key in keys:
            values = [s.get(key) for s in self._history if s.get(key) is not None]
            if values:
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_min"] = min(values)

        return summary

    def reset(self) -> None:
        """Reset the sample history."""
        self._history = []

    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking."""
        reset_peak_memory_stats(self.device)
