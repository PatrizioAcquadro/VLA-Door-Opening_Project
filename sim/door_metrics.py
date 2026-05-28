"""Door-opening task metric extraction.

The active task metrics are defined here so simulation rollouts, evaluation,
and tracking all use the same interpretation of door success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DoorMetricConfig:
    """Thresholds for door-opening success metrics."""

    target_open_angle_rad: float = 1.0472
    angle_tolerance_rad: float = 0.0873
    latch_release_angle_rad: float = 0.35
    contact_threshold: float = 1e-6
    min_contact_stability_fraction: float = 0.5

    @classmethod
    def from_cfg(cls, cfg: Any | None = None) -> DoorMetricConfig:
        """Build thresholds from a Hydra config or mapping."""
        defaults = cls()
        if cfg is None:
            return defaults

        door_cfg = getattr(cfg, "door", None) if hasattr(cfg, "door") else cfg.get("door", {})
        handle_cfg = (
            getattr(cfg, "handle", None) if hasattr(cfg, "handle") else cfg.get("handle", {})
        )
        return cls(
            target_open_angle_rad=float(
                _get(door_cfg, "target_open_angle_rad", defaults.target_open_angle_rad)
            ),
            angle_tolerance_rad=float(
                _get(door_cfg, "angle_tolerance_rad", defaults.angle_tolerance_rad)
            ),
            latch_release_angle_rad=float(
                _get(handle_cfg, "latch_release_angle_rad", defaults.latch_release_angle_rad)
            ),
        )


class DoorMetricsTracker:
    """Accumulates door-opening metrics over one rollout."""

    def __init__(self, config: DoorMetricConfig | None = None) -> None:
        self.config = config or DoorMetricConfig()
        self._door_angles: list[float] = []
        self._door_velocities: list[float] = []
        self._handle_touch: list[float] = []
        self._force_limit_violations = 0
        self._recovery_success = False

    def update_mujoco(self, model: Any, data: Any, recovery_success: bool = False) -> None:
        """Update metrics from a MuJoCo model/data pair."""
        self.update_values(
            door_angle=_sensor_value(model, data, "door_angle"),
            door_angular_velocity=_sensor_value(model, data, "door_angular_velocity"),
            handle_touch=_sensor_value(model, data, "handle_touch"),
            force_limit_violation=_has_ctrlrange_violation(model, data),
            recovery_success=recovery_success,
        )

    def update_values(
        self,
        door_angle: float,
        door_angular_velocity: float = 0.0,
        handle_touch: float = 0.0,
        force_limit_violation: bool = False,
        recovery_success: bool = False,
    ) -> None:
        """Update metrics from scalar values, useful for tests and saved rollouts."""
        self._door_angles.append(float(door_angle))
        self._door_velocities.append(float(door_angular_velocity))
        self._handle_touch.append(float(handle_touch))
        if force_limit_violation:
            self._force_limit_violations += 1
        self._recovery_success = self._recovery_success or bool(recovery_success)

    def finalize(self) -> dict[str, float | bool]:
        """Return the final active door-opening metric dictionary."""
        if not self._door_angles:
            return {
                "handle_contact_success": False,
                "latch_release_success": False,
                "max_door_angle_rad": 0.0,
                "final_angle_error_rad": self.config.target_open_angle_rad,
                "contact_stability": False,
                "force_limit_violation": False,
                "recovery_success": False,
                "target_reached": False,
            }

        angles = np.asarray(self._door_angles, dtype=np.float64)
        touch = np.asarray(self._handle_touch, dtype=np.float64)

        max_angle = float(np.max(angles))
        final_error = abs(float(angles[-1]) - self.config.target_open_angle_rad)
        contact_mask = touch > self.config.contact_threshold
        handle_contact = bool(np.any(contact_mask))
        latch_released = bool(max_angle >= self.config.latch_release_angle_rad)

        if handle_contact:
            first_contact_idx = int(np.argmax(contact_mask))
            stability_fraction = float(np.mean(contact_mask[first_contact_idx:]))
            contact_stability = stability_fraction >= self.config.min_contact_stability_fraction
        else:
            contact_stability = False

        return {
            "handle_contact_success": handle_contact,
            "latch_release_success": latch_released,
            "max_door_angle_rad": max_angle,
            "final_angle_error_rad": float(final_error),
            "contact_stability": bool(contact_stability),
            "force_limit_violation": bool(self._force_limit_violations > 0),
            "recovery_success": bool(self._recovery_success),
            "target_reached": bool(final_error <= self.config.angle_tolerance_rad),
        }


def flatten_door_metrics(metrics: dict[str, float | bool]) -> dict[str, float]:
    """Convert door metrics to numeric values suitable for logging."""
    return {
        f"door/{key}": float(value) if isinstance(value, bool) else float(value)
        for key, value in metrics.items()
    }


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def _sensor_value(model: Any, data: Any, sensor_name: str) -> float:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MuJoCo is required for update_mujoco().") from exc

    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        return 0.0
    adr = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    if dim <= 0:
        return 0.0
    return float(data.sensordata[adr])


def _has_ctrlrange_violation(model: Any, data: Any) -> bool:
    if getattr(model, "nu", 0) == 0:
        return False
    ctrl = np.asarray(data.ctrl[: model.nu], dtype=np.float64)
    ranges = np.asarray(model.actuator_ctrlrange[: model.nu], dtype=np.float64)
    limited = np.asarray(model.actuator_ctrllimited[: model.nu], dtype=bool)
    if not np.any(limited):
        return False
    lo = ranges[:, 0]
    hi = ranges[:, 1]
    return bool(np.any(ctrl[limited] < lo[limited]) or np.any(ctrl[limited] > hi[limited]))
