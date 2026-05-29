"""Frozen VLA door-opening contracts shared by Isaac adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

ACTION_DIM: int = 17
STATE_DIM: int = 52
DEFAULT_CAMERA_WIDTH: int = 320
DEFAULT_CAMERA_HEIGHT: int = 320

CAMERA_NAMES: tuple[str, ...] = (
    "overhead",
    "left_wrist_cam",
    "right_wrist_cam",
    "third_person",
)

DOOR_METRIC_NAMES: tuple[str, ...] = (
    "handle_contact_success",
    "latch_release_success",
    "max_door_angle_rad",
    "final_angle_error_rad",
    "contact_stability",
    "force_limit_violation",
    "recovery_success",
)


class IsaacRuntimeUnavailable(RuntimeError):
    """Raised when a live Isaac-only operation is requested outside Isaac."""


class IsaacContractError(ValueError):
    """Raised when data violates the frozen VLA Isaac contract."""


def validate_action_17d(action: Any) -> np.ndarray:
    """Return a finite ``float64`` action array with shape ``(17,)``."""
    arr = np.asarray(action, dtype=np.float64)
    if arr.shape != (ACTION_DIM,):
        raise IsaacContractError(f"Expected action shape ({ACTION_DIM},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise IsaacContractError("Action contains NaN or Inf")
    return arr


def validate_robot_state_52d(state: Any) -> np.ndarray:
    """Return a finite ``float64`` robot-state array with shape ``(52,)``."""
    arr = np.asarray(state, dtype=np.float64)
    if arr.shape != (STATE_DIM,):
        raise IsaacContractError(f"Expected robot state shape ({STATE_DIM},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise IsaacContractError("Robot state contains NaN or Inf")
    return arr


def validate_door_metrics(metrics: Mapping[str, Any]) -> dict[str, float | bool]:
    """Validate that all frozen door metric names are present.

    Extra keys are allowed so downstream code can carry diagnostics such as
    ``target_angle_error`` without mutating the frozen metric surface.
    """
    missing = [name for name in DOOR_METRIC_NAMES if name not in metrics]
    if missing:
        raise IsaacContractError(f"Missing door metric(s): {missing}")

    validated: dict[str, float | bool] = {}
    for name in tuple(DOOR_METRIC_NAMES) + tuple(
        name for name in metrics if name not in DOOR_METRIC_NAMES
    ):
        value = metrics[name]
        if isinstance(value, (bool, np.bool_)):
            validated[name] = bool(value)
        elif isinstance(value, (int, float, np.number)):
            value_f = float(value)
            if not np.isfinite(value_f):
                raise IsaacContractError(f"Door metric {name!r} is not finite")
            validated[name] = value_f
        else:
            raise IsaacContractError(
                f"Door metric {name!r} must be bool or numeric, got {type(value).__name__}"
            )
    return validated


def validate_render_views(
    views: Mapping[str, Any],
    *,
    width: int = DEFAULT_CAMERA_WIDTH,
    height: int = DEFAULT_CAMERA_HEIGHT,
) -> dict[str, np.ndarray]:
    """Validate the four camera frames and return them as numpy arrays."""
    missing = [name for name in CAMERA_NAMES if name not in views]
    extra = [name for name in views if name not in CAMERA_NAMES]
    if missing or extra:
        raise IsaacContractError(f"Camera view mismatch: missing={missing}, extra={extra}")

    validated: dict[str, np.ndarray] = {}
    for name in CAMERA_NAMES:
        arr = np.asarray(views[name])
        if arr.shape != (height, width, 3):
            raise IsaacContractError(
                f"Camera {name!r} expected shape ({height}, {width}, 3), got {arr.shape}"
            )
        if arr.dtype != np.uint8:
            raise IsaacContractError(f"Camera {name!r} expected dtype uint8, got {arr.dtype}")
        validated[name] = arr
    return validated
