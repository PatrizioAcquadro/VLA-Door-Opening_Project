"""Import-safe Isaac backend contract layer.

This package intentionally avoids importing Isaac Sim, Isaac Lab, ``omni``, or
``pxr`` at module import time. Live Isaac runtime code lives behind explicit
entrypoints such as ``scripts/run_isaac_proof.py``.
"""

from sim.isaac.backend import IsaacDoorBackend
from sim.isaac.contracts import (
    ACTION_DIM,
    CAMERA_NAMES,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DOOR_METRIC_NAMES,
    STATE_DIM,
    IsaacContractError,
    IsaacRuntimeUnavailable,
    validate_action_17d,
    validate_door_metrics,
    validate_render_views,
    validate_robot_state_52d,
)

__all__ = [
    "ACTION_DIM",
    "CAMERA_NAMES",
    "DEFAULT_CAMERA_HEIGHT",
    "DEFAULT_CAMERA_WIDTH",
    "DOOR_METRIC_NAMES",
    "STATE_DIM",
    "IsaacContractError",
    "IsaacDoorBackend",
    "IsaacRuntimeUnavailable",
    "validate_action_17d",
    "validate_door_metrics",
    "validate_render_views",
    "validate_robot_state_52d",
]
