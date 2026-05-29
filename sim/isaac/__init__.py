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
from sim.isaac.native_door import (
    ALEX_DEPENDENT_CAMERA_NAMES,
    DEFAULT_DOOR_ASSET_PATH,
    DEFAULT_HALLWAY_SCENE_PATH,
    LIVE_CAMERA_NAMES,
    IsaacNativeDoorTask,
    LatchProxy,
    NativeDoorPrimPaths,
    NativeDoorTaskConfig,
    build_native_door_metrics,
    compute_target_angle_error,
    summarize_rgb_frame,
    validate_live_camera_request,
)

__all__ = [
    "ACTION_DIM",
    "ALEX_DEPENDENT_CAMERA_NAMES",
    "CAMERA_NAMES",
    "DEFAULT_CAMERA_HEIGHT",
    "DEFAULT_CAMERA_WIDTH",
    "DEFAULT_DOOR_ASSET_PATH",
    "DEFAULT_HALLWAY_SCENE_PATH",
    "DOOR_METRIC_NAMES",
    "STATE_DIM",
    "IsaacContractError",
    "IsaacDoorBackend",
    "IsaacNativeDoorTask",
    "IsaacRuntimeUnavailable",
    "LIVE_CAMERA_NAMES",
    "LatchProxy",
    "NativeDoorPrimPaths",
    "NativeDoorTaskConfig",
    "build_native_door_metrics",
    "compute_target_angle_error",
    "summarize_rgb_frame",
    "validate_action_17d",
    "validate_door_metrics",
    "validate_live_camera_request",
    "validate_render_views",
    "validate_robot_state_52d",
]
