"""Frozen 4-view camera contract for the Alex robot (Phase 1.2.4).

Replaces the Phase 1.1.7 2-view contract (robot_cam + third_person) with a
4-view contract aligned with the VLA dataset spec.

Camera views (frozen):
    - overhead:        Fixed workspace-level camera (in scene worldbody)
    - left_wrist_cam:  Body-attached to left_gripper, tracks left arm
    - right_wrist_cam: Body-attached to right_gripper, tracks right arm
    - third_person:    Fixed external observation camera (in scene worldbody)

Note:
    ``robot_cam`` (head-mounted) remains in the MJCF for interactive debugging
    but is NOT part of this frozen 4-view contract.

Usage::

    from sim.camera import MultiViewRenderer, CAMERA_NAMES

    model = load_scene("alex_upper_body")
    data = mujoco.MjData(model)
    with MultiViewRenderer(model) as mv:
        frame = mv.capture(data, step_index=0)
        frame.views["overhead"].rgb           # (H, W, 3) uint8
        frame.views["left_wrist_cam"].rgb
        frame.metadata["left_wrist_cam"].pos  # live world-frame position
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.offscreen import (
    RenderConfig,
    RenderedFrame,
    create_renderer,
    render_frame,
    resolve_camera_id,
)

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

CAMERA_NAMES: tuple[str, ...] = (
    "overhead",
    "left_wrist_cam",
    "right_wrist_cam",
    "third_person",
)
"""Canonical camera view names (frozen for dataset compatibility).

Order: overhead (fixed), left_wrist (body-attached), right_wrist (body-attached),
third_person (fixed).
"""

NUM_VIEWS: int = 4
"""Number of camera views in the frozen contract."""

DEFAULT_WIDTH: int = 320
"""Default render width in pixels (square format for VLA training)."""

DEFAULT_HEIGHT: int = 320
"""Default render height in pixels (square format for VLA training)."""

DEFAULT_CAPTURE_HZ: float = 20.0
"""Default capture rate aligned with policy control rate (Hz)."""

# ---------------------------------------------------------------------------
# CameraIntrinsics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics derived from MuJoCo fovy and resolution.

    MuJoCo uses a vertical field-of-view (fovy) with square pixels.
    Horizontal FOV is derived from the aspect ratio.

    Attributes:
        fx: Focal length in pixels (x-axis). Equal to fy for square pixels.
        fy: Focal length in pixels (y-axis).
        cx: Principal point x in pixels (image center).
        cy: Principal point y in pixels (image center).
        fovx: Horizontal field of view in degrees.
        fovy: Vertical field of view in degrees.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    fovx: float
    fovy: float

    def to_matrix(self) -> np.ndarray:
        """Return 3x3 camera intrinsic matrix K.

        Returns:
            K: shape (3, 3), dtype float64.
        """
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "fovx": self.fovx,
            "fovy": self.fovy,
        }


def compute_intrinsics(fovy_deg: float, width: int, height: int) -> CameraIntrinsics:
    """Compute pinhole camera intrinsics from MuJoCo vertical FOV.

    MuJoCo cameras use vertical FOV with square pixels::

        fy = (height / 2) / tan(fovy / 2)
        fx = fy   (square pixels)
        fovx = 2 * atan((width / 2) / fx)

    Args:
        fovy_deg: Vertical field of view in degrees.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        CameraIntrinsics with focal lengths, principal point, and both FOVs.
    """
    fovy_rad = np.radians(fovy_deg)
    fy = (height / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    fovx_deg = float(np.degrees(2.0 * np.arctan((width / 2.0) / fx)))
    return CameraIntrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=cx,
        cy=cy,
        fovx=fovx_deg,
        fovy=fovy_deg,
    )


# ---------------------------------------------------------------------------
# CameraMetadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraMetadata:
    """Intrinsic and extrinsic metadata for a single camera at a timestep.

    Attributes:
        name: Camera name in MJCF.
        camera_id: Resolved MuJoCo camera ID.
        fovy: Vertical field of view (degrees).
        pos: Camera position in world frame, shape (3,).
        mat: Camera orientation matrix in world frame, shape (3, 3).
        width: Render width in pixels.
        height: Render height in pixels.
        intrinsics: Pinhole camera intrinsics derived from fovy and resolution.
        is_body_attached: True if attached to a moving body (wrist cams),
            False if fixed in world frame (overhead, third_person).
    """

    name: str
    camera_id: int
    fovy: float
    pos: np.ndarray
    mat: np.ndarray
    width: int
    height: int
    intrinsics: CameraIntrinsics
    is_body_attached: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "name": self.name,
            "camera_id": self.camera_id,
            "fovy": self.fovy,
            "pos": self.pos.tolist(),
            "mat": self.mat.tolist(),
            "width": self.width,
            "height": self.height,
            "intrinsics": self.intrinsics.to_dict(),
            "is_body_attached": self.is_body_attached,
        }


# ---------------------------------------------------------------------------
# MultiViewFrame
# ---------------------------------------------------------------------------


@dataclass
class MultiViewFrame:
    """Synchronized multi-view capture at a single simulation timestep.

    All views are rendered from the exact same simulation state before
    any physics stepping occurs, guaranteeing timestep synchronization.
    Live camera metadata (including extrinsics) is captured alongside
    the pixel data for body-attached cameras like wrist cameras.

    Attributes:
        step_index: Simulation step (control tick) when captured.
        timestamp: Simulation time in seconds.
        views: Dict mapping camera name to RenderedFrame.
        metadata: Dict mapping camera name to CameraMetadata (live extrinsics).
    """

    step_index: int
    timestamp: float
    views: dict[str, RenderedFrame] = field(default_factory=dict)
    metadata: dict[str, CameraMetadata] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def get_camera_metadata(
    model: mujoco.MjModel,  # type: ignore[name-defined]
    data: mujoco.MjData,  # type: ignore[name-defined]
    camera_name: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> CameraMetadata:
    """Extract intrinsic/extrinsic metadata for a named camera.

    Uses ``data.cam_xpos`` / ``data.cam_xmat`` for world-frame pose,
    which is critical for body-attached cameras (wrist cams) whose
    position changes as the robot moves. Call ``mj_forward`` first.

    Args:
        model: Compiled MuJoCo model.
        data: MuJoCo data (must have had ``mj_forward`` called).
        camera_name: MJCF camera name.
        width: Render width.
        height: Render height.

    Returns:
        CameraMetadata with live world-frame pose and pinhole intrinsics.
    """
    cam_id = resolve_camera_id(model, camera_name)
    fovy = float(model.cam_fovy[cam_id])
    intrinsics = compute_intrinsics(fovy, width, height)
    # cam_bodyid == 0 means the camera is in the world body (fixed frame)
    is_body_attached = int(model.cam_bodyid[cam_id]) != 0
    return CameraMetadata(
        name=camera_name,
        camera_id=cam_id,
        fovy=fovy,
        pos=data.cam_xpos[cam_id].copy(),
        mat=data.cam_xmat[cam_id].reshape(3, 3).copy(),
        width=width,
        height=height,
        intrinsics=intrinsics,
        is_body_attached=is_body_attached,
    )


# ---------------------------------------------------------------------------
# MultiViewRenderer
# ---------------------------------------------------------------------------


class MultiViewRenderer:
    """Synchronized multi-view renderer for the frozen 4-camera setup.

    Creates a single MuJoCo ``Renderer`` (shared resolution) and renders
    all frozen camera views from the same simulation state. Live camera
    metadata (including extrinsics for body-attached wrist cameras) is
    captured alongside pixel data in each ``MultiViewFrame``.

    Args:
        model: Compiled MuJoCo model.
        width: Render width (default 320).
        height: Render height (default 320).
        render_depth: Also capture depth maps.
        render_segmentation: Also capture segmentation masks.
    """

    def __init__(
        self,
        model: mujoco.MjModel,  # type: ignore[name-defined]
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        render_depth: bool = False,
        render_segmentation: bool = False,
    ) -> None:
        self._model = model
        self._width = width
        self._height = height

        self._config = RenderConfig(
            width=width,
            height=height,
            render_depth=render_depth,
            render_segmentation=render_segmentation,
        )

        # Resolve camera IDs for all frozen views
        self._camera_ids: dict[str, int] = {}
        for name in CAMERA_NAMES:
            self._camera_ids[name] = resolve_camera_id(model, name)

        # Create single shared renderer
        self._renderer = create_renderer(model, self._config)
        self._closed = False

    @property
    def camera_ids(self) -> dict[str, int]:
        """Mapping of camera name to resolved MuJoCo camera ID."""
        return dict(self._camera_ids)

    @property
    def width(self) -> int:
        """Render width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Render height in pixels."""
        return self._height

    def capture(
        self,
        data: mujoco.MjData,  # type: ignore[name-defined]
        step_index: int,
    ) -> MultiViewFrame:
        """Render all views from the current simulation state.

        All cameras are rendered before any physics stepping, guaranteeing
        timestep synchronization. Live camera metadata (including extrinsics
        for body-attached wrist cameras) is captured in the same call.

        Args:
            data: MuJoCo data at current state.
            step_index: Current simulation step for sync tracking.

        Returns:
            MultiViewFrame with all camera views and per-camera metadata.

        Raises:
            RuntimeError: If renderer has been closed.
        """
        if self._closed:
            raise RuntimeError("MultiViewRenderer has been closed")

        views: dict[str, RenderedFrame] = {}
        metadata: dict[str, CameraMetadata] = {}

        for name, cam_id in self._camera_ids.items():
            # render_frame calls mj_forward internally (idempotent per timestep)
            frame = render_frame(
                self._renderer, self._model, data, step_index, self._config, cam_id
            )
            views[name] = frame
            metadata[name] = get_camera_metadata(self._model, data, name, self._width, self._height)

        return MultiViewFrame(
            step_index=step_index,
            timestamp=float(data.time),
            views=views,
            metadata=metadata,
        )

    def get_metadata(
        self,
        data: mujoco.MjData,  # type: ignore[name-defined]
    ) -> dict[str, CameraMetadata]:
        """Return camera metadata with live world-frame poses.

        For body-attached cameras (wrist cams), the position and orientation
        update as the robot moves. Call ``mj_forward`` before this method.

        Args:
            data: MuJoCo data (must have had ``mj_forward`` called).

        Returns:
            Dict mapping camera name to CameraMetadata.
        """
        return {
            name: get_camera_metadata(self._model, data, name, self._width, self._height)
            for name in CAMERA_NAMES
        }

    def close(self) -> None:
        """Release the renderer resources."""
        if not self._closed:
            self._renderer.close()
            self._closed = True

    def __enter__(self) -> MultiViewRenderer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
