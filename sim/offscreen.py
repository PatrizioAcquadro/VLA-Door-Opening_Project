"""Headless offscreen rendering for MuJoCo simulations (Phase 0.2.3).

Produces RGB, depth, and segmentation frames without a display.
Does NOT import mujoco.viewer — this module is fully headless.

Usage:
    from sim.offscreen import RenderConfig, render_trajectory, save_video

    model = load_model("sim/assets/scenes/test_scene.xml")
    data = mujoco.MjData(model)
    frames = render_trajectory(model, data, n_steps=2500, config=RenderConfig())
    save_video(frames, "output.mp4", fps=30)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


@dataclass
class RenderConfig:
    """Configuration for offscreen rendering.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        camera_name: Named MJCF camera (None = use camera_id).
        camera_id: Numeric camera ID (-1 = free/default camera).
        render_depth: Also capture depth frames.
        render_segmentation: Also capture segmentation masks.
    """

    width: int = 640
    height: int = 480
    camera_name: str | None = None
    camera_id: int = -1
    render_depth: bool = False
    render_segmentation: bool = False


@dataclass
class RenderedFrame:
    """A single rendered frame with all requested modalities.

    Attributes:
        step_index: Simulation step that produced this frame.
        rgb: RGB image array, uint8, shape (H, W, 3).
        depth: Depth map, float32, shape (H, W). None if not requested.
        segmentation: Segmentation mask, int32, shape (H, W, 2). None if not requested.
    """

    step_index: int
    rgb: np.ndarray
    depth: np.ndarray | None = None
    segmentation: np.ndarray | None = None


def resolve_camera_id(
    model: mujoco.MjModel,
    camera_name: str | None = None,
    camera_id: int = -1,
) -> int:
    """Resolve camera specification to a numeric camera ID.

    Args:
        model: Compiled MuJoCo model.
        camera_name: Named camera from MJCF. Takes priority over camera_id.
        camera_id: Numeric camera ID (-1 = free/default camera).

    Returns:
        Resolved camera ID.

    Raises:
        ValueError: If camera_name is not found in the model.
    """
    if camera_name is not None:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")
        return int(cam_id)
    return camera_id


def create_renderer(
    model: mujoco.MjModel,
    config: RenderConfig,
) -> mujoco.Renderer:
    """Create a MuJoCo offscreen renderer.

    Args:
        model: Compiled MuJoCo model.
        config: Rendering configuration.

    Returns:
        Initialized mujoco.Renderer.
    """
    return mujoco.Renderer(model, height=config.height, width=config.width)


def render_frame(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step_index: int,
    config: RenderConfig,
    camera_id: int = -1,
) -> RenderedFrame:
    """Render a single frame from current simulation state.

    Captures RGB and optionally depth/segmentation. The step_index is stored
    in the returned RenderedFrame for synchronization tracking.

    Args:
        renderer: Pre-created MuJoCo Renderer.
        model: Compiled MuJoCo model.
        data: MuJoCo data at current state.
        step_index: Current simulation step (for sync contract).
        config: Render configuration.
        camera_id: Resolved camera ID (-1 = free camera).

    Returns:
        RenderedFrame with all requested modalities.
    """
    # Ensure derived quantities (lighting, contacts, etc.) are up to date
    mujoco.mj_forward(model, data)

    # RGB render
    renderer.update_scene(data, camera_id)
    rgb = renderer.render().copy()

    # Depth render
    depth = None
    if config.render_depth:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera_id)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

    # Segmentation render
    segmentation = None
    if config.render_segmentation:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera_id)
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()

    return RenderedFrame(
        step_index=step_index,
        rgb=rgb,
        depth=depth,
        segmentation=segmentation,
    )


def render_trajectory(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_steps: int,
    config: RenderConfig,
    render_every: int = 1,
) -> list[RenderedFrame]:
    """Step simulation and render frames at regular intervals.

    Enforces the synchronization contract: each RenderedFrame.step_index
    corresponds exactly to the simulation step at which it was captured.

    Args:
        model: Compiled MuJoCo model.
        data: MuJoCo data (will be modified in-place by stepping).
        n_steps: Total simulation steps to run.
        config: Render configuration.
        render_every: Capture a frame every N steps (default=1).

    Returns:
        List of RenderedFrame objects.
    """
    camera_id = resolve_camera_id(model, config.camera_name, config.camera_id)
    renderer = create_renderer(model, config)
    frames: list[RenderedFrame] = []

    try:
        for step in range(n_steps):
            if step % render_every == 0:
                frames.append(render_frame(renderer, model, data, step, config, camera_id))
            mujoco.mj_step(model, data)
    finally:
        renderer.close()

    return frames


def save_video(
    frames: list[RenderedFrame],
    output_path: str | Path,
    fps: float = 30.0,
) -> Path:
    """Write rendered RGB frames to an MP4 video file.

    Args:
        frames: List of RenderedFrame (uses .rgb arrays).
        output_path: Destination file path (should end in .mp4).
        fps: Video frame rate.

    Returns:
        Path to the written video file.

    Raises:
        ImportError: If imageio is not installed.
        ValueError: If frames list is empty.
    """
    import imageio.v3 as iio

    if not frames:
        raise ValueError("Cannot save video from empty frame list")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_stack = np.stack([f.rgb for f in frames])
    iio.imwrite(str(output_path), rgb_stack, fps=fps)

    return output_path


def save_sample_frames(
    frames: list[RenderedFrame],
    output_dir: str | Path,
    indices: list[int] | None = None,
    max_samples: int = 5,
) -> list[Path]:
    """Save sample frames as PNG images.

    Args:
        frames: List of RenderedFrame.
        output_dir: Directory to write images into.
        indices: Specific frame list indices to save. If None, evenly spaced.
        max_samples: Maximum number of samples (used when indices is None).

    Returns:
        List of paths to saved image files.
    """
    import imageio.v3 as iio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        n = len(frames)
        if n <= max_samples:
            indices = list(range(n))
        else:
            indices = [int(i * (n - 1) / (max_samples - 1)) for i in range(max_samples)]

    saved: list[Path] = []
    for idx in indices:
        frame = frames[idx]
        # RGB
        rgb_path = output_dir / f"frame_{frame.step_index:06d}_rgb.png"
        iio.imwrite(str(rgb_path), frame.rgb)
        saved.append(rgb_path)

        # Depth (normalized to 0-255 for visualization)
        if frame.depth is not None:
            depth = frame.depth
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)
            depth_path = output_dir / f"frame_{frame.step_index:06d}_depth.png"
            iio.imwrite(str(depth_path), depth_norm)
            saved.append(depth_path)

    return saved


def validate_backend(model: mujoco.MjModel) -> dict[str, Any]:
    """Validate that offscreen rendering works on this system.

    Creates a renderer, renders one frame, checks shapes and non-emptiness.

    Args:
        model: Compiled MuJoCo model.

    Returns:
        Dict with keys: backend, rgb_shape, depth_shape, seg_shape, success.
    """
    result: dict[str, Any] = {
        "backend": os.environ.get("MUJOCO_GL", "auto"),
        "rgb_shape": None,
        "depth_shape": None,
        "seg_shape": None,
        "success": False,
    }

    config = RenderConfig(render_depth=True, render_segmentation=True)
    data = mujoco.MjData(model)

    try:
        renderer = create_renderer(model, config)
        try:
            frame = render_frame(renderer, model, data, step_index=0, config=config)
            result["rgb_shape"] = frame.rgb.shape
            result["depth_shape"] = frame.depth.shape if frame.depth is not None else None
            result["seg_shape"] = (
                frame.segmentation.shape if frame.segmentation is not None else None
            )
            result["success"] = frame.rgb.size > 0
        finally:
            renderer.close()
    except Exception as e:
        result["error"] = str(e)

    return result
