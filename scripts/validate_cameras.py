#!/usr/bin/env python3
"""Validate the frozen 4-view camera contract (Phase 1.2.4).

Checks:
  - All 4 cameras exist in model and render non-black RGB
  - Wrist cameras track arm motion; overhead/third_person are static
  - Depth and segmentation rendering works for all 4 views
  - Camera intrinsics are computed correctly from fovy
  - Per-capture metadata is populated and live for wrist cameras
  - Dataset sanity: quad-view 2x2 grid video with arm motion

Artifacts are saved to ``logs/camera_validation/``.

Run:
    python scripts/validate_cameras.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from sim.action_space import ACTION_DIM, ARM_DIM, GRIPPER_DIM  # noqa: E402
from sim.asset_loader import load_scene  # noqa: E402
from sim.camera import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_CAPTURE_HZ,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    NUM_VIEWS,
    MultiViewRenderer,
    get_camera_metadata,
)
from sim.offscreen import RenderedFrame, resolve_camera_id, save_video  # noqa: E402
from sim.sim_runner import SimRunner  # noqa: E402

ARTIFACT_DIR = _project_root / "logs" / "camera_validation"


# ---------------------------------------------------------------------------
# Contract summary
# ---------------------------------------------------------------------------


def print_contract() -> None:
    """Print the frozen 4-view camera contract."""
    print("=" * 70)
    print("MULTI-VIEW CAMERA CONTRACT (Phase 1.2.4)")
    print("=" * 70)
    print(f"  Views:           {list(CAMERA_NAMES)}")
    print(f"  Num views:       {NUM_VIEWS}")
    print(f"  Resolution:      {DEFAULT_WIDTH}x{DEFAULT_HEIGHT} (square)")
    print(f"  Capture rate:    {DEFAULT_CAPTURE_HZ} Hz")
    print("  Modalities:      RGB + Depth + Segmentation")
    print()


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def test_render_at_rest(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Render all 4 views at rest and verify non-black RGB."""
    print("-" * 70)
    print("Render at rest (home keyframe) — all 4 views")
    print("-" * 70)

    passed = True
    with MultiViewRenderer(model) as mv:
        frame = mv.capture(data, step_index=0)
        for name in CAMERA_NAMES:
            rgb = frame.views[name].rgb
            shape_ok = rgb.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
            not_black = bool(np.any(rgb > 0))
            mean_val = float(np.mean(rgb))
            status = "OK" if (shape_ok and not_black) else "FAIL"
            print(
                f"  {name:18s}  shape={rgb.shape}  "
                f"not_black={'OK' if not_black else 'FAIL'}  "
                f"mean_px={mean_val:.1f}  [{status}]"
            )
            if not shape_ok or not not_black:
                passed = False

    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_depth_segmentation(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Render depth and segmentation for all 4 views."""
    print("-" * 70)
    print("Depth + segmentation rendering — all 4 views")
    print("-" * 70)

    passed = True
    with MultiViewRenderer(model, render_depth=True, render_segmentation=True) as mv:
        frame = mv.capture(data, step_index=0)
        for name in CAMERA_NAMES:
            view = frame.views[name]
            depth_ok = view.depth is not None and np.all(np.isfinite(view.depth))
            seg_ok = view.segmentation is not None
            status = "OK" if (depth_ok and seg_ok) else "FAIL"
            depth_shape = view.depth.shape if view.depth is not None else "None"
            print(
                f"  {name:18s}  depth={depth_shape}  seg={'OK' if seg_ok else 'None'}  [{status}]"
            )
            if not depth_ok or not seg_ok:
                passed = False

    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_sync(model: mujoco.MjModel) -> bool:
    """Verify all 4 views share the same step_index."""
    print("-" * 70)
    print("Synchronization check (all 4 views)")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    with MultiViewRenderer(model) as mv:
        frame = mv.capture(data, step_index=5)
        indices = {name: frame.views[name].step_index for name in CAMERA_NAMES}
        all_same = len(set(indices.values())) == 1
        print(f"  Step indices:   {indices}")
        print(f"  Frame index:    {frame.step_index}")
        print(f"  Synchronized:   {'OK' if all_same else 'FAIL'}")

    passed = all_same and frame.step_index == 5
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_wrist_camera_movement(model: mujoco.MjModel) -> bool:
    """Verify wrist cams track arm; overhead/third_person are static."""
    print("-" * 70)
    print("Wrist camera movement vs fixed camera stability")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    cam_rest = {name: data.cam_xpos[resolve_camera_id(model, name)].copy() for name in CAMERA_NAMES}

    # Move left shoulder
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_y")
    data.qpos[model.jnt_qposadr[jnt_id]] = -0.5
    mujoco.mj_forward(model, data)

    passed = True
    for name in CAMERA_NAMES:
        cam_id = resolve_camera_id(model, name)
        delta = float(np.linalg.norm(data.cam_xpos[cam_id] - cam_rest[name]))
        body_id = int(model.cam_bodyid[cam_id])
        is_body_attached = body_id != 0

        expected_moved = is_body_attached
        actually_moved = delta > 0.01
        ok = expected_moved == actually_moved

        print(
            f"  {name:18s}  body_attached={is_body_attached}  "
            f"delta={delta:.4f} m  {'moved' if actually_moved else 'static'}  "
            f"[{'OK' if ok else 'FAIL'}]"
        )
        if not ok:
            passed = False

    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_intrinsics(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Verify camera intrinsics computation for all 4 views."""
    print("-" * 70)
    print("Camera intrinsics (pinhole model from fovy)")
    print("-" * 70)

    passed = True
    for name in CAMERA_NAMES:
        meta = get_camera_metadata(model, data, name)
        intr = meta.intrinsics
        K = intr.to_matrix()

        fx_ok = intr.fx > 0
        fy_ok = intr.fy > 0
        K_ok = K.shape == (3, 3) and K[2, 2] == 1.0
        ok = fx_ok and fy_ok and K_ok

        print(
            f"  {name:18s}  fovy={meta.fovy:.1f}°  "
            f"fx={intr.fx:.1f} fy={intr.fy:.1f}  "
            f"cx={intr.cx:.1f} cy={intr.cy:.1f}  "
            f"fovx={intr.fovx:.1f}°  body_attached={meta.is_body_attached}  "
            f"[{'OK' if ok else 'FAIL'}]"
        )
        if not ok:
            passed = False

    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_per_capture_metadata(model: mujoco.MjModel) -> bool:
    """Verify MultiViewFrame.metadata is populated and updates for wrist cams."""
    print("-" * 70)
    print("Per-capture metadata (live extrinsics in MultiViewFrame)")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    with MultiViewRenderer(model) as mv:
        frame1 = mv.capture(data, step_index=0)
        has_meta = len(frame1.metadata) == NUM_VIEWS
        print(
            f"  Frame metadata views:  {len(frame1.metadata)} / {NUM_VIEWS}  "
            f"[{'OK' if has_meta else 'FAIL'}]"
        )

        pos1 = frame1.metadata["left_wrist_cam"].pos.copy()

        # Move arm and recapture
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_y")
        data.qpos[model.jnt_qposadr[jnt_id]] = -0.5
        mujoco.mj_forward(model, data)

        frame2 = mv.capture(data, step_index=1)
        pos2 = frame2.metadata["left_wrist_cam"].pos.copy()
        delta_wrist = float(np.linalg.norm(pos2 - pos1))
        wrist_moved = delta_wrist > 0.01

        pos1_overhead = frame1.metadata["overhead"].pos.copy()
        pos2_overhead = frame2.metadata["overhead"].pos.copy()
        overhead_static = float(np.linalg.norm(pos2_overhead - pos1_overhead)) < 1e-10

        print(f"  left_wrist_cam delta:  {delta_wrist:.4f} m  [{'OK' if wrist_moved else 'FAIL'}]")
        print(
            f"  overhead delta:        {float(np.linalg.norm(pos2_overhead-pos1_overhead)):.2e} m"
            f"  [{'OK' if overhead_static else 'FAIL'}]"
        )

    passed = has_meta and wrist_moved and overhead_static
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------


def save_sample_frames(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Save sample RGB frames from all 4 cameras at rest."""
    import imageio.v3 as iio

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with MultiViewRenderer(model, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT) as mv:
        frame = mv.capture(data, step_index=0)
        for name in CAMERA_NAMES:
            path = ARTIFACT_DIR / f"{name}_rgb.png"
            iio.imwrite(str(path), frame.views[name].rgb)
            print(f"  Saved: {path}")


def save_depth_samples(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Save depth map visualizations from all 4 cameras."""
    import imageio.v3 as iio

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with MultiViewRenderer(
        model, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, render_depth=True
    ) as mv:
        frame = mv.capture(data, step_index=0)
        for name in CAMERA_NAMES:
            depth = frame.views[name].depth
            if depth is None:
                continue
            # Normalize depth to [0, 255] for visualization
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth_vis = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(depth, dtype=np.uint8)
            path = ARTIFACT_DIR / f"{name}_depth.png"
            iio.imwrite(str(path), depth_vis)
            print(f"  Saved: {path}")


def save_quad_view_video(model: mujoco.MjModel, n_steps: int = 100, seed: int = 42) -> None:
    """Save a 2×2 quad-view video showing all 4 cameras simultaneously.

    Layout::

        [ overhead        | left_wrist_cam  ]
        [ right_wrist_cam | third_person     ]

    Each tile is DEFAULT_WIDTH × DEFAULT_HEIGHT; total video is 2W × 2H.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = SimRunner(model, data)
    rng = np.random.default_rng(seed)

    W, H = DEFAULT_WIDTH, DEFAULT_HEIGHT
    combined_frames: list[RenderedFrame] = []

    with MultiViewRenderer(model, width=W, height=H) as mv:
        for step in range(n_steps):
            frame = mv.capture(data, step_index=step)

            # Build 2×2 grid: top-left, top-right, bottom-left, bottom-right
            top = np.concatenate(
                [frame.views["overhead"].rgb, frame.views["left_wrist_cam"].rgb], axis=1
            )
            bottom = np.concatenate(
                [frame.views["right_wrist_cam"].rgb, frame.views["third_person"].rgb], axis=1
            )
            quad = np.concatenate([top, bottom], axis=0)  # (2H, 2W, 3)
            combined_frames.append(RenderedFrame(step_index=step, rgb=quad))

            # Step with random action
            action = rng.uniform(-0.3, 0.3, size=ACTION_DIM)
            action[ARM_DIM:] = rng.uniform(0, 1, size=GRIPPER_DIM)
            runner.step(action)

    video_path = save_video(combined_frames, ARTIFACT_DIR / "quad_view_video.mp4", fps=20.0)
    print(f"  Saved: {video_path}")


def save_metadata_json(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Save camera metadata (intrinsics, extrinsics, flags) to JSON."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with MultiViewRenderer(model) as mv:
        meta = mv.get_metadata(data)
        meta_dict = {name: m.to_dict() for name, m in meta.items()}
        meta_dict["_contract"] = {
            "phase": "1.2.4",
            "num_views": NUM_VIEWS,
            "camera_names": list(CAMERA_NAMES),
            "default_width": DEFAULT_WIDTH,
            "default_height": DEFAULT_HEIGHT,
            "capture_hz": DEFAULT_CAPTURE_HZ,
            "modalities": ["rgb", "depth", "segmentation"],
        }

    json_path = ARTIFACT_DIR / "camera_metadata.json"
    with open(json_path, "w") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"  Saved: {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading Alex upper-body model...")
    model = load_scene("alex_upper_body")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print_contract()

    results: dict[str, bool] = {}
    results["render_at_rest"] = test_render_at_rest(model, data)
    results["depth_segmentation"] = test_depth_segmentation(model, data)
    results["sync"] = test_sync(model)
    results["wrist_cam_movement"] = test_wrist_camera_movement(model)
    results["intrinsics"] = test_intrinsics(model, data)
    results["per_capture_metadata"] = test_per_capture_metadata(model)

    # Artifacts
    print("-" * 70)
    print("Exporting artifacts")
    print("-" * 70)
    save_sample_frames(model, data)
    save_depth_samples(model, data)
    save_quad_view_video(model)
    save_metadata_json(model, data)
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s} {status}")
    print()

    if all_pass:
        print("All validation checks PASSED.")
        print(f"Artifacts saved to: {ARTIFACT_DIR}")
    else:
        print("Some checks FAILED — see details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
