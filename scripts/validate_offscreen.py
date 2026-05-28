"""Phase 0.2.3 validation script: Headless offscreen rendering check.

Run:
    python scripts/validate_offscreen.py

Checks:
    1. Offscreen renderer creates successfully (backend validation)
    2. RGB frame render (correct shape, non-empty)
    3. Depth + segmentation render (correct shapes)
    4. Frame-step synchronization (render_trajectory sync contract)
    5. Video export (MP4 artifact)
    6. Sample frame export (PNG artifacts)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    """Run all Phase 0.2.3 validation checks. Returns 0 on success, 1 on failure."""
    print("=" * 60)
    print("Phase 0.2.3 Validation: Headless Offscreen Rendering")
    print("=" * 60)

    # Check 1: Backend validation
    print("\n[1/6] Validating offscreen rendering backend...")
    try:
        import mujoco  # noqa: F811

        from sim.mujoco_env import load_model
        from sim.offscreen import validate_backend

        scene_path = PROJECT_ROOT / "sim" / "assets" / "scenes" / "test_scene.xml"
        model = load_model(scene_path)
        result = validate_backend(model)
        if result["success"]:
            print(f"  OK: backend={result['backend']}")
            print(f"      rgb_shape={result['rgb_shape']}")
            print(f"      depth_shape={result['depth_shape']}")
            print(f"      seg_shape={result['seg_shape']}")
        else:
            print(f"  FAIL: {result.get('error', 'unknown error')}")
            return 1
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Check 2: RGB frame render
    print("\n[2/6] Rendering single RGB frame...")
    try:
        from sim.offscreen import RenderConfig, create_renderer, render_frame, resolve_camera_id

        camera_name = "overhead"
        cam_id = resolve_camera_id(model, camera_name=camera_name)
        data = mujoco.MjData(model)
        config = RenderConfig(camera_name=camera_name)
        renderer = create_renderer(model, config)
        try:
            frame = render_frame(
                renderer, model, data, step_index=0, config=config, camera_id=cam_id
            )
        finally:
            renderer.close()

        assert frame.rgb.shape == (480, 640, 3), f"Bad shape: {frame.rgb.shape}"
        assert frame.rgb.dtype.name == "uint8", f"Bad dtype: {frame.rgb.dtype}"
        mean_val = frame.rgb.mean()
        assert mean_val > 0, "Frame is all black"
        print(f"  OK: shape={frame.rgb.shape}, dtype={frame.rgb.dtype}, mean={mean_val:.1f}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Check 3: Depth + segmentation
    print("\n[3/6] Rendering depth + segmentation...")
    try:
        config_full = RenderConfig(
            camera_name=camera_name, render_depth=True, render_segmentation=True
        )
        cam_id = resolve_camera_id(model, camera_name=camera_name)
        data = mujoco.MjData(model)
        renderer = create_renderer(model, config_full)
        try:
            frame = render_frame(
                renderer, model, data, step_index=0, config=config_full, camera_id=cam_id
            )
        finally:
            renderer.close()

        assert frame.depth is not None, "Depth is None"
        assert frame.depth.shape == (480, 640), f"Bad depth shape: {frame.depth.shape}"
        print(f"  OK: depth shape={frame.depth.shape}, dtype={frame.depth.dtype}")
        print(f"      depth range=[{frame.depth.min():.3f}, {frame.depth.max():.3f}]")

        assert frame.segmentation is not None, "Segmentation is None"
        print(f"      seg shape={frame.segmentation.shape}, dtype={frame.segmentation.dtype}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Check 4: Frame-step synchronization
    print("\n[4/6] Verifying frame-step synchronization...")
    try:
        from sim.offscreen import render_trajectory

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        sync_config = RenderConfig(camera_name=camera_name)
        frames = render_trajectory(model, data, n_steps=100, config=sync_config, render_every=10)

        expected = list(range(0, 100, 10))
        actual = [f.step_index for f in frames]
        assert actual == expected, f"Sync mismatch: {actual} != {expected}"
        print(f"  OK: {len(frames)} frames, step_indices={actual}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Check 5: Video export
    print("\n[5/6] Exporting smoke-test video (~5s sim time)...")
    try:
        from sim.offscreen import save_video

        fps = 30.0
        sim_duration = 5.0  # seconds of sim time
        timestep = model.opt.timestep
        n_steps = int(sim_duration / timestep)
        render_every = max(1, int(1.0 / (timestep * fps)))

        video_config = RenderConfig(camera_name=camera_name)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        frames = render_trajectory(
            model, data, n_steps=n_steps, config=video_config, render_every=render_every
        )

        video_path = PROJECT_ROOT / "logs" / "offscreen_smoke.mp4"
        save_video(frames, video_path, fps=fps)

        size_kb = video_path.stat().st_size / 1024
        print(f"  OK: {video_path} ({size_kb:.1f} KB, {len(frames)} frames)")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Check 6: Sample frame export
    print("\n[6/6] Exporting sample frames...")
    try:
        from sim.offscreen import save_sample_frames

        frames_dir = PROJECT_ROOT / "logs" / "offscreen_frames"
        saved = save_sample_frames(frames, frames_dir, max_samples=5)
        print(f"  OK: {len(saved)} files saved to {frames_dir}")
        for p in saved:
            print(f"      {p.name}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Save metadata
    print("\n--- Saving metadata ---")
    try:
        from sim.env_meta import collect_metadata

        meta = collect_metadata(PROJECT_ROOT)
        meta["phase"] = "0.2.3"
        meta["render_resolution"] = f"{config.width}x{config.height}"
        meta["video_frames"] = len(frames)
        meta["video_fps"] = fps

        meta_path = PROJECT_ROOT / "logs" / "offscreen_meta.json"
        serializable_meta = {
            k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in meta.items()
        }
        meta_path.write_text(json.dumps(serializable_meta, indent=2))
        print(f"  Metadata saved to {meta_path}")
    except Exception as e:
        print(f"  Warning: metadata save failed: {e}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
