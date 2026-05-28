"""Phase 0.2.4 validation script: Sim Smoke Tests Suite.

One-command smoke test that runs physics, rendering, and I/O checks,
generates artifacts to logs/sim_smoke/, and optionally attaches to W&B.

Run:
    python scripts/validate_sim_smoke.py

Artifacts:
    logs/sim_smoke/smoke_video.mp4
    logs/sim_smoke/frames/*.png
    logs/sim_smoke/sim_smoke_meta.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Configuration ---
SEED = 42
N_STEPS = 2000
MAX_PENETRATION_M = 0.05  # 5 cm (box free-fall onto plane has ~28mm with default solver)
MAX_ENERGY_J = 1000.0
SCENE_PATH = PROJECT_ROOT / "sim" / "assets" / "scenes" / "test_scene.xml"
CAMERA_NAME = "overhead"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "sim_smoke"


def run_step_smoke(model, data) -> dict:
    """Run physics stability checks. Returns result dict."""
    print("\n[1/3] Step Smoke — Physics stability")
    results = {"passed": True, "checks": {}}

    np.random.seed(SEED)
    import mujoco

    # Enable energy computation
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    data_fresh = mujoco.MjData(model)
    mujoco.mj_resetData(model, data_fresh)

    max_penetration = 0.0
    for _step in range(N_STEPS):
        mujoco.mj_step(model, data_fresh)
        for i in range(data_fresh.ncon):
            dist = data_fresh.contact[i].dist
            if dist < 0:
                max_penetration = max(max_penetration, -dist)

    # Check NaN
    has_nan = np.any(np.isnan(data_fresh.qpos)) or np.any(np.isnan(data_fresh.qvel))
    results["checks"]["no_nan"] = not has_nan
    status = "OK" if not has_nan else "FAIL"
    print(f"  [{status}] No NaN in qpos/qvel")

    # Check finite
    is_finite = np.all(np.isfinite(data_fresh.qpos))
    results["checks"]["qpos_finite"] = is_finite
    status = "OK" if is_finite else "FAIL"
    print(f"  [{status}] qpos finite")

    # Check energy
    total_energy = data_fresh.energy[0] + data_fresh.energy[1]
    energy_ok = np.isfinite(total_energy) and abs(total_energy) < MAX_ENERGY_J
    results["checks"]["energy_bounded"] = energy_ok
    status = "OK" if energy_ok else "FAIL"
    print(f"  [{status}] Energy: {total_energy:.2f} J (threshold: {MAX_ENERGY_J} J)")

    # Check penetration
    pen_ok = max_penetration < MAX_PENETRATION_M
    results["checks"]["penetration_bounded"] = pen_ok
    status = "OK" if pen_ok else "FAIL"
    print(
        f"  [{status}] Max penetration: {max_penetration * 1000:.2f} mm "
        f"(threshold: {MAX_PENETRATION_M * 1000:.0f} mm)"
    )

    results["max_penetration_m"] = max_penetration
    results["total_energy_j"] = float(total_energy)
    results["passed"] = all(results["checks"].values())
    return results


def run_render_smoke(model) -> dict:
    """Run rendering correctness checks. Returns result dict."""
    print("\n[2/3] Render Smoke — Rendering correctness")
    results = {"passed": True, "checks": {}}

    import mujoco

    from sim.offscreen import (
        RenderConfig,
        create_renderer,
        render_frame,
        render_trajectory,
        resolve_camera_id,
    )

    config = RenderConfig(camera_name=CAMERA_NAME)
    cam_id = resolve_camera_id(model, camera_name=CAMERA_NAME)

    # RGB check
    data = mujoco.MjData(model)
    renderer = create_renderer(model, config)
    try:
        frame = render_frame(renderer, model, data, step_index=0, config=config, camera_id=cam_id)
        rgb_ok = (
            frame.rgb.shape == (480, 640, 3) and frame.rgb.dtype == np.uint8 and frame.rgb.sum() > 0
        )
        results["checks"]["rgb_valid"] = rgb_ok
        status = "OK" if rgb_ok else "FAIL"
        print(f"  [{status}] RGB: shape={frame.rgb.shape}, mean={frame.rgb.mean():.1f}")
    finally:
        renderer.close()

    # Depth check
    depth_config = RenderConfig(render_depth=True)
    data = mujoco.MjData(model)
    renderer = create_renderer(model, depth_config)
    try:
        frame = render_frame(renderer, model, data, step_index=0, config=depth_config)
        depth_ok = (
            frame.depth is not None
            and frame.depth.shape == (480, 640)
            and np.all(np.isfinite(frame.depth))
            and frame.depth.max() > 0
        )
        results["checks"]["depth_valid"] = depth_ok
        status = "OK" if depth_ok else "FAIL"
        print(
            f"  [{status}] Depth: shape={frame.depth.shape}, range=[{frame.depth.min():.3f}, {frame.depth.max():.3f}]"
        )
    finally:
        renderer.close()

    # Determinism check
    det_config = RenderConfig(width=160, height=120, camera_name=CAMERA_NAME)
    runs = []
    for _ in range(2):
        np.random.seed(SEED)
        m = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        frames = render_trajectory(m, d, n_steps=50, config=det_config, render_every=10)
        runs.append([f.rgb for f in frames])

    # Allow atol=1 for GPU-accelerated renderers (EGL) where rasteriser
    # floating-point non-determinism can cause ±1 pixel-value jitter.
    det_ok = all(np.allclose(a, b, atol=1) for a, b in zip(runs[0], runs[1], strict=True))
    max_diff = max(
        int(np.max(np.abs(a.astype(int) - b.astype(int))))
        for a, b in zip(runs[0], runs[1], strict=True)
    )
    results["checks"]["deterministic"] = det_ok
    results["checks"]["deterministic_max_diff"] = max_diff
    status = "OK" if det_ok else "FAIL"
    print(f"  [{status}] Render determinism (2 runs, 5 frames each, max_diff={max_diff})")

    # Trajectory sync check
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    frames = render_trajectory(model, data, n_steps=100, config=config, render_every=10)
    expected = list(range(0, 100, 10))
    actual = [f.step_index for f in frames]
    sync_ok = actual == expected and len(frames) == 10
    results["checks"]["trajectory_sync"] = sync_ok
    status = "OK" if sync_ok else "FAIL"
    print(f"  [{status}] Trajectory sync: {len(frames)} frames, indices match")

    results["passed"] = all(results["checks"].values())
    return results


def run_io_smoke(model, output_dir: Path) -> dict:
    """Run I/O artifact generation checks. Returns result dict."""
    print("\n[3/3] I/O Smoke — Artifact generation")
    results = {"passed": True, "checks": {}}

    import mujoco

    from sim.offscreen import RenderConfig, render_trajectory, save_sample_frames, save_video

    output_dir.mkdir(parents=True, exist_ok=True)
    config = RenderConfig(camera_name=CAMERA_NAME)

    # Video export
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    fps = 30.0
    sim_duration = 2.0
    timestep = model.opt.timestep
    n_steps = int(sim_duration / timestep)
    render_every = max(1, int(1.0 / (timestep * fps)))

    frames = render_trajectory(
        model, data, n_steps=n_steps, config=config, render_every=render_every
    )
    video_path = output_dir / "smoke_video.mp4"
    save_video(frames, video_path, fps=fps)

    video_ok = video_path.exists() and video_path.stat().st_size > 0
    results["checks"]["video_export"] = video_ok
    size_kb = video_path.stat().st_size / 1024 if video_path.exists() else 0
    status = "OK" if video_ok else "FAIL"
    print(f"  [{status}] Video: {video_path.name} ({size_kb:.1f} KB, {len(frames)} frames)")

    # Frame export
    frames_dir = output_dir / "frames"
    saved = save_sample_frames(frames, frames_dir, max_samples=5)
    frames_ok = len(saved) == 5
    results["checks"]["frame_export"] = frames_ok
    status = "OK" if frames_ok else "FAIL"
    print(f"  [{status}] Frames: {len(saved)} PNGs in {frames_dir.name}/")

    # Metadata JSON
    from sim.env_meta import collect_metadata

    meta = collect_metadata(PROJECT_ROOT)
    smoke_meta = {
        **meta,
        "phase": "0.2.4",
        "seed": SEED,
        "config": {
            "n_steps": N_STEPS,
            "max_penetration_m": MAX_PENETRATION_M,
            "max_energy_j": MAX_ENERGY_J,
            "scene": str(SCENE_PATH),
            "camera": CAMERA_NAME,
            "video_fps": fps,
            "sim_duration_s": sim_duration,
        },
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    meta_path = output_dir / "sim_smoke_meta.json"
    serializable = {
        k: str(v) if not isinstance(v, (str, int, float, bool, dict, list)) else v
        for k, v in smoke_meta.items()
    }
    meta_path.write_text(json.dumps(serializable, indent=2))

    meta_ok = meta_path.exists() and "seed" in json.loads(meta_path.read_text())
    results["checks"]["metadata_json"] = meta_ok
    status = "OK" if meta_ok else "FAIL"
    print(f"  [{status}] Metadata: {meta_path.name}")

    results["passed"] = all(results["checks"].values())
    results["artifacts"] = {
        "video": str(video_path),
        "frames_dir": str(frames_dir),
        "metadata": str(meta_path),
    }
    return results


def attach_wandb_artifacts(output_dir: Path, all_results: dict) -> None:
    """Optionally attach artifacts to W&B if available and enabled."""
    wandb_mode = os.environ.get("WANDB_MODE", "disabled").lower()
    if wandb_mode == "disabled":
        print("\n[W&B] Skipped (WANDB_MODE=disabled). Set WANDB_MODE=online to enable.")
        return

    try:
        import wandb
    except ImportError:
        print("\n[W&B] Skipped (wandb not installed).")
        return

    try:
        run = wandb.init(
            project="vla-lego",
            name=f"smoke-{datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            tags=["phase:0.2.4", "type:smoke-test"],
            config={
                "seed": SEED,
                "n_steps": N_STEPS,
                "max_penetration_m": MAX_PENETRATION_M,
                "max_energy_j": MAX_ENERGY_J,
            },
            job_type="smoke-test",
        )

        # Log results as summary metrics
        for category, result in all_results.items():
            run.summary[f"smoke/{category}/passed"] = result["passed"]
            for check, ok in result.get("checks", {}).items():
                run.summary[f"smoke/{category}/{check}"] = ok

        # Log artifacts
        artifact = wandb.Artifact("sim-smoke-artifacts", type="smoke-test")
        if (output_dir / "smoke_video.mp4").exists():
            artifact.add_file(str(output_dir / "smoke_video.mp4"))
        if (output_dir / "sim_smoke_meta.json").exists():
            artifact.add_file(str(output_dir / "sim_smoke_meta.json"))
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            artifact.add_dir(str(frames_dir), name="frames")
        run.log_artifact(artifact)

        run.finish()
        print(f"\n[W&B] Artifacts logged to run: {run.get_url()}")
    except Exception as e:
        print(f"\n[W&B] Warning: failed to log artifacts: {e}")


def main() -> int:
    """Run all sim smoke tests. Returns 0 on success, 1 on failure."""
    print("=" * 60)
    print("Phase 0.2.4 Validation: Sim Smoke Tests Suite")
    print("=" * 60)

    try:
        import mujoco  # noqa: F811

        from sim.mujoco_env import load_model
    except ImportError as e:
        print(f"FAIL: {e}")
        return 1

    model = load_model(SCENE_PATH)
    data = mujoco.MjData(model)

    all_results = {}

    # 1. Step smoke
    all_results["step"] = run_step_smoke(model, data)

    # 2. Render smoke
    all_results["render"] = run_render_smoke(model)

    # 3. I/O smoke
    all_results["io"] = run_io_smoke(model, OUTPUT_DIR)

    # Update metadata with test results
    meta_path = OUTPUT_DIR / "sim_smoke_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["results"] = {
            cat: {
                "passed": bool(r["passed"]),
                "checks": {k: bool(v) for k, v in r.get("checks", {}).items()},
            }
            for cat, r in all_results.items()
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    # Optional W&B attachment
    attach_wandb_artifacts(OUTPUT_DIR, all_results)

    # Summary
    all_passed = all(r["passed"] for r in all_results.values())
    print("\n" + "=" * 60)
    for cat, r in all_results.items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {cat.upper():>8}: [{status}]")
    print("=" * 60)

    if all_passed:
        print("ALL SMOKE TESTS PASSED")
        print(f"Artifacts: {OUTPUT_DIR}")
    else:
        print("SOME SMOKE TESTS FAILED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
