"""Sim Smoke Tests Suite (Phase 0.2.4).

Consolidated smoke tests covering physics stability, rendering correctness,
and I/O artifact generation. Run with:

    pytest -m smoke -v
    pytest tests/test_sim_smoke.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

ASSETS_DIR = Path(__file__).resolve().parent.parent / "sim" / "assets"
TEST_SCENE = ASSETS_DIR / "scenes" / "test_scene.xml"
TEST_CAMERA = "overhead"

# --- Pass/fail thresholds ---
SEED = 42
N_STEPS = 2000
MAX_PENETRATION_M = 0.05  # 5 cm (box free-fall onto plane has ~28mm with default solver)
MAX_ENERGY_J = 1000.0  # generous bound for a 0.1 kg box


@pytest.fixture
def seeded_model_data():
    """Load test scene and create data with fixed seed."""
    from sim.mujoco_env import load_model

    np.random.seed(SEED)
    model = load_model(TEST_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    return model, data


@pytest.fixture
def render_config():
    """Default RenderConfig for smoke tests."""
    from sim.offscreen import RenderConfig

    return RenderConfig(camera_name=TEST_CAMERA)


# ---------------------------------------------------------------------------
# A. Step Smoke — Physics stability
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.mujoco
class TestStepSmoke:
    """Physics stepping stability checks with explicit thresholds."""

    def test_no_nans(self, seeded_model_data: tuple) -> None:
        """No NaN values in qpos/qvel after N steps."""
        model, data = seeded_model_data
        for _ in range(N_STEPS):
            mujoco.mj_step(model, data)
        assert not np.any(np.isnan(data.qpos)), "NaN in qpos"
        assert not np.any(np.isnan(data.qvel)), "NaN in qvel"

    def test_qpos_finite(self, seeded_model_data: tuple) -> None:
        """All qpos values remain finite (no explosion)."""
        model, data = seeded_model_data
        for _ in range(N_STEPS):
            mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.qpos)), "Non-finite qpos detected"

    def test_energy_bounded(self, seeded_model_data: tuple) -> None:
        """Total energy stays below threshold."""
        model, data = seeded_model_data
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        # Re-create data after model change
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        for _ in range(N_STEPS):
            mujoco.mj_step(model, data)

        total_energy = data.energy[0] + data.energy[1]
        assert np.isfinite(total_energy), "Energy is not finite"
        assert (
            abs(total_energy) < MAX_ENERGY_J
        ), f"Energy {total_energy:.2f} J exceeds threshold {MAX_ENERGY_J} J"

    def test_penetration_bounded(self, seeded_model_data: tuple) -> None:
        """Max contact penetration stays below threshold."""
        model, data = seeded_model_data
        max_penetration = 0.0

        for _ in range(N_STEPS):
            mujoco.mj_step(model, data)
            for i in range(data.ncon):
                dist = data.contact[i].dist
                # Negative dist = penetration
                if dist < 0:
                    max_penetration = max(max_penetration, -dist)

        assert max_penetration < MAX_PENETRATION_M, (
            f"Max penetration {max_penetration * 1000:.2f} mm "
            f"exceeds threshold {MAX_PENETRATION_M * 1000:.0f} mm"
        )


# ---------------------------------------------------------------------------
# B. Render Smoke — Rendering correctness and determinism
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.mujoco
class TestRenderSmoke:
    """Rendering correctness and determinism checks."""

    def test_rgb_shape_and_content(self, seeded_model_data: tuple, render_config) -> None:
        """RGB frame has correct shape, dtype, and is non-empty."""
        from sim.offscreen import create_renderer, render_frame, resolve_camera_id

        model, data = seeded_model_data
        cam_id = resolve_camera_id(model, render_config.camera_name)
        renderer = create_renderer(model, render_config)
        try:
            frame = render_frame(
                renderer, model, data, step_index=0, config=render_config, camera_id=cam_id
            )
            assert frame.rgb.shape == (480, 640, 3)
            assert frame.rgb.dtype == np.uint8
            assert frame.rgb.sum() > 0, "Frame is all black"
        finally:
            renderer.close()

    def test_depth_valid(self, seeded_model_data: tuple) -> None:
        """Depth frame has correct shape, finite values, and non-zero content."""
        from sim.offscreen import RenderConfig, create_renderer, render_frame

        model, data = seeded_model_data
        config = RenderConfig(render_depth=True)
        renderer = create_renderer(model, config)
        try:
            frame = render_frame(renderer, model, data, step_index=0, config=config)
            assert frame.depth is not None
            assert frame.depth.shape == (480, 640)
            assert np.all(np.isfinite(frame.depth)), "Depth has non-finite values"
            assert frame.depth.max() > 0, "Depth is all zero"
        finally:
            renderer.close()

    def test_render_deterministic(self) -> None:
        """Two runs from same state produce identical RGB."""
        from sim.mujoco_env import load_model
        from sim.offscreen import RenderConfig, render_trajectory

        config = RenderConfig(width=160, height=120, camera_name=TEST_CAMERA)
        results = []
        for _ in range(2):
            np.random.seed(SEED)
            model = load_model(TEST_SCENE)
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            frames = render_trajectory(model, data, n_steps=50, config=config, render_every=10)
            results.append([f.rgb for f in frames])

        for a, b in zip(results[0], results[1], strict=True):
            # atol=1 allows for GPU-accelerated renderer (EGL) rounding jitter
            assert np.allclose(a, b, atol=1), (
                f"Rendering is not deterministic (max_diff="
                f"{int(np.max(np.abs(a.astype(int) - b.astype(int))))})"
            )

    def test_trajectory_sync(self, seeded_model_data: tuple, render_config) -> None:
        """Trajectory frame step_indices match expected values."""
        from sim.offscreen import render_trajectory

        model, data = seeded_model_data
        frames = render_trajectory(model, data, n_steps=100, config=render_config, render_every=10)
        assert len(frames) == 10
        expected = list(range(0, 100, 10))
        actual = [f.step_index for f in frames]
        assert actual == expected, f"Sync mismatch: {actual} != {expected}"


# ---------------------------------------------------------------------------
# C. I/O Smoke — Artifact generation and directory layout
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.mujoco
class TestIOSmoke:
    """Artifact I/O and directory layout checks."""

    def test_video_export(self, seeded_model_data: tuple, render_config, tmp_path: Path) -> None:
        """Video export produces valid MP4 file."""
        pytest.importorskip("imageio")
        from sim.offscreen import render_trajectory, save_video

        model, data = seeded_model_data
        frames = render_trajectory(model, data, n_steps=30, config=render_config, render_every=1)
        video_path = tmp_path / "smoke_video.mp4"
        save_video(frames, video_path, fps=10)

        assert video_path.exists(), "Video file not created"
        assert video_path.stat().st_size > 0, "Video file is empty"

    def test_frame_export(self, seeded_model_data: tuple, render_config, tmp_path: Path) -> None:
        """Sample frame export produces correct PNG count."""
        pytest.importorskip("imageio")
        from sim.offscreen import render_trajectory, save_sample_frames

        model, data = seeded_model_data
        frames = render_trajectory(model, data, n_steps=50, config=render_config, render_every=10)
        frames_dir = tmp_path / "smoke_frames"
        saved = save_sample_frames(frames, frames_dir, max_samples=3)

        assert len(saved) == 3
        pngs = list(frames_dir.glob("*.png"))
        assert len(pngs) == 3

    def test_metadata_json(self, tmp_path: Path) -> None:
        """Metadata JSON contains expected keys including seed and config."""
        from sim.env_meta import collect_metadata

        meta = collect_metadata()

        # Add smoke-test-specific fields
        smoke_meta = {
            **meta,
            "phase": "0.2.4",
            "seed": SEED,
            "config": {
                "n_steps": N_STEPS,
                "max_penetration_m": MAX_PENETRATION_M,
                "max_energy_j": MAX_ENERGY_J,
                "scene": str(TEST_SCENE),
            },
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        meta_path = tmp_path / "sim_smoke_meta.json"
        meta_path.write_text(json.dumps(smoke_meta, indent=2))

        # Verify roundtrip
        loaded = json.loads(meta_path.read_text())
        assert "seed" in loaded
        assert "config" in loaded
        assert "mujoco_version" in loaded
        assert "git_commit" in loaded
        assert "timestamp" in loaded
        assert loaded["seed"] == SEED

    def test_artifact_directory_layout(
        self, seeded_model_data: tuple, render_config, tmp_path: Path
    ) -> None:
        """All artifacts land under a single run directory."""
        pytest.importorskip("imageio")
        from sim.env_meta import collect_metadata
        from sim.offscreen import render_trajectory, save_sample_frames, save_video

        model, data = seeded_model_data
        run_dir = tmp_path / "smoke_run"
        run_dir.mkdir()

        # Generate all artifacts in the run directory
        frames = render_trajectory(model, data, n_steps=30, config=render_config, render_every=1)
        save_video(frames, run_dir / "smoke_video.mp4", fps=10)
        save_sample_frames(frames, run_dir / "frames", max_samples=3)

        meta = collect_metadata()
        meta["seed"] = SEED
        (run_dir / "sim_smoke_meta.json").write_text(json.dumps(meta, indent=2))

        # Verify layout
        assert (run_dir / "smoke_video.mp4").exists()
        assert (run_dir / "sim_smoke_meta.json").exists()
        assert len(list((run_dir / "frames").glob("*.png"))) == 3
