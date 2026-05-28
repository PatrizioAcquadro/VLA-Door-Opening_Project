"""Tests for MuJoCo simulation baseline (Phase 0.2.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip entire module if mujoco is not installed
mujoco = pytest.importorskip("mujoco")

ASSETS_DIR = Path(__file__).resolve().parent.parent / "sim" / "assets"
TEST_SCENE = ASSETS_DIR / "scenes" / "test_scene.xml"


@pytest.mark.mujoco
class TestMujocoImport:
    """Test that MuJoCo is importable and functional."""

    def test_import_version(self) -> None:
        """mujoco package imports and has a version string."""
        assert hasattr(mujoco, "__version__")
        assert isinstance(mujoco.__version__, str)


@pytest.mark.mujoco
class TestSceneLoading:
    """Test MJCF scene loading."""

    def test_load_test_scene(self) -> None:
        """Minimal test scene loads without error."""
        from sim.mujoco_env import load_model

        model = load_model(TEST_SCENE)
        assert model.nq > 0  # Has generalized coordinates
        assert model.nv > 0  # Has generalized velocities

    def test_load_nonexistent_raises(self) -> None:
        """Loading a missing file raises FileNotFoundError."""
        from sim.mujoco_env import load_model

        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path.xml")


@pytest.mark.mujoco
class TestDeterministicStepping:
    """Test simulation determinism."""

    def test_step_produces_state_change(self) -> None:
        """Stepping under gravity changes qpos (box falls)."""
        from sim.mujoco_env import load_model, step_n

        model = load_model(TEST_SCENE)
        data = mujoco.MjData(model)
        initial_qpos = data.qpos.copy()

        step_n(model, data, n_steps=100)

        # Box should have fallen (z position decreased)
        assert not np.array_equal(data.qpos, initial_qpos)

    def test_deterministic_stepping(self) -> None:
        """Multiple runs from same initial state produce identical results."""
        from sim.mujoco_env import check_deterministic

        assert check_deterministic(TEST_SCENE, n_steps=1000, n_trials=3)

    def test_no_nans_after_stepping(self) -> None:
        """No NaN values in state after stepping."""
        from sim.mujoco_env import load_model, step_n

        model = load_model(TEST_SCENE)
        data = mujoco.MjData(model)
        step_n(model, data, n_steps=1000)

        assert not np.any(np.isnan(data.qpos))
        assert not np.any(np.isnan(data.qvel))

    def test_energy_bounded(self) -> None:
        """Total energy does not blow up (no simulation explosion)."""
        from sim.mujoco_env import load_model

        model = load_model(TEST_SCENE)
        # Enable energy computation
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        data = mujoco.MjData(model)

        for _ in range(1000):
            mujoco.mj_step(model, data)

        # For a falling box hitting a plane, energy should remain bounded
        energy = data.energy[0] + data.energy[1]  # potential + kinetic
        assert np.isfinite(energy)


@pytest.mark.mujoco
class TestEnvironmentMetadata:
    """Test metadata collection."""

    def test_collect_metadata_returns_expected_keys(self) -> None:
        """Metadata dict contains all required keys."""
        from sim.env_meta import collect_metadata

        meta = collect_metadata()
        expected_keys = {
            "os",
            "python_version",
            "gpu_driver",
            "mujoco_version",
            "git_commit",
            "git_dirty",
            "deps_hash",
        }
        assert expected_keys.issubset(meta.keys())

    def test_mujoco_version_populated(self) -> None:
        """mujoco_version is a real version string, not 'not installed'."""
        from sim.env_meta import collect_metadata

        meta = collect_metadata()
        assert meta["mujoco_version"] != "not installed"
