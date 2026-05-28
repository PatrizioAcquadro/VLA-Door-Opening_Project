"""Tests for sim.viewer (Phase 0.2.2).

All tests run headlessly — none open a window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

ASSETS_DIR = Path(__file__).resolve().parent.parent / "sim" / "assets"
TEST_SCENE = ASSETS_DIR / "scenes" / "test_scene.xml"


@pytest.mark.mujoco
class TestViewerImport:
    """Verify that sim.viewer imports without requiring a display."""

    def test_import_viewer_module(self) -> None:
        import sim.viewer  # noqa: F401

    def test_viewer_config_dataclass(self) -> None:
        from sim.viewer import ViewerConfig

        cfg = ViewerConfig(scene_path=Path("test.xml"))
        assert cfg.passive is False
        assert cfg.show_contacts is False
        assert cfg.show_joints is False
        assert cfg.camera_name is None
        assert cfg.duration == 0.0


@pytest.mark.mujoco
class TestParseArgs:
    """Test CLI argument parsing."""

    def test_minimal_args(self) -> None:
        from sim.viewer import parse_args

        cfg = parse_args(["scene.xml"])
        assert cfg.scene_path == Path("scene.xml")
        assert cfg.passive is False
        assert cfg.show_contacts is False

    def test_all_flags(self) -> None:
        from sim.viewer import parse_args

        cfg = parse_args(
            [
                "robot.xml",
                "--passive",
                "--show-contacts",
                "--show-joints",
                "--camera",
                "overhead",
                "--duration",
                "5.0",
            ]
        )
        assert cfg.passive is True
        assert cfg.show_contacts is True
        assert cfg.show_joints is True
        assert cfg.camera_name == "overhead"
        assert cfg.duration == 5.0

    def test_short_flags(self) -> None:
        from sim.viewer import parse_args

        cfg = parse_args(["scene.xml", "-p", "-c", "-j", "-d", "3.0"])
        assert cfg.passive is True
        assert cfg.show_contacts is True
        assert cfg.show_joints is True
        assert cfg.duration == 3.0


@pytest.mark.mujoco
class TestPreflightChecks:
    """Test preflight checks run headlessly on a loaded model."""

    def test_preflight_returns_expected_checks(self) -> None:
        from sim.mujoco_env import load_model
        from sim.viewer import run_preflight_checks

        model = load_model(TEST_SCENE)
        results = run_preflight_checks(model)
        assert len(results) >= 3
        names = [r.name for r in results]
        assert "gravity" in names
        assert "ground_plane" in names
        assert "geom_summary" in names

    def test_preflight_all_pass_on_test_scene(self) -> None:
        from sim.mujoco_env import load_model
        from sim.viewer import run_preflight_checks

        model = load_model(TEST_SCENE)
        results = run_preflight_checks(model)
        for r in results:
            assert r.passed, f"Check '{r.name}' failed: {r.detail}"

    def test_preflight_report_prints(self, capsys) -> None:
        from sim.mujoco_env import load_model
        from sim.viewer import print_preflight_report, run_preflight_checks

        model = load_model(TEST_SCENE)
        results = run_preflight_checks(model)
        all_ok = print_preflight_report(results)
        assert all_ok is True
        captured = capsys.readouterr()
        assert "Preflight Checks" in captured.out
        assert "[OK]" in captured.out


@pytest.mark.mujoco
class TestVisOptions:
    """Test visualization flag configuration."""

    def test_contacts_flags_set(self) -> None:
        from sim.mujoco_env import load_model
        from sim.viewer import configure_vis_options

        model = load_model(TEST_SCENE)
        opt = mujoco.MjvOption()
        configure_vis_options(model, opt, show_contacts=True)
        assert opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] == 1
        assert opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] == 1

    def test_joints_flag_set(self) -> None:
        from sim.mujoco_env import load_model
        from sim.viewer import configure_vis_options

        model = load_model(TEST_SCENE)
        opt = mujoco.MjvOption()
        configure_vis_options(model, opt, show_joints=True)
        assert opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] == 1

    def test_no_flags_by_default(self) -> None:
        from sim.mujoco_env import load_model
        from sim.viewer import configure_vis_options

        model = load_model(TEST_SCENE)
        opt = mujoco.MjvOption()
        original_contact = opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]
        original_joint = opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]
        configure_vis_options(model, opt)
        assert opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] == original_contact
        assert opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] == original_joint

    def test_returns_same_opt_object(self) -> None:
        from sim.mujoco_env import load_model
        from sim.viewer import configure_vis_options

        model = load_model(TEST_SCENE)
        opt = mujoco.MjvOption()
        returned = configure_vis_options(model, opt, show_contacts=True)
        assert returned is opt
