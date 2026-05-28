"""Tests for asset loading contract (Phase 0.2.5).

Run:
    pytest tests/test_asset_loader.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")


# ---------------------------------------------------------------------------
# A. Directory layout
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
class TestAssetDirectoryLayout:
    """Verify the canonical asset directory structure exists."""

    def test_assets_dir_exists(self) -> None:
        from sim.asset_loader import ASSETS_DIR

        assert ASSETS_DIR.is_dir(), f"Missing: {ASSETS_DIR}"

    def test_scenes_dir_exists(self) -> None:
        from sim.asset_loader import SCENES_DIR

        assert SCENES_DIR.is_dir(), f"Missing: {SCENES_DIR}"

    def test_robots_dir_exists(self) -> None:
        from sim.asset_loader import ROBOTS_DIR

        assert ROBOTS_DIR.is_dir(), f"Missing: {ROBOTS_DIR}"

    def test_test_scene_exists(self) -> None:
        from sim.asset_loader import SCENES_DIR

        assert (SCENES_DIR / "test_scene.xml").exists()

    def test_alex_scene_exists(self) -> None:
        from sim.asset_loader import SCENES_DIR

        assert (SCENES_DIR / "alex_upper_body.xml").exists()

    def test_alex_robot_dir_exists(self) -> None:
        from sim.asset_loader import ROBOTS_DIR

        assert (ROBOTS_DIR / "alex").is_dir()
        assert (ROBOTS_DIR / "alex" / "alex.xml").exists()


# ---------------------------------------------------------------------------
# B. load_scene entrypoint
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
class TestLoadScene:
    """Test the load_scene entrypoint."""

    def test_load_by_name(self) -> None:
        """load_scene("test_scene") loads successfully."""
        from sim.asset_loader import load_scene

        model = load_scene("test_scene")
        assert model.nq > 0

    def test_load_by_name_with_extension(self) -> None:
        """load_scene("test_scene.xml") also works."""
        from sim.asset_loader import load_scene

        model = load_scene("test_scene.xml")
        assert model.nq > 0

    def test_load_nonexistent_raises(self) -> None:
        """load_scene("nonexistent") raises FileNotFoundError."""
        from sim.asset_loader import load_scene

        with pytest.raises(FileNotFoundError):
            load_scene("nonexistent")

    def test_loaded_model_valid(self) -> None:
        """Loaded model has expected properties."""
        from sim.asset_loader import load_scene

        model = load_scene("test_scene")
        assert model.nq > 0
        assert model.nv > 0
        assert model.opt.timestep > 0


# ---------------------------------------------------------------------------
# C. Path resolution
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
class TestResolveScenePath:
    """Test scene path resolution."""

    def test_resolves_to_absolute(self) -> None:
        from sim.asset_loader import resolve_scene_path

        path = resolve_scene_path("test_scene")
        assert path.is_absolute()

    def test_resolves_without_extension(self) -> None:
        from sim.asset_loader import resolve_scene_path

        path = resolve_scene_path("test_scene")
        assert path.name == "test_scene.xml"

    def test_resolves_with_extension(self) -> None:
        from sim.asset_loader import resolve_scene_path

        path = resolve_scene_path("test_scene.xml")
        assert path.name == "test_scene.xml"

    def test_nonexistent_raises(self) -> None:
        from sim.asset_loader import resolve_scene_path

        with pytest.raises(FileNotFoundError):
            resolve_scene_path("no_such_scene")


@pytest.mark.mujoco
class TestResolveRobotPath:
    """Test robot path resolution."""

    def test_nonexistent_robot_raises(self) -> None:
        """Missing robot directory raises FileNotFoundError."""
        from sim.asset_loader import resolve_robot_path

        with pytest.raises(FileNotFoundError):
            resolve_robot_path("nonexistent_robot")

    def test_alex_robot_resolves(self) -> None:
        """resolve_robot_path("alex") returns a valid path."""
        from sim.asset_loader import resolve_robot_path

        path = resolve_robot_path("alex")
        assert path.is_absolute()
        assert path.name == "alex.xml"
        assert path.exists()


# ---------------------------------------------------------------------------
# D. Asset linter
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
class TestAssetLinter:
    """Test the MJCF asset linter."""

    def test_clean_scene_no_issues(self) -> None:
        """test_scene.xml should lint clean (no errors)."""
        from sim.asset_linter import Severity, lint_mjcf
        from sim.asset_loader import SCENES_DIR

        issues = lint_mjcf(SCENES_DIR / "test_scene.xml")
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_absolute_path_detected(self, tmp_path: Path) -> None:
        """MJCF with absolute mesh path produces an ERROR."""
        from sim.asset_linter import Severity, lint_mjcf

        mjcf = tmp_path / "abs.xml"
        mjcf.write_text(
            "<mujoco><asset>"
            '<mesh name="m" file="/absolute/path/mesh.stl"/>'
            "</asset><worldbody/></mujoco>"
        )
        issues = lint_mjcf(mjcf)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert "absolute" in errors[0].message.lower() or "Absolute" in errors[0].message

    def test_missing_mesh_detected(self, tmp_path: Path) -> None:
        """MJCF referencing a nonexistent mesh file produces an ERROR."""
        from sim.asset_linter import Severity, lint_mjcf

        mjcf = tmp_path / "missing.xml"
        mjcf.write_text(
            "<mujoco><asset>"
            '<mesh name="m" file="does_not_exist.stl"/>'
            "</asset><worldbody/></mujoco>"
        )
        issues = lint_mjcf(mjcf)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert "missing" in errors[0].message.lower() or "Missing" in errors[0].message

    def test_suspicious_scale_detected(self, tmp_path: Path) -> None:
        """MJCF with scale="1000 1000 1000" produces a WARNING."""
        from sim.asset_linter import Severity, lint_mjcf

        # Create a dummy mesh so the missing-file check doesn't fire
        (tmp_path / "box.stl").write_bytes(b"dummy")
        mjcf = tmp_path / "scale.xml"
        mjcf.write_text(
            "<mujoco><asset>"
            '<mesh name="big" file="box.stl" scale="1000 1000 1000"/>'
            "</asset><worldbody/></mujoco>"
        )
        issues = lint_mjcf(mjcf)
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert len(warnings) >= 1
        assert "scale" in warnings[0].message.lower() or "suspicious" in warnings[0].message.lower()

    def test_include_missing_detected(self, tmp_path: Path) -> None:
        """MJCF with <include file="missing.xml"/> produces an ERROR."""
        from sim.asset_linter import Severity, lint_mjcf

        mjcf = tmp_path / "inc.xml"
        mjcf.write_text('<mujoco><include file="missing.xml"/><worldbody/></mujoco>')
        issues = lint_mjcf(mjcf)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert "include" in errors[0].message.lower() or "missing" in errors[0].message.lower()

    def test_compiler_meshdir_respected(self, tmp_path: Path) -> None:
        """Linter resolves mesh paths via <compiler meshdir>."""
        from sim.asset_linter import Severity, lint_mjcf

        meshes_dir = tmp_path / "meshes"
        meshes_dir.mkdir()
        (meshes_dir / "cube.stl").write_bytes(b"dummy")

        mjcf = tmp_path / "meshdir.xml"
        mjcf.write_text(
            "<mujoco>"
            '<compiler meshdir="meshes"/>'
            '<asset><mesh name="c" file="cube.stl"/></asset>'
            "<worldbody/></mujoco>"
        )
        issues = lint_mjcf(mjcf)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_nonexistent_file_raises(self) -> None:
        """lint_mjcf raises FileNotFoundError for missing MJCF."""
        from sim.asset_linter import lint_mjcf

        with pytest.raises(FileNotFoundError):
            lint_mjcf(Path("/nonexistent/file.xml"))
