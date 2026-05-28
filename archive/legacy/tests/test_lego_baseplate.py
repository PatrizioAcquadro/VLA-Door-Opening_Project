"""Tests for LEGO baseplate generation, workspace, and contact physics (Phase 1.2.3).

Run:
    pytest tests/test_lego_baseplate.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.lego.connector import get_baseplate_connectors
from sim.lego.constants import (
    BASEPLATE_THICKNESS,
    BASEPLATE_TYPES,
    STUD_HALF_HEIGHT,
    STUD_PITCH,
)
from sim.lego.mass import compute_baseplate_mass


# ---------------------------------------------------------------------------
# TestBaseplateConstants — no MuJoCo required
# ---------------------------------------------------------------------------
class TestBaseplateConstants:
    def test_baseplate_type_exists(self):
        assert "8x8" in BASEPLATE_TYPES

    def test_dimensions_8x8(self):
        bp = BASEPLATE_TYPES["8x8"]
        assert bp.nx_studs == 8
        assert bp.ny_studs == 8
        assert bp.n_studs == 64

    def test_half_dimensions_positive(self):
        bp = BASEPLATE_TYPES["8x8"]
        assert bp.half_x > 0
        assert bp.half_y > 0
        assert bp.half_z > 0

    def test_thickness(self):
        bp = BASEPLATE_TYPES["8x8"]
        assert bp.thickness == BASEPLATE_THICKNESS
        assert abs(bp.thickness - 0.0032) < 1e-8

    def test_square_baseplate(self):
        bp = BASEPLATE_TYPES["8x8"]
        assert abs(bp.half_x - bp.half_y) < 1e-10


# ---------------------------------------------------------------------------
# TestBaseplateMass — no MuJoCo required
# ---------------------------------------------------------------------------
class TestBaseplateMass:
    def test_mass_positive(self):
        mass = compute_baseplate_mass(BASEPLATE_TYPES["8x8"])
        assert mass > 0

    def test_mass_in_range(self):
        mass_g = compute_baseplate_mass(BASEPLATE_TYPES["8x8"]) * 1000
        # 8x8 baseplate: ~64mm x 64mm x 3.2mm ABS + 64 studs
        assert 5.0 <= mass_g <= 30.0, f"8x8 mass={mass_g:.2f}g out of expected range"


# ---------------------------------------------------------------------------
# TestBaseplateConnectors — no MuJoCo required
# ---------------------------------------------------------------------------
class TestBaseplateConnectors:
    def test_stud_count(self):
        bp = BASEPLATE_TYPES["8x8"]
        conn = get_baseplate_connectors(bp)
        assert conn.n_studs == 64

    def test_unique_ids(self):
        bp = BASEPLATE_TYPES["8x8"]
        conn = get_baseplate_connectors(bp)
        ids = [s.id for s in conn.studs]
        assert len(ids) == len(set(ids))

    def test_stud_z_position(self):
        bp = BASEPLATE_TYPES["8x8"]
        conn = get_baseplate_connectors(bp)
        expected_z = bp.thickness + STUD_HALF_HEIGHT
        for s in conn.studs:
            assert abs(s.position[2] - expected_z) < 1e-10

    def test_stud_grid_centered(self):
        bp = BASEPLATE_TYPES["8x8"]
        conn = get_baseplate_connectors(bp)
        positions = conn.stud_positions_array()
        mean_xy = positions[:, :2].mean(axis=0)
        assert abs(mean_xy[0]) < 1e-10
        assert abs(mean_xy[1]) < 1e-10

    def test_stud_grid_spacing(self):
        bp = BASEPLATE_TYPES["8x8"]
        conn = get_baseplate_connectors(bp)
        positions = conn.stud_positions_array()
        # Check that adjacent studs (same row, adjacent columns) have STUD_PITCH spacing
        # First row studs: ix=0, iy=0..7
        row_0 = positions[:8]  # first 8 studs (ix=0, iy=0..7)
        for i in range(7):
            dy = abs(row_0[i + 1, 1] - row_0[i, 1])
            assert abs(dy - STUD_PITCH) < 1e-10

    def test_positions_array_shape(self):
        bp = BASEPLATE_TYPES["8x8"]
        conn = get_baseplate_connectors(bp)
        arr = conn.stud_positions_array()
        assert arr.shape == (64, 3)

    def test_all_studs_kind(self):
        bp = BASEPLATE_TYPES["8x8"]
        conn = get_baseplate_connectors(bp)
        for s in conn.studs:
            assert s.kind == "stud"


# ---------------------------------------------------------------------------
# TestBaseplateGeneration — requires MuJoCo
# ---------------------------------------------------------------------------
@pytest.mark.mujoco
@pytest.mark.lego
class TestBaseplateGeneration:
    def test_loads_without_error(self):
        import mujoco

        from sim.lego.baseplate_generator import generate_baseplate_mjcf

        bp = BASEPLATE_TYPES["8x8"]
        xml = generate_baseplate_mjcf(bp)
        model = mujoco.MjModel.from_xml_string(xml)
        assert model is not None

    def test_no_nan_after_forward(self):
        import mujoco

        from sim.lego.baseplate_generator import generate_baseplate_mjcf

        bp = BASEPLATE_TYPES["8x8"]
        xml = generate_baseplate_mjcf(bp)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert not np.any(np.isnan(data.qpos))

    def test_geom_count(self):
        import mujoco

        from sim.lego.baseplate_generator import generate_baseplate_mjcf

        bp = BASEPLATE_TYPES["8x8"]
        xml = generate_baseplate_mjcf(bp)
        model = mujoco.MjModel.from_xml_string(xml)
        # 1 floor + 1 surface_col + 1 surface_vis + 64 stud_col + 64 stud_vis = 131
        assert model.ngeom == 131

    def test_no_freejoint(self):
        import mujoco

        from sim.lego.baseplate_generator import generate_baseplate_mjcf

        bp = BASEPLATE_TYPES["8x8"]
        xml = generate_baseplate_mjcf(bp)
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.nq == 0
        assert model.nv == 0

    def test_stud_contact_class(self):
        import mujoco

        from sim.lego.baseplate_generator import generate_baseplate_mjcf

        bp = BASEPLATE_TYPES["8x8"]
        xml = generate_baseplate_mjcf(bp)
        model = mujoco.MjModel.from_xml_string(xml)

        # Find a stud geom by name
        stud_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "baseplate_8x8_stud_0_0")
        assert stud_id >= 0
        assert model.geom_contype[stud_id] == 6
        assert model.geom_conaffinity[stud_id] == 7

    def test_surface_contact_class(self):
        import mujoco

        from sim.lego.baseplate_generator import generate_baseplate_mjcf

        bp = BASEPLATE_TYPES["8x8"]
        xml = generate_baseplate_mjcf(bp)
        model = mujoco.MjModel.from_xml_string(xml)

        surface_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "baseplate_8x8_surface")
        assert surface_id >= 0
        assert model.geom_contype[surface_id] == 2
        assert model.geom_conaffinity[surface_id] == 3


# ---------------------------------------------------------------------------
# TestBaseplateContactPhysics — requires MuJoCo
# ---------------------------------------------------------------------------
@pytest.mark.mujoco
@pytest.mark.lego
class TestBaseplateContactPhysics:
    def test_brick_resting_on_baseplate(self):
        """Drop a 2x2 brick on baseplate, verify it lands and stabilizes."""
        import mujoco

        from sim.lego.constants import BRICK_TYPES
        from sim.lego.contact_scene import load_baseplate_insertion_scene

        bp = BASEPLATE_TYPES["8x8"]
        brick = BRICK_TYPES["2x2"]
        model, data = load_baseplate_insertion_scene(bp, brick, approach_gap=0.01)

        # Let brick fall under gravity for 2 seconds
        for _ in range(1000):
            mujoco.mj_step(model, data)

        assert not np.any(np.isnan(data.qpos))

        # Brick should have come to rest somewhere above baseplate
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "top_2x2")
        brick_z = data.xpos[body_id][2]
        assert brick_z > -0.01, "Brick fell through floor"

    def test_brick_insertion_on_baseplate(self):
        """Aligned insertion of 2x2 onto baseplate, verify engagement."""
        from sim.lego.constants import BRICK_TYPES
        from sim.lego.contact_scene import load_baseplate_insertion_scene
        from sim.lego.contact_utils import run_insertion

        bp = BASEPLATE_TYPES["8x8"]
        brick = BRICK_TYPES["2x2"]
        model, data = load_baseplate_insertion_scene(bp, brick)
        result = run_insertion(
            model,
            data,
            base_brick_name=brick.name,
            base_height=0.0,
            base_surface_height=bp.thickness,
        )
        assert result.success, "Insertion onto baseplate failed"
        assert result.max_penetration_m < 0.002

    def test_brick_retention_on_baseplate(self):
        """After insertion, apply pull-off force, verify minimum retention."""
        from sim.lego.constants import BRICK_TYPES
        from sim.lego.contact_scene import load_baseplate_insertion_scene
        from sim.lego.contact_utils import apply_force_ramp, run_insertion

        bp = BASEPLATE_TYPES["8x8"]
        brick = BRICK_TYPES["2x2"]
        model, data = load_baseplate_insertion_scene(bp, brick)
        result = run_insertion(
            model,
            data,
            base_brick_name=brick.name,
            base_height=0.0,
            base_surface_height=bp.thickness,
        )
        if not result.success:
            pytest.skip("Insertion failed, cannot test retention")

        pull_off_force = apply_force_ramp(
            model,
            data,
            body_name="top_2x2",
            direction=np.array([0, 0, 1]),
            force_rate=0.5,
            max_force=5.0,
        )
        # Retention should be at least some non-zero amount
        assert pull_off_force > 0.01, f"Pull-off too low: {pull_off_force:.4f} N"

    def test_brick_static_hold_on_baseplate(self):
        """After insertion, verify no excessive drift over 2s hold."""
        from sim.lego.constants import BRICK_TYPES
        from sim.lego.contact_scene import load_baseplate_insertion_scene
        from sim.lego.contact_utils import measure_position_drift, run_insertion

        bp = BASEPLATE_TYPES["8x8"]
        brick = BRICK_TYPES["2x2"]
        model, data = load_baseplate_insertion_scene(bp, brick)
        result = run_insertion(
            model,
            data,
            base_brick_name=brick.name,
            base_height=0.0,
            base_surface_height=bp.thickness,
        )
        if not result.success:
            pytest.skip("Insertion failed, cannot test hold")

        drift = measure_position_drift(model, data, "top_2x2", duration_s=2.0)
        assert drift < 0.005, f"Drift too high: {drift * 1000:.2f} mm"

    def test_numerical_stability(self):
        """Run 1000 steps, verify no NaN."""
        import mujoco

        from sim.lego.constants import BRICK_TYPES
        from sim.lego.contact_scene import load_baseplate_insertion_scene

        bp = BASEPLATE_TYPES["8x8"]
        brick = BRICK_TYPES["2x2"]
        model, data = load_baseplate_insertion_scene(bp, brick)

        for _ in range(1000):
            mujoco.mj_step(model, data)
            assert not np.any(np.isnan(data.qpos)), "NaN detected in simulation"


# ---------------------------------------------------------------------------
# TestWorkspaceScene — requires MuJoCo
# ---------------------------------------------------------------------------
@pytest.mark.mujoco
@pytest.mark.lego
class TestWorkspaceScene:
    def test_workspace_scene_generates(self):
        """Workspace scene XML generates without error."""
        from sim.lego.contact_scene import generate_workspace_scene

        bp = BASEPLATE_TYPES["8x8"]
        xml = generate_workspace_scene(bp)
        assert "<mujoco" in xml
        assert "baseplate_8x8" in xml

    def test_workspace_scene_loads(self):
        """Workspace scene loads in MuJoCo (requires Alex robot assets)."""
        import mujoco

        from sim.lego.contact_scene import generate_workspace_scene

        bp = BASEPLATE_TYPES["8x8"]
        xml = generate_workspace_scene(bp)

        # Write to temp file in scenes dir for include resolution
        from pathlib import Path

        scenes_dir = Path(__file__).resolve().parent.parent / "sim" / "assets" / "scenes"
        tmp_path = scenes_dir / "_test_workspace_tmp.xml"
        try:
            tmp_path.write_text(xml)
            model = mujoco.MjModel.from_xml_path(str(tmp_path))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            assert not np.any(np.isnan(data.qpos))

            # Verify table body exists
            table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
            assert table_id >= 0

            # Verify baseplate body exists
            bp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "baseplate_8x8")
            assert bp_id >= 0

            # Verify robot actuators present (17 for Alex)
            assert model.nu == 17
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TestBaseplateAssets — requires generated assets
# ---------------------------------------------------------------------------
@pytest.mark.lego
class TestBaseplateAssets:
    def test_generated_file_exists(self):
        from pathlib import Path

        path = (
            Path(__file__).resolve().parent.parent
            / "sim"
            / "assets"
            / "lego"
            / "baseplates"
            / "baseplate_8x8.xml"
        )
        assert path.exists(), f"Baseplate asset not found at {path}"

    def test_asset_loader_resolves(self):
        from sim.asset_loader import resolve_lego_baseplate_path

        path = resolve_lego_baseplate_path("8x8")
        assert path.exists()
        assert path.name == "baseplate_8x8.xml"
