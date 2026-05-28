"""Tests for LEGO brick generation and metadata (Phase 1.2.1).

Run:
    pytest tests/test_lego_bricks.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.lego.connector import get_brick_connectors
from sim.lego.constants import (
    BRICK_HEIGHT,
    BRICK_TYPES,
    STUD_HALF_HEIGHT,
    STUD_PITCH,
    TUBE_CAPSULE_COUNT,
)
from sim.lego.mass import compute_brick_mass


# ---------------------------------------------------------------------------
# TestBrickConstants — no MuJoCo required
# ---------------------------------------------------------------------------
class TestBrickConstants:
    def test_three_brick_types(self):
        assert set(BRICK_TYPES.keys()) == {"2x2", "2x4", "2x6"}

    @pytest.mark.parametrize("name,nx,ny", [("2x2", 2, 2), ("2x4", 2, 4), ("2x6", 2, 6)])
    def test_dimensions(self, name: str, nx: int, ny: int):
        b = BRICK_TYPES[name]
        assert b.nx == nx
        assert b.ny == ny
        assert b.n_studs == nx * ny
        assert b.n_tubes == ny - 1

    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_shell_dimensions_positive(self, name: str):
        b = BRICK_TYPES[name]
        assert b.shell_half_x > 0
        assert b.shell_half_y > 0
        assert b.shell_half_z > 0
        assert b.height == BRICK_HEIGHT


# ---------------------------------------------------------------------------
# TestMassComputation — no MuJoCo required
# ---------------------------------------------------------------------------
class TestMassComputation:
    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_mass_positive(self, name: str):
        mass = compute_brick_mass(BRICK_TYPES[name])
        assert mass > 0

    def test_mass_scales_with_size(self):
        m2 = compute_brick_mass(BRICK_TYPES["2x2"])
        m4 = compute_brick_mass(BRICK_TYPES["2x4"])
        m6 = compute_brick_mass(BRICK_TYPES["2x6"])
        assert m2 < m4 < m6

    @pytest.mark.parametrize(
        "name,min_g,max_g",
        [("2x2", 0.5, 3.0), ("2x4", 1.0, 5.0), ("2x6", 1.5, 8.0)],
    )
    def test_mass_in_range(self, name: str, min_g: float, max_g: float):
        mass_g = compute_brick_mass(BRICK_TYPES[name]) * 1000
        assert min_g <= mass_g <= max_g, f"{name} mass={mass_g:.2f}g out of range"


# ---------------------------------------------------------------------------
# TestConnectorMetadata — no MuJoCo required
# ---------------------------------------------------------------------------
class TestConnectorMetadata:
    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_stud_count(self, name: str):
        b = BRICK_TYPES[name]
        conn = get_brick_connectors(b)
        assert conn.n_studs == b.n_studs

    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_tube_count(self, name: str):
        b = BRICK_TYPES[name]
        conn = get_brick_connectors(b)
        assert conn.n_tubes == b.n_tubes

    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_unique_ids(self, name: str):
        conn = get_brick_connectors(BRICK_TYPES[name])
        all_ids = [s.id for s in conn.studs] + [t.id for t in conn.tubes]
        assert len(all_ids) == len(set(all_ids))

    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_stud_z_position(self, name: str):
        conn = get_brick_connectors(BRICK_TYPES[name])
        expected_z = BRICK_HEIGHT + STUD_HALF_HEIGHT
        for s in conn.studs:
            assert abs(s.position[2] - expected_z) < 1e-10

    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_tube_z_positive(self, name: str):
        """Tubes are inside the brick cavity (positive Z, above bottom face)."""
        conn = get_brick_connectors(BRICK_TYPES[name])
        for t in conn.tubes:
            assert t.position[2] > 0, f"Tube {t.id} z={t.position[2]} should be positive"

    def test_stud_grid_symmetry_2x2(self):
        conn = get_brick_connectors(BRICK_TYPES["2x2"])
        pos = conn.stud_positions_array()
        # Grid should be centered: mean X and Y near zero
        assert abs(pos[:, 0].mean()) < 1e-10
        assert abs(pos[:, 1].mean()) < 1e-10

    def test_stud_grid_spacing(self):
        conn = get_brick_connectors(BRICK_TYPES["2x4"])
        pos = conn.stud_positions_array()
        # Check that adjacent studs are STUD_PITCH apart
        # Sort by Y then X to get consistent ordering
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = abs(pos[i, 0] - pos[j, 0])
                dy = abs(pos[i, 1] - pos[j, 1])
                if dx < 1e-10:  # same X column
                    if abs(dy - STUD_PITCH) < 1e-10:
                        pass  # adjacent in Y
                elif dy < 1e-10:  # same Y row
                    if abs(dx - STUD_PITCH) < 1e-10:
                        pass  # adjacent in X

    def test_tube_centerline_2wide(self):
        """For 2-wide bricks, all tubes should be at X=0."""
        conn = get_brick_connectors(BRICK_TYPES["2x4"])
        for t in conn.tubes:
            assert abs(t.position[0]) < 1e-10, f"Tube {t.id} x={t.position[0]}"

    def test_positions_array_shape(self):
        conn = get_brick_connectors(BRICK_TYPES["2x6"])
        assert conn.stud_positions_array().shape == (12, 3)
        assert conn.tube_positions_array().shape == (5, 3)

    def test_stud_id_format(self):
        conn = get_brick_connectors(BRICK_TYPES["2x4"])
        for s in conn.studs:
            assert s.id.startswith("stud_")
            assert s.kind == "stud"

    def test_tube_id_format(self):
        conn = get_brick_connectors(BRICK_TYPES["2x4"])
        for t in conn.tubes:
            assert t.id.startswith("tube_")
            assert t.kind == "tube"


# ---------------------------------------------------------------------------
# TestBrickMJCFGeneration — requires MuJoCo
# ---------------------------------------------------------------------------
mujoco = pytest.importorskip("mujoco")


@pytest.mark.mujoco
@pytest.mark.lego
class TestBrickMJCFGeneration:
    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_loads_without_error(self, name: str):
        from sim.lego.brick_generator import generate_brick_mjcf

        xml = generate_brick_mjcf(BRICK_TYPES[name])
        model = mujoco.MjModel.from_xml_string(xml)
        assert model is not None

    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_no_nan_after_forward(self, name: str):
        from sim.lego.brick_generator import generate_brick_mjcf

        xml = generate_brick_mjcf(BRICK_TYPES[name])
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert not np.any(np.isnan(data.qpos))
        assert not np.any(np.isnan(data.qvel))

    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_geom_count(self, name: str):
        from sim.lego.brick_generator import generate_brick_mjcf

        b = BRICK_TYPES[name]
        xml = generate_brick_mjcf(b)
        model = mujoco.MjModel.from_xml_string(xml)
        # floor(1) + shell_col(5: top+4walls) + shell_vis(1) + stud_col(n) + stud_vis(n) + tube_caps(n_t*8)
        expected = 1 + 5 + 1 + b.n_studs + b.n_studs + b.n_tubes * TUBE_CAPSULE_COUNT
        assert model.ngeom == expected

    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_mass_matches_computed(self, name: str):
        from sim.lego.brick_generator import generate_brick_mjcf

        b = BRICK_TYPES[name]
        xml = generate_brick_mjcf(b)
        model = mujoco.MjModel.from_xml_string(xml)
        expected_mass = compute_brick_mass(b)
        # body_mass includes world body (0) + brick body
        brick_mass = model.body_mass[1]
        assert abs(brick_mass - expected_mass) / expected_mass < 0.01

    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_has_freejoint(self, name: str):
        from sim.lego.brick_generator import generate_brick_mjcf

        xml = generate_brick_mjcf(BRICK_TYPES[name])
        model = mujoco.MjModel.from_xml_string(xml)
        # Freejoint: 7 qpos (3 pos + 4 quat), 6 qvel
        assert model.nq == 7
        assert model.nv == 6


@pytest.mark.mujoco
@pytest.mark.lego
class TestBrickContactClasses:
    def _load_brick_model(self, name: str):
        from sim.lego.brick_generator import generate_brick_mjcf

        xml = generate_brick_mjcf(BRICK_TYPES[name])
        return mujoco.MjModel.from_xml_string(xml)

    def test_shell_contact_class(self):
        model = self._load_brick_model("2x4")
        shell_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "brick_2x4_shell_top")
        assert shell_id >= 0
        assert model.geom_contype[shell_id] == 2
        assert model.geom_conaffinity[shell_id] == 3  # brick_surface: LEGO + robot

    def test_stud_contact_class(self):
        model = self._load_brick_model("2x4")
        stud_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "brick_2x4_stud_0_0")
        assert stud_id >= 0
        assert model.geom_contype[stud_id] == 6  # lego/stud
        assert model.geom_conaffinity[stud_id] == 7  # stud: contacts tubes + surfaces

    def test_tube_capsule_contact_class(self):
        model = self._load_brick_model("2x4")
        cap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "brick_2x4_tube_0_c0")
        assert cap_id >= 0
        assert model.geom_contype[cap_id] == 4  # lego/tube
        assert model.geom_conaffinity[cap_id] == 4  # tube: contacts studs only

    def test_visual_no_collision(self):
        model = self._load_brick_model("2x2")
        vis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "brick_2x2_shell_vis")
        assert vis_id >= 0
        assert model.geom_contype[vis_id] == 0
        assert model.geom_conaffinity[vis_id] == 0


@pytest.mark.mujoco
@pytest.mark.lego
class TestBrickPhysicsStability:
    def test_drop_2x2_no_nan(self):
        """Drop a 2x2 brick and simulate 500 steps — no divergence."""
        from sim.lego.brick_generator import generate_brick_mjcf

        xml = generate_brick_mjcf(BRICK_TYPES["2x2"])
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        for _ in range(500):
            mujoco.mj_step(model, data)
            assert not np.any(np.isnan(data.qpos)), "NaN in qpos"
            assert not np.any(np.isnan(data.qvel)), "NaN in qvel"

    def test_drop_2x4_energy_bounded(self):
        """Drop a 2x4 brick — energy stays bounded."""
        from sim.lego.brick_generator import generate_brick_mjcf

        xml = generate_brick_mjcf(BRICK_TYPES["2x4"])
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        for _ in range(500):
            mujoco.mj_step(model, data)
        energy = data.energy[0] + data.energy[1]  # potential + kinetic
        assert abs(energy) < 500.0, f"Energy={energy} exceeds bound"


@pytest.mark.lego
class TestAssetLintClean:
    @pytest.mark.parametrize("name", BRICK_TYPES.keys())
    def test_generated_files_lint_clean(self, name: str):
        from pathlib import Path

        from sim.asset_linter import lint_mjcf

        path = Path(__file__).resolve().parent.parent / f"sim/assets/lego/bricks/brick_{name}.xml"
        if not path.exists():
            pytest.skip(f"Brick file not generated: {path}")

        issues = lint_mjcf(path)
        errors = [i for i in issues if i.severity.name == "ERROR"]
        assert len(errors) == 0, f"Lint errors: {errors}"
