#!/usr/bin/env python3
"""Standalone validation for LEGO brick assets (Phase 1.2.1).

Checks:
1. Asset files exist and pass linting
2. MJCF loads in MuJoCo without errors
3. Dimensions and mass are correct
4. Connector metadata is consistent
5. Physics stability (drop test)

Run:
    python scripts/validate_lego_bricks.py

Artifacts are saved to logs/lego_bricks/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from sim.asset_linter import lint_mjcf  # noqa: E402
from sim.lego.brick_generator import generate_brick_mjcf  # noqa: E402
from sim.lego.connector import get_brick_connectors  # noqa: E402
from sim.lego.constants import BRICK_TYPES, TUBE_CAPSULE_COUNT  # noqa: E402
from sim.lego.mass import compute_brick_mass  # noqa: E402

LEGO_BRICKS_DIR = ROOT / "sim" / "assets" / "lego" / "bricks"
LOGS_DIR = ROOT / "logs" / "lego_bricks"

passed = 0
failed = 0


def check(condition: bool, msg: str) -> None:
    global passed, failed
    if condition:
        print(f"  PASS: {msg}")
        passed += 1
    else:
        print(f"  FAIL: {msg}")
        failed += 1


def main() -> None:
    global passed, failed

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Check asset files exist ---
    print("\n=== Asset File Check ===")
    for name in BRICK_TYPES:
        path = LEGO_BRICKS_DIR / f"brick_{name}.xml"
        check(path.exists(), f"brick_{name}.xml exists")

    defaults_path = ROOT / "sim" / "assets" / "lego" / "defaults.xml"
    check(defaults_path.exists(), "defaults.xml exists")

    # --- 2. Lint check ---
    print("\n=== Asset Linting ===")
    for name in BRICK_TYPES:
        path = LEGO_BRICKS_DIR / f"brick_{name}.xml"
        if path.exists():
            issues = lint_mjcf(path)
            errors = [i for i in issues if i.severity.name == "ERROR"]
            check(len(errors) == 0, f"brick_{name}.xml lint clean ({len(errors)} errors)")
            for e in errors:
                print(f"    {e}")

    # --- 3. MJCF loading ---
    print("\n=== MJCF Loading ===")
    for name, brick in BRICK_TYPES.items():
        xml = generate_brick_mjcf(brick)
        try:
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            no_nan = not np.any(np.isnan(data.qpos))
            check(True, f"brick_{name} loads OK")
            check(no_nan, f"brick_{name} no NaN after mj_forward")
        except Exception as e:
            check(False, f"brick_{name} load failed: {e}")
            continue

        # Geom count
        expected_geoms = (
            1 + 1 + 1 + brick.n_studs + brick.n_studs + brick.n_tubes * TUBE_CAPSULE_COUNT
        )
        check(
            model.ngeom == expected_geoms,
            f"brick_{name} geom count: {model.ngeom} (expected {expected_geoms})",
        )

        # Mass
        expected_mass = compute_brick_mass(brick)
        brick_mass = model.body_mass[1]
        mass_err = abs(brick_mass - expected_mass) / expected_mass
        check(
            mass_err < 0.01,
            f"brick_{name} mass: {brick_mass*1000:.2f}g (expected {expected_mass*1000:.2f}g)",
        )

    # --- 4. Connector metadata ---
    print("\n=== Connector Metadata ===")
    for name, brick in BRICK_TYPES.items():
        conn = get_brick_connectors(brick)
        check(conn.n_studs == brick.n_studs, f"brick_{name} stud count: {conn.n_studs}")
        check(conn.n_tubes == brick.n_tubes, f"brick_{name} tube count: {conn.n_tubes}")

        all_ids = [s.id for s in conn.studs] + [t.id for t in conn.tubes]
        check(len(all_ids) == len(set(all_ids)), f"brick_{name} unique IDs")

        pos = conn.stud_positions_array()
        check(pos.shape == (brick.n_studs, 3), f"brick_{name} stud positions shape")

    # --- 5. Physics stability (drop test) ---
    print("\n=== Physics Stability ===")
    for name in BRICK_TYPES:
        xml = generate_brick_mjcf(BRICK_TYPES[name])
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        stable = True
        for _step in range(1000):
            mujoco.mj_step(model, data)
            if np.any(np.isnan(data.qpos)):
                stable = False
                break

        check(stable, f"brick_{name} 1000-step drop test")

    # --- 6. Regeneration consistency ---
    print("\n=== Regeneration Consistency ===")
    for name, brick in BRICK_TYPES.items():
        path = LEGO_BRICKS_DIR / f"brick_{name}.xml"
        if path.exists():
            on_disk = path.read_text()
            regenerated = generate_brick_mjcf(brick)
            # Normalize whitespace for comparison
            check(on_disk.strip() == regenerated.strip(), f"brick_{name} matches on-disk file")

    # --- Summary ---
    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("All checks passed!")


if __name__ == "__main__":
    main()
