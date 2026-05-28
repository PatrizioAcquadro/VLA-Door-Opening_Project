#!/usr/bin/env python3
"""Standalone validation for Phase 1.2.6: MVP-3 Task (multi-step assembly).

Checks task specification, scripted assembly execution, success detection,
evaluation metrics, and deterministic replay.

Artifacts are written to logs/lego_task/.

Usage::

    python scripts/validate_lego_task.py
    python scripts/validate_lego_task.py --n-stress 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _record(results: list[tuple[str, bool, str]], name: str, passed: bool, detail: str) -> None:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {name}: {detail}")
    results.append((name, passed, detail))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1.2.6: MVP-3 Task")
    parser.add_argument(
        "--n-stress",
        type=int,
        default=10,
        help="Number of assemblies for stress test (default: 10)",
    )
    args = parser.parse_args()

    out_dir = _ROOT / "logs" / "lego_task"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts -> {out_dir}/")

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("[SKIP] mujoco not available — cannot run validation")
        return 0

    from sim.lego.constants import BASEPLATE_THICKNESS, BASEPLATE_TYPES, BRICK_TYPES
    from sim.lego.episode_manager import (
        LEVEL_MULTI_STEP,
        LEVEL_SINGLE_CONNECTION,
        EpisodeManager,
    )
    from sim.lego.scripted_assembly import ScriptedAssembler
    from sim.lego.task import (
        _brick_footprints_overlap,
        check_placement,
        compute_target_position,
        generate_assembly_goal,
    )

    results: list[tuple[str, bool, str]] = []

    # Common setup
    bp_type = BASEPLATE_TYPES["8x8"]
    table_pos = (0.45, 0.0, 0.75)
    table_half_z = 0.02
    bp_world_pos = (table_pos[0], table_pos[1], table_pos[2] + table_half_z)

    # -----------------------------------------------------------------------
    print("\n--- 1. Goal Generation ---")

    from sim.lego.episode_manager import EpisodeInfo, SpawnPose

    # Use small bricks (2x2) so they all fit on 8x8 baseplate without overlap
    info = EpisodeInfo(
        seed=42,
        level=3,
        brick_types=["2x2", "2x2", "2x2"],
        spawn_poses=[SpawnPose((0.45, 0.0, 0.85), (1, 0, 0, 0))] * 3,
        settle_steps=100,
        settle_success=True,
    )

    goal = generate_assembly_goal(info, bp_type, bp_world_pos, seed=42)
    _record(results, "Goal generated", len(goal.targets) == 3, f"{len(goal.targets)} targets")

    # Check no overlapping footprints
    overlap_found = False
    for i in range(len(goal.targets)):
        for j in range(i + 1, len(goal.targets)):
            ti, tj = goal.targets[i], goal.targets[j]
            if not ti.base_body_name.startswith("baseplate_"):
                continue
            if not tj.base_body_name.startswith("baseplate_"):
                continue
            bi, bj = BRICK_TYPES[ti.brick_type], BRICK_TYPES[tj.brick_type]
            if _brick_footprints_overlap(
                (ti.target_position[0], ti.target_position[1]),
                bi,
                (tj.target_position[0], tj.target_position[1]),
                bj,
            ):
                overlap_found = True
    _record(results, "No footprint overlaps", not overlap_found, "AABB check on all pairs")

    # -----------------------------------------------------------------------
    print("\n--- 2. Target Position Computation ---")

    pos = compute_target_position(bp_world_pos, bp_type, BRICK_TYPES["2x2"], 3, 3)
    z_ok = math.isclose(pos[2], bp_world_pos[2] + BASEPLATE_THICKNESS, abs_tol=1e-6)
    _record(results, "Target Z correct", z_ok, f"Z={pos[2]:.6f} m")

    # Target should be within baseplate footprint
    bp_half = bp_type.half_x
    xy_in_bounds = (
        abs(pos[0] - bp_world_pos[0]) <= bp_half + 0.001
        and abs(pos[1] - bp_world_pos[1]) <= bp_half + 0.001
    )
    _record(results, "Target XY in bounds", xy_in_bounds, f"pos=({pos[0]:.4f}, {pos[1]:.4f})")

    # -----------------------------------------------------------------------
    print("\n--- 3. Single Brick Baseplate Placement ---")

    em = EpisodeManager(brick_slots=["2x2"], settle_max_steps=300, settle_check_interval=25)
    ep_info = em.reset(seed=42, level=LEVEL_SINGLE_CONNECTION)

    goal_1 = generate_assembly_goal(ep_info, bp_type, bp_world_pos, seed=42)
    assembler = ScriptedAssembler(em.model, em.data)

    t0 = time.perf_counter()
    result_1 = assembler.execute_placement(goal_1.targets[0])
    dt_1 = time.perf_counter() - t0

    _record(
        results,
        "Single brick engaged",
        result_1.z_engaged,
        f"engaged={result_1.z_engaged}, xy_err={result_1.position_error_m:.4f} m, "
        f"steps={result_1.insertion_steps}, time={dt_1:.2f}s",
    )

    # -----------------------------------------------------------------------
    print("\n--- 4. Multi-Brick Baseplate Placement ---")

    em2 = EpisodeManager(
        brick_slots=["2x2", "2x4", "2x2"],
        settle_max_steps=300,
        settle_check_interval=25,
    )
    ep_info2 = em2.reset(seed=10, level=LEVEL_MULTI_STEP, n_active=3)
    goal_2 = generate_assembly_goal(ep_info2, bp_type, bp_world_pos, seed=10)
    assembler2 = ScriptedAssembler(em2.model, em2.data)

    t0 = time.perf_counter()
    result_2 = assembler2.execute_assembly(goal_2, hold_duration_s=0.5)
    dt_2 = time.perf_counter() - t0

    engaged_count = sum(1 for p in result_2.placements if p.z_engaged)
    _record(
        results,
        "Multi-brick placement",
        engaged_count >= 2,
        f"{engaged_count}/{result_2.n_total} engaged, "
        f"stable={result_2.structure_stable}, time={dt_2:.2f}s",
    )

    # -----------------------------------------------------------------------
    print("\n--- 5. Brick-on-Brick Stacking ---")

    em3 = EpisodeManager(
        brick_slots=["2x2", "2x2"],
        settle_max_steps=300,
        settle_check_interval=25,
    )
    ep_info3 = em3.reset(seed=42, level=LEVEL_MULTI_STEP, n_active=2)
    goal_3 = generate_assembly_goal(ep_info3, bp_type, bp_world_pos, seed=42, stacking=True)

    is_stacking = goal_3.targets[1].base_body_name.startswith("brick_")
    _record(
        results, "Stacking goal structure", is_stacking, f"base={goal_3.targets[1].base_body_name}"
    )

    assembler3 = ScriptedAssembler(em3.model, em3.data)
    result_3 = assembler3.execute_assembly(goal_3, hold_duration_s=0.5)

    first_engaged = result_3.placements[0].z_engaged
    _record(
        results,
        "Stacking base engaged",
        first_engaged,
        f"base_engaged={first_engaged}, " f"top_engaged={result_3.placements[1].z_engaged}",
    )

    # -----------------------------------------------------------------------
    print("\n--- 6. Success Detection Accuracy ---")

    em4 = EpisodeManager(brick_slots=["2x2"], settle_max_steps=300, settle_check_interval=25)
    ep_info4 = em4.reset(seed=42)
    goal_4 = generate_assembly_goal(ep_info4, bp_type, bp_world_pos, seed=42)
    target = goal_4.targets[0]

    bt = BRICK_TYPES[target.brick_type]
    jname = f"brick_{target.slot_index}_{bt.name}_joint"
    jnt_id = mujoco.mj_name2id(em4.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    addr = int(em4.model.jnt_qposadr[jnt_id])

    # Place exactly at target
    em4.data.qpos[addr : addr + 3] = target.target_position
    em4.data.qpos[addr + 3 : addr + 7] = target.target_quaternion
    mujoco.mj_forward(em4.model, em4.data)
    ok, err = check_placement(em4.model, em4.data, target)
    _record(results, "Correct placement detected", ok, f"xy_err={err:.6f} m")

    # -----------------------------------------------------------------------
    print("\n--- 7. Failure Detection ---")

    # Place 10mm off
    em4.data.qpos[addr] = target.target_position[0] + 0.01
    mujoco.mj_forward(em4.model, em4.data)
    ok_bad, err_bad = check_placement(em4.model, em4.data, target)
    _record(
        results,
        "Misaligned detected as failure",
        not ok_bad,
        f"success={ok_bad}, xy_err={err_bad:.4f} m",
    )

    # -----------------------------------------------------------------------
    print("\n--- 8. Stability Hold ---")

    em5 = EpisodeManager(brick_slots=["2x2"], settle_max_steps=300, settle_check_interval=25)
    em5.reset(seed=42, level=2)
    goal_5 = generate_assembly_goal(_make_info(["2x2"]), bp_type, bp_world_pos, seed=42)
    assembler5 = ScriptedAssembler(em5.model, em5.data)
    r5 = assembler5.execute_assembly(goal_5, hold_duration_s=1.0)
    _record(
        results,
        "Stability hold",
        r5.stability_hold_steps > 0,
        f"hold_steps={r5.stability_hold_steps}, stable={r5.structure_stable}",
    )

    # -----------------------------------------------------------------------
    print("\n--- 9. Metrics Accuracy ---")

    _record(
        results,
        "Penetration bounded",
        r5.max_penetration_m < 0.005,
        f"max_pen={r5.max_penetration_m:.4f} m",
    )
    _record(
        results,
        "Physics steps positive",
        r5.total_physics_steps > 0,
        f"total_steps={r5.total_physics_steps}",
    )

    # -----------------------------------------------------------------------
    print("\n--- 10. Deterministic Replay ---")

    em6 = EpisodeManager(brick_slots=["2x2"], settle_max_steps=300, settle_check_interval=25)
    em6.reset(seed=99, level=2)
    g6a = generate_assembly_goal(_make_info(["2x2"], seed=99), bp_type, bp_world_pos, seed=99)
    a6a = ScriptedAssembler(em6.model, em6.data)
    r6a = a6a.execute_assembly(g6a, hold_duration_s=0.2)

    em6.reset(seed=99, level=2)
    g6b = generate_assembly_goal(_make_info(["2x2"], seed=99), bp_type, bp_world_pos, seed=99)
    a6b = ScriptedAssembler(em6.model, em6.data)
    r6b = a6b.execute_assembly(g6b, hold_duration_s=0.2)

    replay_match = r6a.n_successful == r6b.n_successful
    if r6a.placements and r6b.placements:
        replay_match = replay_match and math.isclose(
            r6a.placements[0].position_error_m, r6b.placements[0].position_error_m, abs_tol=1e-6
        )
    _record(results, "Deterministic replay", replay_match, "same seed -> same result")

    # -----------------------------------------------------------------------
    print("\n--- 11. Stress Test ---")

    n_stress = args.n_stress
    em_stress = EpisodeManager(
        brick_slots=["2x2", "2x4"],
        settle_max_steps=300,
        settle_check_interval=25,
    )

    successes = 0
    engaged_total = 0
    total_bricks = 0
    t0 = time.perf_counter()

    for i in range(n_stress):
        info_s = em_stress.reset(seed=i * 7, level=LEVEL_SINGLE_CONNECTION)
        goal_s = generate_assembly_goal(info_s, bp_type, bp_world_pos, seed=i * 7)
        a_s = ScriptedAssembler(em_stress.model, em_stress.data)
        r_s = a_s.execute_assembly(goal_s, hold_duration_s=0.2)
        total_bricks += r_s.n_total
        engaged_total += sum(1 for p in r_s.placements if p.z_engaged)
        if r_s.n_successful == r_s.n_total:
            successes += 1

    dt_stress = time.perf_counter() - t0
    engage_rate = engaged_total / total_bricks if total_bricks > 0 else 0.0

    _record(
        results,
        f"Stress test ({n_stress} assemblies)",
        engage_rate >= 0.80,
        f"engaged={engaged_total}/{total_bricks} ({engage_rate:.0%}), "
        f"full_success={successes}/{n_stress}, time={dt_stress:.1f}s",
    )

    # -----------------------------------------------------------------------
    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"Phase 1.2.6 validation: {passed}/{total} checks passed")

    # Write results JSON
    results_json = [{"name": n, "passed": p, "detail": d} for n, p, d in results]
    (out_dir / "validation_results.json").write_text(json.dumps(results_json, indent=2))
    print(f"Results written to {out_dir / 'validation_results.json'}")

    return 0 if passed == total else 1


def _make_info(brick_types, seed=42, level=2):
    """Helper to create EpisodeInfo for validation."""
    from sim.lego.episode_manager import EpisodeInfo, SpawnPose

    return EpisodeInfo(
        seed=seed,
        level=level,
        brick_types=brick_types,
        spawn_poses=[SpawnPose((0.45, 0.0, 0.85), (1, 0, 0, 0))] * len(brick_types),
        settle_steps=100,
        settle_success=True,
    )


if __name__ == "__main__":
    sys.exit(main())
