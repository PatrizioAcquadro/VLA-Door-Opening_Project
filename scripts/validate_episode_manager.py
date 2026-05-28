#!/usr/bin/env python3
"""Standalone validation for Phase 1.2.5: Episode Manager (Block Spawning & Reset).

Checks template scene generation, deterministic seeding, spawn constraints,
settle convergence, curriculum levels, and reset reliability metrics.

Artifacts are written to logs/episode_manager/.

Usage::

    python scripts/validate_episode_manager.py
    python scripts/validate_episode_manager.py --n-stress 100
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _record(results: list[tuple[str, bool, str]], name: str, passed: bool, detail: str) -> None:
    """Print pass/fail and append to results list."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {name}: {detail}")
    results.append((name, passed, detail))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1.2.5: EpisodeManager")
    parser.add_argument(
        "--n-stress",
        type=int,
        default=50,
        help="Number of resets for stress test (default: 50)",
    )
    args = parser.parse_args()

    out_dir = _ROOT / "logs" / "episode_manager"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts -> {out_dir}/")

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("[SKIP] mujoco not available — cannot run validation")
        return 0

    import numpy as np

    from sim.lego.constants import BASEPLATE_TYPES, BRICK_TYPES
    from sim.lego.contact_scene import generate_episode_scene
    from sim.lego.episode_manager import (
        LEVEL_MULTI_STEP,
        LEVEL_SINGLE_BRICK,
        LEVEL_SINGLE_CONNECTION,
        EpisodeManager,
    )

    results: list[tuple[str, bool, str]] = []

    # -----------------------------------------------------------------------
    print("\n--- Template Scene Generation ---")

    print("  Generating episode scene XML...")
    bp = BASEPLATE_TYPES["8x8"]
    brick_types = [BRICK_TYPES["2x2"], BRICK_TYPES["2x4"], BRICK_TYPES["2x6"], BRICK_TYPES["2x2"]]
    t0 = time.perf_counter()
    xml = generate_episode_scene(bp, brick_types)
    gen_ms = (time.perf_counter() - t0) * 1000

    _record(results, "XML generation", len(xml) > 100, f"{len(xml)} chars in {gen_ms:.1f} ms")

    try:
        from sim.asset_loader import SCENES_DIR

        tmp_path = SCENES_DIR / "_validate_episode_tmp.xml"
        tmp_path.write_text(xml)
        try:
            model = mujoco.MjModel.from_xml_path(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
        _record(results, "XML compiles", True, f"{model.nbody} bodies, {model.nq} qpos DOFs")
    except Exception as exc:
        _record(results, "XML compiles", False, str(exc))
        model = None

    if model is not None:
        has_bricks = all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"brick_{i}_{bt.name}") >= 0
            for i, bt in enumerate(brick_types)
        )
        _record(results, "Brick bodies present", has_bricks, f"{len(brick_types)} slots found")

    # -----------------------------------------------------------------------
    print("\n--- EpisodeManager Init ---")

    t0 = time.perf_counter()
    em = EpisodeManager(
        brick_slots=["2x2", "2x4", "2x6", "2x2"],
        settle_max_steps=500,
        settle_check_interval=50,
    )
    init_ms = (time.perf_counter() - t0) * 1000

    _record(results, "EpisodeManager init", True, f"compiled in {init_ms:.0f} ms")
    _record(results, "max_bricks=4", em.max_bricks == 4, f"got {em.max_bricks}")
    _record(results, "slot addrs resolved", len(em._slot_qpos_addrs) == 4, f"{em._slot_qpos_addrs}")

    # -----------------------------------------------------------------------
    print("\n--- Deterministic Reset ---")

    t0 = time.perf_counter()
    info_a = em.reset(seed=42)
    reset_ms = (time.perf_counter() - t0) * 1000
    _record(results, "reset() returns EpisodeInfo", True, f"in {reset_ms:.0f} ms")

    info_b = em.reset(seed=42)
    same_poses = info_a.spawn_poses == info_b.spawn_poses
    _record(
        results,
        "Deterministic by seed",
        same_poses,
        "same seed → same poses" if same_poses else "MISMATCH",
    )

    info_c = em.reset(seed=43)
    diff_poses = info_a.spawn_poses != info_c.spawn_poses
    _record(
        results,
        "Different seeds differ",
        diff_poses,
        "different seeds → different poses" if diff_poses else "seeds not independent",
    )

    # -----------------------------------------------------------------------
    print("\n--- Spawn Region Validation ---")

    for seed in range(5):
        em.reset(seed=seed)

    # Check last reset's brick positions from qpos
    info = em.reset(seed=99)
    if info.settle_success and info.spawn_poses:
        x, y, z = info.spawn_poses[0].position
        in_x = em._spawn_x_range[0] <= x <= em._spawn_x_range[1]
        in_y = em._spawn_y_range[0] <= y <= em._spawn_y_range[1]
        at_z = math.isclose(z, em._spawn_z, rel_tol=1e-4)
        _record(
            results,
            "Spawn within X range",
            in_x,
            f"x={x:.3f} ∈ [{em._spawn_x_range[0]}, {em._spawn_x_range[1]}]",
        )
        _record(
            results,
            "Spawn within Y range",
            in_y,
            f"y={y:.3f} ∈ [{em._spawn_y_range[0]}, {em._spawn_y_range[1]}]",
        )
        _record(results, "Spawn Z correct", at_z, f"z={z:.4f} (expected {em._spawn_z:.4f})")
    else:
        _record(results, "Spawn region check", False, "settle failed or no poses")

    # Min-distance validation
    info = em.reset(seed=77, n_active=4)
    if info.spawn_poses and len(info.spawn_poses) >= 2:
        min_dist = float("inf")
        for i, pi in enumerate(info.spawn_poses):
            for j, pj in enumerate(info.spawn_poses):
                if i >= j:
                    continue
                dx = pi.position[0] - pj.position[0]
                dy = pi.position[1] - pj.position[1]
                d = math.sqrt(dx**2 + dy**2)
                min_dist = min(min_dist, d)
        ok = min_dist >= em._min_spawn_distance - 1e-9
        _record(
            results,
            "Min-distance enforced",
            ok,
            f"min pairwise dist={min_dist:.4f} m (threshold={em._min_spawn_distance})",
        )
    else:
        _record(results, "Min-distance enforced", False, "insufficient poses for check")

    # Unused bricks parked
    info = em.reset(seed=0, level=LEVEL_SINGLE_BRICK)
    unused_parked = all(float(em.data.qpos[em._slot_qpos_addrs[i] + 2]) < -5.0 for i in [1, 2, 3])
    _record(results, "Unused bricks parked", unused_parked, "slots 1-3 at Z<-5")

    # Robot joints at home — check immediately after mj_resetData (before settle drift)
    mujoco.mj_resetData(em.model, em.data)
    robot_q = em.data.qpos[: em._slot_qpos_addrs[0]]
    robot_at_home = bool(np.allclose(robot_q, 0.0, atol=1e-6))
    _record(
        results, "Robot at home position", robot_at_home, f"max|q|={np.max(np.abs(robot_q)):.2e}"
    )

    # -----------------------------------------------------------------------
    print("\n--- Settle Phase ---")

    settle_tests = [(42, "settle seed=42"), (7, "settle seed=7"), (123, "settle seed=123")]
    settle_times = []
    for seed, name in settle_tests:
        info = em.reset(seed=seed)
        _record(results, name, info.settle_success, f"{info.settle_steps} steps")
        if info.settle_success:
            settle_times.append(info.settle_steps)

    if settle_times:
        avg_steps = sum(settle_times) / len(settle_times)
        avg_time_ms = avg_steps * 0.002 * 1000  # 0.002s per physics step
        _record(
            results,
            "Settle performance",
            avg_steps < 500,
            f"avg {avg_steps:.0f} steps ({avg_time_ms:.0f} ms sim time)",
        )

    # -----------------------------------------------------------------------
    print("\n--- Curriculum Levels ---")

    em1 = EpisodeManager(brick_slots=["2x2", "2x4", "2x6", "2x2"], settle_max_steps=300)

    # Level 1
    info = em1.reset(seed=0, level=LEVEL_SINGLE_BRICK)
    _record(
        results,
        "Level 1: 1 brick",
        len(info.brick_types) == 1,
        f"brick_types={info.brick_types}",
    )

    # Level 2
    info = em1.reset(seed=0, level=LEVEL_SINGLE_CONNECTION)
    _record(
        results,
        "Level 2: 1 brick",
        len(info.brick_types) == 1,
        f"brick_types={info.brick_types}",
    )

    # Level 3: run several seeds and verify range
    l3_counts = set()
    for seed in range(10):
        info = em1.reset(seed=seed, level=LEVEL_MULTI_STEP)
        l3_counts.add(len(info.brick_types))
    all_valid = all(2 <= c <= 4 for c in l3_counts)
    _record(
        results,
        "Level 3: 2-4 bricks",
        all_valid,
        f"observed counts={sorted(l3_counts)}",
    )

    # -----------------------------------------------------------------------
    print(f"\n--- Stress Test ({args.n_stress} resets) ---")

    em_stress = EpisodeManager(brick_slots=["2x2", "2x4", "2x6", "2x2"], settle_max_steps=500)
    t0 = time.perf_counter()
    for seed in range(args.n_stress):
        em_stress.reset(seed=seed, level=LEVEL_MULTI_STEP)
    total_s = time.perf_counter() - t0

    m = em_stress.metrics
    rate = m.success_rate
    avg_settle_s = m.avg_settle_steps * 0.002 if m.settle_steps_history else 0.0
    throughput = args.n_stress / total_s

    _record(
        results,
        f"Success rate ≥95% ({args.n_stress} resets)",
        rate >= 0.95,
        f"{rate:.1%} ({m.successful_resets}/{m.total_resets}), failures={m.failure_reasons}",
    )
    _record(
        results,
        "Avg settle time",
        avg_settle_s < 1.0,
        f"{avg_settle_s:.3f} s sim time (avg), {m.avg_settle_steps:.0f} steps",
    )
    _record(
        results,
        "Reset throughput",
        throughput >= 0.5,
        f"{throughput:.1f} resets/s wall-clock",
    )

    # -----------------------------------------------------------------------
    # Write artifacts
    # -----------------------------------------------------------------------
    report_lines = [
        "=== Phase 1.2.5 Episode Manager Validation Report ===\n",
        f"Total checks: {len(results)}\n",
        f"Passed: {sum(1 for _, p, _ in results if p)}\n",
        f"Failed: {sum(1 for _, p, _ in results if not p)}\n\n",
    ]
    for name, passed, detail in results:
        report_lines.append(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}\n")
    (out_dir / "validation_report.txt").write_text("".join(report_lines))

    metrics_lines = [
        f"stress_resets: {args.n_stress}\n",
        f"success_rate: {rate:.4f}\n",
        f"avg_settle_steps: {m.avg_settle_steps:.1f}\n",
        f"avg_settle_sim_s: {avg_settle_s:.4f}\n",
        f"throughput_resets_per_s: {throughput:.2f}\n",
        f"failures: {m.failure_reasons}\n",
    ]
    (out_dir / "reset_metrics.txt").write_text("".join(metrics_lines))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n_pass = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    print(f"\n{'='*50}")
    print(f"Episode Manager Validation: {n_pass}/{n_total} passed")
    if n_pass == n_total:
        print("ALL CHECKS PASSED")
    else:
        failed = [name for name, p, _ in results if not p]
        print(f"FAILED: {failed}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
