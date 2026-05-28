"""Interactive MuJoCo viewer for debug inspection.

Launch:
    python sim/viewer.py sim/assets/scenes/test_scene.xml
    vla-viewer sim/assets/scenes/test_scene.xml --show-contacts --show-joints

This module is for interactive debugging ONLY and must never be imported
by training or runtime code paths.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass
class ViewerConfig:
    """Configuration for the interactive viewer.

    Attributes:
        scene_path: Path to MJCF XML file.
        passive: Use non-blocking passive viewer.
        show_contacts: Enable contact point/force visualization.
        show_joints: Enable joint axis visualization.
        camera_name: Name of MJCF-defined camera to use initially.
        duration: Seconds to run in passive mode (0 = unlimited).
    """

    scene_path: Path
    passive: bool = False
    show_contacts: bool = False
    show_joints: bool = False
    camera_name: str | None = None
    duration: float = 0.0


def parse_args(argv: list[str] | None = None) -> ViewerConfig:
    """Parse command-line arguments into a ViewerConfig.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Populated ViewerConfig.
    """
    parser = argparse.ArgumentParser(
        description="Interactive MuJoCo viewer for debug inspection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              vla-viewer sim/assets/scenes/test_scene.xml
              vla-viewer robot.xml --show-contacts --show-joints
              vla-viewer robot.xml --passive --duration 10
        """),
    )
    parser.add_argument("scene", type=Path, help="Path to MJCF XML file")
    parser.add_argument(
        "--passive",
        "-p",
        action="store_true",
        help="Use non-blocking passive viewer (steps simulation in a loop)",
    )
    parser.add_argument(
        "--show-contacts",
        "-c",
        action="store_true",
        help="Enable contact point and force visualization",
    )
    parser.add_argument(
        "--show-joints",
        "-j",
        action="store_true",
        help="Enable joint axis visualization",
    )
    parser.add_argument(
        "--camera",
        "--cam",
        type=str,
        default=None,
        help="Named camera defined in the MJCF to start from",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=0.0,
        help="In passive mode, run for N seconds then exit (0 = unlimited)",
    )
    args = parser.parse_args(argv)
    return ViewerConfig(
        scene_path=args.scene,
        passive=args.passive,
        show_contacts=args.show_contacts,
        show_joints=args.show_joints,
        camera_name=args.camera,
        duration=args.duration,
    )


@dataclass
class PreflightResult:
    """Result of a single preflight check.

    Attributes:
        name: Human-readable check name.
        passed: Whether the check passed.
        detail: Description of what was found.
    """

    name: str
    passed: bool
    detail: str


def configure_vis_options(
    model: mujoco.MjModel,
    opt: mujoco.MjvOption,
    *,
    show_contacts: bool = False,
    show_joints: bool = False,
) -> mujoco.MjvOption:
    """Set MjvOption flags for debug visualization.

    Args:
        model: Compiled MuJoCo model.
        opt: MjvOption to modify in place.
        show_contacts: Enable contact point and force rendering.
        show_joints: Enable joint axis rendering.

    Returns:
        The modified opt (same object, returned for convenience).
    """
    if show_contacts:
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    if show_joints:
        opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    return opt


def run_preflight_checks(model: mujoco.MjModel) -> list[PreflightResult]:
    """Run automated sanity checks on a loaded model before viewing.

    Checks gravity, ground plane presence, and collision/visual geom breakdown.

    Args:
        model: Compiled MuJoCo model.

    Returns:
        List of PreflightResult for each check.
    """
    results: list[PreflightResult] = []

    # 1. Gravity check
    grav = model.opt.gravity
    grav_mag = float(np.linalg.norm(grav))
    if grav_mag < 1e-6:
        results.append(PreflightResult("gravity", False, f"Gravity is zero: {grav}"))
    elif grav[2] >= 0:
        results.append(
            PreflightResult(
                "gravity",
                False,
                f"Gravity Z component is non-negative ({grav[2]:.3f}); expected negative",
            )
        )
    else:
        results.append(
            PreflightResult(
                "gravity", True, f"gravity = [{grav[0]:.2f}, {grav[1]:.2f}, {grav[2]:.2f}]"
            )
        )

    # 2. Ground plane check
    plane_count = sum(
        1 for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
    )
    if plane_count == 0:
        results.append(PreflightResult("ground_plane", False, "No plane geom found in scene"))
    else:
        results.append(PreflightResult("ground_plane", True, f"{plane_count} plane geom(s) found"))

    # 3. Collision vs visual geom summary
    visual_count = sum(
        1
        for i in range(model.ngeom)
        if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0
    )
    collision_count = model.ngeom - visual_count
    results.append(
        PreflightResult(
            "geom_summary",
            True,
            f"{model.ngeom} total geoms: {collision_count} collision, {visual_count} visual-only",
        )
    )

    return results


def print_preflight_report(results: list[PreflightResult]) -> bool:
    """Print preflight check results to stdout.

    Args:
        results: List of PreflightResult from run_preflight_checks.

    Returns:
        True if all checks passed.
    """
    print("--- Preflight Checks ---")
    all_ok = True
    for r in results:
        status = "OK" if r.passed else "WARN"
        if not r.passed:
            all_ok = False
        print(f"  [{status}] {r.name}: {r.detail}")
    print("------------------------")
    return all_ok


def launch_viewer(config: ViewerConfig) -> None:
    """Load a scene and open the interactive MuJoCo viewer.

    Args:
        config: ViewerConfig with all options.

    Raises:
        FileNotFoundError: If scene_path does not exist.
        RuntimeError: If the viewer fails to launch (e.g., no display).
    """
    import mujoco.viewer  # noqa: F811 — deferred import to avoid display requirement

    from sim.mujoco_env import load_model

    model = load_model(config.scene_path)
    data = mujoco.MjData(model)

    # Run and print preflight checks
    results = run_preflight_checks(model)
    print_preflight_report(results)

    # These options are applied via the passive viewer handle; if the user
    # requests them without --passive, switch modes instead of ignoring flags.
    needs_passive = bool(
        config.camera_name or config.show_contacts or config.show_joints or config.duration > 0
    )
    if needs_passive and not config.passive:
        print(
            "Note: --camera/--show-contacts/--show-joints/--duration "
            "require passive mode; enabling --passive.",
            file=sys.stderr,
        )
        config.passive = True

    if config.passive:
        _launch_passive(model, data, config)
    else:
        _launch_interactive(model, data, config)


def _launch_interactive(model: mujoco.MjModel, data: mujoco.MjData, config: ViewerConfig) -> None:
    """Open the blocking interactive viewer."""
    import mujoco.viewer

    # mujoco.viewer.launch is blocking — vis options and camera can be set
    # via the viewer UI (keyboard shortcuts) during the session.
    # For programmatic defaults, we use key_callback or launch_passive instead.
    mujoco.viewer.launch(model, data)


def _launch_passive(model: mujoco.MjModel, data: mujoco.MjData, config: ViewerConfig) -> None:
    """Open the non-blocking passive viewer with a step loop."""
    import time

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_vis_options(
            model,
            viewer.opt,
            show_contacts=config.show_contacts,
            show_joints=config.show_joints,
        )
        if config.camera_name:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, config.camera_name)
            if cam_id >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = cam_id
            else:
                print(
                    f"Warning: camera '{config.camera_name}' not found in MJCF, "
                    "using default view",
                    file=sys.stderr,
                )

        start = time.monotonic()
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            if config.duration > 0 and (time.monotonic() - start) > config.duration:
                break


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the interactive viewer.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 success, 1 failure).
    """
    config = parse_args(argv)
    try:
        launch_viewer(config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Viewer error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
