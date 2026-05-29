#!/usr/bin/env python3
"""Prove that local Alex assets can load and run in Isaac Sim/Lab.

The default mode is an orchestrator that writes machine-readable evidence and
launches Isaac subprocesses. ``--worker-minimal`` is the Isaac-side worker used
by the orchestrator; it is intentionally kept in this file so the proof remains
one repeatable command.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ALEX_ROOT = Path("/home/pacquadr/Desktop/Alex-robot")
DEFAULT_ISAAC_WRAPPER = Path("/home/pacquadr/Desktop/isaac_suitcase/bin/isaaclab-alex-run")
DEFAULT_DISPLAY = ":1"
DEFAULT_OVERLAY_ALEX = Path(
    "/home/pacquadr/Desktop/isaac_suitcase/IsaacLab/source/isaaclab_assets/"
    "isaaclab_assets/ihmc/robots/alex/alex.py"
)
DEFAULT_ONNX_SCRIPT = (
    DEFAULT_ALEX_ROOT / "isaac-sim-rl-bringup/scripts/alex_room_explore/alex_onnx_walking_policy.py"
)

FULL_BODY_PRIORITY = (
    "alex_v1.rlModel_fullBody_robotAccurate_torsoFootCollisions.urdf",
    "alex_v1.rlModel_fullBody_robotAccurate_fullCollisions.urdf",
)
NUB_FOREARMS_PRIORITY = (
    "alex_v1.rlModel_nubForearms_robotAccurate_torsoFootCollisions.urdf",
    "alex_v1.rlModel_nubForearms_robotAccurate_fullCollisions.urdf",
)
EXPECTED_FULL_BODY_JOINTS = (
    "LEFT_HIP_X",
    "LEFT_HIP_Z",
    "LEFT_HIP_Y",
    "LEFT_KNEE_Y",
    "LEFT_ANKLE_Y",
    "LEFT_ANKLE_X",
    "RIGHT_HIP_X",
    "RIGHT_HIP_Z",
    "RIGHT_HIP_Y",
    "RIGHT_KNEE_Y",
    "RIGHT_ANKLE_Y",
    "RIGHT_ANKLE_X",
    "SPINE_Z",
    "LEFT_SHOULDER_Y",
    "LEFT_SHOULDER_X",
    "LEFT_SHOULDER_Z",
    "LEFT_ELBOW_Y",
    "LEFT_WRIST_Z",
    "LEFT_WRIST_X",
    "LEFT_GRIPPER_Z",
    "NECK_Z",
    "NECK_Y",
    "RIGHT_SHOULDER_Y",
    "RIGHT_SHOULDER_X",
    "RIGHT_SHOULDER_Z",
    "RIGHT_ELBOW_Y",
    "RIGHT_WRIST_Z",
    "RIGHT_WRIST_X",
    "RIGHT_GRIPPER_Z",
)
ONNX_SUCCESS_MARKERS = (
    "Joint map built: 23/23 joints found",
    "Smoke complete",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker_minimal:
        return _run_minimal_worker(args)
    return _run_orchestrator(args)


def _run_orchestrator(args: argparse.Namespace) -> int:
    repo_root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    alex_root = args.alex_root.resolve()
    output_dir = _make_output_dir(repo_root, args.output_dir)
    command_result_path = output_dir / "command_result.json"

    evidence: dict[str, Any] = {
        "timestamp": _utc_now(),
        "command": sys.argv,
        "repo_root": str(repo_root),
        "alex_root": str(alex_root),
        "isaac_wrapper": str(args.isaac_wrapper),
        "host": _host_summary(),
        "alex_repo": _git_summary(alex_root),
        "isaac_overlay": _overlay_summary(DEFAULT_OVERLAY_ALEX, alex_root),
        "asset_inventory": _asset_inventory(alex_root),
        "selected_assets": {},
        "display_probe": {},
        "proof_items": {
            "minimal_isaac_lab_alex_load": {"passed": False},
            "onnx_bringup_smoke": {"passed": False},
            "gui_or_visual_evidence": {"passed": False},
        },
    }
    _write_json(output_dir / "alex_isaac_resources.json", evidence)

    selected_full_body = _select_preferred_urdf(
        evidence["asset_inventory"]["full_body_urdfs"], FULL_BODY_PRIORITY
    )
    selected_nub = _select_preferred_urdf(
        evidence["asset_inventory"]["nub_forearms_urdfs"], NUB_FOREARMS_PRIORITY
    )
    evidence["selected_assets"] = {
        "minimal_full_body_urdf": selected_full_body,
        "onnx_nub_forearms_urdf": selected_nub,
    }

    command_result: dict[str, Any] = {
        "timestamp": _utc_now(),
        "command": sys.argv,
        "cwd": str(repo_root),
        "output_dir": str(output_dir),
        "evidence": str(output_dir / "alex_isaac_resources.json"),
        "returncode": None,
        "completed_at": None,
    }
    _write_json(command_result_path, command_result)

    try:
        _run_minimal_proof(args, repo_root, alex_root, output_dir, evidence, selected_full_body)
        _write_json(output_dir / "alex_isaac_resources.json", evidence)

        _run_gui_probe_and_optional_proof(
            args, repo_root, alex_root, output_dir, evidence, selected_full_body
        )
        _write_json(output_dir / "alex_isaac_resources.json", evidence)

        if args.skip_onnx:
            evidence["proof_items"]["onnx_bringup_smoke"] = {
                "passed": False,
                "skipped": True,
                "reason": "--skip-onnx was requested",
            }
        else:
            _run_onnx_smoke(args, repo_root, alex_root, output_dir, evidence)
        _write_json(output_dir / "alex_isaac_resources.json", evidence)
    except BaseException as exc:
        evidence["fatal_error"] = _format_exception(exc)
        _write_json(output_dir / "alex_isaac_resources.json", evidence)

    all_passed = all(
        bool(evidence["proof_items"][name].get("passed"))
        for name in (
            "minimal_isaac_lab_alex_load",
            "onnx_bringup_smoke",
            "gui_or_visual_evidence",
        )
    )
    returncode = 0 if all_passed else 1
    command_result.update(
        {
            "completed_at": _utc_now(),
            "returncode": returncode,
            "latest_evidence": str(output_dir / "alex_isaac_resources.json"),
        }
    )
    _write_json(command_result_path, command_result)
    latest_dir = repo_root / "logs" / "alex_isaac_proof"
    latest_dir.mkdir(parents=True, exist_ok=True)
    _write_json(latest_dir / "latest_command_result.json", command_result)
    _write_json(latest_dir / "latest_alex_isaac_resources.json", evidence)
    print(json.dumps(command_result, indent=2, sort_keys=True))
    return returncode


def _run_minimal_proof(
    args: argparse.Namespace,
    repo_root: Path,
    alex_root: Path,
    output_dir: Path,
    evidence: dict[str, Any],
    selected_full_body: dict[str, Any] | None,
) -> None:
    if not selected_full_body:
        evidence["proof_items"]["minimal_isaac_lab_alex_load"] = {
            "passed": False,
            "blocker": "No full-body Alex URDF candidate was found.",
        }
        return

    worker_json = output_dir / "minimal_headless_result.json"
    command = [
        str(args.isaac_wrapper.resolve()),
        str(Path(__file__).resolve()),
        "--worker-minimal",
        "--repo-root",
        str(repo_root),
        "--alex-root",
        str(alex_root),
        "--output-dir",
        str(output_dir),
        "--urdf",
        selected_full_body["path"],
        "--headless",
        "--worker-label",
        "minimal_headless",
        "--screenshot-name",
        "alex_loaded_rgb.png",
        "--worker-steps",
        str(args.minimal_steps),
    ]
    result = _run_logged_child(
        "minimal_isaac_lab",
        command,
        cwd=repo_root,
        output_dir=output_dir,
        timeout_s=args.timeout_s,
        env=_child_env(repo_root),
    )
    worker = _read_json_if_exists(worker_json)
    evidence["proof_items"]["minimal_isaac_lab_alex_load"] = {
        "passed": result["returncode"] == 0 and bool(worker.get("passed")),
        "child_process": result,
        "worker_evidence": str(worker_json),
        "worker_summary": worker,
    }


def _run_gui_probe_and_optional_proof(
    args: argparse.Namespace,
    repo_root: Path,
    alex_root: Path,
    output_dir: Path,
    evidence: dict[str, Any],
    selected_full_body: dict[str, Any] | None,
) -> None:
    display_probe = _probe_display(args.display)
    evidence["display_probe"] = display_probe
    offscreen_screenshot = output_dir / "alex_loaded_rgb.png"

    if args.skip_gui:
        item = {
            "passed": offscreen_screenshot.exists(),
            "skipped": True,
            "reason": "--skip-gui was requested",
            "offscreen_screenshot": str(offscreen_screenshot),
        }
        evidence["proof_items"]["gui_or_visual_evidence"] = item
        return

    if not display_probe["passed"]:
        blocker_path = output_dir / "display_blocker.json"
        _write_json(blocker_path, display_probe)
        evidence["proof_items"]["gui_or_visual_evidence"] = {
            "passed": offscreen_screenshot.exists(),
            "mode": "display_blocked_offscreen_screenshot_used",
            "display_blocker": str(blocker_path),
            "offscreen_screenshot": str(offscreen_screenshot),
        }
        return

    if not selected_full_body:
        evidence["proof_items"]["gui_or_visual_evidence"] = {
            "passed": False,
            "blocker": "Display is reachable, but no full-body Alex URDF candidate was found.",
        }
        return

    worker_json = output_dir / "gui_display_result.json"
    env = _child_env(repo_root)
    env["DISPLAY"] = args.display
    command = [
        str(args.isaac_wrapper.resolve()),
        str(Path(__file__).resolve()),
        "--worker-minimal",
        "--repo-root",
        str(repo_root),
        "--alex-root",
        str(alex_root),
        "--output-dir",
        str(output_dir),
        "--urdf",
        selected_full_body["path"],
        "--no-headless",
        "--worker-label",
        "gui_display",
        "--screenshot-name",
        "desktop_gui_screenshot.png",
        "--worker-steps",
        str(args.minimal_steps),
    ]
    result = _run_logged_child(
        "gui_display",
        command,
        cwd=repo_root,
        output_dir=output_dir,
        timeout_s=args.gui_timeout_s,
        env=env,
    )
    worker = _read_json_if_exists(worker_json)
    evidence["proof_items"]["gui_or_visual_evidence"] = {
        "passed": result["returncode"] == 0 and bool(worker.get("passed")),
        "mode": "desktop_display",
        "child_process": result,
        "worker_evidence": str(worker_json),
        "worker_summary": worker,
        "screenshot": str(output_dir / "desktop_gui_screenshot.png"),
    }


def _run_onnx_smoke(
    args: argparse.Namespace,
    repo_root: Path,
    alex_root: Path,
    output_dir: Path,
    evidence: dict[str, Any],
) -> None:
    command = [
        str(args.isaac_wrapper.resolve()),
        str(args.onnx_script.resolve()),
        "--headless",
        "--smoke-steps",
        str(args.onnx_smoke_steps),
        "scene=groundplane",
        "rerun=disabled",
    ]
    result = _run_logged_child(
        "onnx_smoke",
        command,
        cwd=alex_root,
        output_dir=output_dir,
        timeout_s=args.timeout_s,
        env=_child_env(repo_root),
    )
    stdout = Path(result["stdout_log"]).read_text(errors="replace") if result["stdout_log"] else ""
    stderr = Path(result["stderr_log"]).read_text(errors="replace") if result["stderr_log"] else ""
    combined = stdout + "\n" + stderr
    markers = {marker: marker in combined for marker in ONNX_SUCCESS_MARKERS}
    evidence["proof_items"]["onnx_bringup_smoke"] = {
        "passed": result["returncode"] == 0 and all(markers.values()),
        "child_process": result,
        "success_markers": markers,
        "onnx_script": str(args.onnx_script.resolve()),
    }


def _run_minimal_worker(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.worker_label}_result.json"
    evidence: dict[str, Any] = {
        "timestamp": _utc_now(),
        "label": args.worker_label,
        "command": sys.argv,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "environment": {
            "DISPLAY": os.environ.get("DISPLAY"),
            "XDG_SESSION_TYPE": os.environ.get("XDG_SESSION_TYPE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "urdf_source_path": str(args.urdf.resolve()),
        "headless": args.headless,
        "proof_items": {
            "app_launcher": {"passed": False},
            "alex_module_import": {"passed": False},
            "articulation_spawn": {"passed": False},
            "joint_state": {"passed": False},
            "foot_contacts": {"passed": False},
            "screenshot": {"passed": False},
        },
        "passed": False,
    }
    _write_json(result_path, evidence)

    simulation_app = None
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher({"headless": args.headless, "enable_cameras": True})
        simulation_app = app_launcher.app
        evidence["proof_items"]["app_launcher"] = {
            "passed": True,
            "headless": args.headless,
        }
        _write_json(result_path, evidence)

        import copy

        import isaaclab.sim as sim_utils
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        import torch
        from isaaclab.assets import Articulation
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationContext
        from isaacsim.core.api.objects import FixedCuboid

        alex_isaac_dir = args.alex_root / "alex_models" / "alex_V1_isaacsim"
        sys.path.insert(0, str(alex_isaac_dir.resolve()))
        import alex as alex_cfg  # noqa: PLC0415

        evidence["proof_items"]["alex_module_import"] = {
            "passed": True,
            "module_file": getattr(alex_cfg, "__file__", None),
            "sim_dt": getattr(alex_cfg, "SIM_DT", None),
        }
        _write_json(result_path, evidence)

        resolved_urdf = _rewrite_urdf_package_paths(args.urdf.resolve(), args.alex_root, output_dir)
        evidence["resolved_urdf_path"] = str(resolved_urdf)

        sim_dt = float(getattr(alex_cfg, "SIM_DT", 0.005))
        sim = SimulationContext(sim_utils.SimulationCfg(dt=sim_dt))
        sim.set_camera_view(eye=[3.0, -2.0, 1.8], target=[0.0, 0.0, 0.85])
        FixedCuboid(
            prim_path="/World/proof_ground",
            name="proof_ground",
            position=np.array([0.0, 0.0, -0.05]),
            scale=np.array([10.0, 10.0, 0.1]),
            size=1.0,
            color=np.array([0.35, 0.35, 0.35]),
        )
        sim_utils.DomeLightCfg(intensity=2000.0).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
        )

        robot_cfg = copy.deepcopy(alex_cfg.ALEX_V1_FULLBODY_DEFAULT_CFG)
        robot_cfg.spawn.asset_path = str(resolved_urdf)
        robot = Articulation(robot_cfg.replace(prim_path="/World/Alex"))
        left_contact = ContactSensor(
            ContactSensorCfg(prim_path="/World/Alex/LEFT_FOOT", update_period=0.0, history_length=1)
        )
        right_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/Alex/RIGHT_FOOT", update_period=0.0, history_length=1
            )
        )

        camera = rep.create.camera(position=(3.0, -2.0, 1.8), look_at=(0.0, 0.0, 0.85))
        render_product = rep.create.render_product(camera, (640, 480))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])

        sim.reset()
        left_contact.reset()
        right_contact.reset()

        rgb = None
        for step in range(args.worker_steps):
            sim.step(render=True)
            robot.update(sim_dt)
            left_contact.update(sim_dt)
            right_contact.update(sim_dt)
            if step >= 10:
                rgb = _annotator_rgb_array(rgb_annotator.get_data())

        stage = omni.usd.get_context().get_stage()
        prim_valid = stage is not None and stage.GetPrimAtPath("/World/Alex").IsValid()
        evidence["proof_items"]["articulation_spawn"] = {
            "passed": bool(prim_valid and robot.num_joints > 0),
            "prim_path": "/World/Alex",
            "prim_valid": bool(prim_valid),
            "num_joints": int(robot.num_joints),
            "num_bodies": int(robot.num_bodies),
        }

        joint_pos = robot.data.joint_pos[0].detach().cpu()
        joint_vel = robot.data.joint_vel[0].detach().cpu()
        root_pos = robot.data.root_pos_w[0].detach().cpu()
        root_quat = robot.data.root_quat_w[0].detach().cpu()
        joint_presence = _joint_presence(robot)
        state_finite = bool(
            torch.isfinite(joint_pos).all()
            and torch.isfinite(joint_vel).all()
            and torch.isfinite(root_pos).all()
            and torch.isfinite(root_quat).all()
        )
        missing_joints = [
            name for name, present in joint_presence["expected_full_body"].items() if not present
        ]
        evidence["proof_items"]["joint_state"] = {
            "passed": bool(state_finite and not missing_joints),
            "state_finite": state_finite,
            "missing_expected_joints": missing_joints,
            "joint_pos_shape": list(joint_pos.shape),
            "joint_vel_shape": list(joint_vel.shape),
            "root_pos_w": [float(value) for value in root_pos.tolist()],
            "root_quat_w": [float(value) for value in root_quat.tolist()],
            "joint_presence": joint_presence,
        }

        contact_summary = {
            "left": _contact_summary(left_contact),
            "right": _contact_summary(right_contact),
        }
        contacts_readable = all(item["readable"] for item in contact_summary.values())
        evidence["proof_items"]["foot_contacts"] = {
            "passed": contacts_readable,
            "contacts": contact_summary,
        }

        screenshot_path = output_dir / args.screenshot_name
        screenshot_summary = _write_rgb_png(rgb, screenshot_path)
        evidence["proof_items"]["screenshot"] = screenshot_summary

    except BaseException as exc:
        evidence["fatal_error"] = _format_exception(exc)

    evidence["passed"] = all(item.get("passed") for item in evidence["proof_items"].values())
    _write_json(result_path, evidence)
    sys.stdout.flush()
    sys.stderr.flush()
    if simulation_app is not None:
        os._exit(0 if evidence["passed"] else 1)
    return 0 if evidence["passed"] else 1


def _asset_inventory(alex_root: Path) -> dict[str, Any]:
    urdf_root = alex_root / "alex_models" / "alex_V1_description" / "rl_urdf"
    urdfs = (
        sorted(path.resolve() for path in urdf_root.glob("*.urdf")) if urdf_root.exists() else []
    )
    return {
        "urdf_root": str(urdf_root),
        "all_urdfs": [str(path) for path in urdfs],
        "full_body_urdfs": [str(path) for path in urdfs if "fullBody" in path.name],
        "nub_forearms_urdfs": [str(path) for path in urdfs if "nubForearms" in path.name],
        "isaac_alex_py": str(alex_root / "alex_models" / "alex_V1_isaacsim" / "alex.py"),
        "isaac_mjcf": str(
            alex_root / "alex_models" / "alex_V1_isaacsim" / "alex_v1_full_body_isaacsim.xml"
        ),
    }


def _select_preferred_urdf(paths: list[str], priority: tuple[str, ...]) -> dict[str, Any] | None:
    by_name = {Path(path).name: path for path in paths}
    for name in priority:
        path = by_name.get(name)
        if path:
            return {"path": path, "reason": f"Preferred available URDF: {name}"}
    return {"path": paths[0], "reason": "Fallback to first sorted URDF"} if paths else None


def _overlay_summary(overlay_path: Path, alex_root: Path) -> dict[str, Any]:
    expected_target = alex_root / "alex_models" / "alex_V1_isaacsim" / "alex.py"
    summary = {
        "path": str(overlay_path),
        "exists": overlay_path.exists(),
        "is_symlink": overlay_path.is_symlink(),
        "target": None,
        "expected_target": str(expected_target),
        "matches_expected": False,
    }
    if overlay_path.is_symlink():
        target = overlay_path.resolve()
        summary["target"] = str(target)
        summary["matches_expected"] = target == expected_target.resolve()
    return summary


def _git_summary(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "branch": _run_git(path, ["branch", "--show-current"]),
        "commit": _run_git(path, ["rev-parse", "HEAD"]),
        "status_short_branch": _run_git(path, ["status", "--short", "--branch"], split_lines=True),
        "remote": _run_git(path, ["remote", "-v"], split_lines=True),
    }


def _run_git(path: Path, args: list[str], split_lines: bool = False) -> Any:
    proc = subprocess.run(
        ["git", *args],
        cwd=path,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if proc.returncode != 0:
        return {"returncode": proc.returncode, "stderr": proc.stderr.strip()}
    output = proc.stdout.strip()
    return output.splitlines() if split_lines else output


def _probe_display(display: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["DISPLAY"] = display
    result = _run_capture(["xset", "q"], env=env, timeout_s=10)
    return {
        "display": display,
        "command": result["command"],
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "passed": result["returncode"] == 0,
        "diagnosis": (
            None
            if result["returncode"] == 0
            else "Host display is not reachable from this SSH/session context."
        ),
    }


def _run_logged_child(
    label: str,
    command: list[str],
    cwd: Path,
    output_dir: Path,
    timeout_s: int,
    env: dict[str, str],
) -> dict[str, Any]:
    stdout_path = output_dir / f"{label}_stdout.log"
    stderr_path = output_dir / f"{label}_stderr.log"
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = _to_text(exc.stdout)
        stderr = _to_text(exc.stderr)
        returncode = 124
        timed_out = True

    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    return {
        "label": label,
        "command": command,
        "cwd": str(cwd),
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def _child_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("ACCEPT_EULA", "Y")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "Yes")
    env.setdefault("PRIVACY_CONSENT", "Y")
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _rewrite_urdf_package_paths(urdf: Path, alex_root: Path, output_dir: Path) -> Path:
    text = urdf.read_text()
    package_prefix = "package://alex_V1_description/"
    if package_prefix not in text:
        return urdf
    abs_prefix = str(alex_root / "alex_models" / "alex_V1_description") + "/"
    rewritten = output_dir / f"{urdf.stem}_abs_paths.urdf"
    rewritten.write_text(text.replace(package_prefix, abs_prefix))
    return rewritten


def _joint_presence(robot: Any) -> dict[str, dict[str, bool]]:
    presence: dict[str, bool] = {}
    for name in EXPECTED_FULL_BODY_JOINTS:
        idx_list, _ = robot.find_joints(name)
        presence[name] = bool(idx_list)
    return {"expected_full_body": presence}


def _contact_summary(sensor: Any) -> dict[str, Any]:
    data = getattr(sensor, "data", None)
    forces = getattr(data, "net_forces_w", None)
    if forces is None:
        return {"readable": False, "shape": None, "max_force_norm": None}
    finite = bool(forces.isfinite().all().item())
    norm = forces.norm(dim=-1)
    return {
        "readable": finite,
        "shape": list(forces.shape),
        "max_force_norm": float(norm.max().item()) if norm.numel() else 0.0,
    }


def _annotator_rgb_array(data: Any) -> Any:
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def _write_rgb_png(rgb: Any, path: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    if rgb is None:
        return {"passed": False, "path": str(path), "shape": None, "reason": "No RGB data"}
    array = np.asarray(rgb)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    nonblank = bool(array.size and np.isfinite(array).all() and float(np.std(array)) > 0.0)
    if array.ndim == 3 and array.shape[-1] == 3:
        Image.fromarray(array).save(path)
    return {
        "passed": bool(nonblank and path.exists()),
        "path": str(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(np.mean(array)) if array.size else None,
        "std": float(np.std(array)) if array.size else None,
        "nonblank": nonblank,
    }


def _host_summary() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "environment": {
            "DISPLAY": os.environ.get("DISPLAY"),
            "XDG_SESSION_TYPE": os.environ.get("XDG_SESSION_TYPE"),
            "SSH_CONNECTION": os.environ.get("SSH_CONNECTION"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "gpu": _run_capture(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            timeout_s=10,
        ),
    }


def _run_capture(
    command: list[str], env: dict[str, str] | None = None, timeout_s: int = 15
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def _make_output_dir(repo_root: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        path = output_dir.resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = repo_root / "logs" / "alex_isaac_proof" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _format_exception(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--alex-root", type=Path, default=DEFAULT_ALEX_ROOT)
    parser.add_argument("--isaac-wrapper", type=Path, default=DEFAULT_ISAAC_WRAPPER)
    parser.add_argument("--onnx-script", type=Path, default=DEFAULT_ONNX_SCRIPT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timeout-s", type=int, default=420)
    parser.add_argument("--gui-timeout-s", type=int, default=420)
    parser.add_argument("--onnx-smoke-steps", type=int, default=20)
    parser.add_argument("--minimal-steps", type=int, default=160)
    parser.add_argument("--display", default=DEFAULT_DISPLAY)
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-gui", action="store_true")

    parser.add_argument("--worker-minimal", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--urdf", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--worker-label", default="minimal_headless", help=argparse.SUPPRESS)
    parser.add_argument("--screenshot-name", default="alex_loaded_rgb.png", help=argparse.SUPPRESS)
    parser.add_argument("--worker-steps", type=int, default=160, help=argparse.SUPPRESS)
    args, unknown = parser.parse_known_args(argv)
    unsupported = [item for item in unknown if item != "--kit_args" and not item.startswith("--/")]
    if unsupported:
        parser.error(f"unrecognized arguments: {' '.join(unsupported)}")
    if args.worker_minimal and args.urdf is None:
        parser.error("--worker-minimal requires --urdf")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
