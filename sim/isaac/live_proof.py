"""Live Isaac proof worker.

Run through ``scripts/run_isaac_proof.py`` so stdout/stderr and process status
are captured even if Kit exits early.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.isaac.assets import inventory_alex_asset_candidates, select_preferred_alex_asset
from sim.isaac.contracts import CAMERA_NAMES, DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = args.repo_root.resolve()

    evidence: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "repo_root": str(repo_root),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment": {
            "DISPLAY": os.environ.get("DISPLAY"),
            "XDG_SESSION_TYPE": os.environ.get("XDG_SESSION_TYPE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "gpu": _run_capture(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "proof_items": {
            "isaac_sim_launch": {"passed": False},
            "isaac_lab_python": {"passed": False},
            "alex_asset_load_or_conversion": {"passed": False},
            "physx_hinged_door": {"passed": False},
            "programmatic_camera_reads": {"passed": False},
            "programmatic_contact_reads": {"passed": False},
        },
    }

    inventory = inventory_alex_asset_candidates(repo_root)
    selected_asset = select_preferred_alex_asset(inventory)
    evidence["asset_inventory"] = inventory
    evidence["selected_alex_asset"] = selected_asset
    _write_json(output_dir / "isaac_proof.json", evidence)

    app = None
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": args.headless})
        app.update()
        evidence["proof_items"]["isaac_sim_launch"] = {
            "passed": True,
            "headless": args.headless,
            "runtime_path": sys.executable,
        }
        _write_json(output_dir / "isaac_proof.json", evidence)

        _prove_lab_import(evidence)
        _write_json(output_dir / "isaac_proof.json", evidence)

        _prove_asset_path(evidence, output_dir)
        _write_json(output_dir / "isaac_proof.json", evidence)

        _prove_physics_camera_contact(evidence)
        _write_json(output_dir / "isaac_proof.json", evidence)
    except BaseException as exc:
        evidence["fatal_error"] = _format_exception(exc)
        _write_json(output_dir / "isaac_proof.json", evidence)
        return 2
    finally:
        if app is not None:
            try:
                app.close()
            except Exception as exc:  # pragma: no cover - teardown diagnostics only
                evidence["app_close_error"] = _format_exception(exc)
                _write_json(output_dir / "isaac_proof.json", evidence)

    all_passed = all(bool(item.get("passed")) for item in evidence["proof_items"].values())
    return 0 if all_passed else 1


def _prove_lab_import(evidence: dict[str, Any]) -> None:
    item = evidence["proof_items"]["isaac_lab_python"]
    try:
        import isaaclab
        import torch

        item.update(
            {
                "passed": True,
                "isaaclab_module": getattr(isaaclab, "__file__", None),
                "torch_version": torch.__version__,
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except Exception as exc:
        item.update({"passed": False, "error": _format_exception(exc)})


def _prove_asset_path(evidence: dict[str, Any], output_dir: Path) -> None:
    item = evidence["proof_items"]["alex_asset_load_or_conversion"]
    selected = evidence["selected_alex_asset"]
    path = selected.get("path")
    kind = selected.get("kind")
    attempts: list[dict[str, Any]] = []
    item["attempts"] = attempts

    if not path or not kind:
        item.update({"passed": False, "blocker": "No Alex USD, URDF, or MJCF candidate found."})
        return

    try:
        if kind == "usd":
            _try_open_usd(path, attempts)
        elif kind == "urdf":
            _try_import_urdf(path, output_dir, attempts)
        elif kind == "mjcf":
            _try_import_mjcf(path, output_dir, attempts)
        else:
            raise RuntimeError(f"Unsupported asset kind: {kind}")
    except Exception as exc:
        attempts.append(
            {
                "kind": kind,
                "path": path,
                "passed": False,
                "error": _format_exception(exc),
            }
        )

    successful = [attempt for attempt in attempts if attempt.get("passed") is True]
    item.update(
        {
            "passed": bool(successful),
            "selected_path": path,
            "selected_kind": kind,
            "result": successful[-1] if successful else None,
            "blocker": None if successful else f"{kind} path could not be loaded/converted.",
        }
    )


def _try_open_usd(path: str, attempts: list[dict[str, Any]]) -> None:
    from pxr import Usd

    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open returned None for {path}")
    prims = [str(prim.GetPath()) for prim in stage.Traverse()]
    attempts.append(
        {
            "kind": "usd",
            "path": path,
            "passed": True,
            "prim_count": len(prims),
            "sample_prims": prims[:20],
        }
    )


def _try_import_urdf(path: str, output_dir: Path, attempts: list[dict[str, Any]]) -> None:
    import omni.kit.app
    import omni.kit.commands
    from isaacsim.core.utils.stage import create_new_stage, update_stage
    from pxr import Usd

    create_new_stage()
    update_stage()

    manager = omni.kit.app.get_app().get_extension_manager()
    if manager.get_enabled_extension_id("isaacsim.asset.importer.urdf") is None:
        manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
        update_stage()

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.collision_from_visuals = False

    usd_path = output_dir / "alex_urdf_import.usd"
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=path,
        import_config=import_config,
        dest_path=str(usd_path),
        get_articulation_root=True,
    )
    update_stage()

    stage = Usd.Stage.Open(str(usd_path))
    prims = [str(prim.GetPath()) for prim in stage.Traverse()] if stage else []
    attempts.append(
        {
            "kind": "urdf",
            "path": path,
            "command": "URDFParseAndImportFile",
            "passed": bool(status and usd_path.exists() and prims),
            "usd_path": str(usd_path),
            "imported_prim_path": str(prim_path),
            "prim_count": len(prims),
            "sample_prims": prims[:20],
        }
    )


def _try_import_mjcf(path: str, output_dir: Path, attempts: list[dict[str, Any]]) -> None:
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    cfg = MjcfConverterCfg(asset_path=path, usd_dir=str(output_dir), force_usd_conversion=True)
    converter = MjcfConverter(cfg)
    usd_path = Path(converter.usd_path)
    attempts.append(
        {
            "kind": "mjcf",
            "path": path,
            "command": "isaaclab.sim.converters.MjcfConverter",
            "passed": usd_path.exists(),
            "usd_path": str(usd_path),
        }
    )


def _prove_physics_camera_contact(evidence: dict[str, Any]) -> None:
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.sensors.physics import ContactSensor
    from pxr import Gf, Sdf, UsdPhysics

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    frame = world.scene.add(
        FixedCuboid(
            prim_path="/World/door_frame",
            name="door_frame",
            position=np.array([0.0, 0.0, 1.0]),
            scale=np.array([0.05, 0.1, 2.0]),
            size=1.0,
            color=np.array([0.1, 0.1, 0.1]),
        )
    )
    door = world.scene.add(
        DynamicCuboid(
            prim_path="/World/door_panel",
            name="door_panel",
            mass=5.0,
            position=np.array([0.5, 0.0, 1.0]),
            scale=np.array([1.0, 0.05, 1.8]),
            size=1.0,
            color=np.array([0.0, 0.2, 0.8]),
        )
    )
    stage = world.stage
    joint = UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path("/World/door_hinge"))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(frame.prim_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(door.prim_path)])
    joint.CreateAxisAttr().Set(UsdPhysics.Tokens.z)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5, 0.0, 0.0))

    world.scene.add(
        FixedCuboid(
            prim_path="/World/contact_base",
            name="contact_base",
            position=np.array([0.0, -2.0, 0.25]),
            scale=np.array([0.5, 0.5, 0.5]),
            size=1.0,
            color=np.array([0.2, 0.2, 0.2]),
        )
    )
    contact_top = world.scene.add(
        DynamicCuboid(
            prim_path="/World/contact_top",
            name="contact_top",
            mass=1.0,
            position=np.array([0.0, -2.0, 1.0]),
            scale=np.array([0.5, 0.5, 0.5]),
            size=1.0,
            color=np.array([0.8, 0.1, 0.1]),
        )
    )
    contact_sensor = world.scene.add(
        ContactSensor(
            prim_path=f"{contact_top.prim_path}/contact_sensor",
            name="contact_sensor",
            min_threshold=0.0,
            max_threshold=100000.0,
            radius=-1.0,
        )
    )

    camera = rep.create.camera(position=(3.0, 2.0, 2.0), look_at=(0.3, 0.0, 0.8))
    render_product = rep.create.render_product(
        camera, (DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT)
    )
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])

    world.reset()
    door.set_angular_velocity(np.array([0.0, 0.0, 1.2]))

    angle_samples: list[float] = []
    rgb = None
    for step in range(160):
        world.step(render=True)
        if step % 10 == 0:
            _, quat = door.get_world_pose()
            angle_samples.append(_quat_yaw_rad(quat))
        if step == 20:
            rgb = np.asarray(rgb_annotator.get_data())

    _, quat = door.get_world_pose()
    angle_samples.append(_quat_yaw_rad(quat))
    contact_frame = contact_sensor.get_current_frame()

    rgb_summary = _summarize_rgb(rgb)
    contact_summary = {
        "sensor_prim_path": contact_sensor.prim_path,
        "body_prim_path": contact_top.prim_path,
        "in_contact": bool(contact_frame.get("in_contact", False)),
        "force": float(contact_frame.get("force", 0.0)),
        "number_of_contacts": int(contact_frame.get("number_of_contacts", 0)),
        "physics_step": float(contact_frame.get("physics_step", 0.0)),
    }

    angle_delta = float(max(angle_samples) - min(angle_samples)) if angle_samples else 0.0
    evidence["proof_items"]["physx_hinged_door"] = {
        "passed": bool(angle_delta > 1e-4),
        "joint_prim_path": "/World/door_hinge",
        "door_prim_path": door.prim_path,
        "angle_samples_rad": angle_samples,
        "angle_delta_rad": angle_delta,
    }
    evidence["proof_items"]["programmatic_camera_reads"] = {
        "passed": rgb_summary["non_fake_frame_check"],
        "camera_names": list(CAMERA_NAMES),
        "proof_camera": "overhead",
        **rgb_summary,
    }
    evidence["proof_items"]["programmatic_contact_reads"] = {
        "passed": bool(contact_summary["in_contact"] and contact_summary["force"] > 0.0),
        **contact_summary,
    }


def _summarize_rgb(rgb: np.ndarray | None) -> dict[str, Any]:
    if rgb is None:
        return {
            "shape": None,
            "dtype": None,
            "non_fake_frame_check": False,
            "mean": None,
            "std": None,
        }
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    expected_shape = (DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3)
    return {
        "shape": list(rgb.shape),
        "dtype": str(rgb.dtype),
        "mean": float(np.mean(rgb)) if rgb.size else None,
        "std": float(np.std(rgb)) if rgb.size else None,
        "non_fake_frame_check": bool(
            rgb.shape == expected_shape
            and rgb.size > 0
            and np.isfinite(rgb).all()
            and float(np.std(rgb)) > 0.0
        ),
    }


def _quat_yaw_rad(quat: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _run_capture(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except Exception as exc:
        return {"command": cmd, "returncode": None, "error": str(exc)}
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _format_exception(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
