"""Live HallwayScene Isaac-native door proof worker."""

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

import numpy as np

from sim.isaac.native_door import (
    DEFAULT_DOOR_ASSET_PATH,
    DEFAULT_HALLWAY_SCENE_PATH,
    IsaacNativeDoorTask,
    NativeDoorTaskConfig,
    summarize_rgb_frame,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = args.repo_root.resolve()
    evidence_path = output_dir / "isaac_native_door_proof.json"

    evidence: dict[str, Any] = {
        "timestamp": _utc_now(),
        "command": sys.argv,
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
            "hallway_scene_open": {"passed": False},
            "door_asset_inspection": {"passed": False},
            "native_task_door_prims": {"passed": False},
            "physx_revolute_hinge_motion": {"passed": False},
            "handle_contact_readings": {"passed": False},
            "latch_proxy_transitions": {"passed": False},
            "target_angle_error_metric": {"passed": False},
            "live_camera_reads": {"passed": False},
            "episode_recording": {"passed": False},
        },
    }
    _write_json(evidence_path, evidence)

    app = None
    skip_close = False
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": args.headless})
        app.update()
        evidence["proof_items"]["isaac_sim_launch"] = {
            "passed": True,
            "headless": args.headless,
            "runtime_path": sys.executable,
        }
        _write_json(evidence_path, evidence)

        config = NativeDoorTaskConfig(
            hallway_scene_path=args.hallway_scene.resolve(),
            door_asset_path=args.door_asset.resolve(),
        )
        task = IsaacNativeDoorTask(config)
        reset_summary = task.reset()
        evidence["task_config"] = config.to_json_dict()
        evidence["reset_summary"] = reset_summary
        evidence["selected_paths"] = reset_summary["selected_task_door"]["prim_paths"]
        evidence["hinge_axis"] = reset_summary["selected_task_door"]["hinge_axis"]
        evidence["hinge_limits_deg"] = reset_summary["selected_task_door"]["hinge_limits_deg"]

        evidence["proof_items"]["hallway_scene_open"] = {
            "passed": bool(reset_summary["hallway_scene"]["opened"]),
            **reset_summary["hallway_scene"],
        }
        evidence["proof_items"]["door_asset_inspection"] = {
            "passed": bool(reset_summary["door_asset_inspection"]["opened"]),
            **reset_summary["door_asset_inspection"],
        }
        evidence["proof_items"]["native_task_door_prims"] = {
            "passed": True,
            "frame_prim_path": task.paths.frame,
            "panel_prim_path": task.paths.panel,
            "hinge_joint_prim_path": task.paths.hinge_joint,
            "handle_prim_path": task.paths.handle,
            "handle_region_prim_path": task.paths.handle_region,
            "contact_sensor_prim_path": task.paths.contact_sensor,
            "camera_prim_paths": task.paths.cameras,
        }
        _write_json(evidence_path, evidence)

        actions = _proof_actions(args.steps)
        observations = []
        for action in actions:
            observations.append(task.step(action))

        frames = task.render_views()
        frame_summaries = {name: summarize_rgb_frame(frame) for name, frame in frames.items()}
        episode_path = task.record_episode(
            output_dir=output_dir / "episode",
            actions_17d=_proof_actions(max(4, args.episode_steps)),
        )

        angle_samples = task.angle_samples
        target_error_samples = task.target_error_samples
        contact_samples = task.contact_samples
        metrics = task.get_door_metrics()
        angle_delta = float(max(angle_samples) - min(angle_samples)) if angle_samples else 0.0
        max_contact_force = (
            float(max(sample["force_n"] for sample in contact_samples)) if contact_samples else 0.0
        )

        evidence.update(
            {
                "angle_samples_rad": angle_samples,
                "target_angle_rad": config.target_open_angle_rad,
                "target_angle_error_samples": target_error_samples,
                "contact_readings": contact_samples,
                "latch_state_transitions": task.latch.transitions,
                "final_metrics": metrics,
                "camera_frame_summaries": frame_summaries,
                "episode_artifact_path": str(episode_path),
                "observations_sample": _json_safe(observations[:5]),
            }
        )
        evidence["proof_items"]["physx_revolute_hinge_motion"] = {
            "passed": bool(angle_delta > 1e-4),
            "door_angle_delta_rad": angle_delta,
            "angle_samples_rad": angle_samples,
            "hinge_joint_prim_path": task.paths.hinge_joint,
            "panel_prim_path": task.paths.panel,
        }
        evidence["proof_items"]["handle_contact_readings"] = {
            "passed": bool(max_contact_force > config.contact_threshold_n),
            "max_contact_force_n": max_contact_force,
            "contact_threshold_n": config.contact_threshold_n,
            "contact_sensor_prim_path": task.paths.contact_sensor,
            "handle_prim_path": task.paths.handle,
            "contact_readings": contact_samples,
        }
        evidence["proof_items"]["latch_proxy_transitions"] = {
            "passed": bool(task.latch.transitions)
            and any(
                item["to"] in {"released", "opening", "target_reached"}
                for item in task.latch.transitions
            ),
            "state": task.latch.state,
            "transitions": task.latch.transitions,
            "policy": "locked until measured handle-region contact force exceeds threshold",
        }
        evidence["proof_items"]["target_angle_error_metric"] = {
            "passed": bool(
                target_error_samples
                and "target_angle_error" in metrics
                and "final_angle_error_rad" in metrics
                and np.isfinite(target_error_samples).all()
            ),
            "target_angle_error_samples": target_error_samples,
            "final_angle_error_rad": metrics.get("final_angle_error_rad"),
            "target_angle_error": metrics.get("target_angle_error"),
        }
        evidence["proof_items"]["live_camera_reads"] = {
            "passed": bool(frame_summaries)
            and all(bool(summary["nonfake_check"]) for summary in frame_summaries.values()),
            "camera_prim_paths": task.paths.cameras,
            "camera_frame_summaries": frame_summaries,
            "unavailable_views": reset_summary["unavailable_views"],
        }
        evidence["proof_items"]["episode_recording"] = {
            "passed": episode_path.exists(),
            "episode_artifact_path": str(episode_path),
        }
        _write_json(evidence_path, evidence)
    except BaseException as exc:
        evidence["fatal_error"] = _format_exception(exc)
        _write_json(evidence_path, evidence)
        skip_close = True
        return 2
    finally:
        if app is not None and not skip_close:
            try:
                app.close()
            except Exception as exc:  # pragma: no cover - teardown diagnostics only
                evidence["app_close_error"] = _format_exception(exc)
                _write_json(evidence_path, evidence)

    all_passed = all(bool(item.get("passed")) for item in evidence["proof_items"].values())
    return 0 if all_passed else 1


def _proof_actions(num_steps: int) -> np.ndarray:
    actions = np.zeros((num_steps, 17), dtype=np.float64)
    actions[:, 0] = 1.0
    actions[:, 1] = 0.9
    return actions


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hallway-scene", type=Path, default=DEFAULT_HALLWAY_SCENE_PATH)
    parser.add_argument("--door-asset", type=Path, default=DEFAULT_DOOR_ASSET_PATH)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--episode-steps", type=int, default=12)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
