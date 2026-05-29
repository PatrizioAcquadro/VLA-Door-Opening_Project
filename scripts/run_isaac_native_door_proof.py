"""Run the live HallwayScene Isaac-native door proof and capture evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ISAAC_WRAPPER = Path("/home/pacquadr/Desktop/isaac_suitcase/bin/isaaclab-run")
DEFAULT_HALLWAY_SCENE_PATH = Path("/home/pacquadr/Desktop/HallwayScene/Hallway.usdc")
DEFAULT_DOOR_ASSET_PATH = Path("/home/pacquadr/Desktop/HallwayScene/Objects/DoorObject_v2.usda")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = REPO_ROOT
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (repo_root / "logs" / "isaac_native_door" / timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = args.isaac_wrapper.resolve()
    command = [
        str(wrapper),
        "-m",
        "sim.isaac.native_door_proof",
        "--repo-root",
        str(repo_root),
        "--output-dir",
        str(output_dir),
        "--hallway-scene",
        str(args.hallway_scene.resolve()),
        "--door-asset",
        str(args.door_asset.resolve()),
        "--steps",
        str(args.steps),
        "--episode-steps",
        str(args.episode_steps),
    ]
    if not args.headless:
        command.append("--no-headless")

    env = os.environ.copy()
    env.setdefault("ACCEPT_EULA", "Y")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "Yes")
    env.setdefault("PRIVACY_CONSENT", "Y")
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    command_result_path = output_dir / "command_result.json"
    command_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cwd": str(repo_root),
        "output_dir": str(output_dir),
        "python_executable": sys.executable,
    }
    _write_json(command_result_path, command_payload)

    try:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout_s,
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        returncode = 124
        timed_out = True

    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)

    command_payload.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
            "timed_out": timed_out,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "worker_evidence": str(output_dir / "isaac_native_door_proof.json"),
        }
    )
    _write_json(command_result_path, command_payload)

    latest_dir = repo_root / "logs" / "isaac_native_door"
    _write_json(latest_dir / "latest_command_result.json", command_payload)
    return returncode


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-wrapper", type=Path, default=DEFAULT_ISAAC_WRAPPER)
    parser.add_argument("--hallway-scene", type=Path, default=DEFAULT_HALLWAY_SCENE_PATH)
    parser.add_argument("--door-asset", type=Path, default=DEFAULT_DOOR_ASSET_PATH)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--episode-steps", type=int, default=12)
    parser.add_argument("--timeout-s", type=int, default=420)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
