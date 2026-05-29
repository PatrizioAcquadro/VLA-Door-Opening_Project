"""Run the live Isaac proof worker and capture repo-local evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ISAAC_WRAPPER = Path("/home/pacquadr/Desktop/isaac_suitcase/bin/isaaclab-run")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (repo_root / "logs" / "isaac_proof" / timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = args.isaac_wrapper.resolve()
    command = [
        str(wrapper),
        "-m",
        "sim.isaac.live_proof",
        "--repo-root",
        str(repo_root),
        "--output-dir",
        str(output_dir),
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
    command_result_path.write_text(json.dumps(command_payload, indent=2, sort_keys=True) + "\n")

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

    (output_dir / "stdout.log").write_text(stdout)
    (output_dir / "stderr.log").write_text(stderr)

    command_payload.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
            "timed_out": timed_out,
            "stdout_log": str(output_dir / "stdout.log"),
            "stderr_log": str(output_dir / "stderr.log"),
            "worker_evidence": str(output_dir / "isaac_proof.json"),
        }
    )
    command_result_path.write_text(json.dumps(command_payload, indent=2, sort_keys=True) + "\n")

    latest_dir = repo_root / "logs" / "isaac_proof"
    (latest_dir / "latest_command_result.json").write_text(
        json.dumps(command_payload, indent=2, sort_keys=True) + "\n"
    )
    return returncode


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-wrapper", type=Path, default=DEFAULT_ISAAC_WRAPPER)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
