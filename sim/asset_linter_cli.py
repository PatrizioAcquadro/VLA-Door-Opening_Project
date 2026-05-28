"""CLI entrypoint for the asset linter (vla-lint-assets).

Usage:
    vla-lint-assets                                     # Lint all MJCF files
    vla-lint-assets sim/assets/scenes/test_scene.xml    # Lint a specific file
    vla-lint-assets --strict                            # Treat warnings as errors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sim.asset_linter import Severity, lint_mjcf
from sim.asset_loader import ASSETS_DIR


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for asset linting.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Exit code: 0 = clean, 1 = errors found.
    """
    parser = argparse.ArgumentParser(
        prog="vla-lint-assets",
        description="Lint MJCF files for asset path issues.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="MJCF files to lint. If none given, lints all XML under sim/assets/.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (non-zero exit).",
    )
    args = parser.parse_args(argv)

    # Collect files to lint
    if args.files:
        mjcf_files = args.files
    else:
        mjcf_files = sorted(ASSETS_DIR.rglob("*.xml"))

    if not mjcf_files:
        print("No MJCF files found.")
        return 0

    total_errors = 0
    total_warnings = 0

    for f in mjcf_files:
        issues = lint_mjcf(f)
        for issue in issues:
            prefix = issue.severity.value
            print(f"[{prefix}] {issue.file_path}: {issue.message}")
            if issue.severity == Severity.ERROR:
                total_errors += 1
            else:
                total_warnings += 1

    # Summary
    n_files = len(mjcf_files)
    print(f"\nLinted {n_files} file(s): {total_errors} error(s), {total_warnings} warning(s).")

    if total_errors > 0:
        return 1
    if args.strict and total_warnings > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
