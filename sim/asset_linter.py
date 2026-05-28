"""Asset linter for MJCF files (Phase 0.2.5).

Walks MJCF XML elements and validates:
- No absolute paths in file/mesh/texture references
- All referenced files (meshes, textures, includes) exist on disk
- No suspicious scale factors on meshes

Usage:
    from sim.asset_linter import lint_mjcf

    issues = lint_mjcf(Path("sim/assets/scenes/test_scene.xml"))
    for issue in issues:
        print(f"[{issue.severity.value}] {issue.message}")
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Severity(Enum):
    """Lint issue severity level."""

    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class LintIssue:
    """A single lint finding.

    Attributes:
        severity: ERROR or WARNING.
        element_tag: XML tag where the issue was found.
        attribute: XML attribute name (if applicable).
        message: Human-readable description.
        file_path: Path to the MJCF file containing the issue.
    """

    severity: Severity
    element_tag: str
    attribute: str
    message: str
    file_path: Path


def _is_absolute(path_str: str) -> bool:
    """Check if a path string is absolute (Unix or Windows)."""
    return path_str.startswith("/") or (len(path_str) >= 2 and path_str[1] == ":")


# Attributes that contain file paths, keyed by element tag.
_FILE_ATTRS: dict[str, list[str]] = {
    "mesh": ["file"],
    "texture": ["file", "filename"],
    "include": ["file"],
    "compiler": ["meshdir", "texturedir"],
    "hfield": ["file"],
}


def _check_absolute_paths(tree: ET.ElementTree[ET.Element], mjcf_path: Path) -> list[LintIssue]:
    """Check for absolute paths in file-referencing attributes."""
    issues: list[LintIssue] = []
    root = tree.getroot()

    for tag, attrs in _FILE_ATTRS.items():
        for elem in root.iter(tag):
            for attr in attrs:
                value = elem.get(attr)
                if value and _is_absolute(value):
                    issues.append(
                        LintIssue(
                            severity=Severity.ERROR,
                            element_tag=tag,
                            attribute=attr,
                            message=f'Absolute path in <{tag} {attr}="{value}">',
                            file_path=mjcf_path,
                        )
                    )
    return issues


def _get_compiler_dirs(tree: ET.ElementTree[ET.Element]) -> tuple[str | None, str | None]:
    """Extract meshdir and texturedir from <compiler> element if present."""
    root = tree.getroot()
    compiler = root.find(".//compiler")
    if compiler is None:
        return None, None
    return compiler.get("meshdir"), compiler.get("texturedir")


def _check_missing_files(tree: ET.ElementTree[ET.Element], mjcf_path: Path) -> list[LintIssue]:
    """Check that all referenced files exist relative to the MJCF file."""
    issues: list[LintIssue] = []
    root = tree.getroot()
    base_dir = mjcf_path.parent
    meshdir, texturedir = _get_compiler_dirs(tree)

    # Check mesh files
    mesh_base = base_dir / meshdir if meshdir else base_dir
    for elem in root.iter("mesh"):
        file_val = elem.get("file")
        if file_val and not _is_absolute(file_val):
            resolved = mesh_base / file_val
            if not resolved.exists():
                issues.append(
                    LintIssue(
                        severity=Severity.ERROR,
                        element_tag="mesh",
                        attribute="file",
                        message=f"Missing mesh file: {file_val} (resolved to {resolved})",
                        file_path=mjcf_path,
                    )
                )

    # Check texture files
    tex_base = base_dir / texturedir if texturedir else base_dir
    for elem in root.iter("texture"):
        for attr in ("file", "filename"):
            file_val = elem.get(attr)
            if file_val and not _is_absolute(file_val):
                resolved = tex_base / file_val
                if not resolved.exists():
                    issues.append(
                        LintIssue(
                            severity=Severity.ERROR,
                            element_tag="texture",
                            attribute=attr,
                            message=f"Missing texture file: {file_val} (resolved to {resolved})",
                            file_path=mjcf_path,
                        )
                    )

    # Check hfield files
    for elem in root.iter("hfield"):
        file_val = elem.get("file")
        if file_val and not _is_absolute(file_val):
            resolved = base_dir / file_val
            if not resolved.exists():
                issues.append(
                    LintIssue(
                        severity=Severity.ERROR,
                        element_tag="hfield",
                        attribute="file",
                        message=f"Missing hfield file: {file_val} (resolved to {resolved})",
                        file_path=mjcf_path,
                    )
                )

    return issues


def _check_include_files(tree: ET.ElementTree[ET.Element], mjcf_path: Path) -> list[LintIssue]:
    """Check that all <include> file references exist."""
    issues: list[LintIssue] = []
    root = tree.getroot()
    base_dir = mjcf_path.parent

    for elem in root.iter("include"):
        file_val = elem.get("file")
        if file_val and not _is_absolute(file_val):
            resolved = base_dir / file_val
            if not resolved.exists():
                issues.append(
                    LintIssue(
                        severity=Severity.ERROR,
                        element_tag="include",
                        attribute="file",
                        message=f"Missing include file: {file_val} (resolved to {resolved})",
                        file_path=mjcf_path,
                    )
                )
    return issues


def _check_mesh_scales(tree: ET.ElementTree[ET.Element], mjcf_path: Path) -> list[LintIssue]:
    """Check for suspicious mesh scale factors."""
    issues: list[LintIssue] = []
    root = tree.getroot()

    for elem in root.iter("mesh"):
        scale_str = elem.get("scale")
        if not scale_str:
            continue
        try:
            components = [float(x) for x in scale_str.split()]
        except ValueError:
            issues.append(
                LintIssue(
                    severity=Severity.ERROR,
                    element_tag="mesh",
                    attribute="scale",
                    message=f'Unparseable mesh scale: "{scale_str}"',
                    file_path=mjcf_path,
                )
            )
            continue

        for val in components:
            abs_val = abs(val)
            if abs_val > 100 or (abs_val > 0 and abs_val < 0.001):
                mesh_name = elem.get("name", "<unnamed>")
                issues.append(
                    LintIssue(
                        severity=Severity.WARNING,
                        element_tag="mesh",
                        attribute="scale",
                        message=(
                            f"Suspicious scale on mesh '{mesh_name}': "
                            f"{scale_str} (possible unit mismatch)"
                        ),
                        file_path=mjcf_path,
                    )
                )
                break  # One warning per mesh is enough

    return issues


def lint_mjcf(mjcf_path: Path) -> list[LintIssue]:
    """Lint an MJCF XML file for asset path issues.

    Checks:
        1. Absolute paths in mesh/texture/include file attributes.
        2. Missing referenced files (mesh, texture, include, hfield).
        3. Suspicious mesh scale factors (component > 100 or < 0.001).

    Args:
        mjcf_path: Path to the MJCF XML file.

    Returns:
        List of LintIssue findings (empty = clean).

    Raises:
        FileNotFoundError: If mjcf_path does not exist.
        ET.ParseError: If XML is malformed.
    """
    mjcf_path = Path(mjcf_path)
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")

    tree = ET.parse(mjcf_path)  # noqa: S314 — trusted local MJCF files
    issues: list[LintIssue] = []
    issues.extend(_check_absolute_paths(tree, mjcf_path))
    issues.extend(_check_missing_files(tree, mjcf_path))
    issues.extend(_check_include_files(tree, mjcf_path))
    issues.extend(_check_mesh_scales(tree, mjcf_path))
    return issues
