"""Live asset inventory helpers for the first Isaac backend slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetCandidate:
    """A concrete robot asset path found on disk."""

    kind: str
    path: str
    source: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def inventory_alex_asset_candidates(
    repo_root: Path,
    external_roots: tuple[Path, ...] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Inventory Alex USD/URDF/MJCF candidates from current disk state."""
    repo_root = repo_root.resolve()
    external_roots = external_roots or (Path("/home/pacquadr/Desktop/Alex-robot"),)

    candidates: dict[str, list[AssetCandidate]] = {
        "usd": [],
        "urdf": [],
        "mjcf": [],
        "door_usd": [],
    }

    for path in _iter_files(repo_root, (".usd", ".usda", ".usdc")):
        lower = path.name.lower()
        if "alex" in lower:
            candidates["usd"].append(
                AssetCandidate("usd", str(path), "repo", "Alex-named USD candidate")
            )
        elif "door" in lower:
            candidates["door_usd"].append(
                AssetCandidate("door_usd", str(path), "repo", "Door USD candidate")
            )

    for path in _iter_files(repo_root / "sim" / "assets", (".urdf",)):
        candidates["urdf"].append(AssetCandidate("urdf", str(path), "repo", "Repo URDF"))

    for path in _iter_files(repo_root / "sim" / "assets", (".xml",)):
        if "alex" in path.name.lower() or "alex" in str(path.parent).lower():
            candidates["mjcf"].append(
                AssetCandidate("mjcf", str(path), "repo", "Repo Alex MuJoCo/MJCF asset")
            )

    for root in external_roots:
        if not root.exists():
            continue
        for path in _iter_files(root, (".usd", ".usda", ".usdc")):
            lower_name = path.name.lower()
            lower_rel = _relative_text(path, root).lower()
            if "alex" in lower_rel:
                candidates["usd"].append(
                    AssetCandidate("usd", str(path), str(root), "External Alex USD candidate")
                )
            elif "door" in lower_name:
                candidates["door_usd"].append(
                    AssetCandidate("door_usd", str(path), str(root), "External door USD candidate")
                )
        for path in _iter_files(root, (".urdf",)):
            if "alex" in _relative_text(path, root).lower():
                candidates["urdf"].append(
                    AssetCandidate("urdf", str(path), str(root), "External Alex URDF candidate")
                )
        for path in _iter_files(root, (".xml",)):
            if "alex" in _relative_text(path, root).lower():
                candidates["mjcf"].append(
                    AssetCandidate("mjcf", str(path), str(root), "External Alex MJCF candidate")
                )

    return {
        key: [item.to_dict() for item in _sort_candidates(value)]
        for key, value in candidates.items()
    }


def select_preferred_alex_asset(
    inventory: dict[str, list[dict[str, str]]],
) -> dict[str, str | None]:
    """Select the cleanest current Alex asset path by the required decision tree."""
    if inventory.get("usd"):
        candidate = inventory["usd"][0]
        return {
            "kind": "usd",
            "path": candidate["path"],
            "reason": "Existing Alex USD is preferred over conversion paths.",
        }
    if inventory.get("urdf"):
        candidate = inventory["urdf"][0]
        return {
            "kind": "urdf",
            "path": candidate["path"],
            "reason": "No Alex USD was found; URDF import is the next cleanest Isaac path.",
        }
    if inventory.get("mjcf"):
        candidate = inventory["mjcf"][0]
        return {
            "kind": "mjcf",
            "path": candidate["path"],
            "reason": "No Alex USD or URDF was found; MJCF conversion is the fallback path.",
        }
    return {
        "kind": None,
        "path": None,
        "reason": "No Alex USD, URDF, or MJCF candidates were found on disk.",
    }


def _iter_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    return [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def _sort_candidates(candidates: list[AssetCandidate]) -> list[AssetCandidate]:
    def key(candidate: AssetCandidate) -> tuple[int, str]:
        text = candidate.path.lower()
        score = 100
        if "fullbody_robotaccurate_torsofootcollisions" in text:
            score = 0
        elif "fullbody_robotaccurate_fullcollisions" in text:
            score = 1
        elif "full_body" in text or "fullbody" in text:
            score = 2
        elif "nubforearms" in text:
            score = 3
        return score, candidate.path

    return sorted(candidates, key=key)


def _relative_text(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
