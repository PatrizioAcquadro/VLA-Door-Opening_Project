"""Dataset classes for VLA door-opening data."""

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

REQUIRED_VLA_FIELDS = (
    "input_ids",
    "attention_mask",
    "robot_states",
    "action_chunks",
    "chunk_masks",
    "text_labels",
)


class SimulationDataset(Dataset):
    """Manifest-backed dataset for door-opening VLA training episodes.

    The manifest must live at ``<data_path>/<split>_manifest.json`` and contain
    a list of records. Each record can either contain a ``path`` field pointing
    to a ``.pt``/``.pth``/``.npz`` sample file, or inline tensor-like fields.

    Required sample fields match ``models.vla_model.VLAModel.forward``:
    ``input_ids``, ``attention_mask``, ``robot_states``, ``action_chunks``,
    ``chunk_masks``, and ``text_labels``. Optional fields such as
    ``pixel_values``, ``image_grid_thw``, and ``door_metrics`` are preserved.

    Args:
        data_path: Path to processed data directory
        max_length: Maximum sequence length
        split: One of "train", "val", "test"
    """

    def __init__(
        self,
        data_path: str | Path,
        max_length: int = 1024,
        split: str = "train",
    ) -> None:
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.split = split

        # Load data index/manifest
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load data manifest/index."""
        manifest_path = self.data_path / f"{self.split}_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {self.split} manifest: {manifest_path}. "
                "Door-opening training requires recorded VLA episodes; use "
                'dataset.name="dummy" only for explicit smoke tests.'
            )

        with open(manifest_path) as f:
            samples = json.load(f)

        if not isinstance(samples, list):
            raise ValueError(f"Manifest must be a list of records: {manifest_path}")
        if not samples:
            raise ValueError(f"Manifest contains no samples: {manifest_path}")

        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and validate a single VLA sample."""
        record = self._samples[idx]
        sample = self._load_record(record)
        return self._normalize_sample(sample)

    def _load_record(self, record: Any) -> dict[str, Any]:
        """Load a sample from a manifest record."""
        if not isinstance(record, dict):
            raise ValueError(f"Manifest record must be a dict, got {type(record).__name__}")

        if "path" not in record:
            return dict(record)

        sample_path = Path(record["path"])
        if not sample_path.is_absolute():
            sample_path = self.data_path / sample_path
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")

        if sample_path.suffix in {".pt", ".pth"}:
            loaded = torch.load(sample_path, map_location="cpu")
        elif sample_path.suffix == ".npz":
            import numpy as np

            with np.load(sample_path, allow_pickle=False) as npz:
                loaded = {key: npz[key] for key in npz.files}
        else:
            raise ValueError(f"Unsupported sample file format: {sample_path.suffix}")

        if not isinstance(loaded, dict):
            raise ValueError(f"Sample file must contain a dict: {sample_path}")
        return loaded

    def _normalize_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Convert sample values to tensors and verify the VLA batch contract."""
        missing = [field for field in REQUIRED_VLA_FIELDS if field not in sample]
        if missing:
            raise KeyError(f"VLA sample missing required field(s): {missing}")

        input_ids = _as_tensor(sample["input_ids"], torch.long)
        raw_attention_mask = _as_tensor(sample["attention_mask"], torch.long)
        robot_states = _as_tensor(sample["robot_states"], torch.float32)
        action_chunks = _as_tensor(sample["action_chunks"], torch.float32)
        chunk_masks = _as_tensor(sample["chunk_masks"], torch.float32)
        text_labels = _as_tensor(sample["text_labels"], torch.long)

        if input_ids.ndim != 1:
            raise ValueError("input_ids must be 1-D per sample")
        if text_labels.ndim != 1:
            raise ValueError("text_labels must be 1-D per sample")
        if raw_attention_mask.ndim != 1:
            raise ValueError("attention_mask must be 1-D per sample")
        if input_ids.shape != text_labels.shape:
            raise ValueError("input_ids and text_labels must have matching lengths")
        if robot_states.ndim != 2:
            raise ValueError("robot_states must be 2-D per sample")
        if action_chunks.ndim not in {2, 3}:
            raise ValueError("action_chunks must be 2-D or 3-D per sample")

        original_text_len = input_ids.shape[0]
        seq_len = min(self.max_length, original_text_len)
        input_ids = input_ids[:seq_len]
        text_labels = text_labels[:seq_len]

        if robot_states.shape[-1] != 52:
            raise ValueError("robot_states last dimension must be 52")
        if action_chunks.shape[-1] != 17:
            raise ValueError("action_chunks last dimension must be 17")
        if action_chunks.ndim == 3:
            n_action_tokens = action_chunks.shape[0] * action_chunks.shape[1]
        else:
            n_action_tokens = action_chunks.shape[0]
        if chunk_masks.numel() != n_action_tokens:
            raise ValueError(
                "chunk_masks must contain one mask value per action token; "
                f"got {chunk_masks.numel()} for {n_action_tokens} action tokens"
            )

        n_state_tokens = robot_states.shape[0]
        attention_mask = _normalize_attention_mask(
            raw_attention_mask,
            original_text_len=original_text_len,
            seq_len=seq_len,
            n_state_tokens=n_state_tokens,
            n_action_tokens=n_action_tokens,
        )

        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "robot_states": robot_states,
            "action_chunks": action_chunks,
            "chunk_masks": chunk_masks,
            "text_labels": text_labels,
        }

        if "pixel_values" in sample and sample["pixel_values"] is not None:
            out["pixel_values"] = _as_tensor(sample["pixel_values"], torch.float32)
        if "image_grid_thw" in sample and sample["image_grid_thw"] is not None:
            out["image_grid_thw"] = _as_tensor(sample["image_grid_thw"], torch.long)
        if "door_metrics" in sample:
            out["door_metrics"] = sample["door_metrics"]

        return out


class DummyDataset(Dataset):
    """Explicit dummy dataset for transformer smoke tests only."""

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 512,
        state_dim: int = 256,
    ) -> None:
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.state_dim = state_dim

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.randn(self.seq_length, self.state_dim),
            "labels": torch.randn(self.seq_length, self.state_dim),
            "attention_mask": torch.ones(self.seq_length, dtype=torch.long),
        }


def _as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Convert tensor-like values to a CPU tensor with the requested dtype."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _normalize_attention_mask(
    attention_mask: torch.Tensor,
    *,
    original_text_len: int,
    seq_len: int,
    n_state_tokens: int,
    n_action_tokens: int,
) -> torch.Tensor:
    """Normalize per-sample attention masks to the full VLA sequence length."""
    seq_total = seq_len + n_state_tokens + n_action_tokens
    original_seq_total = original_text_len + n_state_tokens + n_action_tokens

    if attention_mask.shape[0] == seq_total:
        return attention_mask

    if attention_mask.shape[0] == original_seq_total:
        return torch.cat(
            [
                attention_mask[:seq_len],
                attention_mask[original_text_len:original_seq_total],
            ]
        )

    if attention_mask.shape[0] == original_text_len:
        return torch.cat(
            [
                attention_mask[:seq_len],
                torch.ones(n_state_tokens + n_action_tokens, dtype=torch.long),
            ]
        )

    raise ValueError(
        "attention_mask length must match text length or full VLA sequence length; "
        f"got {attention_mask.shape[0]}, expected one of "
        f"{original_text_len}, {original_seq_total}, {seq_total}"
    )
