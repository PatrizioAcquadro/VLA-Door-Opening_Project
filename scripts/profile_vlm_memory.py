#!/usr/bin/env python3
"""Profile VLM backbone VRAM consumption (Phase 3.1.4).

Auto-detects GPU and selects an appropriate sweep:
  - <=32 GB (e.g. RTX 4090 24 GB): 7 configs, max seq=4096, max batch=2 training
  - >32 GB  (e.g. A100 80 GB):    11 configs, adds seq=8192 + batch=4 training

Artifacts are written to logs/vlm_memory/.

Usage::

    python scripts/profile_vlm_memory.py                          # full auto-detect sweep
    python scripts/profile_vlm_memory.py --quick                  # 2 configs only
    python scripts/profile_vlm_memory.py --model-config vlm       # production config (A100)
    python scripts/profile_vlm_memory.py --output-dir /tmp/prof   # custom output dir
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProfileConfig:
    """A single profiling configuration to sweep."""

    seq_len: int
    batch_size: int
    mode: str  # "inference" or "training"
    n_images: int
    label: str


@dataclass
class ProfileResult:
    """Result from a single profiling run."""

    seq_len: int
    batch_size: int
    mode: str
    n_images: int
    peak_vram_gb: float
    remaining_vram_gb: float
    wall_time_ms: float
    kv_cache_estimate_gb: float
    status: str  # "ok" or "OOM"


# ---------------------------------------------------------------------------
# GPU detection and sweep construction
# ---------------------------------------------------------------------------


def _get_gpu_info() -> dict:
    """Return GPU name and total memory."""
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": round(props.total_memory / (1024**3), 1),
    }


def _build_sweep(total_mem_gb: float, quick: bool = False) -> list[ProfileConfig]:
    """Build sweep configs based on available GPU memory."""
    if quick:
        return [
            ProfileConfig(2048, 1, "inference", 4, "Quick inference"),
            ProfileConfig(2048, 1, "training", 4, "Quick training"),
        ]

    # Base sweep — fits on 4090-class GPUs (<=32 GB)
    configs = [
        ProfileConfig(1024, 1, "inference", 0, "Text-only baseline"),
        ProfileConfig(1024, 1, "inference", 4, "Quad-view inference"),
        ProfileConfig(2048, 1, "inference", 4, "Typical VLA sequence"),
        ProfileConfig(4096, 1, "inference", 4, "Max dev context"),
        ProfileConfig(2048, 1, "training", 4, "Training baseline"),
        ProfileConfig(2048, 2, "training", 4, "Training batch=2"),
        ProfileConfig(4096, 1, "training", 4, "Long training"),
    ]

    # Extended sweep for A100-class GPUs (>32 GB)
    if total_mem_gb > 32:
        configs.extend(
            [
                ProfileConfig(8192, 1, "inference", 4, "Max context"),
                ProfileConfig(2048, 4, "training", 4, "Training batch=4"),
                ProfileConfig(4096, 2, "training", 4, "Long training batch=2"),
                ProfileConfig(4096, 4, "training", 4, "Long training batch=4"),
            ]
        )

    return configs


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

_FORWARD_KEYS = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}


def _make_profiling_inputs(
    backbone: object, config: ProfileConfig, device: torch.device
) -> dict[str, torch.Tensor]:
    """Create dummy inputs at a specific (seq_len, batch_size, n_images) configuration.

    For image configs: preprocesses synthetic images through the real processor,
    then pads input_ids to the target seq_len and replicates across batch.
    For text-only configs: creates random token IDs at the target shape.
    """
    from models.vlm_backbone import preprocess_images

    if config.n_images > 0:
        images = [
            np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8) for _ in range(config.n_images)
        ]
        base = preprocess_images(backbone, images, "x", device=device)

        base_seq = base["input_ids"].shape[1]

        # Pad or truncate to target seq_len
        if config.seq_len > base_seq:
            tokenizer = getattr(backbone.processor, "tokenizer", backbone.processor)
            pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
            pad_len = config.seq_len - base_seq
            pad_ids = torch.full((1, pad_len), pad_id, dtype=base["input_ids"].dtype, device=device)
            input_ids = torch.cat([base["input_ids"], pad_ids], dim=1)
            attn_pad = torch.zeros(1, pad_len, dtype=base["attention_mask"].dtype, device=device)
            attention_mask = torch.cat([base["attention_mask"], attn_pad], dim=1)
        else:
            input_ids = base["input_ids"][:, : config.seq_len]
            attention_mask = base["attention_mask"][:, : config.seq_len]

        # Expand to batch_size
        inputs: dict[str, torch.Tensor] = {
            "input_ids": input_ids.expand(config.batch_size, -1).contiguous(),
            "attention_mask": attention_mask.expand(config.batch_size, -1).contiguous(),
        }

        # Replicate pixel_values and image_grid_thw across batch
        pv = base["pixel_values"]
        inputs["pixel_values"] = pv.repeat(config.batch_size, *([1] * (pv.ndim - 1)))
        if "image_grid_thw" in base:
            inputs["image_grid_thw"] = base["image_grid_thw"].repeat(config.batch_size, 1)

        return inputs
    else:
        # Text-only: random token IDs
        input_ids = torch.randint(0, 1000, (config.batch_size, config.seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# KV cache analytical estimate
# ---------------------------------------------------------------------------


def _compute_kv_cache_gb(model_config: object, seq_len: int, batch_size: int) -> float:
    """Analytical KV cache estimate in GB.

    Formula: 2 (K+V) * num_layers * batch * seq * num_kv_heads * head_dim * dtype_bytes
    """
    tc = model_config.text_config
    num_layers = tc.num_hidden_layers
    num_kv_heads = getattr(tc, "num_key_value_heads", tc.num_attention_heads)
    head_dim = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
    dtype_bytes = 2  # bf16

    kv_bytes = 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * dtype_bytes
    return kv_bytes / (1024**3)


# ---------------------------------------------------------------------------
# Single profile execution
# ---------------------------------------------------------------------------


def _run_single_profile(
    backbone: object,
    config: ProfileConfig,
    device: torch.device,
    total_mem_gb: float,
) -> ProfileResult:
    """Run a single profiling configuration. Returns ProfileResult."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        inputs = _make_profiling_inputs(backbone, config, device)
        forward_inputs = {k: v for k, v in inputs.items() if k in _FORWARD_KEYS}

        if config.mode == "inference":
            with torch.no_grad():
                t0 = time.perf_counter()
                _ = backbone.forward(**forward_inputs)
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - t0) * 1000
        else:
            # Training: unfreeze for gradient computation
            backbone.unfreeze_backbone()
            t0 = time.perf_counter()
            out = backbone.forward(**forward_inputs)
            loss = out["logits"].float().sum()
            loss.backward()
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - t0) * 1000
            del out, loss
            backbone.freeze_backbone()
            backbone._model.zero_grad(set_to_none=True)

        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        remaining_gb = total_mem_gb - peak_gb
        kv_gb = _compute_kv_cache_gb(backbone._model.config, config.seq_len, config.batch_size)

        del inputs, forward_inputs
        gc.collect()
        torch.cuda.empty_cache()

        return ProfileResult(
            config.seq_len,
            config.batch_size,
            config.mode,
            config.n_images,
            round(peak_gb, 2),
            round(remaining_gb, 2),
            round(wall_ms, 1),
            round(kv_gb, 3),
            "ok",
        )

    except torch.cuda.OutOfMemoryError:
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        remaining_gb = total_mem_gb - peak_gb
        try:
            backbone.freeze_backbone()
            backbone._model.zero_grad(set_to_none=True)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        return ProfileResult(
            config.seq_len,
            config.batch_size,
            config.mode,
            config.n_images,
            round(peak_gb, 2),
            round(remaining_gb, 2),
            0.0,
            0.0,
            "OOM",
        )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format_table(results: list[ProfileResult], gpu_info: dict, weight_vram_gb: float) -> str:
    """Format results as ASCII table."""
    lines: list[str] = []
    name = gpu_info["name"]
    total = gpu_info["total_memory_gb"]

    lines.append(f"VLM BACKBONE MEMORY PROFILE: Qwen/Qwen3.5-4B (bf16) on {name} {total:.0f}GB")
    lines.append("=" * 92)
    lines.append(f"Weight-only VRAM baseline: {weight_vram_gb:.1f} GB")
    lines.append("")

    header = (
        f"{'Seq Len':>8} | {'Batch':>5} | {'Mode':>10} | {'Images':>6} "
        f"| {'Peak VRAM':>10} | {'Remaining':>10} | {'Time':>9} | {'Status':>6}"
    )
    lines.append(header)
    lines.append("-" * 92)

    for r in results:
        peak = f"{r.peak_vram_gb:.1f} GB"
        remaining = f"{r.remaining_vram_gb:.1f} GB" if r.status == "ok" else "---"
        wall = f"{r.wall_time_ms:.0f} ms" if r.status == "ok" else "---"
        lines.append(
            f"{r.seq_len:>8} | {r.batch_size:>5} | {r.mode:>10} | {r.n_images:>6} "
            f"| {peak:>10} | {remaining:>10} | {wall:>9} | {r.status:>6}"
        )

    lines.append("=" * 92)

    # Action head budget
    training_ok = [r for r in results if r.mode == "training" and r.status == "ok"]
    if training_ok:
        lines.append("")
        lines.append("ACTION HEAD BUDGET (training mode, 4 images, bf16):")
        for r in training_ok:
            lines.append(
                f"  seq={r.seq_len}, batch={r.batch_size}: "
                f"{r.remaining_vram_gb:.1f} GB remaining"
            )

    # KV cache analysis
    ok_results = [r for r in results if r.status == "ok"]
    if ok_results:
        lines.append("")
        lines.append("KV CACHE ANALYSIS (analytical estimate):")
        for r in ok_results:
            lines.append(
                f"  seq={r.seq_len}, batch={r.batch_size}: " f"{r.kv_cache_estimate_gb:.3f} GB"
            )

    # Notes
    lines.append("")
    lines.append("NOTES:")
    lines.append("  - Training mode measures full backbone gradient computation (worst case)")
    lines.append("  - Does NOT include optimizer states (Adam adds ~2x model size = ~16 GB bf16)")
    lines.append("  - Action head memory is additional to these numbers")
    lines.append("  - LoRA fine-tuning uses less memory than full backbone backward")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile VLM backbone VRAM consumption (Phase 3.1.4)"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="vlm_dev",
        help="Hydra model config name (default: vlm_dev)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 2 configs only (inference + training at seq=2048, batch=1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: logs/vlm_memory/)",
    )
    args = parser.parse_args()

    # --- GPU check ---
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. Memory profiling requires a GPU.")
        return 1

    gpu_info = _get_gpu_info()
    total_mem_gb = gpu_info["total_memory_gb"]
    tier = "A100+" if total_mem_gb > 32 else "4090-class"

    out_dir = Path(args.output_dir) if args.output_dir else _ROOT / "logs" / "vlm_memory"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {gpu_info['name']} ({total_mem_gb:.0f} GB)")
    print(f"Tier: {tier}")
    print(f"Artifacts -> {out_dir}/")

    # --- Build sweep ---
    configs = _build_sweep(total_mem_gb, quick=args.quick)
    print(f"Sweep: {len(configs)} configurations")
    print()

    # --- Load backbone ---
    print("Loading backbone...")
    from hydra import compose, initialize

    from models.vlm_backbone import load_vlm_backbone

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"model={args.model_config}", "cluster=local"],
        )

    t0 = time.time()
    backbone = load_vlm_backbone(cfg)
    load_time = time.time() - t0
    device = next(backbone._model.parameters()).device

    # Weight-only VRAM baseline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Force a small allocation to refresh stats
    _ = torch.empty(1, device=device)
    weight_vram_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)

    print(f"Backbone loaded in {load_time:.1f}s")
    print(f"Weight-only VRAM: {weight_vram_gb:.1f} GB")
    print(f"Device: {device}")
    print()

    # --- Warmup ---
    print("Warmup forward pass...")
    warmup_ids = torch.randint(0, 1000, (1, 64), device=device)
    with torch.no_grad():
        _ = backbone.forward(input_ids=warmup_ids, attention_mask=torch.ones_like(warmup_ids))
    del warmup_ids
    gc.collect()
    torch.cuda.empty_cache()
    print()

    # --- Run sweep ---
    print("Running memory sweep:")
    results: list[ProfileResult] = []
    for i, config in enumerate(configs):
        label = (
            f"[{i + 1}/{len(configs)}] seq={config.seq_len}, "
            f"batch={config.batch_size}, {config.mode}, {config.n_images} img"
        )
        print(f"  {label} ... ", end="", flush=True)
        result = _run_single_profile(backbone, config, device, total_mem_gb)
        if result.status == "ok":
            print(f"peak={result.peak_vram_gb:.1f} GB, {result.wall_time_ms:.0f} ms")
        else:
            print(f"OOM (peak before crash: {result.peak_vram_gb:.1f} GB)")
        results.append(result)

    # --- Format and print table ---
    print()
    table = _format_table(results, gpu_info, weight_vram_gb)
    print(table)

    # --- Save artifacts ---
    # 1. JSON report
    report = {
        "phase": "3.1.4",
        "gpu": gpu_info,
        "tier": tier,
        "model_config": args.model_config,
        "weight_only_vram_gb": weight_vram_gb,
        "load_time_s": round(load_time, 1),
        "results": [asdict(r) for r in results],
    }
    (out_dir / "memory_profile.json").write_text(json.dumps(report, indent=2) + "\n")

    # 2. ASCII table
    (out_dir / "memory_table.txt").write_text(table + "\n")

    # 3. Action head budget
    budget: dict[str, dict] = {}
    for r in results:
        if r.mode == "training" and r.status == "ok":
            key = f"seq{r.seq_len}_batch{r.batch_size}"
            budget[key] = {
                "peak_backbone_vram_gb": r.peak_vram_gb,
                "remaining_vram_gb": r.remaining_vram_gb,
                "total_gpu_memory_gb": total_mem_gb,
            }
    (out_dir / "action_head_budget.json").write_text(json.dumps(budget, indent=2) + "\n")

    print()
    print(f"Artifacts saved to {out_dir}/")
    print(f"  memory_profile.json     — full results ({len(results)} configs)")
    print("  memory_table.txt        — formatted table")
    print("  action_head_budget.json — training VRAM budget")

    torch.cuda.empty_cache()

    n_ok = sum(1 for r in results if r.status == "ok")
    n_oom = len(results) - n_ok
    print(f"\n{n_ok}/{len(results)} configurations completed ({n_oom} OOM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
