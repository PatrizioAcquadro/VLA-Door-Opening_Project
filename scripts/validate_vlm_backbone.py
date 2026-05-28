#!/usr/bin/env python3
"""Validate VLM backbone end-to-end inference (Phase 3.1.3).

Runs 8 sequential checks verifying that the loaded Qwen3.5-4B backbone
produces valid logits, non-NaN hidden states, and coherent text generation
with actual or synthetic images.

Artifacts are written to logs/vlm_backbone/.

Usage::

    python scripts/validate_vlm_backbone.py
    python scripts/validate_vlm_backbone.py --model-config vlm_dev
    python scripts/validate_vlm_backbone.py --model-config vlm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _record(results: list[tuple[str, bool, str]], name: str, passed: bool, detail: str) -> None:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {name}: {detail}")
    results.append((name, passed, detail))


# ---------------------------------------------------------------------------
# Forward pass key filtering
# ---------------------------------------------------------------------------

_FORWARD_KEYS = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}


def _filter_forward_inputs(inputs: dict) -> dict:
    """Extract only the keys that backbone.forward() / get_hidden_states() accept."""
    return {k: v for k, v in inputs.items() if k in _FORWARD_KEYS}


# ---------------------------------------------------------------------------
# Image acquisition — sim images if MuJoCo available, else synthetic
# ---------------------------------------------------------------------------


def _get_test_images(n: int = 4) -> tuple[list[np.ndarray], str]:
    """Return test images and a label describing the source.

    Tries MuJoCo sim images first (alex_upper_body, rest keyframe).
    Falls back to synthetic 320x320 uint8 arrays (EO-1 eval_policy.py pattern).
    """
    try:
        import mujoco  # noqa: F401

        from sim.asset_loader import load_scene  # noqa: E402
        from sim.camera import CAMERA_NAMES, MultiViewRenderer  # noqa: E402

        model, data = load_scene("alex_upper_body")
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rest")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        with MultiViewRenderer(model) as mv:
            frame = mv.capture(data, step_index=0)
            names = list(CAMERA_NAMES)[:n]
            images = [frame.views[name].rgb for name in names]
            return images, "mujoco_sim"
    except (ImportError, Exception):
        pass

    images = [np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8) for _ in range(n)]
    return images, "synthetic"


# ---------------------------------------------------------------------------
# Contract summary
# ---------------------------------------------------------------------------

_EXPECTED_HIDDEN_SIZE = 2560
_EXPECTED_VOCAB_SIZE = 248320
_TEST_PROMPT = (
    "The robot is performing a LEGO assembly task. "
    "Describe what you see and what the robot should do next."
)


def _print_contract(model_config: str) -> None:
    print("=" * 70)
    print("VLM BACKBONE INFERENCE SANITY CHECK (Phase 3.1.3)")
    print("=" * 70)
    print(f"  Model config:    {model_config}")
    print("  Expected model:  Qwen/Qwen3.5-4B")
    print(f"  Hidden size:     {_EXPECTED_HIDDEN_SIZE}")
    print(f"  Vocab size:      {_EXPECTED_VOCAB_SIZE}")
    print("  Image resolution: 320x320 (frozen camera contract)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VLM backbone inference (Phase 3.1.3)")
    parser.add_argument(
        "--model-config",
        type=str,
        default="vlm_dev",
        help="Hydra model config name (default: vlm_dev)",
    )
    args = parser.parse_args()

    out_dir = _ROOT / "logs" / "vlm_backbone"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts -> {out_dir}/")

    _print_contract(args.model_config)

    results: list[tuple[str, bool, str]] = []
    output_tensors: list[tuple[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Check 1: Model loading
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 1: Model loading")
    print("-" * 70)
    try:
        from hydra import compose, initialize

        from models.vlm_backbone import load_vlm_backbone, verify_backbone  # noqa: E402

        with initialize(config_path="../configs", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[f"model={args.model_config}", "cluster=local"],
            )

        t0 = time.time()
        backbone = load_vlm_backbone(cfg)
        load_time = time.time() - t0

        verify_backbone(backbone)
        device = next(backbone._model.parameters()).device

        detail = (
            f"{backbone.info.param_count_total / 1e9:.2f}B params, "
            f"hidden_size={backbone.hidden_size}, "
            f"dtype={backbone.info.dtype}, "
            f"device={device}, "
            f"loaded in {load_time:.1f}s"
        )
        _record(results, "model_loading", True, detail)
    except Exception as e:
        _record(results, "model_loading", False, str(e))
        # Cannot proceed without backbone
        print("\n[ABORT] Cannot proceed without a loaded backbone.")
        return 1

    # ------------------------------------------------------------------
    # Check 2: Processor — image preprocessing
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 2: Processor — image preprocessing")
    print("-" * 70)
    try:
        from models.vlm_backbone import preprocess_images  # noqa: E402

        test_images, img_source = _get_test_images(1)
        print(f"  Image source: {img_source}")

        inputs_check2 = preprocess_images(
            backbone, test_images, "Describe what you see.", device=device
        )

        has_input_ids = "input_ids" in inputs_check2
        has_pixels = "pixel_values" in inputs_check2
        pixel_numel = inputs_check2["pixel_values"].numel() if has_pixels else 0
        seq_len = inputs_check2["input_ids"].shape[1] if has_input_ids else 0

        ok = has_input_ids and has_pixels and pixel_numel > 0
        detail = (
            f"input_ids shape={list(inputs_check2.get('input_ids', torch.tensor([])).shape)}, "
            f"pixel_values numel={pixel_numel}, "
            f"seq_len={seq_len}"
        )
        _record(results, "processor", ok, detail)
    except Exception as e:
        _record(results, "processor", False, str(e))

    # ------------------------------------------------------------------
    # Check 3: Text-only forward pass
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 3: Text-only forward pass")
    print("-" * 70)
    try:
        # Text-only: use tokenizer directly (no image in chat template)
        processor = backbone.processor
        tokenizer = getattr(processor, "tokenizer", processor)
        text_inputs = tokenizer("The robot is idle.", return_tensors="pt", padding=True)
        text_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in text_inputs.items()
        }

        with torch.no_grad():
            outputs_text = backbone.forward(**_filter_forward_inputs(text_inputs))

        logits = outputs_text["logits"]
        ok = (
            logits.ndim == 3
            and logits.shape[-1] == _EXPECTED_VOCAB_SIZE
            and not torch.isnan(logits.float()).any()
        )
        detail = (
            f"logits shape={list(logits.shape)}, no_nan={not torch.isnan(logits.float()).any()}"
        )
        _record(results, "text_only_forward", ok, detail)
        output_tensors.append(("text_only_logits", logits.detach()))
    except Exception as e:
        _record(results, "text_only_forward", False, str(e))

    # ------------------------------------------------------------------
    # Check 4: Single-image forward pass
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 4: Single-image forward pass")
    print("-" * 70)
    try:
        single_images, _ = _get_test_images(1)
        inputs_single = preprocess_images(
            backbone, single_images, "Describe the robot workspace.", device=device
        )

        with torch.no_grad():
            outputs_single = backbone.forward(**_filter_forward_inputs(inputs_single))

        logits = outputs_single["logits"]
        ok = logits.ndim == 3 and logits.shape[-1] == _EXPECTED_VOCAB_SIZE
        detail = f"logits shape={list(logits.shape)}"
        _record(results, "single_image_forward", ok, detail)
        output_tensors.append(("single_image_logits", logits.detach()))
    except Exception as e:
        _record(results, "single_image_forward", False, str(e))

    # ------------------------------------------------------------------
    # Check 5: Multi-view forward pass (4 images)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 5: Multi-view forward pass (4 images)")
    print("-" * 70)
    try:
        quad_images, img_source = _get_test_images(4)
        print(f"  Image source: {img_source}")

        inputs_quad = preprocess_images(backbone, quad_images, _TEST_PROMPT, device=device)

        with torch.no_grad():
            outputs_quad = backbone.forward(**_filter_forward_inputs(inputs_quad))

        logits = outputs_quad["logits"]
        ok = logits.ndim == 3 and logits.shape[-1] == _EXPECTED_VOCAB_SIZE
        seq_len_quad = logits.shape[1]
        detail = f"logits shape={list(logits.shape)}, seq_len={seq_len_quad}"
        _record(results, "multi_view_forward", ok, detail)
        output_tensors.append(("multi_view_logits", logits.detach()))
    except Exception as e:
        _record(results, "multi_view_forward", False, str(e))

    # ------------------------------------------------------------------
    # Check 6: Hidden state extraction
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 6: Hidden state extraction")
    print("-" * 70)
    try:
        # Reuse quad inputs from check 5
        with torch.no_grad():
            hidden = backbone.get_hidden_states(**_filter_forward_inputs(inputs_quad))

        shape_ok = hidden.ndim == 3 and hidden.shape[-1] == _EXPECTED_HIDDEN_SIZE
        dtype_ok = hidden.dtype == torch.bfloat16
        ok = shape_ok and dtype_ok

        shape_info = {
            "shape": list(hidden.shape),
            "dtype": str(hidden.dtype),
            "hidden_size": hidden.shape[-1],
            "seq_len": hidden.shape[1],
        }
        (out_dir / "hidden_states_shape.json").write_text(json.dumps(shape_info, indent=2) + "\n")

        detail = (
            f"shape={list(hidden.shape)}, dtype={hidden.dtype}, "
            f"expected hidden_size={_EXPECTED_HIDDEN_SIZE}"
        )
        _record(results, "hidden_state_extraction", ok, detail)
        output_tensors.append(("hidden_states", hidden.detach()))
    except Exception as e:
        _record(results, "hidden_state_extraction", False, str(e))

    # ------------------------------------------------------------------
    # Check 7: Text generation
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 7: Text generation")
    print("-" * 70)
    try:
        gen_images, _ = _get_test_images(1)
        gen_inputs = preprocess_images(backbone, gen_images, _TEST_PROMPT, device=device)

        with torch.no_grad():
            generated_ids = backbone._model.generate(
                **gen_inputs, max_new_tokens=50, do_sample=False
            )

        # Validate token IDs
        vocab_size = backbone.info.vocab_size
        ids_valid = (generated_ids >= 0).all() and (generated_ids < vocab_size).all()

        # Decode
        generated_text = backbone.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text_nonempty = len(generated_text.strip()) > 0

        ok = ids_valid and text_nonempty

        # Save generation artifact
        gen_artifact = (
            f"Prompt: {_TEST_PROMPT}\n\n"
            f"Generated ({generated_ids.shape[1]} tokens):\n{generated_text}\n\n"
            f"Token IDs valid: {ids_valid}\n"
        )
        (out_dir / "sample_generation.txt").write_text(gen_artifact)

        detail = (
            f"tokens={generated_ids.shape[1]}, "
            f"text_len={len(generated_text.strip())}, "
            f"ids_valid={bool(ids_valid)}"
        )
        _record(results, "text_generation", ok, detail)
    except Exception as e:
        _record(results, "text_generation", False, str(e))

    # ------------------------------------------------------------------
    # Check 8: Numerical sanity (all outputs)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 8: Numerical sanity (all output tensors)")
    print("-" * 70)
    try:
        all_clean = True
        nan_details = []
        for name, tensor in output_tensors:
            t_float = tensor.float()
            has_nan = bool(torch.isnan(t_float).any())
            has_inf = bool(torch.isinf(t_float).any())
            if has_nan or has_inf:
                all_clean = False
                nan_details.append(f"{name}: nan={has_nan}, inf={has_inf}")

        if all_clean:
            detail = f"All {len(output_tensors)} tensors clean (no NaN/Inf)"
        else:
            detail = f"Issues found: {'; '.join(nan_details)}"
        _record(results, "numerical_sanity", all_clean, detail)
    except Exception as e:
        _record(results, "numerical_sanity", False, str(e))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    n_pass = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    print(f"RESULT: {n_pass}/{n_total} checks passed")
    print("=" * 70)

    # Save JSON report
    report = {
        "phase": "3.1.3",
        "model_config": args.model_config,
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in results],
        "summary": {"passed": n_pass, "total": n_total, "all_passed": n_pass == n_total},
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report saved to {out_dir / 'validation_report.json'}")

    # Clean up GPU memory
    torch.cuda.empty_cache()

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
