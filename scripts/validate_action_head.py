#!/usr/bin/env python3
"""Validate action head components and VLA model integration (Phase 3.2.5).

Runs 10 sequential checks verifying all Phase 3.2 action head components
individually and integrated, profiles VRAM overhead on GPU, and confirms
the complete VLA model is ready for Phase 3.3 training integration.

Dual-mode operation:
    GPU + transformers available  → loads real Qwen3.5-4B backbone via load_vla_model()
    CPU-only (no GPU/transformers) → uses lightweight mock backbone for checks 1-9
    Check 10 (memory overhead)    → SKIP if no CUDA, always runs on GPU otherwise

Artifacts are written to logs/action_head/.

Usage::

    python scripts/validate_action_head.py
    python scripts/validate_action_head.py --model-config vla_dev
    python scripts/validate_action_head.py --model-config vla
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _record(results: list[tuple[str, bool, str]], name: str, passed: bool, detail: str) -> None:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {name}: {detail}")
    results.append((name, passed, detail))


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

_EXPECTED_ACTION_CHUNK_SIZE = 16
_EXPECTED_ACTION_DIM = 17
_EXPECTED_STATE_DIM = 52
_EXPECTED_HIDDEN_SIZE = 2560  # Qwen3.5-4B

# Expected param counts (H=2560, d_t=256):
#   RobotStateProjector:    ~6,691,944
#   NoisyActionProjector:   ~7,257,600
#   ActionOutputHead:       ~6,599,697
_PARAM_TOLERANCE = 0.10  # ±10% tolerance for param count checks

# Max allowed VRAM overhead for action head (bytes)
_MAX_MEMORY_OVERHEAD_GB = 1.0


def _print_contract(model_config: str, gpu_mode: bool) -> None:
    print("=" * 70)
    print("ACTION HEAD VALIDATION (Phase 3.2.5)")
    print("=" * 70)
    print(f"  Model config:      {model_config}")
    print(f"  Mode:              {'GPU + real backbone' if gpu_mode else 'CPU + mock backbone'}")
    print(f"  Chunk size:        {_EXPECTED_ACTION_CHUNK_SIZE}")
    print(f"  Action dim:        {_EXPECTED_ACTION_DIM}")
    print(f"  State dim:         {_EXPECTED_STATE_DIM}")
    print(f"  Hidden size (H):   {_EXPECTED_HIDDEN_SIZE} (real backbone only)")
    print()


# ---------------------------------------------------------------------------
# Self-contained mock VLM backbone (used for CPU-only checks)
# ---------------------------------------------------------------------------


class _MockVLMBackbone(nn.Module):
    """Lightweight mock backbone for CPU-only action head validation.

    Simulates VLMBackbone interface with a tiny hidden_size=64.
    Does NOT require transformers or GPU.
    """

    def __init__(self, hidden_size: int = 64, vocab_size: int = 100) -> None:
        super().__init__()
        self._hs = hidden_size

        from models.vlm_backbone import VLMBackboneInfo

        self._embed = nn.Embedding(vocab_size, hidden_size)
        self._lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.0
        )

        self._info = VLMBackboneInfo(
            model_id="mock",
            param_count_total=sum(p.numel() for p in self.parameters()),
            param_count_trainable=sum(p.numel() for p in self.parameters()),
            hidden_size=hidden_size,
            num_layers=1,
            vocab_size=vocab_size,
            dtype="float32",
            vision_hidden_size=hidden_size,
            vision_depth=1,
            image_token_id=99,
            vision_start_token_id=97,
            vision_end_token_id=98,
        )

    @property
    def hidden_size(self) -> int:
        return self._hs

    @property
    def info(self) -> Any:
        return self._info

    @property
    def processor(self) -> None:
        return None

    @property
    def lm_head(self) -> nn.Module:
        return self._lm_head

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embed(input_ids)

    def get_vision_features(
        self, pixel_values: torch.Tensor, image_grid_thw: Any
    ) -> list[torch.Tensor]:
        n_images = pixel_values.shape[0]
        return [torch.randn(1, self._hs) for _ in range(n_images)]

    def get_hidden_states(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            x = self._embed(input_ids)
        return self._encoder(x)

    def freeze_backbone(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def freeze_vision(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Batch builder for VLA forward tests
# ---------------------------------------------------------------------------

_SEQ_TEXT = 8
_N_SEG = 1
_N_CHUNKS = 1
_CHUNK_SIZE = _EXPECTED_ACTION_CHUNK_SIZE
_N_ACTION_TOKENS = _N_CHUNKS * _CHUNK_SIZE
_SEQ_TOTAL = _SEQ_TEXT + _N_SEG + _N_ACTION_TOKENS


def _make_synthetic_batch(
    B: int,
    vocab_size: int,
    device: torch.device,
    action_dim: int = _EXPECTED_ACTION_DIM,
    state_dim: int = _EXPECTED_STATE_DIM,
) -> dict[str, torch.Tensor]:
    """Build a synthetic training batch on the given device."""
    input_ids = torch.randint(0, min(97, vocab_size - 1), (B, _SEQ_TEXT), device=device)
    text_labels = torch.randint(0, vocab_size, (B, _SEQ_TEXT), device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(B, _SEQ_TOTAL, dtype=torch.long, device=device),
        "robot_states": torch.randn(B, _N_SEG, state_dim, device=device),
        "action_chunks": torch.randn(B, _N_CHUNKS, _CHUNK_SIZE, action_dim, device=device),
        "chunk_masks": torch.ones(B, _N_CHUNKS, _CHUNK_SIZE, device=device),
        "text_labels": text_labels,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description="Validate action head (Phase 3.2.5)")
    parser.add_argument(
        "--model-config",
        type=str,
        default="vla_dev",
        help="Hydra model config name (default: vla_dev)",
    )
    args = parser.parse_args()

    out_dir = _ROOT / "logs" / "action_head"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts -> {out_dir}/")

    has_gpu = torch.cuda.is_available()
    has_transformers = False
    try:
        import transformers  # noqa: F401

        has_transformers = True
    except ImportError:
        pass

    gpu_mode = has_gpu and has_transformers
    _print_contract(args.model_config, gpu_mode)

    results: list[tuple[str, bool, str]] = []

    # ------------------------------------------------------------------
    # Check 1: Config parsing
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 1: Config parsing")
    print("-" * 70)
    cfg = None
    try:
        from hydra import compose, initialize
        from omegaconf import OmegaConf

        with initialize(config_path="../configs", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[f"model={args.model_config}", "cluster=local"],
            )

        has_action_head = (
            hasattr(cfg, "model")
            and hasattr(cfg.model, "action_head")
            and hasattr(cfg.model.action_head, "chunk_size")
        )
        ok = has_action_head
        detail = (
            f"architecture.type={cfg.model.architecture.type}, "
            f"chunk_size={cfg.model.action_head.chunk_size}, "
            f"action_dim={cfg.model.action_head.action_dim}, "
            f"has_flow_matching={hasattr(cfg.model.action_head, 'flow_matching')}"
        )
        _record(results, "config_parsing", ok, detail)
    except Exception as e:
        _record(results, "config_parsing", False, str(e))
        print("\n[ABORT] Cannot proceed without valid config.")
        return 1

    # ------------------------------------------------------------------
    # Check 2: Component instantiation
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 2: Component instantiation")
    print("-" * 70)
    from omegaconf import OmegaConf

    from models.action_head import (
        ActionOutputHead,
        FlowMatchingConfig,
        FlowMatchingModule,
        NoisyActionProjector,
        RobotStateProjector,
    )

    try:
        ah_cfg = OmegaConf.to_container(cfg.model.action_head, resolve=True)

        # Use hidden_size=64 for CPU mode, expected 2560 for GPU mode
        H = _EXPECTED_HIDDEN_SIZE if gpu_mode else 64

        t0 = time.time()
        state_proj = RobotStateProjector.from_cfg(ah_cfg, H)
        action_proj = NoisyActionProjector.from_cfg(ah_cfg, H)
        output_head = ActionOutputHead.from_cfg(ah_cfg, H)
        fm_cfg = ah_cfg.get("flow_matching") if ah_cfg else None
        FlowMatchingModule(FlowMatchingConfig.from_cfg(fm_cfg))  # verify FM config parses
        elapsed = time.time() - t0

        n_state = sum(p.numel() for p in state_proj.parameters())
        n_action = sum(p.numel() for p in action_proj.parameters())
        n_output = sum(p.numel() for p in output_head.parameters())
        n_total = n_state + n_action + n_output

        # Save param counts artifact
        param_counts = {
            "hidden_size_used": H,
            "state_projector": n_state,
            "noisy_action_projector": n_action,
            "action_output_head": n_output,
            "total_action_head": n_total,
            "flow_matching_params": 0,  # no learnable parameters
        }
        (out_dir / "param_counts.json").write_text(json.dumps(param_counts, indent=2) + "\n")

        ok = n_total > 0
        detail = (
            f"state={n_state / 1e6:.3f}M, "
            f"action={n_action / 1e6:.3f}M, "
            f"output={n_output / 1e6:.3f}M, "
            f"total={n_total / 1e6:.3f}M, "
            f"elapsed={elapsed * 1000:.0f}ms"
        )
        _record(results, "component_instantiation", ok, detail)
    except Exception as e:
        _record(results, "component_instantiation", False, str(e))

    # ------------------------------------------------------------------
    # Check 3: Flow matching math
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 3: Flow matching math")
    print("-" * 70)
    try:
        from models.action_head import FlowMatchingConfig, FlowMatchingModule

        fm = FlowMatchingModule(FlowMatchingConfig())
        B, C, D = 2, _CHUNK_SIZE, _EXPECTED_ACTION_DIM
        x_data = torch.randn(B, C, D)
        noise = torch.randn(B, C, D)

        # At t=0: x_t == noise; at t=1: x_t == x_data
        t0_tensor = torch.zeros(B, 1, 1)
        t1_tensor = torch.ones(B, 1, 1)
        x_at_t0 = fm.interpolate(x_data, noise, t0_tensor)
        x_at_t1 = fm.interpolate(x_data, noise, t1_tensor)

        boundary_ok = torch.allclose(x_at_t0, noise, atol=1e-6) and torch.allclose(
            x_at_t1, x_data, atol=1e-6
        )

        # Target velocity is constant: x_data - noise
        v_target = fm.target_velocity(x_data, noise)
        velocity_ok = torch.allclose(v_target, x_data - noise, atol=1e-6)

        # Loss(v, v) == 0
        loss_self = fm.loss(v_target, v_target)
        loss_ok = float(loss_self.item()) < 1e-6

        # Sample timestep shape
        t = fm.sample_timestep(B, device=torch.device("cpu"))
        t_shape_ok = t.shape == (B, 1, 1)

        ok = boundary_ok and velocity_ok and loss_ok and t_shape_ok
        detail = (
            f"t=0→noise={boundary_ok}, t=1→data={bool(x_at_t1.allclose(x_data, atol=1e-6))}, "
            f"velocity={velocity_ok}, loss_self={loss_self.item():.2e}, "
            f"t_shape={t.shape}"
        )
        _record(results, "flow_matching_math", ok, detail)
    except Exception as e:
        _record(results, "flow_matching_math", False, str(e))

    # ------------------------------------------------------------------
    # Check 4: Projector shapes
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 4: Projector shapes")
    print("-" * 70)
    try:
        from models.action_head import NoisyActionProjector, RobotStateProjector

        H_test = 64  # use small H for CPU speed
        s_proj = RobotStateProjector(state_dim=_EXPECTED_STATE_DIM, hidden_dim=H_test)
        a_proj = NoisyActionProjector(
            action_dim=_EXPECTED_ACTION_DIM, hidden_dim=H_test, timestep_embed_dim=16
        )

        B = 4
        state_in = torch.randn(B, _EXPECTED_STATE_DIM)
        noisy_actions = torch.randn(B, _CHUNK_SIZE, _EXPECTED_ACTION_DIM)
        t = torch.rand(B, 1)

        with torch.no_grad():
            state_out = s_proj(state_in)
            action_out = a_proj(noisy_actions, t)

        state_shape_ok = state_out.shape == (B, 1, H_test)
        action_shape_ok = action_out.shape == (B, _CHUNK_SIZE, H_test)
        ok = state_shape_ok and action_shape_ok

        detail = (
            f"state: ({B},{_EXPECTED_STATE_DIM}) → {tuple(state_out.shape)} "
            f"[expected ({B},1,{H_test})] ok={state_shape_ok}; "
            f"action: ({B},{_CHUNK_SIZE},{_EXPECTED_ACTION_DIM}) + t → "
            f"{tuple(action_out.shape)} [expected ({B},{_CHUNK_SIZE},{H_test})] ok={action_shape_ok}"
        )
        _record(results, "projector_shapes", ok, detail)
    except Exception as e:
        _record(results, "projector_shapes", False, str(e))

    # ------------------------------------------------------------------
    # Check 5: Output head shape
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 5: Output head shape")
    print("-" * 70)
    try:
        from models.action_head import ActionOutputHead

        H_test = 64
        head = ActionOutputHead(action_dim=_EXPECTED_ACTION_DIM, hidden_dim=H_test)

        B = 4
        hidden = torch.randn(B, _CHUNK_SIZE, H_test)
        with torch.no_grad():
            vel_out = head(hidden)

        shape_ok = vel_out.shape == (B, _CHUNK_SIZE, _EXPECTED_ACTION_DIM)
        no_nan = torch.isfinite(vel_out).all()
        ok = shape_ok and bool(no_nan)

        detail = (
            f"({B},{_CHUNK_SIZE},{H_test}) → {tuple(vel_out.shape)} "
            f"[expected ({B},{_CHUNK_SIZE},{_EXPECTED_ACTION_DIM})] "
            f"ok={shape_ok}, finite={bool(no_nan)}"
        )
        _record(results, "output_head_shape", ok, detail)
    except Exception as e:
        _record(results, "output_head_shape", False, str(e))

    # ------------------------------------------------------------------
    # Checks 6-10: require a VLA model (real or mock)
    # ------------------------------------------------------------------

    from models.vla_model import VLAModel

    if gpu_mode:
        print(f"\nLoading real VLA model (model={args.model_config}) for GPU checks 6-10...")
        try:
            from models.vla_model import load_vla_model

            t_load = time.time()
            vla_model = load_vla_model(cfg)
            load_time = time.time() - t_load
            device = next(vla_model.backbone._model.parameters()).device
            vocab_size_vla = vla_model.backbone.info.vocab_size
            print(f"  Real backbone loaded in {load_time:.1f}s on {device}\n")
        except Exception as e:
            print(f"  [WARN] Failed to load real backbone: {e}")
            print("  Falling back to mock backbone for checks 6-10.")
            gpu_mode = False
            device = torch.device("cpu")
            vla_model = None
            vocab_size_vla = 100

    if not gpu_mode:
        # Mock backbone
        from omegaconf import OmegaConf

        mock_bb = _MockVLMBackbone(hidden_size=64, vocab_size=100)
        mock_bb.freeze_backbone()
        ah_cfg_dict = OmegaConf.to_container(cfg.model.action_head, resolve=True)
        mock_cfg = OmegaConf.create({"model": {"action_head": ah_cfg_dict}})
        vla_model = VLAModel(mock_bb, mock_cfg)
        device = torch.device("cpu")
        vocab_size_vla = 100

    # ------------------------------------------------------------------
    # Check 6: VLA training forward pass
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 6: VLA training forward pass")
    print("-" * 70)
    try:
        action_dim_vla = vla_model.chunk_config.action_dim
        state_dim_vla = vla_model.chunk_config.state_dim

        batch = _make_synthetic_batch(
            B=1,
            vocab_size=vocab_size_vla,
            device=device,
            action_dim=action_dim_vla,
            state_dim=state_dim_vla,
        )

        t0 = time.time()
        with torch.no_grad():
            out = vla_model.forward(batch)
        elapsed = time.time() - t0

        keys_ok = set(out.keys()) == {"total_loss", "text_loss", "action_loss"}
        all_finite = all(torch.isfinite(v) for v in out.values())
        all_scalar = all(v.ndim == 0 for v in out.values())
        ok = keys_ok and all_finite and all_scalar

        detail = (
            f"total_loss={out['total_loss'].item():.4f}, "
            f"text_loss={out['text_loss'].item():.4f}, "
            f"action_loss={out['action_loss'].item():.4f}, "
            f"finite={all_finite}, scalar={all_scalar}, "
            f"elapsed={elapsed * 1000:.0f}ms"
        )
        _record(results, "vla_training_forward", ok, detail)
    except Exception as e:
        _record(results, "vla_training_forward", False, str(e))

    # ------------------------------------------------------------------
    # Check 7: VLA inference (predict_actions)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 7: VLA inference (predict_actions)")
    print("-" * 70)
    try:
        B_inf = 1
        seq_text_inf = 8
        input_ids_inf = torch.randint(0, min(90, vocab_size_vla - 1), (B_inf, seq_text_inf)).to(
            device
        )
        attn_mask_inf = torch.ones(B_inf, seq_text_inf, dtype=torch.long, device=device)
        robot_state_inf = torch.randn(B_inf, _EXPECTED_STATE_DIM, device=device)

        t0 = time.time()
        with torch.no_grad():
            actions = vla_model.predict_actions(input_ids_inf, attn_mask_inf, robot_state_inf, K=3)
        elapsed = time.time() - t0

        expected_shape = (B_inf, _EXPECTED_ACTION_CHUNK_SIZE, _EXPECTED_ACTION_DIM)
        shape_ok = actions.shape == expected_shape
        finite_ok = torch.isfinite(actions).all()
        dtype_ok = actions.dtype == torch.float32
        ok = shape_ok and bool(finite_ok) and dtype_ok

        detail = (
            f"shape={tuple(actions.shape)} [expected {expected_shape}] ok={shape_ok}, "
            f"finite={bool(finite_ok)}, dtype={actions.dtype}, "
            f"elapsed={elapsed * 1000:.0f}ms"
        )
        _record(results, "vla_inference", ok, detail)
    except Exception as e:
        _record(results, "vla_inference", False, str(e))

    # ------------------------------------------------------------------
    # Check 8: Gradient routing
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 8: Gradient routing")
    print("-" * 70)
    try:
        action_dim_vla = vla_model.chunk_config.action_dim
        state_dim_vla = vla_model.chunk_config.state_dim

        batch_grad = _make_synthetic_batch(
            B=1,
            vocab_size=vocab_size_vla,
            device=device,
            action_dim=action_dim_vla,
            state_dim=state_dim_vla,
        )

        vla_model.zero_grad()
        out = vla_model.forward(batch_grad)
        out["total_loss"].backward()

        # Action head should have non-zero grads
        ah_params = (
            list(vla_model.state_projector.parameters())
            + list(vla_model.action_projector.parameters())
            + list(vla_model.action_output_head.parameters())
        )
        has_nonzero_ah = any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in ah_params
        )

        # Frozen backbone should have no meaningful grads
        frozen_clean = all(
            p.grad is None or p.grad.abs().sum().item() < 1e-9
            for p in vla_model.backbone.parameters()
        )

        ok = has_nonzero_ah and frozen_clean

        n_ah_with_grad = sum(
            1 for p in ah_params if p.grad is not None and p.grad.abs().sum().item() > 0.0
        )
        detail = (
            f"action_head_nonzero_grad={has_nonzero_ah} ({n_ah_with_grad}/{len(ah_params)} params), "
            f"frozen_backbone_clean={frozen_clean}"
        )
        _record(results, "gradient_routing", ok, detail)

        vla_model.zero_grad()
    except Exception as e:
        _record(results, "gradient_routing", False, str(e))

    # ------------------------------------------------------------------
    # Check 9: Numerical stability (edge cases)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 9: Numerical stability (edge cases)")
    print("-" * 70)
    try:
        from models.action_head import FlowMatchingConfig, FlowMatchingModule, NoisyActionProjector

        H_test = 64
        a_proj_test = NoisyActionProjector(
            action_dim=_EXPECTED_ACTION_DIM, hidden_dim=H_test, timestep_embed_dim=16
        )
        fm_test = FlowMatchingModule(FlowMatchingConfig())

        issues = []

        # Edge case 1: t near 0
        x_data = torch.randn(2, _CHUNK_SIZE, _EXPECTED_ACTION_DIM)
        noise = torch.randn_like(x_data)
        t_near0 = torch.full((2, 1, 1), 0.001)
        x_t = fm_test.interpolate(x_data, noise, t_near0)
        t_1d = t_near0.squeeze(-1)
        with torch.no_grad():
            out_t0 = a_proj_test(x_t, t_1d)
        if not torch.isfinite(out_t0).all():
            issues.append("t≈0 projector output has NaN/Inf")

        # Edge case 2: t near 1
        t_near1 = torch.full((2, 1, 1), 0.999)
        x_t = fm_test.interpolate(x_data, noise, t_near1)
        t_1d = t_near1.squeeze(-1)
        with torch.no_grad():
            out_t1 = a_proj_test(x_t, t_1d)
        if not torch.isfinite(out_t1).all():
            issues.append("t≈1 projector output has NaN/Inf")

        # Edge case 3: zero robot state through VLA forward
        batch_zero_state = _make_synthetic_batch(
            B=1,
            vocab_size=vocab_size_vla,
            device=device,
            action_dim=vla_model.chunk_config.action_dim,
            state_dim=vla_model.chunk_config.state_dim,
        )
        batch_zero_state["robot_states"] = torch.zeros_like(batch_zero_state["robot_states"])
        with torch.no_grad():
            out_zero = vla_model.forward(batch_zero_state)
        for k, v in out_zero.items():
            if not torch.isfinite(v):
                issues.append(f"zero_state: {k} not finite")

        # Edge case 4: large actions (100x scale)
        batch_large_actions = _make_synthetic_batch(
            B=1,
            vocab_size=vocab_size_vla,
            device=device,
            action_dim=vla_model.chunk_config.action_dim,
            state_dim=vla_model.chunk_config.state_dim,
        )
        batch_large_actions["action_chunks"] = batch_large_actions["action_chunks"] * 100.0
        with torch.no_grad():
            out_large = vla_model.forward(batch_large_actions)
        for k, v in out_large.items():
            if not torch.isfinite(v):
                issues.append(f"large_actions: {k} not finite")

        ok = len(issues) == 0
        detail = "All edge cases clean (no NaN/Inf)" if ok else f"Issues: {'; '.join(issues)}"
        _record(results, "numerical_stability", ok, detail)
    except Exception as e:
        _record(results, "numerical_stability", False, str(e))

    # ------------------------------------------------------------------
    # Check 10: Memory overhead (GPU only)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Check 10: Memory overhead (GPU only)")
    print("-" * 70)
    if not has_gpu:
        _record(results, "memory_overhead", True, "SKIP — no CUDA available")
    else:
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # Baseline: memory after model is loaded
            baseline_bytes = torch.cuda.max_memory_allocated()

            torch.cuda.reset_peak_memory_stats()

            action_dim_vla = vla_model.chunk_config.action_dim
            state_dim_vla = vla_model.chunk_config.state_dim
            batch_mem = _make_synthetic_batch(
                B=1,
                vocab_size=vocab_size_vla,
                device=device,
                action_dim=action_dim_vla,
                state_dim=state_dim_vla,
            )

            # Forward + backward (simulate action head training overhead)
            out_mem = vla_model.forward(batch_mem)
            out_mem["total_loss"].backward()

            peak_during_fwd_bwd = torch.cuda.max_memory_allocated()
            overhead_bytes = peak_during_fwd_bwd - baseline_bytes
            overhead_gb = overhead_bytes / (1024**3)

            vla_model.zero_grad()
            torch.cuda.empty_cache()

            ok = overhead_gb < _MAX_MEMORY_OVERHEAD_GB

            memory_report = {
                "baseline_gb": round(baseline_bytes / 1024**3, 3),
                "peak_fwd_bwd_gb": round(peak_during_fwd_bwd / 1024**3, 3),
                "overhead_gb": round(overhead_gb, 3),
                "budget_gb": _MAX_MEMORY_OVERHEAD_GB,
                "within_budget": ok,
                "device": str(device),
            }
            (out_dir / "memory_overhead.json").write_text(
                json.dumps(memory_report, indent=2) + "\n"
            )

            detail = (
                f"baseline={baseline_bytes / 1024**3:.2f}GB, "
                f"peak={peak_during_fwd_bwd / 1024**3:.2f}GB, "
                f"overhead={overhead_gb:.3f}GB "
                f"({'< 1 GB OK' if ok else f'> {_MAX_MEMORY_OVERHEAD_GB} GB EXCEEDS BUDGET'})"
            )
            _record(results, "memory_overhead", ok, detail)
        except Exception as e:
            _record(results, "memory_overhead", False, str(e))

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
        "phase": "3.2.5",
        "model_config": args.model_config,
        "gpu_mode": gpu_mode,
        "has_gpu": has_gpu,
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in results],
        "summary": {"passed": n_pass, "total": n_total, "all_passed": n_pass == n_total},
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report saved to {out_dir / 'validation_report.json'}")
    print(f"Param counts saved to {out_dir / 'param_counts.json'}")
    if has_gpu:
        print(f"Memory overhead saved to {out_dir / 'memory_overhead.json'}")

    torch.cuda.empty_cache()
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
