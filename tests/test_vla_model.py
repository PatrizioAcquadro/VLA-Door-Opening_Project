"""Integration tests for VLAModel (Phase 3.2.4).

CPU tests (``action_head`` marker) use a lightweight MockVLMBackbone that
simulates the VLMBackbone interface without loading the 4B parameter model.

GPU tests (``vlm``, ``gpu``, ``slow`` markers) load the real Qwen3.5-4B
backbone via ``load_vla_model()`` and run full end-to-end forward/inference.

Test classes:
    TestVLAModelConstruction  -- Component composition, param counts, float32 head.
    TestSequenceAssembly      -- Assemble sequence output shape, ordering, dtype.
    TestVLAModelForward       -- Training forward pass: losses, keys, NaN checks.
    TestVLAModelGradients     -- Gradient routing: action head vs frozen backbone.
    TestVLAModelInference     -- predict_actions shape, finite values, K parameter.
    TestBackwardCompatibility -- Existing get_model() paths unaffected.
    TestVLAModelGPU           -- Real backbone: loading, forward, inference, gradients.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from models.action_head import (
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    STATE_DIM,
    TokenType,
)
from models.vla_model import VLAModel
from models.vlm_backbone import VLMBackboneInfo

# ---------------------------------------------------------------------------
# Mock backbone
# ---------------------------------------------------------------------------


class MockVLMBackbone(nn.Module):
    """Lightweight mock of VLMBackbone for CPU-only VLAModel testing.

    Provides all properties and methods that VLAModel calls, but uses a tiny
    hidden_size and a simple transformer encoder layer instead of the 4B model.
    Has trainable parameters so freeze/unfreeze behaviour can be verified.
    """

    def __init__(self, hidden_size: int = 64, vocab_size: int = 100) -> None:
        super().__init__()
        self._hs = hidden_size

        # Trainable components (used to verify freeze/unfreeze)
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

    # --- Properties ---

    @property
    def hidden_size(self) -> int:
        return self._hs

    @property
    def info(self) -> VLMBackboneInfo:
        return self._info

    @property
    def processor(self) -> None:
        return None

    @property
    def lm_head(self) -> nn.Module:
        return self._lm_head

    # --- VLAModel-facing methods ---

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embed(input_ids)

    def get_vision_features(
        self, pixel_values: torch.Tensor, image_grid_thw: Any
    ) -> list[torch.Tensor]:
        # Return per-image feature tensors (1 token per image for simplicity)
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
            assert input_ids is not None
            x = self._embed(input_ids)
        return self._encoder(x)

    # --- Freeze / unfreeze ---

    def freeze_backbone(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def freeze_vision(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

_ACTION_HEAD_CFG: dict = {
    "chunk_size": ACTION_CHUNK_SIZE,
    "action_dim": ACTION_DIM,
    "state_dim": STATE_DIM,
    "tokens_per_action_step": 1,
    "projector": {
        "hidden_dim": None,  # resolved at runtime to backbone.hidden_size
        "n_layers": 2,
        "activation": "silu",
        "dropout": 0.0,
    },
    "timestep_embed_dim": 16,  # small for fast CPU tests
    "flow_matching": {
        "n_denoising_steps": 3,  # fast for tests
        "solver": "euler",
        "sigma_min": 0.001,
        "time_sampling": "uniform",
        "time_alpha": 1.5,
        "time_beta": 1.0,
        "time_min": 0.001,
        "time_max": 0.999,
    },
    "loss": {
        "lambda_text": 1.0,
        "lambda_action": 1.0,
    },
    "inference": {"execute_steps": 8},
    "float32_head": True,
}


class _FakeCfg:
    """Minimal cfg-like object exposing cfg.model.action_head."""

    class _Model:
        def __init__(self, ah_cfg: dict) -> None:
            self.action_head = _DictAttr(ah_cfg)

    def __init__(self, ah_cfg: dict) -> None:
        self.model = _FakeCfg._Model(ah_cfg)


class _DictAttr:
    """Dict wrapper with attribute-style access (mimics OmegaConf)."""

    def __init__(self, d: dict) -> None:
        self._d = d

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        return self._d[key]


def _make_cfg() -> _FakeCfg:
    """Create a minimal cfg with action_head settings."""
    return _FakeCfg(_ACTION_HEAD_CFG)


def _make_model(hidden_size: int = 64) -> VLAModel:
    """Create a VLAModel with mock backbone and test action-head config."""
    from omegaconf import OmegaConf

    backbone = MockVLMBackbone(hidden_size=hidden_size)
    cfg = OmegaConf.create({"model": {"action_head": _ACTION_HEAD_CFG}})
    return VLAModel(backbone, cfg)


# ---------------------------------------------------------------------------
# Synthetic batch helper
# ---------------------------------------------------------------------------

_B = 2
_SEQ_TEXT = 8  # short text sequence (includes 2 image token placeholders)
_N_SEG = 1  # single state token (one segment)
_N_CHUNKS = 1
_CHUNK_SIZE = ACTION_CHUNK_SIZE
_N_ACTION_TOKENS = _N_CHUNKS * _CHUNK_SIZE
_SEQ_TOTAL = _SEQ_TEXT + _N_SEG + _N_ACTION_TOKENS


def _make_batch(B: int = _B, vocab_size: int = 100) -> dict[str, torch.Tensor]:
    """Build a synthetic training batch with correct shapes and types."""
    # input_ids: put image token placeholders at positions 2 and 4
    input_ids = torch.randint(0, 97, (B, _SEQ_TEXT))  # token IDs 0..96
    input_ids[:, 2] = 99  # image_token_id
    input_ids[:, 4] = 99  # image_token_id

    # text_labels: -100 at image positions, valid IDs elsewhere
    text_labels = torch.randint(0, vocab_size, (B, _SEQ_TEXT))
    text_labels[:, 2] = -100
    text_labels[:, 4] = -100

    # token_type_ids: TEXT=0, IMAGE=1, STATE=2, ACTION=3
    # Layout: [TEXT(0..7) | STATE(8) | ACTION(9..24)]
    token_type_ids = torch.zeros(B, _SEQ_TOTAL, dtype=torch.long)
    token_type_ids[:, 2] = TokenType.IMAGE
    token_type_ids[:, 4] = TokenType.IMAGE
    token_type_ids[:, _SEQ_TEXT : _SEQ_TEXT + _N_SEG] = TokenType.STATE
    token_type_ids[:, _SEQ_TEXT + _N_SEG :] = TokenType.ACTION

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(B, _SEQ_TOTAL, dtype=torch.long),
        "robot_states": torch.randn(B, _N_SEG, STATE_DIM),
        "action_chunks": torch.randn(B, _N_CHUNKS, _CHUNK_SIZE, ACTION_DIM),
        "chunk_masks": torch.ones(B, _N_CHUNKS, _CHUNK_SIZE),
        "token_type_ids": token_type_ids,
        "text_labels": text_labels,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.action_head
class TestVLAModelConstruction:
    """VLAModel composes components correctly from config."""

    def test_components_present(self) -> None:
        """All required sub-modules exist after construction."""
        model = _make_model()
        assert hasattr(model, "backbone")
        assert hasattr(model, "state_projector")
        assert hasattr(model, "action_projector")
        assert hasattr(model, "action_output_head")
        assert hasattr(model, "flow_matching")
        assert hasattr(model, "chunk_config")

    def test_hidden_size_property(self) -> None:
        """hidden_size matches backbone."""
        model = _make_model(hidden_size=64)
        assert model.hidden_size == 64

    def test_float32_head(self) -> None:
        """Action head components remain in float32 when float32_head=True."""
        model = _make_model()
        for m in (model.state_projector, model.action_projector, model.action_output_head):
            for p in m.parameters():
                assert p.dtype == torch.float32, f"Expected float32, got {p.dtype}"

    def test_action_head_param_count_nonzero(self) -> None:
        """Action head has a positive number of trainable parameters."""
        model = _make_model()
        n = model._count_action_head_params()
        assert n > 0


@pytest.mark.action_head
class TestSequenceAssembly:
    """VLAModel.assemble_sequence() and _scatter_vision_features()."""

    def test_assemble_sequence_shape(self) -> None:
        """assemble_sequence concatenates to correct total length."""
        model = _make_model()
        H = model.hidden_size
        B = 2
        text = torch.randn(B, _SEQ_TEXT, H)
        state = torch.randn(B, _N_SEG, H)
        action = torch.randn(B, _N_ACTION_TOKENS, H)
        out = model.assemble_sequence(text, state, action)
        assert out.shape == (B, _SEQ_TEXT + _N_SEG + _N_ACTION_TOKENS, H)

    def test_assemble_sequence_ordering(self) -> None:
        """assemble_sequence places text, state, action in the correct order."""
        model = _make_model()
        H = model.hidden_size
        B = 1
        text = torch.ones(B, _SEQ_TEXT, H) * 1.0
        state = torch.ones(B, _N_SEG, H) * 2.0
        action = torch.ones(B, _N_ACTION_TOKENS, H) * 3.0
        out = model.assemble_sequence(text, state, action)

        assert out[:, :_SEQ_TEXT, :].mean().item() == pytest.approx(1.0, abs=0.01)
        assert out[:, _SEQ_TEXT : _SEQ_TEXT + _N_SEG, :].mean().item() == pytest.approx(
            2.0, abs=0.01
        )
        assert out[:, _SEQ_TEXT + _N_SEG :, :].mean().item() == pytest.approx(3.0, abs=0.01)

    def test_assemble_sequence_dtype(self) -> None:
        """assemble_sequence casts all inputs to backbone dtype."""
        model = _make_model()
        H = model.hidden_size
        B = 2
        # Backbone mock uses float32; state/action projectors also float32 here
        text = torch.randn(B, _SEQ_TEXT, H)
        state = torch.randn(B, _N_SEG, H)
        action = torch.randn(B, _N_ACTION_TOKENS, H)
        out = model.assemble_sequence(text, state, action)
        assert out.dtype == text.dtype


@pytest.mark.action_head
class TestVLAModelForward:
    """Training forward pass returns correct loss structure."""

    def test_forward_returns_dict_keys(self) -> None:
        """forward() returns a dict with total_loss, text_loss, action_loss."""
        model = _make_model()
        batch = _make_batch()
        out = model.forward(batch)
        assert set(out.keys()) == {"total_loss", "text_loss", "action_loss"}

    def test_forward_losses_finite(self) -> None:
        """All returned losses are finite (no NaN or Inf)."""
        model = _make_model()
        batch = _make_batch()
        out = model.forward(batch)
        for name, val in out.items():
            assert torch.isfinite(val), f"{name} is not finite: {val}"

    def test_forward_losses_scalar(self) -> None:
        """All returned losses are 0-D scalar tensors."""
        model = _make_model()
        batch = _make_batch()
        out = model.forward(batch)
        for name, val in out.items():
            assert val.ndim == 0, f"{name} has ndim {val.ndim}, expected 0"

    def test_forward_total_loss_requires_grad(self) -> None:
        """total_loss is differentiable (requires_grad=True)."""
        model = _make_model()
        batch = _make_batch()
        out = model.forward(batch)
        assert out["total_loss"].requires_grad


@pytest.mark.action_head
class TestVLAModelGradients:
    """Gradient routing: action head receives grads; frozen backbone does not."""

    def _run_backward(self, frozen_backbone: bool) -> VLAModel:
        model = _make_model()
        if frozen_backbone:
            model.freeze_backbone()
        batch = _make_batch()
        out = model.forward(batch)
        out["total_loss"].backward()
        return model

    def test_action_head_grads_nonzero(self) -> None:
        """Action head parameters have non-zero gradients after backward."""
        model = self._run_backward(frozen_backbone=True)
        ah_params = (
            list(model.state_projector.parameters())
            + list(model.action_projector.parameters())
            + list(model.action_output_head.parameters())
        )
        has_nonzero_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in ah_params
        )
        assert has_nonzero_grad, "Expected non-zero gradient on at least one action head param"

    def test_frozen_backbone_no_grads(self) -> None:
        """Frozen backbone parameters have no gradients after backward."""
        model = self._run_backward(frozen_backbone=True)
        for p in model.backbone.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum().item() == pytest.approx(
                    0.0, abs=1e-9
                ), "Frozen backbone param has non-zero gradient"

    def test_backward_does_not_error(self) -> None:
        """backward() completes without error."""
        self._run_backward(frozen_backbone=False)
        # If we got here, backward succeeded

    def test_freeze_unfreeze_toggle(self) -> None:
        """freeze_backbone() and unfreeze_backbone() correctly toggle requires_grad."""
        model = _make_model()
        model.freeze_backbone()
        for p in model.backbone.parameters():
            assert not p.requires_grad, "Expected requires_grad=False after freeze"

        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad, "Expected requires_grad=True after unfreeze"


@pytest.mark.action_head
class TestVLAModelInference:
    """predict_actions() returns correct shape and finite values."""

    def test_predict_actions_shape(self) -> None:
        """predict_actions returns (B, chunk_size, action_dim)."""
        model = _make_model()
        B = 2
        input_ids = torch.randint(0, 97, (B, _SEQ_TEXT))
        attention_mask = torch.ones(B, _SEQ_TEXT, dtype=torch.long)
        robot_state = torch.randn(B, STATE_DIM)

        with torch.no_grad():
            actions = model.predict_actions(input_ids, attention_mask, robot_state, K=3)

        assert actions.shape == (B, ACTION_CHUNK_SIZE, ACTION_DIM)

    def test_predict_actions_finite(self) -> None:
        """predict_actions returns finite values (no NaN or Inf)."""
        model = _make_model()
        B = 1
        input_ids = torch.randint(0, 97, (B, _SEQ_TEXT))
        attention_mask = torch.ones(B, _SEQ_TEXT, dtype=torch.long)
        robot_state = torch.randn(B, STATE_DIM)

        with torch.no_grad():
            actions = model.predict_actions(input_ids, attention_mask, robot_state, K=3)

        assert torch.isfinite(actions).all(), "predict_actions returned NaN or Inf"

    def test_predict_actions_k_parameter(self) -> None:
        """K parameter controls the number of denoising steps."""
        model = _make_model()
        B = 1
        input_ids = torch.randint(0, 97, (B, _SEQ_TEXT))
        attention_mask = torch.ones(B, _SEQ_TEXT, dtype=torch.long)
        robot_state = torch.randn(B, STATE_DIM)

        # K=1 and K=5 both succeed and return correct shape
        for K in (1, 5):
            with torch.no_grad():
                out = model.predict_actions(input_ids, attention_mask, robot_state, K=K)
            assert out.shape == (B, ACTION_CHUNK_SIZE, ACTION_DIM)


@pytest.mark.action_head
class TestBackwardCompatibility:
    """Existing get_model() paths are unaffected by Phase 3.2.4 changes."""

    def test_get_model_transformer_unaffected(self) -> None:
        """get_model('transformer') still works with a valid config."""
        from omegaconf import OmegaConf

        from models.utils import get_model

        cfg = OmegaConf.create(
            {
                "model": {
                    "architecture": {
                        "type": "transformer",
                        "hidden_size": 64,
                        "num_layers": 1,
                        "num_attention_heads": 2,
                        "intermediate_size": 128,
                        "hidden_dropout": 0.0,
                        "activation": "relu",
                        "use_pre_norm": False,
                        "max_seq_length": 32,
                    }
                }
            }
        )
        model = get_model(cfg)
        assert model is not None

    def test_get_model_unknown_type_raises(self) -> None:
        """get_model raises ValueError for unknown model types."""
        from omegaconf import OmegaConf

        from models.utils import get_model

        cfg = OmegaConf.create({"model": {"architecture": {"type": "nonexistent_model"}}})
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model(cfg)


# ---------------------------------------------------------------------------
# GPU tests with real Qwen3.5-4B backbone
# ---------------------------------------------------------------------------


@pytest.mark.vlm
@pytest.mark.gpu
@pytest.mark.slow
class TestVLAModelGPU:
    """End-to-end VLA model tests with the real Qwen3.5-4B backbone.

    Uses class-scoped fixture to load the model once for all tests.
    Uses vla_dev config (sdpa attention, 4096 context) for RTX 4090.
    """

    @pytest.fixture(scope="class")
    def vla_model(self):
        """Load VLA model with real backbone once for all tests."""
        from hydra import compose, initialize

        from models.vla_model import load_vla_model

        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="config", overrides=["model=vla_dev", "cluster=local"])

        model = load_vla_model(cfg)
        return model

    @pytest.fixture(scope="class")
    def device(self, vla_model):
        """Get the device the backbone is on."""
        return next(vla_model.backbone._model.parameters()).device

    def _make_gpu_batch(
        self, vla_model, device: torch.device, B: int = 1
    ) -> dict[str, torch.Tensor]:
        """Build a synthetic training batch on GPU."""
        seq_text = 16
        n_seg = 1
        n_chunks = 1
        chunk_size = ACTION_CHUNK_SIZE
        n_action_tokens = n_chunks * chunk_size
        seq_total = seq_text + n_seg + n_action_tokens

        # Use token IDs in safe range (avoid special tokens)
        input_ids = torch.randint(100, 1000, (B, seq_text), device=device)
        # No image tokens in this batch (vision-free, simpler for GPU test)

        text_labels = torch.randint(100, 1000, (B, seq_text), device=device)

        token_type_ids = torch.zeros(B, seq_total, dtype=torch.long, device=device)
        token_type_ids[:, seq_text : seq_text + n_seg] = TokenType.STATE
        token_type_ids[:, seq_text + n_seg :] = TokenType.ACTION

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(B, seq_total, dtype=torch.long, device=device),
            "robot_states": torch.randn(B, n_seg, STATE_DIM, device=device),
            "action_chunks": torch.randn(B, n_chunks, chunk_size, ACTION_DIM, device=device),
            "chunk_masks": torch.ones(B, n_chunks, chunk_size, device=device),
            "token_type_ids": token_type_ids,
            "text_labels": text_labels,
        }

    def test_vla_model_loads(self, vla_model) -> None:
        """VLA model loads with real backbone without error."""
        assert vla_model is not None
        assert vla_model.hidden_size == 2560

    def test_action_head_param_count(self, vla_model) -> None:
        """Action head has ~20.5M params with H=2560."""
        n = vla_model._count_action_head_params()
        # Expected: ~20.5M (state ~6.7M + action ~7.3M + output ~6.6M)
        assert (
            15_000_000 < n < 30_000_000
        ), f"Action head params {n/1e6:.1f}M outside expected range"

    def test_float32_action_head_on_gpu(self, vla_model) -> None:
        """Action head components are float32 even though backbone is bf16."""
        for name, mod in [
            ("state_projector", vla_model.state_projector),
            ("action_projector", vla_model.action_projector),
            ("action_output_head", vla_model.action_output_head),
        ]:
            for p in mod.parameters():
                assert (
                    p.dtype == torch.float32
                ), f"{name} param dtype is {p.dtype}, expected float32"

    def test_backbone_is_frozen(self, vla_model) -> None:
        """Backbone parameters are frozen after loading with default config."""
        trainable = sum(1 for p in vla_model.backbone.parameters() if p.requires_grad)
        assert trainable == 0, f"Expected 0 trainable backbone params, got {trainable}"

    def test_forward_losses_finite(self, vla_model, device) -> None:
        """Training forward pass produces finite losses with real backbone."""
        batch = self._make_gpu_batch(vla_model, device, B=1)
        out = vla_model.forward(batch)

        assert set(out.keys()) == {"total_loss", "text_loss", "action_loss"}
        for name, val in out.items():
            assert torch.isfinite(val), f"{name} is not finite: {val.item():.6f}"
            assert val.ndim == 0, f"{name} is not scalar: ndim={val.ndim}"

    def test_forward_gradient_routing(self, vla_model, device) -> None:
        """Action head gets gradients; frozen backbone does not."""
        batch = self._make_gpu_batch(vla_model, device, B=1)
        out = vla_model.forward(batch)
        out["total_loss"].backward()

        # Action head should have non-zero grads
        ah_params = (
            list(vla_model.state_projector.parameters())
            + list(vla_model.action_projector.parameters())
            + list(vla_model.action_output_head.parameters())
        )
        has_nonzero = any(p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in ah_params)
        assert has_nonzero, "Action head has no non-zero gradients"

        # Frozen backbone should have no grads
        for p in vla_model.backbone.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum().item() == pytest.approx(
                    0.0, abs=1e-9
                ), "Frozen backbone param has non-zero gradient"

        # Clean up grads for other tests
        vla_model.zero_grad()

    def test_predict_actions_shape_and_finite(self, vla_model, device) -> None:
        """predict_actions returns (B, 16, 17) with finite values."""
        B = 1
        input_ids = torch.randint(100, 1000, (B, 16), device=device)
        attention_mask = torch.ones(B, 16, dtype=torch.long, device=device)
        robot_state = torch.randn(B, STATE_DIM, device=device)

        actions = vla_model.predict_actions(input_ids, attention_mask, robot_state, K=3)

        assert actions.shape == (
            B,
            ACTION_CHUNK_SIZE,
            ACTION_DIM,
        ), f"Expected ({B}, {ACTION_CHUNK_SIZE}, {ACTION_DIM}), got {actions.shape}"
        assert torch.isfinite(actions).all(), "predict_actions returned NaN or Inf"
        assert actions.dtype == torch.float32, f"Expected float32 actions, got {actions.dtype}"

    def test_predict_actions_batch_size_2(self, vla_model, device) -> None:
        """predict_actions works with batch_size > 1."""
        B = 2
        input_ids = torch.randint(100, 1000, (B, 12), device=device)
        attention_mask = torch.ones(B, 12, dtype=torch.long, device=device)
        robot_state = torch.randn(B, STATE_DIM, device=device)

        actions = vla_model.predict_actions(input_ids, attention_mask, robot_state, K=2)

        assert actions.shape == (B, ACTION_CHUNK_SIZE, ACTION_DIM)
        assert torch.isfinite(actions).all()

    def test_hidden_states_with_inputs_embeds(self, vla_model, device) -> None:
        """Backbone accepts inputs_embeds (Scenario C path) and returns correct shape."""
        B = 1
        seq_len = 20
        H = vla_model.hidden_size
        inputs_embeds = torch.randn(B, seq_len, H, device=device, dtype=torch.bfloat16)
        attn_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)

        with torch.no_grad():
            hidden = vla_model.backbone.get_hidden_states(
                inputs_embeds=inputs_embeds, attention_mask=attn_mask
            )

        assert hidden.shape == (B, seq_len, H)
        assert hidden.dtype == torch.bfloat16
        assert torch.isfinite(hidden.float()).all(), "NaN/Inf in hidden states"
