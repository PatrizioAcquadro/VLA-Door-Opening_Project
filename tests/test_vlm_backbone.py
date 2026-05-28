"""Tests for VLM backbone loading pipeline (Phase 3.1.1) and processor (Phase 3.1.2).

Test classes:
    TestVLMBackboneConfig     -- Config parsing (CPU, no model download). Marker: vlm
    TestVLMBackboneLoading    -- Actual model loading.  Markers: vlm, gpu, slow
    TestVLMBackboneInfo       -- Dataclass validation.  Marker: vlm
    TestBackwardCompatibility -- Existing model unchanged. No markers (always runs)
    TestResolveDtype          -- dtype resolution helper. Marker: vlm
    TestProcessorInfo         -- ProcessorInfo dataclass (CPU). Marker: vlm
    TestContextBudget         -- Context budget computation (CPU). Marker: vlm
    TestProcessorFunctions    -- Processor utilities (GPU). Markers: vlm, gpu, slow
"""

import pytest
import torch
from hydra import compose, initialize

# ---------------------------------------------------------------------------
# Config / dataclass tests (CPU, no model download)
# ---------------------------------------------------------------------------


@pytest.mark.vlm
class TestVLMBackboneConfig:
    """Config parsing and dataclass construction — CPU-safe, no weights needed."""

    @pytest.fixture
    def vlm_cfg(self):
        with initialize(config_path="../configs", version_base=None):
            return compose(config_name="config", overrides=["model=vlm", "cluster=local"])

    @pytest.fixture
    def vlm_dev_cfg(self):
        with initialize(config_path="../configs", version_base=None):
            return compose(config_name="config", overrides=["model=vlm_dev", "cluster=local"])

    def test_vlm_config_parses(self, vlm_cfg):
        """VLM config has architecture.type='vlm' and vlm section."""
        assert vlm_cfg.model.architecture.type == "vlm"
        assert vlm_cfg.model.vlm.model_id == "Qwen/Qwen3.5-4B"

    def test_vlm_dev_config_parses(self, vlm_dev_cfg):
        """VLM dev config uses sdpa attention and shorter context."""
        assert vlm_dev_cfg.model.vlm.attn_implementation == "sdpa"
        assert vlm_dev_cfg.model.vlm.max_seq_length == 4096

    def test_vlm_config_freeze_defaults(self, vlm_cfg):
        """Backbone and vision encoder are frozen by default."""
        assert vlm_cfg.model.vlm.freeze.backbone is True
        assert vlm_cfg.model.vlm.freeze.vision_encoder is True

    def test_vlm_config_image_resolution(self, vlm_cfg):
        """Image resolution matches frozen camera contract (320×320)."""
        assert vlm_cfg.model.vlm.processor.image_resolution == 320

    def test_vlm_backbone_info_dataclass(self):
        """VLMBackboneInfo can be constructed and is frozen (immutable)."""
        from models.vlm_backbone import VLMBackboneInfo

        info = VLMBackboneInfo(
            model_id="test-model",
            param_count_total=4_539_000_000,
            param_count_trainable=0,
            hidden_size=2560,
            num_layers=32,
            vocab_size=248320,
            dtype="bfloat16",
            vision_hidden_size=1024,
            vision_depth=24,
            image_token_id=248056,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
        )
        assert info.hidden_size == 2560
        assert info.num_layers == 32
        assert info.dtype == "bfloat16"
        # Frozen dataclass: mutation must raise FrozenInstanceError
        with pytest.raises(AttributeError):
            info.hidden_size = 512  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Actual model loading tests (GPU + transformers required)
# ---------------------------------------------------------------------------


@pytest.mark.vlm
@pytest.mark.gpu
@pytest.mark.slow
class TestVLMBackboneLoading:
    """Model loading, verification, and freeze controls.

    Uses class-scoped fixture to load the model once for all 8 tests,
    avoiding repeated 10-30s loads and excessive VRAM allocation.
    Uses vlm_dev config (sdpa attention) to avoid flash-attn dependency.
    """

    @pytest.fixture(scope="class")
    def backbone(self):
        """Load backbone once for all tests in this class."""
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="config", overrides=["model=vlm_dev", "cluster=local"])
        from models.vlm_backbone import load_vlm_backbone

        return load_vlm_backbone(cfg)

    def test_backbone_loads(self, backbone):
        """Backbone loads without error."""
        assert backbone is not None

    def test_backbone_hidden_size(self, backbone):
        """Hidden size matches Qwen3.5-4B text config (2560)."""
        assert backbone.hidden_size == 2560

    def test_backbone_info_param_count(self, backbone):
        """Total params approximately 4.54B (3.6B–5.5B acceptable range)."""
        total = backbone.info.param_count_total
        assert 3_600_000_000 < total < 5_500_000_000, (
            f"Unexpected param count: {total / 1e9:.2f}B — " "expected Qwen3.5-4B ~4.54B"
        )

    def test_backbone_info_dtype(self, backbone):
        """Info records bfloat16 dtype."""
        assert backbone.info.dtype == "bfloat16"

    def test_backbone_frozen_by_default(self, backbone):
        """All params frozen after loading with default freeze config."""
        trainable = sum(1 for p in backbone.parameters() if p.requires_grad)
        assert trainable == 0, f"Expected 0 trainable params, got {trainable}"

    def test_backbone_has_processor(self, backbone):
        """Processor is attached to backbone."""
        assert backbone.processor is not None

    def test_backbone_verify_passes(self, backbone):
        """verify_backbone returns True for a correctly loaded backbone."""
        from models.vlm_backbone import verify_backbone

        result = verify_backbone(backbone)
        assert result is True

    def test_backbone_unfreeze_refreeze(self, backbone):
        """unfreeze_backbone then freeze_backbone restores zero trainable params."""
        backbone.unfreeze_backbone()
        trainable_after_unfreeze = sum(1 for p in backbone.parameters() if p.requires_grad)
        assert trainable_after_unfreeze > 0, "Expected params to be unfrozen"

        backbone.freeze_backbone()
        trainable_after_refreeze = sum(1 for p in backbone.parameters() if p.requires_grad)
        assert trainable_after_refreeze == 0, "Expected params to be re-frozen"


# ---------------------------------------------------------------------------
# VLMBackboneInfo dataclass validation
# ---------------------------------------------------------------------------


@pytest.mark.vlm
class TestVLMBackboneInfo:
    """Dataclass serialization and immutability checks."""

    @pytest.fixture
    def sample_info(self):
        from models.vlm_backbone import VLMBackboneInfo

        return VLMBackboneInfo(
            model_id="Qwen/Qwen3.5-4B",
            param_count_total=4_539_000_000,
            param_count_trainable=0,
            hidden_size=2560,
            num_layers=32,
            vocab_size=248320,
            dtype="bfloat16",
            vision_hidden_size=1024,
            vision_depth=24,
            image_token_id=248056,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
        )

    def test_info_fields_accessible(self, sample_info):
        """All fields are accessible and have expected values."""
        assert sample_info.model_id == "Qwen/Qwen3.5-4B"
        assert sample_info.hidden_size == 2560
        assert sample_info.num_layers == 32
        assert sample_info.vocab_size == 248320
        assert sample_info.image_token_id == 248056
        assert sample_info.vision_start_token_id == 248053
        assert sample_info.vision_end_token_id == 248054

    def test_info_immutable(self, sample_info):
        """VLMBackboneInfo is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            sample_info.model_id = "other"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            sample_info.hidden_size = 512  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Backward compatibility — existing models must be unaffected
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Existing get_model() behavior must be unchanged for non-VLM types."""

    def test_transformer_model_still_works(self):
        """get_model with model=base returns TransformerModel."""
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="config", overrides=["model=base", "cluster=local"])
        from models import get_model
        from models.transformer import TransformerModel

        model = get_model(cfg)
        assert isinstance(model, TransformerModel)

    def test_unknown_model_type_raises(self):
        """get_model with an unknown architecture type raises ValueError."""
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="config", overrides=["model=base", "cluster=local"])
        from omegaconf import OmegaConf

        # Override type to something unrecognised
        OmegaConf.update(cfg, "model.architecture.type", "nonexistent_model_type")

        from models import get_model

        with pytest.raises(ValueError, match="Unknown model type"):
            get_model(cfg)


# ---------------------------------------------------------------------------
# Helper dtype resolution
# ---------------------------------------------------------------------------


@pytest.mark.vlm
class TestResolveDtype:
    """Unit tests for the internal _resolve_dtype helper."""

    def test_bfloat16(self):
        from models.vlm_backbone import _resolve_dtype

        assert _resolve_dtype("bfloat16") == torch.bfloat16

    def test_float16(self):
        from models.vlm_backbone import _resolve_dtype

        assert _resolve_dtype("float16") == torch.float16

    def test_float32(self):
        from models.vlm_backbone import _resolve_dtype

        assert _resolve_dtype("float32") == torch.float32

    def test_invalid_raises(self):
        from models.vlm_backbone import _resolve_dtype

        with pytest.raises(ValueError, match="Unknown dtype"):
            _resolve_dtype("int8")


# ---------------------------------------------------------------------------
# ProcessorInfo dataclass (CPU, no model download)
# ---------------------------------------------------------------------------


@pytest.mark.vlm
class TestProcessorInfo:
    """ProcessorInfo dataclass construction and immutability."""

    def test_processor_info_construction(self):
        """ProcessorInfo can be constructed with all fields."""
        from models.vlm_backbone import ProcessorInfo

        info = ProcessorInfo(
            vocab_size=248320,
            bos_token_id=None,
            eos_token_id=248001,
            pad_token_id=248000,
            image_token_id=248056,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
            estimated_vision_tokens_per_image=196,
            image_resolution=320,
        )
        assert info.vocab_size == 248320
        assert info.estimated_vision_tokens_per_image == 196
        assert info.image_resolution == 320
        assert info.bos_token_id is None
        assert info.eos_token_id == 248001

    def test_processor_info_immutable(self):
        """ProcessorInfo is frozen (immutable)."""
        from models.vlm_backbone import ProcessorInfo

        info = ProcessorInfo(
            vocab_size=248320,
            bos_token_id=None,
            eos_token_id=248001,
            pad_token_id=248000,
            image_token_id=248056,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
            estimated_vision_tokens_per_image=196,
            image_resolution=320,
        )
        with pytest.raises(AttributeError):
            info.vocab_size = 0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            info.estimated_vision_tokens_per_image = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Context budget computation (CPU, pure function)
# ---------------------------------------------------------------------------


@pytest.mark.vlm
class TestContextBudget:
    """Context window budget computation — pure function, no model needed."""

    def test_budget_default_params(self):
        """Default budget with 196 tokens/image leaves ample action tokens."""
        from models.vlm_backbone import compute_context_budget

        budget = compute_context_budget(vision_tokens_per_image=196)
        assert budget["max_seq_length"] == 8192
        assert budget["n_images"] == 4
        assert budget["vision_tokens_per_image"] == 196
        assert budget["vision_tokens_total"] == 784
        assert budget["text_tokens_narration"] == 200  # 25 * 8
        assert budget["text_tokens_total"] == 245  # 40 + 200 + 5
        assert budget["tokens_used"] == 1029  # 784 + 245
        assert budget["tokens_remaining_for_actions"] == 8192 - 1029
        assert budget["tokens_remaining_for_actions"] > 5000

    def test_budget_custom_seq_length(self):
        """Budget with dev context (4096) still has action tokens remaining."""
        from models.vlm_backbone import compute_context_budget

        budget = compute_context_budget(
            vision_tokens_per_image=256,
            n_images=4,
            max_seq_length=4096,
        )
        assert budget["vision_tokens_total"] == 1024
        assert budget["tokens_remaining_for_actions"] == 4096 - 1024 - 245

    def test_budget_all_expected_keys(self):
        """Budget dict contains all expected keys."""
        from models.vlm_backbone import compute_context_budget

        budget = compute_context_budget(vision_tokens_per_image=100)
        expected_keys = {
            "max_seq_length",
            "n_images",
            "vision_tokens_per_image",
            "vision_tokens_total",
            "text_tokens_task",
            "text_tokens_narration",
            "text_tokens_outcome",
            "text_tokens_total",
            "tokens_used",
            "tokens_remaining_for_actions",
        }
        assert set(budget.keys()) == expected_keys

    def test_budget_single_image(self):
        """Budget with single image (overhead only)."""
        from models.vlm_backbone import compute_context_budget

        budget = compute_context_budget(vision_tokens_per_image=196, n_images=1)
        assert budget["vision_tokens_total"] == 196
        assert budget["tokens_remaining_for_actions"] > budget["max_seq_length"] // 2


# ---------------------------------------------------------------------------
# Processor functions (GPU + transformers required)
# ---------------------------------------------------------------------------


@pytest.mark.vlm
@pytest.mark.gpu
@pytest.mark.slow
class TestProcessorFunctions:
    """Processor utility tests requiring loaded backbone.

    Uses class-scoped fixture to load backbone once for all tests.
    Uses vlm_dev config (sdpa attention) to avoid flash-attn dependency.
    """

    @pytest.fixture(scope="class")
    def backbone(self):
        """Load backbone once for all processor tests."""
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="config", overrides=["model=vlm_dev", "cluster=local"])
        from models.vlm_backbone import load_vlm_backbone

        return load_vlm_backbone(cfg)

    def test_estimate_vision_tokens(self, backbone):
        """Vision token count for 320x320 is positive and in expected range."""
        from models.vlm_backbone import estimate_vision_tokens

        n_tokens = estimate_vision_tokens(backbone, 320, 320)
        assert isinstance(n_tokens, int)
        assert (
            50 < n_tokens < 500
        ), f"Vision token count {n_tokens} outside expected range [50, 500]"

    def test_get_processor_info(self, backbone):
        """get_processor_info returns fully populated ProcessorInfo."""
        from models.vlm_backbone import ProcessorInfo, get_processor_info

        info = get_processor_info(backbone, image_resolution=320)
        assert isinstance(info, ProcessorInfo)
        assert info.vocab_size == 248320
        assert info.estimated_vision_tokens_per_image > 0
        assert info.image_resolution == 320
        assert info.image_token_id == backbone.info.image_token_id

    def test_preprocess_single_image(self, backbone):
        """Single numpy uint8 image preprocesses to valid tensors."""
        import numpy as np

        from models.vlm_backbone import preprocess_images

        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        result = preprocess_images(backbone, [img], "Describe what you see.")

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "pixel_values" in result
        assert result["input_ids"].ndim == 2  # (B, seq_len)
        assert result["pixel_values"].numel() > 0

    def test_preprocess_quad_view(self, backbone):
        """Four images (quad view) preprocess to valid combined tensors."""
        import numpy as np

        from models.vlm_backbone import preprocess_images

        images = [np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8) for _ in range(4)]
        result = preprocess_images(
            backbone, images, "The robot is performing a LEGO assembly task."
        )

        assert "input_ids" in result
        assert "pixel_values" in result
        assert result["pixel_values"].numel() > 0
        # 4 images should produce more vision tokens than 1
        image_token_id = backbone.info.image_token_id
        n_vision = int((result["input_ids"][0] == image_token_id).sum().item())
        single_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        single_result = preprocess_images(backbone, [single_img], "test")
        n_single = int((single_result["input_ids"][0] == image_token_id).sum().item())
        assert n_vision > n_single, "4 images should have more vision tokens than 1"

    def test_preprocess_text_tokenization(self, backbone):
        """Robot manipulation vocabulary is tokenizable and round-trips."""
        processor = backbone.processor
        tokenizer = getattr(processor, "tokenizer", processor)

        terms = ["gripper", "baseplate", "stud", "brick", "press-fit", "LEGO"]
        for term in terms:
            tokens = tokenizer.encode(term)
            assert len(tokens) > 0, f"Failed to tokenize: {term}"
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            assert (
                term.lower() in decoded.lower()
            ), f"Round-trip failed for '{term}': decoded='{decoded}'"

    def test_preprocess_invalid_dtype_raises(self, backbone):
        """Non-uint8 numpy array raises ValueError."""
        import numpy as np

        from models.vlm_backbone import preprocess_images

        img = np.zeros((320, 320, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            preprocess_images(backbone, [img], "test")

    def test_preprocess_invalid_type_raises(self, backbone):
        """Non-image input raises TypeError."""
        from models.vlm_backbone import preprocess_images

        with pytest.raises(TypeError, match="Expected numpy array or PIL Image"):
            preprocess_images(backbone, ["not_an_image"], "test")

    def test_context_budget_with_measured_tokens(self, backbone):
        """Context budget with actually measured vision token count."""
        from models.vlm_backbone import compute_context_budget, estimate_vision_tokens

        n_tokens = estimate_vision_tokens(backbone, 320, 320)
        budget = compute_context_budget(
            vision_tokens_per_image=n_tokens,
            n_images=4,
            max_seq_length=8192,
        )

        assert (
            budget["tokens_remaining_for_actions"] > 0
        ), f"No tokens remaining for actions: {budget}"


# ---------------------------------------------------------------------------
# Inference sanity checks (GPU + transformers required)
# ---------------------------------------------------------------------------

# Keys accepted by backbone.forward() / get_hidden_states()
_FORWARD_KEYS = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}


@pytest.mark.vlm
@pytest.mark.gpu
@pytest.mark.slow
class TestVLMInference:
    """End-to-end forward pass tests with synthetic images (Phase 3.1.3).

    Uses class-scoped fixture to load backbone once for all 7 tests.
    Uses synthetic images (no MuJoCo dependency — keeps tests runnable
    on GPU-only machines without sim deps).
    """

    PROMPT = (
        "The robot is performing a LEGO assembly task. "
        "Describe what you see and what the robot should do next."
    )

    @pytest.fixture(scope="class")
    def backbone(self):
        """Load backbone once for all inference tests."""
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["model=vlm_dev", "cluster=local"],
            )
        from models.vlm_backbone import load_vlm_backbone

        return load_vlm_backbone(cfg)

    def test_forward_text_only(self, backbone):
        """Text-only forward pass produces valid logits."""
        tokenizer = getattr(backbone.processor, "tokenizer", backbone.processor)
        device = next(backbone._model.parameters()).device
        inputs = tokenizer("The robot is idle.", return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            fwd_inputs = {k: v for k, v in inputs.items() if k in _FORWARD_KEYS}
            outputs = backbone.forward(**fwd_inputs)

        logits = outputs["logits"]
        assert logits.ndim == 3, f"Expected 3D logits, got {logits.ndim}D"
        assert logits.shape[-1] == 248320, f"Vocab dim mismatch: {logits.shape[-1]}"
        assert not torch.isnan(logits.float()).any(), "NaN in text-only logits"

    def test_forward_single_image(self, backbone):
        """Single-image forward pass produces valid logits with vision tokens."""
        import numpy as np

        from models.vlm_backbone import preprocess_images

        device = next(backbone._model.parameters()).device
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        inputs = preprocess_images(backbone, [img], self.PROMPT, device=device)

        with torch.no_grad():
            fwd = {k: v for k, v in inputs.items() if k in _FORWARD_KEYS}
            outputs = backbone.forward(**fwd)

        logits = outputs["logits"]
        assert logits.ndim == 3
        assert logits.shape[-1] == 248320

    def test_forward_multi_view(self, backbone):
        """Quad-view forward pass has more vision tokens than single-image."""
        import numpy as np

        from models.vlm_backbone import preprocess_images

        device = next(backbone._model.parameters()).device
        quad = [np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8) for _ in range(4)]
        inputs = preprocess_images(backbone, quad, self.PROMPT, device=device)

        with torch.no_grad():
            fwd = {k: v for k, v in inputs.items() if k in _FORWARD_KEYS}
            outputs = backbone.forward(**fwd)

        logits = outputs["logits"]
        assert logits.ndim == 3
        assert logits.shape[-1] == 248320

        # Verify 4 images produce more vision tokens than 1
        image_token_id = backbone.info.image_token_id
        n_vision_quad = int((inputs["input_ids"][0] == image_token_id).sum().item())

        single = [np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)]
        single_inputs = preprocess_images(backbone, single, "test", device=device)
        n_vision_single = int((single_inputs["input_ids"][0] == image_token_id).sum().item())
        assert n_vision_quad > n_vision_single, (
            f"4 images should have more vision tokens than 1: "
            f"{n_vision_quad} vs {n_vision_single}"
        )

    def test_hidden_state_shape(self, backbone):
        """get_hidden_states() returns (1, seq, 2560) in bf16."""
        import numpy as np

        from models.vlm_backbone import preprocess_images

        device = next(backbone._model.parameters()).device
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        inputs = preprocess_images(backbone, [img], self.PROMPT, device=device)

        with torch.no_grad():
            fwd = {k: v for k, v in inputs.items() if k in _FORWARD_KEYS}
            hidden = backbone.get_hidden_states(**fwd)

        assert hidden.ndim == 3, f"Expected 3D hidden states, got {hidden.ndim}D"
        assert hidden.shape[0] == 1, f"Batch dim should be 1, got {hidden.shape[0]}"
        assert hidden.shape[-1] == 2560, f"Hidden size mismatch: {hidden.shape[-1]}"
        assert hidden.dtype == torch.bfloat16, f"Expected bf16, got {hidden.dtype}"

    def test_hidden_state_no_nan(self, backbone):
        """Hidden states contain no NaN or Inf."""
        import numpy as np

        from models.vlm_backbone import preprocess_images

        device = next(backbone._model.parameters()).device
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        inputs = preprocess_images(backbone, [img], self.PROMPT, device=device)

        with torch.no_grad():
            fwd = {k: v for k, v in inputs.items() if k in _FORWARD_KEYS}
            hidden = backbone.get_hidden_states(**fwd)

        hidden_f32 = hidden.float()
        assert not torch.isnan(hidden_f32).any(), "NaN in hidden states"
        assert not torch.isinf(hidden_f32).any(), "Inf in hidden states"

    def test_generate_text(self, backbone):
        """model.generate() produces non-empty text with valid token IDs."""
        import numpy as np

        from models.vlm_backbone import preprocess_images

        device = next(backbone._model.parameters()).device
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        inputs = preprocess_images(backbone, [img], self.PROMPT, device=device)

        with torch.no_grad():
            generated_ids = backbone._model.generate(**inputs, max_new_tokens=50, do_sample=False)

        # All token IDs within vocab range
        vocab_size = backbone.info.vocab_size
        assert (generated_ids >= 0).all(), "Negative token IDs in generation"
        assert (
            generated_ids < vocab_size
        ).all(), f"Token IDs exceed vocab size {vocab_size}: max={generated_ids.max().item()}"

        # Decoded text is non-empty
        text = backbone.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        assert len(text.strip()) > 0, "Generated text is empty"

    def test_logits_numerical_sanity(self, backbone):
        """Logits from forward pass contain no NaN or Inf."""
        import numpy as np

        from models.vlm_backbone import preprocess_images

        device = next(backbone._model.parameters()).device
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        inputs = preprocess_images(backbone, [img], self.PROMPT, device=device)

        with torch.no_grad():
            fwd = {k: v for k, v in inputs.items() if k in _FORWARD_KEYS}
            outputs = backbone.forward(**fwd)

        logits_f32 = outputs["logits"].float()
        assert not torch.isnan(logits_f32).any(), "NaN in logits"
        assert not torch.isinf(logits_f32).any(), "Inf in logits"
