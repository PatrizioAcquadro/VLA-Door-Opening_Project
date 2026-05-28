"""VLM backbone wrapper for Qwen3.5-4B.

This module wraps the HuggingFace Qwen3.5-4B model for use as the VLA backbone.
It is a lazy import — only imported when architecture.type == "vlm" in the config.

Usage:
    from models.vlm_backbone import load_vlm_backbone, VLMBackbone, VLMBackboneInfo
    backbone = load_vlm_backbone(cfg)
    hidden = backbone.get_hidden_states(input_ids, attention_mask, pixel_values, image_grid_thw)
    # → (B, seq_len, 2560) for Qwen3.5-4B

Processor utilities (Phase 3.1.2):
    from models.vlm_backbone import (
        ProcessorInfo, get_processor_info, preprocess_images,
        estimate_vision_tokens, compute_context_budget,
    )
    proc_info = get_processor_info(backbone, image_resolution=320)
    inputs = preprocess_images(backbone, [numpy_img], "Describe the scene.")
    budget = compute_context_budget(proc_info.estimated_vision_tokens_per_image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# --- Frozen architecture constants (Qwen3.5-4B) ---------------------------------

# Expected param range for verification: 4.54B ± ~20%
_PARAM_COUNT_MIN = 3_600_000_000
_PARAM_COUNT_MAX = 5_500_000_000


# --- VLMBackboneInfo dataclass --------------------------------------------------


@dataclass(frozen=True)
class VLMBackboneInfo:
    """Frozen architecture metadata for the loaded VLM backbone.

    All fields are populated during loading and are read-only thereafter.
    Phase 3.2 uses hidden_size to size the action head input dimension.
    """

    model_id: str
    param_count_total: int
    param_count_trainable: int
    hidden_size: int
    num_layers: int
    vocab_size: int
    dtype: str
    vision_hidden_size: int
    vision_depth: int
    image_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int


# --- ProcessorInfo dataclass ---------------------------------------------------


@dataclass(frozen=True)
class ProcessorInfo:
    """Frozen metadata about the processor and tokenizer configuration.

    Populated by get_processor_info(). Provides the vision token count
    and special token IDs needed for Phase 3.2 context window budgeting.
    """

    vocab_size: int
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    image_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    estimated_vision_tokens_per_image: int
    image_resolution: int


# --- VLMBackbone nn.Module ------------------------------------------------------


class VLMBackbone(nn.Module):
    """Wrapper around a HuggingFace Qwen3.5-4B model for VLA use.

    Exposes the interface Phase 3.2 depends on:
        backbone.hidden_size         → int (2560 for Qwen3.5-4B)
        backbone.info                → VLMBackboneInfo
        backbone.processor           → Qwen3VLProcessor
        backbone.get_hidden_states() → (B, seq_len, hidden_size) tensor
        backbone.forward()           → dict with "logits" and "hidden_states"
        backbone.freeze_backbone()   → freezes all model params
        backbone.unfreeze_backbone() → unfreezes all model params
        backbone.freeze_vision()     → freezes only vision encoder params
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        info: VLMBackboneInfo,
        device_mapped: bool = False,
    ) -> None:
        super().__init__()
        self._model = model
        self._processor = processor
        self._info = info
        self._device_mapped = device_mapped

    # --- Properties -------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        """Hidden dimension of the text backbone (2560 for Qwen3.5-4B)."""
        return self._info.hidden_size

    @property
    def info(self) -> VLMBackboneInfo:
        """Architecture metadata."""
        return self._info

    @property
    def processor(self) -> Any:
        """HuggingFace processor (Qwen3VLProcessor) for input preparation."""
        return self._processor

    # --- Core forward pass methods ----------------------------------------------

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text token embeddings from the backbone's embedding layer.

        Used by VLAModel (Phase 3.2.4) to obtain text embeddings independently,
        so that state and action tokens can be injected before the backbone forward.

        Args:
            input_ids: Token IDs, shape (B, seq_len).

        Returns:
            Text embeddings of shape (B, seq_len, hidden_size).
        """
        model = cast(Any, self._model)
        embedding_layer = cast(nn.Module, model.get_input_embeddings())
        return cast(torch.Tensor, embedding_layer(input_ids))

    def get_vision_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Extract vision features from raw pixel values via the backbone's vision encoder.

        Used by VLAModel (Phase 3.2.4) for Scenario C (full embedding control) —
        vision features are extracted separately so they can be scattered into the
        manually assembled ``inputs_embeds``.

        Args:
            pixel_values: Preprocessed image tensors from processor.
            image_grid_thw: Image grid dimensions tensor (n_images, 3).

        Returns:
            List of per-image feature tensors. Each element has shape
            ``(n_tokens_i, hidden_size)`` where ``n_tokens_i`` is the number of
            vision tokens for image i.
        """
        model = cast(Any, self._model)
        outputs = model.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        return cast(list[torch.Tensor], outputs.pooler_output)

    @property
    def lm_head(self) -> torch.nn.Module:
        """The language model head: Linear(hidden_size, vocab_size).

        Used by VLAModel (Phase 3.2.4) to compute text logits for AR loss.
        Weight-tied with the text embedding layer.
        """
        model = cast(Any, self._model)
        return cast(nn.Module, model.lm_head)

    def get_hidden_states(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract last hidden state from the backbone.

        Supports two modes:
        - **Standard mode** (``inputs_embeds=None``): passes ``input_ids`` +
          ``pixel_values`` through the full HuggingFace model (vision encoding,
          embedding, transformer). Used by standalone VLM inference.
        - **Embedding mode** (``inputs_embeds`` provided): passes pre-assembled
          embeddings directly to the language model, bypassing the embedding
          layer and vision encoder. Used by VLAModel (Phase 3.2.4) for Scenario C
          injection of state and action tokens. ``pixel_values`` and ``input_ids``
          are ignored in this mode.

        Args:
            input_ids: Token IDs, shape (B, seq_len). Ignored when ``inputs_embeds``
                is provided (HuggingFace XOR constraint).
            attention_mask: Boolean mask, shape (B, seq_len or seq_total).
            pixel_values: Preprocessed image tensors. Ignored when ``inputs_embeds``
                is provided.
            image_grid_thw: Image grid dimensions (temporal, height, width). Ignored
                when ``inputs_embeds`` is provided.
            inputs_embeds: Pre-assembled embeddings, shape (B, seq_total, hidden_size).
                When provided, bypasses the embedding layer and vision encoder.
            **kwargs: Additional kwargs forwarded to the HuggingFace model.

        Returns:
            Last hidden state tensor of shape (B, seq_len, hidden_size).
        """
        if inputs_embeds is not None:
            # Scenario C: full embedding control — state/action tokens already injected
            outputs = self._model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
        else:
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                **kwargs,
            )
        return outputs.hidden_states[-1]  # type: ignore[no-any-return]

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass returning dict with logits and hidden states.

        Returns a dict compatible with the existing TransformerModel interface:
            {"logits": (B, seq_len, vocab_size), "hidden_states": (B, seq_len, hidden_size)}

        Args:
            input_ids: Token IDs, shape (B, seq_len). Ignored when ``inputs_embeds``
                is provided.
            attention_mask: Boolean mask, shape (B, seq_len).
            pixel_values: Preprocessed image tensors from processor. Ignored when
                ``inputs_embeds`` is provided.
            image_grid_thw: Image grid dimensions (temporal, height, width). Ignored
                when ``inputs_embeds`` is provided.
            inputs_embeds: Pre-assembled embeddings (B, seq_total, hidden_size). When
                provided, bypasses the backbone's embedding and vision encoder.
            **kwargs: Additional kwargs forwarded to the HuggingFace model.

        Returns:
            Dict with "logits" and "hidden_states" tensors.
        """
        if inputs_embeds is not None:
            outputs = self._model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
        else:
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                **kwargs,
            )
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states[-1],
        }

    # --- Freeze / unfreeze methods ----------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (requires_grad=False)."""
        for param in self._model.parameters():
            param.requires_grad = False
        logger.info("VLMBackbone: all parameters frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (requires_grad=True)."""
        for param in self._model.parameters():
            param.requires_grad = True
        logger.info("VLMBackbone: all parameters unfrozen")

    def freeze_vision(self) -> None:
        """Freeze only the vision encoder parameters.

        Targets self._model.model.visual (Qwen3_5VisionModel, ~334M params).
        Text backbone parameters are unaffected.
        """
        vision_encoder = self._model.model.visual  # type: ignore[union-attr]
        for param in vision_encoder.parameters():  # type: ignore[union-attr]
            param.requires_grad = False
        n_frozen = sum(p.numel() for p in vision_encoder.parameters())  # type: ignore[union-attr,misc]
        logger.info("VLMBackbone: vision encoder frozen (%s params)", f"{n_frozen / 1e6:.0f}M")

    # --- Device placement override ---------------------------------------------

    def to(self, *args: Any, **kwargs: Any) -> VLMBackbone:
        """Override to() to be a no-op when device_map was used.

        When loaded with device_map='auto', HuggingFace has already placed the
        model on the appropriate device(s). The trainer's .to(device) call would
        conflict, so we make it a safe no-op. Phase 3.3 will update the trainer
        to skip .to() for VLM models.
        """
        if self._device_mapped:
            return self
        return super().to(*args, **kwargs)  # type: ignore[return-value]


# --- Loading pipeline -----------------------------------------------------------


def load_vlm_backbone(cfg: DictConfig) -> VLMBackbone:
    """Load and configure the VLM backbone from Hydra config.

    End-to-end loading pipeline:
      1. Resolve dtype and config values from cfg.model.vlm
      2. Download/load model via AutoModelForCausalLM.from_pretrained
      3. Load processor via AutoProcessor.from_pretrained
      4. Build VLMBackboneInfo from loaded model config
      5. Apply freeze policy from cfg.model.vlm.freeze
      6. Verify backbone integrity
      7. Return configured VLMBackbone

    Args:
        cfg: Full Hydra configuration (must have cfg.model.vlm section).

    Returns:
        Configured and verified VLMBackbone.

    Raises:
        ImportError: If transformers is not installed.
        RuntimeError: If backbone verification fails.
    """
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        raise ImportError(
            "transformers is required for VLM backbone. " "Install with: pip install -e '.[vlm]'"
        ) from e

    vlm_cfg = cfg.model.vlm
    model_id: str = vlm_cfg.model_id
    torch_dtype = _resolve_dtype(vlm_cfg.torch_dtype)
    revision = vlm_cfg.revision if vlm_cfg.revision else None
    cache_dir: str = vlm_cfg.cache_dir

    logger.info("Loading VLM backbone: %s (dtype=%s)", model_id, vlm_cfg.torch_dtype)

    # Load model weights (vision-language model, not text-only CausalLM)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        device_map=vlm_cfg.device_map,
        trust_remote_code=vlm_cfg.trust_remote_code,
        attn_implementation=vlm_cfg.attn_implementation,
        cache_dir=cache_dir,
    )

    # Load processor (text tokenizer + image preprocessor)
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=vlm_cfg.trust_remote_code,
        cache_dir=cache_dir,
    )

    # Build architecture metadata from loaded model config
    mc = model.config
    tc = mc.text_config
    vc = mc.vision_config

    param_count_total = sum(p.numel() for p in model.parameters())

    info = VLMBackboneInfo(
        model_id=model_id,
        param_count_total=param_count_total,
        param_count_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        hidden_size=tc.hidden_size,
        num_layers=tc.num_hidden_layers,
        vocab_size=tc.vocab_size,
        dtype=vlm_cfg.torch_dtype,
        vision_hidden_size=vc.hidden_size,
        vision_depth=vc.depth,
        image_token_id=mc.image_token_id,
        vision_start_token_id=mc.vision_start_token_id,
        vision_end_token_id=mc.vision_end_token_id,
    )

    logger.info(
        "Loaded %s: %.2fB params, hidden_size=%d, vision_depth=%d",
        model_id,
        param_count_total / 1e9,
        info.hidden_size,
        info.vision_depth,
    )

    backbone = VLMBackbone(model, processor, info, device_mapped=True)

    # Apply freeze policy
    freeze_cfg = vlm_cfg.freeze
    if freeze_cfg.backbone:
        backbone.freeze_backbone()
    if freeze_cfg.vision_encoder and not freeze_cfg.backbone:
        # Only freeze vision separately if backbone is not already fully frozen
        backbone.freeze_vision()

    # Update trainable count after freezing
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters after freeze policy: %s", f"{trainable / 1e6:.1f}M")

    # Verify integrity
    verify_backbone(backbone)

    return backbone


def verify_backbone(backbone: VLMBackbone) -> bool:
    """Verify the loaded backbone meets expected integrity criteria.

    Checks:
      1. Total param count is in the expected range for Qwen3.5-4B
      2. A sample of parameters have the expected dtype (bf16)
      3. No NaN or Inf in a sample of weight tensors
      4. backbone.hidden_size matches info.hidden_size

    Args:
        backbone: Loaded VLMBackbone to verify.

    Returns:
        True if all checks pass.

    Raises:
        RuntimeError: If any check fails, with details about the failure.
    """
    info = backbone.info

    # Check 1: param count in expected range
    total = info.param_count_total
    if not (_PARAM_COUNT_MIN <= total <= _PARAM_COUNT_MAX):
        raise RuntimeError(
            f"VLM backbone param count {total / 1e9:.2f}B is outside expected range "
            f"[{_PARAM_COUNT_MIN / 1e9:.1f}B, {_PARAM_COUNT_MAX / 1e9:.1f}B]. "
            "Check that the correct model was loaded."
        )

    # Check 2: dtype correctness on sample of params
    expected_dtype = _resolve_dtype(info.dtype)
    named_params = list(backbone._model.named_parameters())
    sample = named_params[:10]
    for name, param in sample:
        if param.dtype != expected_dtype:
            raise RuntimeError(
                f"Parameter '{name}' has dtype {param.dtype}, expected {expected_dtype}. "
                "Check torch_dtype config."
            )

    # Check 3: no NaN / Inf in sample
    for name, param in sample:
        data = param.data.float()  # cast to float32 for nan/inf check (bf16 may not support isnan)
        if torch.isnan(data).any():
            raise RuntimeError(f"NaN detected in parameter '{name}'. Possible corrupt download.")
        if torch.isinf(data).any():
            raise RuntimeError(f"Inf detected in parameter '{name}'. Possible corrupt download.")

    # Check 4: hidden_size consistency
    if backbone.hidden_size != info.hidden_size:
        raise RuntimeError(
            f"backbone.hidden_size ({backbone.hidden_size}) != "
            f"info.hidden_size ({info.hidden_size})"
        )

    logger.info(
        "VLMBackbone verification passed: %.2fB params, dtype=%s, hidden_size=%d",
        total / 1e9,
        info.dtype,
        info.hidden_size,
    )
    return True


# --- Processor utilities (Phase 3.1.2) ------------------------------------------


def estimate_vision_tokens(backbone: VLMBackbone, width: int = 320, height: int = 320) -> int:
    """Measure actual vision token count for a given image resolution.

    Creates a dummy image, runs it through the processor's chat template
    and tokenization pipeline, then counts image_pad tokens in the output.

    Args:
        backbone: Loaded VLMBackbone with processor attached.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Number of vision tokens per image at the given resolution.

    Raises:
        ImportError: If Pillow or numpy are not installed.
        RuntimeError: If no vision tokens are found in the processed output.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow and numpy are required for vision token estimation. "
            "Install with: pip install -e '.[vlm]'"
        ) from e

    processor = backbone.processor
    dummy = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "x"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[dummy], return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"][0]
    image_token_id = backbone.info.image_token_id
    n_vision = int((input_ids == image_token_id).sum().item())

    if n_vision == 0:
        raise RuntimeError(
            f"No vision tokens (id={image_token_id}) found in processed input. "
            "Check processor configuration and chat template."
        )

    logger.info("Vision token count for %dx%d: %d tokens/image", width, height, n_vision)
    return n_vision


def get_processor_info(backbone: VLMBackbone, image_resolution: int = 320) -> ProcessorInfo:
    """Extract processor metadata and measure vision token count.

    Args:
        backbone: Loaded VLMBackbone with processor attached.
        image_resolution: Square image resolution to measure vision tokens for.

    Returns:
        Frozen ProcessorInfo with all metadata populated.
    """
    processor = backbone.processor
    tokenizer = getattr(processor, "tokenizer", processor)
    info = backbone.info

    vision_tokens = estimate_vision_tokens(backbone, image_resolution, image_resolution)

    return ProcessorInfo(
        vocab_size=info.vocab_size,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        image_token_id=info.image_token_id,
        vision_start_token_id=info.vision_start_token_id,
        vision_end_token_id=info.vision_end_token_id,
        estimated_vision_tokens_per_image=vision_tokens,
        image_resolution=image_resolution,
    )


def preprocess_images(
    backbone: VLMBackbone,
    images: list[Any],
    text: str,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Preprocess images and text into model-ready tensors.

    Takes numpy uint8 RGB arrays (MuJoCo render output) or PIL Images
    plus a text prompt, and returns tensors ready for the VLM backbone.

    Args:
        backbone: Loaded VLMBackbone with processor attached.
        images: List of numpy uint8 RGB arrays or PIL Images.
        text: Text prompt for the model.
        device: Optional device to move output tensors to.

    Returns:
        Dict with model-ready tensors: input_ids, attention_mask,
        pixel_values, image_grid_thw, etc.

    Raises:
        ImportError: If Pillow or numpy are not installed.
        ValueError: If image arrays are not uint8.
        TypeError: If image is not a numpy array or PIL Image.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow and numpy are required for image preprocessing. "
            "Install with: pip install -e '.[vlm]'"
        ) from e

    processor = backbone.processor

    # Convert numpy arrays to PIL Images
    pil_images: list[Any] = []
    for img in images:
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                raise ValueError(
                    f"Expected uint8 image array, got {img.dtype}. "
                    "MuJoCo renders should be uint8 RGB."
                )
            pil_images.append(Image.fromarray(img))
        elif isinstance(img, Image.Image):
            pil_images.append(img)
        else:
            raise TypeError(f"Expected numpy array or PIL Image, got {type(img)}")

    # Build message content with image placeholders and text
    content: list[dict[str, str]] = []
    for _ in pil_images:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": content}]

    # Apply chat template to get formatted text with image placeholders
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process text and images into model tensors
    inputs = processor(
        text=[text_input],
        images=pil_images if pil_images else None,
        return_tensors="pt",
        padding=True,
    )

    result = dict(inputs)

    # Move to device if specified
    if device is not None:
        result = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in result.items()}

    return result


def compute_context_budget(
    vision_tokens_per_image: int,
    n_images: int = 4,
    max_seq_length: int = 8192,
    text_tokens_task: int = 40,
    text_tokens_per_step: int = 25,
    n_narration_steps: int = 8,
    text_tokens_outcome: int = 5,
) -> dict[str, int]:
    """Compute context window budget breakdown for a VLA sequence.

    Pure computation — no model or processor needed.

    Args:
        vision_tokens_per_image: Measured vision tokens per image.
        n_images: Number of camera views (default: 4, frozen contract).
        max_seq_length: Model context window size.
        text_tokens_task: Estimated tokens for task description.
        text_tokens_per_step: Estimated tokens per step narration.
        n_narration_steps: Number of step narrations in sequence.
        text_tokens_outcome: Estimated tokens for outcome text.

    Returns:
        Dict with budget breakdown including tokens_remaining_for_actions.
    """
    vision_total = n_images * vision_tokens_per_image
    narration_total = text_tokens_per_step * n_narration_steps
    text_total = text_tokens_task + narration_total + text_tokens_outcome
    used = vision_total + text_total
    remaining = max_seq_length - used

    return {
        "max_seq_length": max_seq_length,
        "n_images": n_images,
        "vision_tokens_per_image": vision_tokens_per_image,
        "vision_tokens_total": vision_total,
        "text_tokens_task": text_tokens_task,
        "text_tokens_narration": narration_total,
        "text_tokens_outcome": text_tokens_outcome,
        "text_tokens_total": text_total,
        "tokens_used": used,
        "tokens_remaining_for_actions": remaining,
    }


# --- Utilities ------------------------------------------------------------------


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Map dtype string from config to torch.dtype.

    Args:
        dtype_str: One of "bfloat16", "float16", "float32".

    Returns:
        Corresponding torch.dtype.

    Raises:
        ValueError: If dtype_str is not a recognized dtype.
    """
    dtype_map: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Expected one of: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]
