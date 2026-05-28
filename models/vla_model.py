"""VLA model assembly — composes VLM backbone with action head components.

Phase 3.2.4: Wires the Phase 3.1 VLM backbone (Qwen3.5-4B) together with all
Phase 3.2 action head components into a single trainable model with Transfusion-
style dual loss (AR text + flow matching action) and multi-step ODE denoising
inference.

Multimodal injection strategy: Scenario C (full embedding control).
The backbone's XOR constraint prevents passing both ``input_ids`` and
``inputs_embeds`` simultaneously (Qwen3_5Model line 1736). VLAModel therefore
assembles the full embedding sequence manually:
    1. text embeddings via backbone.get_text_embeddings(input_ids)
    2. vision features via backbone.get_vision_features(pixel_values, ...)
       scattered at image_token_id positions in text embeddings
    3. state token via state_projector(robot_state)
    4. action tokens via action_projector(noisy_actions, t)
    5. concatenated → inputs_embeds → backbone.get_hidden_states(inputs_embeds=...)

Position IDs default to 1D sequential when inputs_embeds is provided without
input_ids (backbone fallback). Loses 3D spatial RoPE for vision tokens —
acceptable with frozen backbone; optimize in Phase 3.3 if needed.

Usage::

    from models.vla_model import VLAModel, load_vla_model

    model = load_vla_model(cfg)

    # Training
    losses = model.forward(batch)  # {total_loss, text_loss, action_loss}
    losses["total_loss"].backward()

    # Inference
    actions = model.predict_actions(input_ids, attention_mask, robot_state)
    # → (B, 16, 17) denoised action chunk
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from models.action_head import (
    ActionChunkConfig,
    ActionOutputHead,
    FlowMatchingConfig,
    FlowMatchingModule,
    NoisyActionProjector,
    RobotStateProjector,
)
from models.vlm_backbone import VLMBackboneInfo

logger = logging.getLogger(__name__)


class VLABackbone(Protocol):
    @property
    def hidden_size(self) -> int: ...

    @property
    def info(self) -> VLMBackboneInfo: ...

    @property
    def lm_head(self) -> nn.Module: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def get_vision_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
    ) -> list[torch.Tensor]: ...

    def get_hidden_states(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor: ...

    def freeze_backbone(self) -> None: ...

    def unfreeze_backbone(self) -> None: ...

    def freeze_vision(self) -> None: ...


class VLAModel(nn.Module):
    """Vision-Language-Action model with Transfusion-style dual loss.

    Composes:
        backbone          — Phase 3.1 VLMBackbone (Qwen3.5-4B, frozen by default)
        state_projector   — Phase 3.2.2 RobotStateProjector (52-D → H)
        action_projector  — Phase 3.2.3 NoisyActionProjector ((17-D + t) → H)
        action_output_head— Phase 3.2.3 ActionOutputHead (H → 17-D velocity)
        flow_matching     — Phase 3.2.1 FlowMatchingModule (CFM math, no params)

    Sequence layout (per segment)::

        [text_tokens + scattered_vision | state_token(1) | action_tokens(16)]
        ←————————— seq_text ——————————→ ←—— n_seg ———→ ←— n_action_tokens —→

    Forward pass returns Transfusion dual loss:
        text_loss   — AR cross-entropy at text positions
        action_loss — FM velocity MSE at action positions
        total_loss  — lambda_text * text_loss + lambda_action * action_loss

    Action head components are kept in float32 for numerical stability (EO-1
    pattern). Their outputs are cast to backbone dtype (bf16) for sequence
    assembly, then cast back to float32 for velocity prediction.

    Args:
        backbone: Loaded VLMBackbone from Phase 3.1.
        cfg: Full Hydra config with model.action_head section.

    Example::

        model = VLAModel(backbone, cfg)

        # Training
        losses = model.forward(batch)
        losses["total_loss"].backward()

        # Inference
        actions = model.predict_actions(input_ids, attn_mask, robot_state)
        # → (B, chunk_size, action_dim)
    """

    def __init__(self, backbone: VLABackbone, cfg: DictConfig) -> None:
        super().__init__()

        self.backbone: VLABackbone = backbone
        H: int = backbone.hidden_size

        ah_cfg = _extract_action_head_cfg(cfg)

        # Action head components (trainable, kept in float32)
        self.state_projector: RobotStateProjector = RobotStateProjector.from_cfg(ah_cfg, H)
        self.action_projector: NoisyActionProjector = NoisyActionProjector.from_cfg(ah_cfg, H)
        self.action_output_head: ActionOutputHead = ActionOutputHead.from_cfg(ah_cfg, H)

        # Flow matching module (no learnable parameters)
        fm_cfg = ah_cfg.get("flow_matching") if ah_cfg else None
        self.flow_matching = FlowMatchingModule(FlowMatchingConfig.from_cfg(fm_cfg))

        # Frozen config values
        self.chunk_config = ActionChunkConfig.from_cfg(ah_cfg)
        self._lambda_text = float((ah_cfg or {}).get("loss", {}).get("lambda_text", 1.0))
        self._lambda_action = float((ah_cfg or {}).get("loss", {}).get("lambda_action", 1.0))

        # Float32 action head (EO-1 pattern for numerical stability)
        float32_head = bool((ah_cfg or {}).get("float32_head", True))
        if float32_head:
            self.state_projector.float()
            self.action_projector.float()
            self.action_output_head.float()

        # Move action head to same device as backbone (handles device_map="auto")
        bb_device = self._detect_backbone_device()
        if bb_device.type != "cpu":
            self.state_projector.to(bb_device)
            self.action_projector.to(bb_device)
            self.action_output_head.to(bb_device)

        n_ah = self._count_action_head_params()
        logger.info(
            "VLAModel assembled: hidden_size=%d, action_head_params=%.1fM",
            H,
            n_ah / 1e6,
        )

    # --- Properties -------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        """Hidden dimension of the VLM backbone."""
        return self.backbone.hidden_size

    # --- Private helpers --------------------------------------------------------

    def _detect_backbone_device(self) -> torch.device:
        """Detect the device the backbone is on (first parameter's device)."""
        for p in self.backbone.parameters():
            return p.device
        return torch.device("cpu")

    def _count_action_head_params(self) -> int:
        return sum(
            p.numel()
            for m in (self.state_projector, self.action_projector, self.action_output_head)
            for p in m.parameters()
        )

    def _backbone_dtype(self) -> torch.dtype:
        """Return the backbone's working dtype for embedding concatenation."""
        for p in self.backbone.parameters():
            return p.dtype
        return torch.float32

    def _scatter_vision_features(
        self,
        text_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        vision_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Scatter vision encoder features at image token positions in text embeddings.

        Replicates the backbone's native ``masked_scatter`` behaviour for Scenario C
        so that vision tokens occupy the correct positions in the assembled sequence.

        Args:
            text_embeds: ``(B, seq_text, H)`` — from ``embed_tokens(input_ids)``.
            input_ids: ``(B, seq_text)`` — used to locate ``image_token_id`` positions.
            vision_features: List of per-image feature tensors (from
                ``backbone.get_vision_features()``). Each has shape
                ``(n_tokens_i, H)``.

        Returns:
            Modified ``text_embeds`` with vision features at image token positions,
            shape ``(B, seq_text, H)``.
        """
        if not vision_features:
            return text_embeds

        image_token_id = self.backbone.info.image_token_id
        image_mask = input_ids.eq(image_token_id)  # (B, seq_text) bool tensor

        # Concatenate all per-image features into a single flat tensor
        all_features = torch.cat(vision_features, dim=0).to(  # (total_vis, H)
            text_embeds.device, text_embeds.dtype
        )

        # 3-D mask for masked_scatter: (B, seq_text, H)
        image_mask_3d = image_mask.unsqueeze(-1).expand_as(text_embeds)
        return text_embeds.masked_scatter(image_mask_3d, all_features)

    def assemble_sequence(
        self,
        text_embeds: torch.Tensor,
        state_embeds: torch.Tensor,
        action_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the full VLA input sequence from per-modality embeddings.

        Layout: ``[text_with_vision(seq_text) | state(n_seg) | action(n_act)]``

        State and action embeddings are cast to backbone dtype (bf16) before
        concatenation.

        Args:
            text_embeds: ``(B, seq_text, H)`` — text embeddings with vision
                features scattered in at image token positions.
            state_embeds: ``(B, n_seg, H)`` — robot state tokens (float32 from
                projector).
            action_embeds: ``(B, n_act, H)`` — noisy action tokens (float32 from
                projector).

        Returns:
            ``(B, seq_total, H)`` full embedding sequence in backbone dtype.
        """
        bb_dtype = self._backbone_dtype()
        return torch.cat(
            [
                text_embeds,
                state_embeds.to(bb_dtype),
                action_embeds.to(bb_dtype),
            ],
            dim=1,
        )

    # --- Training forward pass --------------------------------------------------

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Training forward pass with Transfusion dual loss.

        Args:
            batch: Dict with the following keys:

                ``input_ids`` ``(B, seq_text)``
                    Text token IDs, may contain ``image_token_id`` placeholders.

                ``pixel_values`` *(optional)*
                    Preprocessed image tensors for the vision encoder.

                ``image_grid_thw`` *(optional)*
                    Image grid dimensions ``(n_images, 3)``.

                ``attention_mask`` ``(B, seq_total)``
                    1 = attend, 0 = ignore. Must cover the full assembled sequence.

                ``robot_states`` ``(B, n_seg, 52)``
                    Normalised robot state per sequence segment.

                ``action_chunks`` ``(B, n_chunks, chunk_size, action_dim)``
                    Ground-truth action chunks (or pre-flattened
                    ``(B, n_action_tokens, action_dim)``).

                ``chunk_masks`` ``(B, n_chunks, chunk_size)`` or ``(B, n_action_tokens)``
                    Binary mask — 1 for real steps, 0 for padding.

                ``text_labels`` ``(B, seq_text)``
                    Next-token targets. -100 at positions to ignore (image tokens,
                    padding, non-text positions).

        Returns:
            Dict with scalar tensors:
                ``total_loss`` — differentiable combined loss.
                ``text_loss``  — AR cross-entropy (detached for logging).
                ``action_loss``— FM velocity MSE (detached for logging).
        """
        input_ids = batch["input_ids"]
        B = input_ids.shape[0]
        device = input_ids.device

        # --- 1. Flow matching setup ---
        action_chunks = batch["action_chunks"]
        chunk_masks = batch["chunk_masks"]

        # Flatten to (B, n_action_tokens, action_dim)
        if action_chunks.ndim == 4:
            n_total = action_chunks.shape[1] * action_chunks.shape[2]
            actions_flat = action_chunks.reshape(B, n_total, self.chunk_config.action_dim)
            masks_flat = chunk_masks.reshape(B, n_total)
        else:
            # Already (B, n_action_tokens, action_dim)
            actions_flat = action_chunks
            masks_flat = chunk_masks

        actions_flat = actions_flat.float()
        n_action_tokens = actions_flat.shape[1]

        t = self.flow_matching.sample_timestep(B, device=device)  # (B, 1, 1)
        noise = torch.randn_like(actions_flat)
        noisy_actions = self.flow_matching.interpolate(actions_flat, noise, t)
        target_velocity = self.flow_matching.target_velocity(actions_flat, noise)

        # --- 2. Get text embeddings ---
        text_embeds = self.backbone.get_text_embeddings(input_ids)
        seq_text = text_embeds.shape[1]

        # --- 3. Scatter vision features (optional) ---
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            vision_features = self.backbone.get_vision_features(
                pixel_values, batch.get("image_grid_thw")
            )
            text_embeds = self._scatter_vision_features(text_embeds, input_ids, vision_features)

        # --- 4. Project state tokens ---
        robot_states = batch["robot_states"].float()  # (B, n_seg, 52)
        B_s, n_seg, _ = robot_states.shape
        states_flat_input = robot_states.reshape(B_s * n_seg, -1)  # (B*n_seg, 52)
        state_tokens = self.state_projector(states_flat_input)  # (B*n_seg, 1, H)
        state_embeds = state_tokens.reshape(B_s, n_seg, -1)  # (B, n_seg, H)

        # --- 5. Project noisy action tokens ---
        t_for_proj = t.squeeze(-1)  # (B, 1) — NoisyActionProjector expects (B, 1)
        action_embeds = self.action_projector(noisy_actions, t_for_proj)  # (B, n_act, H)

        # --- 6. Assemble full sequence ---
        full_embeds = self.assemble_sequence(text_embeds, state_embeds, action_embeds)

        # --- 7. Forward through backbone (Scenario C) ---
        hidden_states = self.backbone.get_hidden_states(
            inputs_embeds=full_embeds,
            attention_mask=batch.get("attention_mask"),
        )  # (B, seq_total, H)

        # --- 8. Text loss (AR cross-entropy) ---
        text_hidden = hidden_states[:, :seq_text, :]  # (B, seq_text, H)
        lm_head = cast(nn.Module, self.backbone.lm_head)
        text_logits = cast(torch.Tensor, lm_head(text_hidden))
        # (B, seq_text, vocab_size)

        text_labels = batch["text_labels"]  # (B, seq_text)
        shift_logits = text_logits[:, :-1, :].contiguous()
        shift_labels = text_labels[:, 1:].contiguous()
        text_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        # --- 9. Action loss (flow matching MSE) ---
        action_start = seq_text + n_seg
        action_end = action_start + n_action_tokens
        action_hidden = hidden_states[:, action_start:action_end, :]  # (B, n_act, H)

        pred_velocity = self.action_output_head(action_hidden.float())  # (B, n_act, 17)
        action_loss = self.flow_matching.loss(pred_velocity, target_velocity, masks_flat)

        # --- 10. Combined loss ---
        total_loss = self._lambda_text * text_loss + self._lambda_action * action_loss

        return {
            "total_loss": total_loss,
            "text_loss": text_loss.detach(),
            "action_loss": action_loss.detach(),
        }

    # --- Inference --------------------------------------------------------------

    @torch.no_grad()
    def predict_actions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        robot_state: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        K: int | None = None,
    ) -> torch.Tensor:
        """Predict a denoised action chunk via multi-step ODE integration.

        Iteratively denoises from Gaussian noise to a clean action chunk using
        K steps of the configured ODE solver (Euler by default). The context
        embeddings (text + vision + state) are computed once and reused across
        all K denoising iterations.

        Args:
            input_ids: ``(B, seq_text)`` text token IDs with image placeholders.
            attention_mask: ``(B, seq_text)`` mask for text tokens. Extended to
                cover the full sequence inside this method.
            robot_state: ``(B, 52)`` normalised robot state for this observation.
            pixel_values: Optional image tensors for vision processing.
            image_grid_thw: Optional image grid dimensions ``(n_images, 3)``.
            K: Number of ODE denoising steps. Defaults to
                ``flow_matching.config.n_denoising_steps`` (10).

        Returns:
            ``(B, chunk_size, action_dim)`` denoised action chunk in float32.
        """
        B = input_ids.shape[0]
        device = input_ids.device
        chunk_size = self.chunk_config.chunk_size
        action_dim = self.chunk_config.action_dim

        if K is None:
            K = self.flow_matching.config.n_denoising_steps

        # --- Compute context embeddings once (cacheable for KV-cache opt) ---
        text_embeds = self.backbone.get_text_embeddings(input_ids)
        if pixel_values is not None:
            vision_features = self.backbone.get_vision_features(pixel_values, image_grid_thw)
            text_embeds = self._scatter_vision_features(text_embeds, input_ids, vision_features)

        # State embedding: (B, 52) → (B, 1, H)
        state_embeds = self.state_projector(robot_state.float())  # (B, 1, H)

        # Build attention mask for full sequence (all ones — no KV cache yet)
        seq_text = text_embeds.shape[1]

        def predict_velocity(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Velocity prediction closure for ODE solver.

            Args:
                x_t: Noisy actions ``(B, chunk_size, action_dim)`` at time t.
                t: Timestep tensor ``(B, 1, 1)``.

            Returns:
                Predicted velocity ``(B, chunk_size, action_dim)`` in float32.
            """
            t_proj = t.squeeze(-1)  # (B, 1)
            action_embeds = self.action_projector(x_t, t_proj)  # (B, C, H)

            full_embeds = self.assemble_sequence(text_embeds, state_embeds, action_embeds)
            seq_total = full_embeds.shape[1]
            attn_mask = torch.ones(B, seq_total, dtype=torch.long, device=device)

            hidden = self.backbone.get_hidden_states(
                inputs_embeds=full_embeds,
                attention_mask=attn_mask,
            )

            action_start = seq_text + state_embeds.shape[1]
            action_hidden = hidden[:, action_start : action_start + chunk_size, :]
            return cast(torch.Tensor, self.action_output_head(action_hidden.float()))

        action_chunk = self.flow_matching.denoise(
            predict_fn=predict_velocity,
            shape=(B, chunk_size, action_dim),
            K=K,
            device=device,
        )

        return action_chunk

    # --- Freeze / unfreeze delegation -------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        self.backbone.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        self.backbone.unfreeze_backbone()

    def freeze_vision(self) -> None:
        """Freeze backbone vision encoder only."""
        self.backbone.freeze_vision()


# --- Config helpers -------------------------------------------------------------


def _extract_action_head_cfg(cfg: DictConfig) -> dict[str, Any] | None:
    """Extract the action_head config section as a plain dict.

    Args:
        cfg: Full Hydra config. Expects ``cfg.model.action_head`` section.

    Returns:
        Plain dict with action head config, or None if section is absent.
    """
    try:
        if hasattr(cfg, "model") and hasattr(cfg.model, "action_head"):
            resolved = OmegaConf.to_container(cfg.model.action_head, resolve=True)
            if isinstance(resolved, dict):
                return cast(dict[str, Any], resolved)
    except Exception:
        pass
    return None


# --- Factory function -----------------------------------------------------------


def load_vla_model(cfg: DictConfig) -> VLAModel:
    """Load and assemble the full VLA model from Hydra config.

    Loads the Phase 3.1 VLM backbone, then wraps it with all Phase 3.2 action
    head components into a ``VLAModel``.

    Args:
        cfg: Full Hydra config with ``model.vlm`` and ``model.action_head``
             sections (e.g. from ``configs/model/vla.yaml``).

    Returns:
        Assembled and ready-to-use ``VLAModel``.

    Raises:
        ImportError: If transformers is not installed.
        RuntimeError: If backbone verification fails.
    """
    from models.vlm_backbone import load_vlm_backbone

    backbone = load_vlm_backbone(cfg)
    return VLAModel(backbone, cfg)
