"""Loss contracts and utilities for Phase 4.1.

Defines the standard loss output interface (``LossOutput``), re-exports
``TokenType`` from Phase 3.2.0, and provides shape verification utilities
used by all downstream loss modules (``VLATextLoss``, ``VLAActionLoss``,
``VLACombinedLoss``) implemented in Phase 4.1.1–4.1.3.

Design notes:
- ``TokenType`` canonical definition lives in ``models.action_head`` (Phase 3.2.0).
  It is re-exported here so Phase 4.1+ consumers can import from ``models.losses``
  without changing Phase 3.2 import paths.
- ``LossOutput`` separates the differentiable ``loss`` tensor (used for backprop)
  from detached ``metrics`` floats (used for logging), ensuring monitoring never
  introduces unintended gradient paths.
- Shape verifiers are ``assert``-based debug helpers compiled away with
  ``python -O``. They are not runtime validators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export from Phase 3.2.0 (canonical definition stays in action_head)
from models.action_head import TokenType

__all__ = [
    "TokenType",
    "LossOutput",
    "verify_text_loss_inputs",
    "verify_action_loss_inputs",
    "VLATextLoss",
    "VLAActionLoss",
]

MetricValue = float | list[float]


# --- LossOutput ---------------------------------------------------------------


@dataclass
class LossOutput:
    """Standard return type for all VLA loss modules.

    All loss modules (``VLATextLoss``, ``VLAActionLoss``, ``VLACombinedLoss``)
    return a ``LossOutput``. The ``loss`` field participates in backpropagation;
    the ``metrics`` dict contains monitoring quantities that are already
    ``.detach().item()``-ed at creation time.

    Args:
        loss: Scalar differentiable loss tensor for backprop.
        metrics: Detached monitoring metrics as plain Python floats or small
                 float lists. Keys follow the ``loss/<name>`` W&B panel convention.

    Example::

        output = VLATextLoss()(logits, labels)
        output.loss.backward()         # differentiable
        print(output.metrics)          # {"perplexity": 12.3, "accuracy_top1": 0.42}
    """

    loss: torch.Tensor
    metrics: dict[str, MetricValue] = field(default_factory=dict)


# --- Shape verification utilities --------------------------------------------


def verify_text_loss_inputs(logits: torch.Tensor, labels: torch.Tensor) -> None:
    """Assert text loss input shapes are correct.

    Debug helper — compiled away with ``python -O``.

    Checks:
        - ``logits`` is 3-D: ``(B, S, V)``
        - ``labels`` is 2-D: ``(B, S)``
        - batch and sequence dimensions match between ``logits`` and ``labels``
        - ``labels`` dtype is ``torch.long``

    Args:
        logits: LM head output ``(B, S, vocab_size)``.
        labels: Next-token targets ``(B, S)``; ``-100`` at ignored positions.

    Raises:
        AssertionError: If any shape contract is violated (only in non-optimized mode).
    """
    assert logits.ndim == 3, f"logits must be 3-D (B, S, V), got {logits.ndim}-D"  # nosec B101
    assert labels.ndim == 2, f"labels must be 2-D (B, S), got {labels.ndim}-D"  # nosec B101
    assert (  # nosec B101
        logits.shape[0] == labels.shape[0]
    ), f"batch dim mismatch: logits {logits.shape[0]} vs labels {labels.shape[0]}"
    assert (  # nosec B101
        logits.shape[1] == labels.shape[1]
    ), f"seq dim mismatch: logits {logits.shape[1]} vs labels {labels.shape[1]}"
    assert (
        labels.dtype == torch.long
    ), f"labels must be torch.long, got {labels.dtype}"  # nosec B101


def verify_action_loss_inputs(
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    """Assert action loss input shapes are correct.

    Debug helper — compiled away with ``python -O``.

    Checks:
        - ``pred_v`` and ``target_v`` are 3-D: ``(B, T, action_dim)``
        - ``pred_v.shape == target_v.shape``
        - if ``mask`` is provided: 2-D ``(B, T)`` with values in ``{0, 1}``

    Args:
        pred_v: Predicted velocity ``(B, T, action_dim)``.
        target_v: Target velocity ``(B, T, action_dim)``.
        mask: Optional binary validity mask ``(B, T)`` — 1=valid, 0=padding.

    Raises:
        AssertionError: If any shape contract is violated (only in non-optimized mode).
    """
    assert pred_v.ndim == 3, f"pred_v must be 3-D (B, T, D), got {pred_v.ndim}-D"  # nosec B101
    assert (
        target_v.ndim == 3
    ), f"target_v must be 3-D (B, T, D), got {target_v.ndim}-D"  # nosec B101
    assert (  # nosec B101
        pred_v.shape == target_v.shape
    ), f"shape mismatch: pred_v {pred_v.shape} vs target_v {target_v.shape}"
    if mask is not None:
        assert mask.ndim == 2, f"mask must be 2-D (B, T), got {mask.ndim}-D"  # nosec B101
        assert (  # nosec B101
            mask.shape[0] == pred_v.shape[0]
        ), f"batch dim mismatch: mask {mask.shape[0]} vs pred_v {pred_v.shape[0]}"
        assert (  # nosec B101
            mask.shape[1] == pred_v.shape[1]
        ), f"seq dim mismatch: mask {mask.shape[1]} vs pred_v {pred_v.shape[1]}"
        unique_vals = mask.unique()
        assert all(  # nosec B101
            v in (0, 1) for v in unique_vals.tolist()
        ), f"mask values must be 0 or 1, got {unique_vals.tolist()}"


# --- VLATextLoss --------------------------------------------------------------


class VLATextLoss(nn.Module):
    """Autoregressive cross-entropy loss for text token positions.

    Handles causal shift internally: position i predicts token i+1.
    Operates on full-sequence logits with ``ignore_index`` masking — not on
    compacted text-only spans — to avoid cross-span prediction bugs at
    non-text boundaries (IMAGE, STATE, ACTION positions in the interleaved
    sequence become ignored positions in the shifted labels, which is correct).

    Returns ``LossOutput`` with perplexity and top-1 accuracy metrics.
    Has zero learnable parameters.

    Args:
        ignore_index: Token ID to exclude from loss computation (default: -100).
                      Set at non-text positions, BOS tokens, and padding.
        label_smoothing: Label smoothing factor in ``[0, 1)`` (default: 0.0).
                         Passed directly to ``F.cross_entropy``.

    Example::

        text_loss = VLATextLoss(ignore_index=-100)
        logits = torch.randn(4, 128, 150000)  # (B, S, vocab_size)
        labels = torch.full((4, 128), -100, dtype=torch.long)
        labels[:, 10:20] = torch.randint(0, 150000, (4, 10))  # text positions
        output = text_loss(logits, labels)
        output.loss.backward()
        print(output.metrics)  # {"perplexity": ..., "accuracy_top1": ..., "n_valid_tokens": 10}
    """

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> LossOutput:
        """Compute autoregressive cross-entropy loss.

        Args:
            logits: LM head output for the full sequence, shape ``(B, S, vocab_size)``.
                    All positions including non-text spans.
            labels: Next-token target IDs, shape ``(B, S)``, dtype ``torch.long``.
                    Non-text positions (IMAGE, STATE, ACTION), BOS tokens, and
                    padding are set to ``ignore_index=-100``.

        Returns:
            ``LossOutput`` with:

            - ``loss``: Scalar cross-entropy averaged over non-ignored tokens
              (differentiable).
            - ``metrics["perplexity"]``: ``exp(loss)`` — detached float.
            - ``metrics["accuracy_top1"]``: Top-1 accuracy on non-ignored tokens
              — detached float.
            - ``metrics["n_valid_tokens"]``: Count of non-ignored positions — int.
        """
        verify_text_loss_inputs(logits, labels)

        # Causal shift: position i predicts token i+1
        # Operates on full sequence; non-text boundaries produce ignored positions.
        shift_logits = logits[:, :-1, :].contiguous()  # (B, S-1, V)
        shift_labels = labels[:, 1:].contiguous()  # (B, S-1)

        # All-ignored edge case — explicit guard rather than relying on framework behavior
        valid_mask = shift_labels != self.ignore_index  # (B, S-1) bool
        n_valid = valid_mask.sum()
        if n_valid == 0:
            return LossOutput(
                loss=torch.tensor(0.0, device=logits.device, requires_grad=True),
                metrics={"perplexity": 1.0, "accuracy_top1": 0.0, "n_valid_tokens": 0},
            )

        # Cross-entropy — reduction="mean" averages over non-ignored tokens only
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        # Detached monitoring metrics
        perplexity = torch.exp(loss).detach().item()
        preds = shift_logits.detach().argmax(dim=-1)  # (B, S-1)
        correct = (preds == shift_labels) & valid_mask
        accuracy = correct.sum().float() / n_valid.float()

        return LossOutput(
            loss=loss,
            metrics={
                "perplexity": perplexity,
                "accuracy_top1": accuracy.item(),
                "n_valid_tokens": int(n_valid.item()),
            },
        )


# --- VLAActionLoss ------------------------------------------------------------


class VLAActionLoss(nn.Module):
    """Flow matching velocity MSE loss for action token positions.

    Computes masked MSE between predicted and target velocity fields, averaged
    over both valid (non-padded) action positions and action dimensions.
    Provides per-joint MSE breakdown and velocity norm diagnostics.

    Independent of ``FlowMatchingModule.loss()`` — both compute masked MSE but
    this module returns a ``LossOutput`` with monitoring metrics, whereas
    ``FlowMatchingModule.loss()`` returns a plain scalar tensor.

    Has zero learnable parameters.

    Args:
        action_dim: Action space dimension (default: 17).

    Example::

        action_loss = VLAActionLoss(action_dim=17)
        pred_v = torch.randn(4, 16, 17)     # (B, T, action_dim)
        target_v = torch.randn(4, 16, 17)
        mask = torch.ones(4, 16)
        mask[:, -2:] = 0                    # last 2 steps padded
        output = action_loss(pred_v, target_v, mask)
        output.loss.backward()
        print(output.metrics)  # {"per_joint_mse": [...], "mean_pred_velocity_norm": ..., ...}
    """

    def __init__(self, action_dim: int = 17) -> None:
        super().__init__()
        self.action_dim = action_dim

    def forward(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute masked MSE loss between predicted and target velocity.

        Args:
            pred_velocity: Predicted velocity from ``ActionOutputHead``,
                           shape ``(B, T, action_dim)``.
            target_velocity: Target velocity from ``FlowMatchingModule``,
                             shape ``(B, T, action_dim)``.
            chunk_mask: Binary mask, shape ``(B, T)`` — 1 for valid action
                        steps, 0 for padded positions. None means all
                        positions are valid.

        Returns:
            ``LossOutput`` with:

            - ``loss``: Scalar masked MSE averaged over valid positions and
              action dimensions (differentiable).
            - ``metrics["per_joint_mse"]``: ``list[float]`` of length
              ``action_dim`` — per-dimension MSE. Ordering matches the frozen
              17-D action space (derive joint names from
              ``sim.action_space.ARM_ACTUATOR_NAMES`` at logging time).
            - ``metrics["mean_pred_velocity_norm"]``: Mean L2 norm of predicted
              velocity vectors — detached float.
            - ``metrics["mean_target_velocity_norm"]``: Mean L2 norm of target
              velocity vectors — detached float.
            - ``metrics["mask_fraction_valid"]``: Fraction of valid (non-padded)
              positions — 1.0 when ``chunk_mask`` is None.
        """
        verify_action_loss_inputs(pred_velocity, target_velocity, chunk_mask)

        sq_error = (pred_velocity - target_velocity) ** 2  # (B, T, action_dim)

        if chunk_mask is not None:
            mask_3d = chunk_mask.unsqueeze(-1)  # (B, T, 1) — broadcasts over action_dim
            n_valid = chunk_mask.sum()

            if n_valid == 0:
                # All positions masked — explicit guard to avoid division by zero
                return LossOutput(
                    loss=torch.tensor(0.0, device=pred_velocity.device, requires_grad=True),
                    metrics={
                        "per_joint_mse": [0.0] * self.action_dim,
                        "mean_pred_velocity_norm": 0.0,
                        "mean_target_velocity_norm": 0.0,
                        "mask_fraction_valid": 0.0,
                    },
                )

            loss = (sq_error * mask_3d).sum() / (n_valid * self.action_dim)
            # Per-joint: masked mean over (B, T) positions, per dimension
            per_joint = (sq_error * mask_3d).sum(dim=(0, 1)) / n_valid  # (action_dim,)
            mask_fraction = chunk_mask.float().mean().item()
        else:
            loss = sq_error.mean()
            per_joint = sq_error.mean(dim=(0, 1))  # (action_dim,)
            mask_fraction = 1.0

        # Velocity norm diagnostics (detached)
        mean_pred_norm = pred_velocity.detach().norm(dim=-1).mean().item()
        mean_target_norm = target_velocity.detach().norm(dim=-1).mean().item()

        per_joint_values = [float(x) for x in per_joint.detach().tolist()]

        return LossOutput(
            loss=loss,
            metrics={
                "per_joint_mse": per_joint_values,
                "mean_pred_velocity_norm": mean_pred_norm,
                "mean_target_velocity_norm": mean_target_norm,
                "mask_fraction_valid": mask_fraction,
            },
        )
