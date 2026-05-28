"""Tests for loss contracts and utilities (Phase 4.1.0).

Tests the ``LossOutput`` dataclass, ``TokenType`` re-export from ``models.losses``,
and shape verification utilities (``verify_text_loss_inputs``,
``verify_action_loss_inputs``).

All tests are CPU-only with no external dependencies.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.losses import (
    LossOutput,
    TokenType,
    VLAActionLoss,
    VLATextLoss,
    verify_action_loss_inputs,
    verify_text_loss_inputs,
)

# --- TokenType re-export tests -----------------------------------------------


class TestTokenType:
    """Verify TokenType re-exported from models.losses matches Phase 3.2.0 contract."""

    def test_token_type_reexport_values(self):
        """TokenType enum values match Phase 3.2.0 contract."""
        assert TokenType.TEXT == 0
        assert TokenType.IMAGE == 1
        assert TokenType.STATE == 2
        assert TokenType.ACTION == 3
        assert len(TokenType) == 4

    def test_token_type_identity(self):
        """Re-exported TokenType is the same class as action_head.TokenType."""
        from models.action_head import TokenType as OriginalTokenType

        assert TokenType is OriginalTokenType


# --- LossOutput tests ---------------------------------------------------------


class TestLossOutput:
    """Verify LossOutput dataclass construction and field access."""

    def test_construction_with_metrics(self):
        """LossOutput constructs with a loss tensor and metrics dict."""
        loss = torch.tensor(2.5, requires_grad=True)
        metrics = {"perplexity": 12.18, "accuracy_top1": 0.42}
        output = LossOutput(loss=loss, metrics=metrics)

        assert output.loss is loss
        assert output.loss.requires_grad
        assert output.metrics == metrics
        assert output.metrics["perplexity"] == pytest.approx(12.18)

    def test_default_metrics(self):
        """LossOutput with only loss has empty metrics dict by default."""
        loss = torch.tensor(1.0)
        output = LossOutput(loss=loss)

        assert output.loss is loss
        assert output.metrics == {}
        assert isinstance(output.metrics, dict)

    def test_metrics_mutable(self):
        """Metrics dict is mutable (can be updated after construction)."""
        output = LossOutput(loss=torch.tensor(1.0))
        output.metrics["new_metric"] = 0.5
        assert output.metrics["new_metric"] == 0.5


# --- verify_text_loss_inputs tests -------------------------------------------


class TestVerifyTextLossInputs:
    """Verify shape checker for text loss inputs."""

    def test_accepts_correct_shapes(self):
        """Correct shapes pass without error."""
        B, S, V = 4, 20, 1000
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))
        verify_text_loss_inputs(logits, labels)  # should not raise

    def test_rejects_wrong_logits_ndim(self):
        """2-D logits raise AssertionError."""
        logits = torch.randn(4, 1000)  # missing seq dim
        labels = torch.randint(0, 1000, (4, 20))
        with pytest.raises(AssertionError, match="logits must be 3-D"):
            verify_text_loss_inputs(logits, labels)

    def test_rejects_wrong_labels_ndim(self):
        """3-D labels raise AssertionError."""
        logits = torch.randn(4, 20, 1000)
        labels = torch.randint(0, 1000, (4, 20, 1))  # extra dim
        with pytest.raises(AssertionError, match="labels must be 2-D"):
            verify_text_loss_inputs(logits, labels)

    def test_rejects_batch_dim_mismatch(self):
        """Mismatched batch dimensions raise AssertionError."""
        logits = torch.randn(4, 20, 1000)
        labels = torch.randint(0, 1000, (2, 20))
        with pytest.raises(AssertionError, match="batch dim mismatch"):
            verify_text_loss_inputs(logits, labels)

    def test_rejects_seq_dim_mismatch(self):
        """Mismatched sequence dimensions raise AssertionError."""
        logits = torch.randn(4, 20, 1000)
        labels = torch.randint(0, 1000, (4, 10))
        with pytest.raises(AssertionError, match="seq dim mismatch"):
            verify_text_loss_inputs(logits, labels)

    def test_rejects_wrong_labels_dtype(self):
        """Float labels raise AssertionError."""
        logits = torch.randn(4, 20, 1000)
        labels = torch.randn(4, 20)  # float instead of long
        with pytest.raises(AssertionError, match="labels must be torch.long"):
            verify_text_loss_inputs(logits, labels)


# --- verify_action_loss_inputs tests -----------------------------------------


class TestVerifyActionLossInputs:
    """Verify shape checker for action loss inputs."""

    def test_accepts_correct_shapes_no_mask(self):
        """Correct shapes without mask pass without error."""
        B, T, D = 4, 16, 17
        pred_v = torch.randn(B, T, D)
        target_v = torch.randn(B, T, D)
        verify_action_loss_inputs(pred_v, target_v)  # should not raise

    def test_accepts_correct_shapes_with_mask(self):
        """Correct shapes with valid binary mask pass without error."""
        B, T, D = 4, 16, 17
        pred_v = torch.randn(B, T, D)
        target_v = torch.randn(B, T, D)
        mask = torch.ones(B, T)
        mask[:, -2:] = 0  # last 2 steps padded
        verify_action_loss_inputs(pred_v, target_v, mask)  # should not raise

    def test_rejects_wrong_pred_ndim(self):
        """2-D pred_v raises AssertionError."""
        pred_v = torch.randn(4, 17)  # missing time dim
        target_v = torch.randn(4, 16, 17)
        with pytest.raises(AssertionError, match="pred_v must be 3-D"):
            verify_action_loss_inputs(pred_v, target_v)

    def test_rejects_shape_mismatch(self):
        """Mismatched pred_v and target_v shapes raise AssertionError."""
        pred_v = torch.randn(4, 16, 17)
        target_v = torch.randn(4, 16, 15)  # wrong action dim
        with pytest.raises(AssertionError, match="shape mismatch"):
            verify_action_loss_inputs(pred_v, target_v)

    def test_rejects_wrong_mask_ndim(self):
        """3-D mask raises AssertionError."""
        pred_v = torch.randn(4, 16, 17)
        target_v = torch.randn(4, 16, 17)
        mask = torch.ones(4, 16, 1)  # extra dim
        with pytest.raises(AssertionError, match="mask must be 2-D"):
            verify_action_loss_inputs(pred_v, target_v, mask)

    def test_rejects_mask_batch_mismatch(self):
        """Mask with wrong batch dim raises AssertionError."""
        pred_v = torch.randn(4, 16, 17)
        target_v = torch.randn(4, 16, 17)
        mask = torch.ones(2, 16)
        with pytest.raises(AssertionError, match="batch dim mismatch"):
            verify_action_loss_inputs(pred_v, target_v, mask)

    def test_rejects_invalid_mask_values(self):
        """Mask with values outside {0, 1} raises AssertionError."""
        pred_v = torch.randn(4, 16, 17)
        target_v = torch.randn(4, 16, 17)
        mask = torch.full((4, 16), 0.5)  # invalid values
        with pytest.raises(AssertionError, match="mask values must be 0 or 1"):
            verify_action_loss_inputs(pred_v, target_v, mask)


# --- VLATextLoss tests -------------------------------------------------------


class TestVLATextLoss:
    """Verify VLATextLoss: causal shift, CE correctness, metrics, edge cases."""

    def test_correct_cross_entropy(self):
        """Loss matches hand-computed cross-entropy on a small known example.

        3-token sequence (B=1, S=3, V=4). Causal shift: position 0 predicts
        label[1], position 1 predicts label[2]. Position 2 has no target.
        """
        B, S, V = 1, 3, 4
        # One-hot logits: argmax at known positions
        logits = torch.zeros(B, S, V)
        logits[0, 0, 2] = 10.0  # predicts class 2
        logits[0, 1, 1] = 10.0  # predicts class 1

        # Labels: shifted — label[1]=2 (correct for pos 0), label[2]=3 (wrong for pos 1)
        labels = torch.tensor([[0, 2, 3]], dtype=torch.long)  # (1, 3)

        module = VLATextLoss()
        output = module(logits, labels)

        # Shift: shift_logits[:,0,:] predicts shift_labels[:,0]=2 (correct)
        #        shift_logits[:,1,:] predicts shift_labels[:,1]=3 (wrong, pred=1)
        expected_loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), labels[:, 1:].reshape(-1))
        assert output.loss.item() == pytest.approx(expected_loss.item(), rel=1e-5)
        assert output.metrics["n_valid_tokens"] == 2

    def test_ignore_index_respected(self):
        """Positions with -100 labels do not contribute to loss."""
        B, S, V = 2, 6, 10
        logits = torch.randn(B, S, V)
        labels = torch.full((B, S), -100, dtype=torch.long)
        # Only positions 2 and 3 are valid text (predict labels[3] and labels[4])
        labels[:, 3] = torch.randint(0, V, (B,))
        labels[:, 4] = torch.randint(0, V, (B,))

        module = VLATextLoss()
        output = module(logits, labels)

        # Reference: compute manually on only the valid shifted positions
        shift_logits = logits[:, :-1, :]  # (B, S-1, V)
        shift_labels = labels[:, 1:]  # (B, S-1)
        ref = F.cross_entropy(
            shift_logits.reshape(-1, V), shift_labels.reshape(-1), ignore_index=-100
        )

        assert output.loss.item() == pytest.approx(ref.item(), rel=1e-5)
        assert output.metrics["n_valid_tokens"] == B * 2

    def test_perplexity_equals_exp_loss(self):
        """Perplexity metric equals exp(loss)."""
        B, S, V = 2, 8, 20
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))

        module = VLATextLoss()
        output = module(logits, labels)

        expected_ppl = torch.exp(output.loss).item()
        assert output.metrics["perplexity"] == pytest.approx(expected_ppl, rel=1e-5)

    def test_accuracy_computation(self):
        """Top-1 accuracy is correct for known predictions."""
        B, S, V = 1, 5, 4
        # Logits with argmax at class 0 for all positions
        logits = torch.zeros(B, S, V)
        logits[:, :, 0] = 10.0  # predicts class 0 everywhere

        # Labels: shift means position i predicts labels[i+1]
        # shift_labels = labels[:, 1:] = [1, 0, 0, 0]  → 3 correct out of 4
        labels = torch.tensor([[99, 1, 0, 0, 0]], dtype=torch.long)
        # shift_labels: [1, 0, 0, 0] — pred=0 matches [False, True, True, True] → 3/4

        module = VLATextLoss()
        output = module(logits, labels)

        assert output.metrics["accuracy_top1"] == pytest.approx(3 / 4, rel=1e-5)

    def test_label_smoothing_changes_loss(self):
        """label_smoothing > 0 produces a different CE than label_smoothing = 0."""
        B, S, V = 2, 10, 5
        # Sharp logits amplify the smoothing effect (random logits near uniform
        # produce a negligible difference for large V).
        torch.manual_seed(42)
        logits = torch.randn(B, S, V) * 5.0
        labels = torch.randint(0, V, (B, S))

        no_smooth = VLATextLoss(label_smoothing=0.0)
        with_smooth = VLATextLoss(label_smoothing=0.1)

        out_no = no_smooth(logits, labels)
        out_sm = with_smooth(logits, labels)

        assert out_no.loss.item() != pytest.approx(out_sm.loss.item(), rel=1e-3)

    def test_all_ignored_returns_zero(self):
        """All-ignored batch returns loss=0, perplexity=1, accuracy=0, no NaN."""
        B, S, V = 3, 12, 100
        logits = torch.randn(B, S, V)
        labels = torch.full((B, S), -100, dtype=torch.long)

        module = VLATextLoss()
        output = module(logits, labels)

        assert output.loss.item() == pytest.approx(0.0)
        assert output.loss.requires_grad  # graph valid for backward
        assert output.metrics["perplexity"] == pytest.approx(1.0)
        assert output.metrics["accuracy_top1"] == pytest.approx(0.0)
        assert output.metrics["n_valid_tokens"] == 0
        assert not torch.isnan(output.loss)

    def test_batch_consistency(self):
        """Same per-sample logits/labels produce consistent loss across batch sizes."""
        S, V = 8, 30
        logits_1 = torch.randn(1, S, V)
        labels_1 = torch.randint(0, V, (1, S))

        # Repeat across batch dimension
        logits_2 = logits_1.expand(4, S, V)
        labels_2 = labels_1.expand(4, S)

        module = VLATextLoss()
        out1 = module(logits_1, labels_1)
        out2 = module(logits_2.contiguous(), labels_2.contiguous())

        assert out1.loss.item() == pytest.approx(out2.loss.item(), rel=1e-5)

    def test_gradient_flows_to_logits(self):
        """backward() produces non-zero gradient on logits tensor."""
        B, S, V = 2, 6, 20
        logits = torch.randn(B, S, V, requires_grad=True)
        labels = torch.randint(0, V, (B, S))

        module = VLATextLoss()
        output = module(logits, labels)
        output.loss.backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0.0


# --- VLAActionLoss tests ------------------------------------------------------


class TestVLAActionLoss:
    """Verify VLAActionLoss: masked MSE, per-joint breakdown, diagnostics."""

    def test_correct_mse(self):
        """Loss matches hand-computed MSE on a small known example.

        B=1, T=2, D=3. Known pred and target values so MSE is exact.
        """
        B, T, D = 1, 2, 3
        # pred = [[1, 2, 3], [4, 5, 6]], target = [[0, 0, 0], [0, 0, 0]]
        pred_v = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
        target_v = torch.zeros(B, T, D)
        # sq_error = [[1, 4, 9], [16, 25, 36]], mean = (1+4+9+16+25+36) / 6 = 91/6
        expected = (1 + 4 + 9 + 16 + 25 + 36) / 6.0

        module = VLAActionLoss(action_dim=D)
        output = module(pred_v, target_v)

        assert output.loss.item() == pytest.approx(expected, rel=1e-5)

    def test_chunk_mask_excludes_padding(self):
        """Padded positions (mask=0) do not contribute to loss."""
        B, T, D = 2, 4, 17
        pred_v = torch.randn(B, T, D)
        target_v = torch.zeros(B, T, D)

        # Only first 2 of 4 positions are valid
        mask = torch.zeros(B, T)
        mask[:, :2] = 1.0

        module = VLAActionLoss(action_dim=D)
        output_masked = module(pred_v, target_v, mask)

        # Reference: compute only on the first 2 positions
        sq_valid = (pred_v[:, :2, :] - target_v[:, :2, :]) ** 2
        expected = sq_valid.sum().item() / (mask.sum().item() * D)

        assert output_masked.loss.item() == pytest.approx(expected, rel=1e-5)

    def test_all_masked_returns_zero(self):
        """All-masked batch returns loss=0, no NaN, requires_grad=True."""
        B, T, D = 2, 16, 17
        pred_v = torch.randn(B, T, D)
        target_v = torch.randn(B, T, D)
        mask = torch.zeros(B, T)

        module = VLAActionLoss(action_dim=D)
        output = module(pred_v, target_v, mask)

        assert output.loss.item() == pytest.approx(0.0)
        assert output.loss.requires_grad
        assert not torch.isnan(output.loss)
        assert output.metrics["per_joint_mse"] == [0.0] * D
        assert output.metrics["mask_fraction_valid"] == pytest.approx(0.0)

    def test_per_joint_mse_consistency(self):
        """sum(per_joint_mse) ≈ action_dim * loss (within float precision)."""
        B, T, D = 3, 16, 17
        pred_v = torch.randn(B, T, D)
        target_v = torch.randn(B, T, D)
        mask = torch.ones(B, T)
        mask[:, -4:] = 0.0  # some padding

        module = VLAActionLoss(action_dim=D)
        output = module(pred_v, target_v, mask)

        per_joint_sum = sum(output.metrics["per_joint_mse"])
        expected = D * output.loss.item()
        assert per_joint_sum == pytest.approx(expected, rel=1e-4)

    def test_velocity_norm_diagnostics(self):
        """Velocity norm metrics are correct for known inputs."""
        B, T, D = 2, 4, 3
        # pred: all 1s → norm per vector = sqrt(3)
        pred_v = torch.ones(B, T, D)
        # target: all zeros → norm = 0
        target_v = torch.zeros(B, T, D)

        module = VLAActionLoss(action_dim=D)
        output = module(pred_v, target_v)

        expected_pred_norm = 3**0.5  # sqrt(1^2 + 1^2 + 1^2)
        assert output.metrics["mean_pred_velocity_norm"] == pytest.approx(
            expected_pred_norm, rel=1e-5
        )
        assert output.metrics["mean_target_velocity_norm"] == pytest.approx(0.0, abs=1e-7)

    def test_mask_fraction_diagnostic(self):
        """mask_fraction_valid correctly reflects the fraction of valid positions."""
        B, T, D = 2, 8, 17
        pred_v = torch.randn(B, T, D)
        target_v = torch.randn(B, T, D)

        # 6 valid out of 8 per sample → 6/8 = 0.75
        mask = torch.ones(B, T)
        mask[:, -2:] = 0.0

        module = VLAActionLoss(action_dim=D)
        output = module(pred_v, target_v, mask)

        assert output.metrics["mask_fraction_valid"] == pytest.approx(6 / 8, rel=1e-5)

        # None mask → fraction = 1.0
        output_no_mask = module(pred_v, target_v, None)
        assert output_no_mask.metrics["mask_fraction_valid"] == pytest.approx(1.0)

    def test_none_mask_equals_all_ones(self):
        """None mask produces the same loss as an all-ones mask."""
        B, T, D = 3, 16, 17
        pred_v = torch.randn(B, T, D)
        target_v = torch.randn(B, T, D)
        mask_all_ones = torch.ones(B, T)

        module = VLAActionLoss(action_dim=D)
        out_none = module(pred_v, target_v, None)
        out_ones = module(pred_v, target_v, mask_all_ones)

        assert out_none.loss.item() == pytest.approx(out_ones.loss.item(), rel=1e-5)

    def test_gradient_flows_to_pred_velocity(self):
        """backward() produces non-zero gradient on pred_velocity tensor."""
        B, T, D = 2, 16, 17
        pred_v = torch.randn(B, T, D, requires_grad=True)
        target_v = torch.randn(B, T, D)
        mask = torch.ones(B, T)
        mask[:, -2:] = 0.0

        module = VLAActionLoss(action_dim=D)
        output = module(pred_v, target_v, mask)
        output.loss.backward()

        assert pred_v.grad is not None
        assert pred_v.grad.abs().sum().item() > 0.0
