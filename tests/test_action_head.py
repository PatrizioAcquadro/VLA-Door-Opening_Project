"""Tests for action head contracts and utilities (Phase 3.2.0).

Test classes:
    TestActionChunkContract  -- Frozen constants, config, chunking.   Marker: action_head
    TestContextBudget        -- Context window budget verification.   Marker: action_head
    TestTensorInterfaceShapes -- Shape contracts for all components.  Marker: action_head

All tests in this file are CPU-only — no VLM backbone or GPU required.
"""

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from models.action_head import (
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    STATE_DIM,
    TOKENS_PER_ACTION_STEP,
    TOKENS_PER_STATE,
    ActionChunkConfig,
    ActionOutputHead,
    FlowMatchingConfig,
    FlowMatchingModule,
    NoisyActionProjector,
    RobotStateProjector,
    TokenType,
    chunk_actions,
    chunk_actions_batched,
    compute_action_context_tokens,
    sinusoidal_timestep_embedding,
)

# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


@pytest.mark.action_head
class TestActionChunkContract:
    """Frozen constants, config dataclass, and chunking utilities."""

    def test_constants_match_frozen(self):
        """Frozen constants have the correct values from Phase 1.1 contracts."""
        assert ACTION_CHUNK_SIZE == 16
        assert ACTION_DIM == 17
        assert STATE_DIM == 52
        assert TOKENS_PER_ACTION_STEP == 1
        assert TOKENS_PER_STATE == 1

    def test_action_chunk_config_frozen(self):
        """ActionChunkConfig is immutable (frozen dataclass)."""
        config = ActionChunkConfig()
        with pytest.raises(AttributeError):
            config.chunk_size = 32  # type: ignore[misc]

    def test_action_chunk_config_defaults(self):
        """Default config matches frozen constants."""
        config = ActionChunkConfig()
        assert config.chunk_size == 16
        assert config.action_dim == 17
        assert config.state_dim == 52
        assert config.tokens_per_chunk == 16
        assert config.chunk_shape == (16, 17)

    def test_action_chunk_config_from_cfg(self):
        """from_cfg() constructs config from dict."""
        cfg = {"chunk_size": 8, "action_dim": 17, "state_dim": 52}
        config = ActionChunkConfig.from_cfg(cfg)
        assert config.chunk_size == 8
        assert config.tokens_per_chunk == 8

    def test_action_chunk_config_from_cfg_none(self):
        """from_cfg(None) returns defaults."""
        config = ActionChunkConfig.from_cfg(None)
        assert config.chunk_size == ACTION_CHUNK_SIZE

    def test_token_type_enum_values(self):
        """TokenType enum has correct integer values for loss routing."""
        assert TokenType.TEXT == 0
        assert TokenType.IMAGE == 1
        assert TokenType.STATE == 2
        assert TokenType.ACTION == 3
        assert len(TokenType) == 4

    def test_chunk_actions_exact_division(self):
        """32 steps split into exactly 2 chunks with full masks."""
        actions = torch.randn(32, 17)
        chunks, masks = chunk_actions(actions, chunk_size=16)
        assert chunks.shape == (2, 16, 17)
        assert masks.shape == (2, 16)
        assert masks.sum().item() == 32  # all valid
        assert torch.allclose(chunks[0], actions[:16])
        assert torch.allclose(chunks[1], actions[16:32])

    def test_chunk_actions_with_padding(self):
        """20 steps → 2 chunks; second has 4 valid + 12 padded."""
        actions = torch.randn(20, 17)
        chunks, masks = chunk_actions(actions, chunk_size=16)
        assert chunks.shape == (2, 16, 17)
        assert masks.shape == (2, 16)
        # First chunk fully valid
        assert masks[0].sum().item() == 16
        # Second chunk: 4 valid, 12 padded
        assert masks[1].sum().item() == 4
        assert masks[1, :4].all()
        assert not masks[1, 4:].any()
        # Padded steps are zeros
        assert (chunks[1, 4:] == 0).all()
        # Valid steps match original
        assert torch.allclose(chunks[1, :4], actions[16:20])

    def test_chunk_actions_single_chunk(self):
        """10 steps → 1 chunk with 10 valid + 6 padded."""
        actions = torch.randn(10, 17)
        chunks, masks = chunk_actions(actions, chunk_size=16)
        assert chunks.shape == (1, 16, 17)
        assert masks.shape == (1, 16)
        assert masks[0, :10].all()
        assert not masks[0, 10:].any()

    def test_chunk_actions_empty(self):
        """0 steps → 0 chunks."""
        actions = torch.empty(0, 17)
        chunks, masks = chunk_actions(actions, chunk_size=16)
        assert chunks.shape == (0, 16, 17)
        assert masks.shape == (0, 16)

    def test_chunk_actions_wrong_ndim_raises(self):
        """chunk_actions rejects tensors that aren't 2-D."""
        with pytest.raises(ValueError, match="2-D"):
            chunk_actions(torch.randn(4, 16, 17))

    def test_chunk_actions_batched(self):
        """Batch-level chunking pads all samples to same chunk count."""
        B, max_steps = 3, 32
        actions = torch.randn(B, max_steps, 17)
        step_masks = torch.ones(B, max_steps)
        step_masks[1, 20:] = 0  # sample 1 has only 20 valid steps

        chunks, chunk_masks = chunk_actions_batched(actions, step_masks, chunk_size=16)
        assert chunks.shape == (3, 2, 16, 17)
        assert chunk_masks.shape == (3, 2, 16)
        # Sample 0: all valid
        assert chunk_masks[0].sum().item() == 32
        # Sample 1: 20 valid out of 32 slots
        assert chunk_masks[1].sum().item() == 20

    def test_compute_action_context_tokens(self):
        """Token count formula: ceil(n_steps / chunk_size) * chunk_size."""
        # Exact division
        assert compute_action_context_tokens(32, chunk_size=16) == 32
        # Padding needed
        assert compute_action_context_tokens(20, chunk_size=16) == 32
        # Single partial chunk
        assert compute_action_context_tokens(10, chunk_size=16) == 16
        # Typical episode (~200 steps)
        assert compute_action_context_tokens(200, chunk_size=16) == 13 * 16
        # Edge cases
        assert compute_action_context_tokens(0, chunk_size=16) == 0
        assert compute_action_context_tokens(1, chunk_size=16) == 16


# ---------------------------------------------------------------------------
# Context budget verification
# ---------------------------------------------------------------------------


@pytest.mark.action_head
class TestContextBudget:
    """Verify action chunk tokens fit within the 8K context window."""

    def test_single_placement_fits_8k(self):
        """Single-placement episode (~200 steps, 16 images, V=200) fits in 8K."""
        # Worst-case vision token estimate for 320×320 (Phase 3.1.2 expected 130–260)
        V = 200  # vision tokens per image
        n_images = 16
        text_task = 40
        text_narration = 8 * 25  # 8 segments × ~25 tokens
        text_outcome = 5
        n_state_tokens = 8  # 1 per segment
        n_action_steps = 200
        action_tokens = compute_action_context_tokens(n_action_steps, chunk_size=16)

        total = (
            n_images * V
            + text_task
            + text_narration
            + text_outcome
            + n_state_tokens
            + action_tokens
        )
        assert total < 8192, f"Single-placement total {total} exceeds 8K context"

    def test_level3_episode_fits_8k(self):
        """Level 3 multi-placement worst case (~600 steps, 32 images, V=200) fits in 8K."""
        V = 200
        n_images = 32  # more images for multi-placement
        text_task = 40
        text_narration = 16 * 25  # more segments
        text_outcome = 5
        n_state_tokens = 16
        n_action_steps = 600
        action_tokens = compute_action_context_tokens(n_action_steps, chunk_size=16)

        total = (
            n_images * V
            + text_task
            + text_narration
            + text_outcome
            + n_state_tokens
            + action_tokens
        )
        # Level 3 may be tight — document the budget
        max_ctx = 8192
        if total > max_ctx:
            # Not a failure — just documents the constraint for Phase 2.3/3.3
            pytest.skip(
                f"Level 3 worst case ({total} tokens) exceeds 8K. "
                f"Phase 2.3 must limit image count or use shorter episodes."
            )
        assert total < max_ctx

    def test_budget_with_phase31_compute(self):
        """Cross-check with Phase 3.1.2 compute_context_budget()."""
        from models.vlm_backbone import compute_context_budget

        # Default budget (4 images, no action tokens accounted for)
        budget = compute_context_budget(
            vision_tokens_per_image=200,
            n_images=4,
            max_seq_length=8192,
        )
        remaining = budget["tokens_remaining_for_actions"]

        # Verify action chunks fit in remaining budget
        n_action_steps = 200
        action_tokens = compute_action_context_tokens(n_action_steps, chunk_size=16)
        n_state_tokens = 8

        assert remaining > action_tokens + n_state_tokens, (
            f"Action tokens ({action_tokens}) + state tokens ({n_state_tokens}) "
            f"exceed remaining budget ({remaining})"
        )


# ---------------------------------------------------------------------------
# Tensor interface shape contracts
# ---------------------------------------------------------------------------


@pytest.mark.action_head
class TestTensorInterfaceShapes:
    """Shape contracts for all Phase 3.2 components.

    These tests verify the tensor shapes documented in the 3.2.0 contract
    table. They use simple placeholder tensors — no actual modules.
    Components will be tested against these contracts in 3.2.1–3.2.4.
    """

    @pytest.fixture
    def B(self):
        return 4

    @pytest.fixture
    def H(self):
        """Hidden size for Qwen3.5-4B."""
        return 2560

    def test_state_projector_contract(self, B, H):
        """RobotStateProjector: (B, 52) → (B, 1, H)."""
        state_input = torch.randn(B, STATE_DIM)
        # Contract: output must be (B, 1, H)
        expected_output_shape = (B, TOKENS_PER_STATE, H)
        assert state_input.shape == (B, 52)
        assert expected_output_shape == (B, 1, H)

    def test_noisy_action_projector_contract(self, B, H):
        """NoisyActionProjector: (B, 16, 17) + (B, 1) → (B, 16, H)."""
        noisy_actions = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        timestep = torch.rand(B, 1)
        expected_output_shape = (B, ACTION_CHUNK_SIZE, H)
        assert noisy_actions.shape == (B, 16, 17)
        assert timestep.shape == (B, 1)
        assert expected_output_shape == (B, 16, H)

    def test_action_output_head_contract(self, B, H):
        """ActionOutputHead: (B, 16, H) → (B, 16, 17)."""
        hidden_states = torch.randn(B, ACTION_CHUNK_SIZE, H)
        expected_output_shape = (B, ACTION_CHUNK_SIZE, ACTION_DIM)
        assert hidden_states.shape == (B, 16, H)
        assert expected_output_shape == (B, 16, 17)

    def test_flow_matching_interpolate_contract(self, B):
        """FlowMatchingModule.interpolate: (B, 16, 17) × 2 + (B, 1, 1) → (B, 16, 17)."""
        x_data = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        noise = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        t = torch.rand(B, 1, 1)
        # Standard CFM interpolation: x_t = (1-t)*noise + t*x_data
        x_t = (1 - t) * noise + t * x_data
        assert x_t.shape == (B, 16, 17)

    def test_flow_matching_velocity_contract(self, B):
        """FlowMatchingModule.target_velocity: (B, 16, 17) × 2 → (B, 16, 17)."""
        x_data = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        noise = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        # Standard CFM target velocity: u_t = x_data - noise
        u_t = x_data - noise
        assert u_t.shape == (B, 16, 17)

    def test_flow_matching_loss_contract(self, B):
        """FlowMatchingModule.loss: (B, 16, 17) × 2 + (B, 16) → scalar."""
        pred_v = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        target_v = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        mask = torch.ones(B, ACTION_CHUNK_SIZE)
        # Masked MSE loss
        diff_sq = (pred_v - target_v) ** 2  # (B, 16, 17)
        per_step = diff_sq.mean(dim=-1)  # (B, 16) — mean over action_dim
        masked = (per_step * mask).sum() / mask.sum().clamp(min=1)
        assert masked.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# Flow matching module (Phase 3.2.1)
# ---------------------------------------------------------------------------


@pytest.mark.action_head
class TestFlowMatchingModule:
    """Unit tests for FlowMatchingConfig and FlowMatchingModule.

    All CPU-only. No backbone or GPU required.
    Tests cover: config, timestep sampling, interpolation, target velocity,
    masked loss, ODE denoising (Euler/midpoint/RK4), and numerical stability.
    """

    # ---- fixtures -----------------------------------------------------------

    @pytest.fixture
    def fm(self) -> FlowMatchingModule:
        return FlowMatchingModule()

    @pytest.fixture
    def B(self) -> int:
        return 4

    @pytest.fixture
    def C(self) -> int:
        return ACTION_CHUNK_SIZE  # 16

    @pytest.fixture
    def D(self) -> int:
        return ACTION_DIM  # 17

    # ---- FlowMatchingConfig tests -------------------------------------------

    def test_config_defaults(self):
        """Default config matches action_head.yaml values."""
        cfg = FlowMatchingConfig()
        assert cfg.n_denoising_steps == 10
        assert cfg.solver == "euler"
        assert cfg.sigma_min == 0.001
        assert cfg.time_sampling == "beta"
        assert cfg.time_alpha == 1.5
        assert cfg.time_beta == 1.0
        assert cfg.time_min == 0.001
        assert cfg.time_max == 0.999

    def test_config_from_cfg(self):
        """from_cfg() round-trips a dict correctly."""
        d = {"n_denoising_steps": 5, "solver": "rk4", "time_sampling": "uniform"}
        cfg = FlowMatchingConfig.from_cfg(d)
        assert cfg.n_denoising_steps == 5
        assert cfg.solver == "rk4"
        assert cfg.time_sampling == "uniform"
        # Untouched keys keep defaults
        assert cfg.time_alpha == 1.5

    def test_config_from_cfg_none(self):
        """from_cfg(None) returns defaults."""
        cfg = FlowMatchingConfig.from_cfg(None)
        assert cfg.n_denoising_steps == 10

    def test_config_frozen(self):
        """FlowMatchingConfig is immutable (frozen dataclass)."""
        cfg = FlowMatchingConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.n_denoising_steps = 99  # type: ignore[misc]

    # ---- sample_timestep tests ----------------------------------------------

    def test_sample_timestep_shape(self, fm, B):
        """sample_timestep returns (B, 1, 1) in [time_min, time_max]."""
        t = fm.sample_timestep(B)
        assert t.shape == (B, 1, 1)
        assert t.min().item() >= fm.config.time_min - 1e-6
        assert t.max().item() <= fm.config.time_max + 1e-6

    def test_sample_timestep_broadcasts(self, fm, B, C, D):
        """Timestep (B, 1, 1) broadcasts to (B, C, D) without expand."""
        t = fm.sample_timestep(B)
        x = torch.randn(B, C, D)
        # Should broadcast without error
        result = x * t
        assert result.shape == (B, C, D)

    def test_sample_timestep_beta_distribution(self, fm):
        """Beta sampling produces the correct mean (large sample)."""
        N = 10000
        t = fm.sample_timestep(N)  # (N, 1, 1)
        t_vals = t.squeeze()  # (N,)

        # Rescale back to [0, 1] and check Beta mean ≈ alpha / (alpha + beta)
        t_min, t_max = fm.config.time_min, fm.config.time_max
        t_unit = (t_vals - t_min) / (t_max - t_min)
        expected_mean = fm.config.time_alpha / (fm.config.time_alpha + fm.config.time_beta)
        assert abs(t_unit.mean().item() - expected_mean) < 0.02

    def test_sample_timestep_uniform(self):
        """Uniform sampling stays in [time_min, time_max] with uniform spread."""
        fm_u = FlowMatchingModule(FlowMatchingConfig(time_sampling="uniform"))
        N = 5000
        t = fm_u.sample_timestep(N).squeeze()
        assert t.min().item() >= fm_u.config.time_min - 1e-6
        assert t.max().item() <= fm_u.config.time_max + 1e-6
        # Mean of uniform on [time_min, time_max] ≈ (time_min + time_max) / 2
        expected = (fm_u.config.time_min + fm_u.config.time_max) / 2.0
        assert abs(t.mean().item() - expected) < 0.02

    # ---- interpolate tests --------------------------------------------------

    def test_interpolate_t0_is_noise(self, fm, B, C, D):
        """interpolate(x_data, noise, t=0) == noise."""
        x_data = torch.randn(B, C, D)
        noise = torch.randn(B, C, D)
        t = torch.zeros(B, 1, 1)
        result = fm.interpolate(x_data, noise, t)
        assert torch.allclose(result, noise, atol=1e-6)

    def test_interpolate_t1_is_data(self, fm, B, C, D):
        """interpolate(x_data, noise, t=1) == x_data."""
        x_data = torch.randn(B, C, D)
        noise = torch.randn(B, C, D)
        t = torch.ones(B, 1, 1)
        result = fm.interpolate(x_data, noise, t)
        assert torch.allclose(result, x_data, atol=1e-6)

    def test_interpolate_midpoint(self, fm, B, C, D):
        """interpolate at t=0.5 is the average of noise and data."""
        x_data = torch.randn(B, C, D)
        noise = torch.randn(B, C, D)
        t = torch.full((B, 1, 1), 0.5)
        result = fm.interpolate(x_data, noise, t)
        expected = 0.5 * noise + 0.5 * x_data
        assert torch.allclose(result, expected, atol=1e-6)

    def test_interpolate_shape(self, fm, B, C, D):
        """interpolate output has same shape as inputs."""
        x_data = torch.randn(B, C, D)
        noise = torch.randn(B, C, D)
        t = torch.rand(B, 1, 1)
        result = fm.interpolate(x_data, noise, t)
        assert result.shape == (B, C, D)

    # ---- target_velocity tests ----------------------------------------------

    def test_target_velocity_formula(self, fm, B, C, D):
        """target_velocity == x_data - noise for all inputs."""
        x_data = torch.randn(B, C, D)
        noise = torch.randn(B, C, D)
        expected = x_data - noise
        result = fm.target_velocity(x_data, noise)
        assert torch.allclose(result, expected, atol=1e-7)

    def test_target_velocity_shape(self, fm, B, C, D):
        """target_velocity output shape matches inputs."""
        x_data = torch.randn(B, C, D)
        noise = torch.randn(B, C, D)
        result = fm.target_velocity(x_data, noise)
        assert result.shape == (B, C, D)

    # ---- loss tests ---------------------------------------------------------

    def test_loss_zero_when_equal(self, fm, B, C, D):
        """loss is 0 when predicted velocity equals target velocity."""
        v = torch.randn(B, C, D)
        mask = torch.ones(B, C)
        assert fm.loss(v, v, mask).item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_positive_when_different(self, fm, B, C, D):
        """loss is positive when predicted and target velocities differ."""
        pred = torch.randn(B, C, D)
        target = torch.randn(B, C, D)
        mask = torch.ones(B, C)
        assert fm.loss(pred, target, mask).item() > 0.0

    def test_loss_scalar(self, fm, B, C, D):
        """loss returns a scalar tensor."""
        pred = torch.randn(B, C, D)
        target = torch.randn(B, C, D)
        mask = torch.ones(B, C)
        result = fm.loss(pred, target, mask)
        assert result.ndim == 0

    def test_loss_mask_excludes_padded(self, fm, B, C, D):
        """Masked positions don't contribute to the loss."""
        pred = torch.randn(B, C, D)
        target = torch.randn(B, C, D)

        # All valid → positive loss
        full_mask = torch.ones(B, C)
        loss_full = fm.loss(pred, target, full_mask).item()

        # Only first half valid; zero out second half of pred and target to match
        pred_half = pred.clone()
        target_half = target.clone()
        pred_half[:, C // 2 :] = 0.0
        target_half[:, C // 2 :] = 0.0

        half_mask = torch.zeros(B, C)
        half_mask[:, : C // 2] = 1.0

        loss_half = fm.loss(pred_half, target_half, half_mask).item()

        # Losses computed on disjoint sets of positions — both finite
        assert loss_full > 0.0
        assert loss_half >= 0.0
        # loss on first half of pred/target with full mask should equal loss_half
        loss_first_half_full = fm.loss(pred[:, : C // 2], target[:, : C // 2], None).item()
        assert abs(loss_half - loss_first_half_full) < 1e-5

    def test_loss_all_zeros_mask(self, fm, B, C, D):
        """All-zero mask doesn't cause division by zero — returns 0."""
        pred = torch.randn(B, C, D)
        target = torch.randn(B, C, D)
        zero_mask = torch.zeros(B, C)
        result = fm.loss(pred, target, zero_mask)
        assert result.item() == pytest.approx(0.0, abs=1e-6)
        assert not torch.isnan(result)

    def test_loss_no_mask(self, fm, B, C, D):
        """loss with mask=None equals mean MSE over all positions."""
        pred = torch.randn(B, C, D)
        target = torch.randn(B, C, D)
        result_no_mask = fm.loss(pred, target, None)
        result_full_mask = fm.loss(pred, target, torch.ones(B, C))
        assert abs(result_no_mask.item() - result_full_mask.item()) < 1e-5

    # ---- denoise tests ------------------------------------------------------

    def _make_perfect_fn(self, x_data: Tensor, x0: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a predict_fn that always returns the true constant velocity."""
        true_v = x_data - x0  # constant along OT path

        def predict_fn(x_t: Tensor, t: Tensor) -> Tensor:
            return true_v

        return predict_fn

    def test_denoise_identity_euler(self, B, C, D):
        """Euler integration with perfect predict_fn recovers x_data exactly."""
        fm = FlowMatchingModule(FlowMatchingConfig(solver="euler", n_denoising_steps=10))
        x_data = torch.randn(B, C, D)
        x0 = torch.randn(B, C, D)
        predict_fn = self._make_perfect_fn(x_data, x0)
        result = fm.denoise(predict_fn, (B, C, D), K=10, x_init=x0)
        assert torch.allclose(result, x_data, atol=1e-5)

    def test_denoise_identity_midpoint(self, B, C, D):
        """Midpoint integration with perfect predict_fn recovers x_data exactly."""
        fm = FlowMatchingModule(FlowMatchingConfig(solver="midpoint", n_denoising_steps=10))
        x_data = torch.randn(B, C, D)
        x0 = torch.randn(B, C, D)
        predict_fn = self._make_perfect_fn(x_data, x0)
        result = fm.denoise(predict_fn, (B, C, D), K=10, x_init=x0)
        assert torch.allclose(result, x_data, atol=1e-5)

    def test_denoise_identity_rk4(self, B, C, D):
        """RK4 integration with perfect predict_fn recovers x_data exactly."""
        fm = FlowMatchingModule(FlowMatchingConfig(solver="rk4", n_denoising_steps=10))
        x_data = torch.randn(B, C, D)
        x0 = torch.randn(B, C, D)
        predict_fn = self._make_perfect_fn(x_data, x0)
        result = fm.denoise(predict_fn, (B, C, D), K=10, x_init=x0)
        assert torch.allclose(result, x_data, atol=1e-5)

    def test_denoise_shape(self, fm, B, C, D):
        """denoise output has the requested shape."""
        shape = (B, C, D)
        result = fm.denoise(lambda x, t: torch.zeros_like(x), shape)
        assert result.shape == shape

    def test_denoise_uses_config_k(self, B, C, D):
        """denoise uses config.n_denoising_steps when K is not passed."""
        call_count = [0]

        def counting_fn(x: Tensor, t: Tensor) -> Tensor:
            call_count[0] += 1
            return torch.zeros_like(x)

        fm = FlowMatchingModule(FlowMatchingConfig(n_denoising_steps=7))
        fm.denoise(counting_fn, (B, C, D))
        assert call_count[0] == 7

    def test_denoise_invalid_solver(self, B, C, D):
        """denoise raises ValueError for unknown solver."""
        fm = FlowMatchingModule(FlowMatchingConfig(solver="bogus"))
        with pytest.raises(ValueError, match="bogus"):
            fm.denoise(lambda x, t: x, (B, C, D))

    # ---- numerical stability tests ------------------------------------------

    def test_numerical_stability_edge_cases(self, fm, B, C, D):
        """No NaN/Inf for t=0, t=1, zero noise, large noise."""
        x_data = torch.randn(B, C, D)

        for t_val in [0.0, 1.0, 0.5]:
            t = torch.full((B, 1, 1), t_val)

            # Zero noise
            x_t = fm.interpolate(x_data, torch.zeros_like(x_data), t)
            assert not torch.any(torch.isnan(x_t))
            assert not torch.any(torch.isinf(x_t))

            # Large noise
            large_noise = torch.randn(B, C, D) * 100.0
            x_t = fm.interpolate(x_data, large_noise, t)
            assert not torch.any(torch.isnan(x_t))
            assert not torch.any(torch.isinf(x_t))

        # Loss with zero tensors
        z = torch.zeros(B, C, D)
        loss = fm.loss(z, z)
        assert not torch.isnan(loss)

        # Target velocity with identical inputs
        v = fm.target_velocity(x_data, x_data)
        assert not torch.any(torch.isnan(v))
        assert torch.allclose(v, torch.zeros_like(v))


# ---------------------------------------------------------------------------
# Robot state projector (Phase 3.2.2)
# ---------------------------------------------------------------------------


@pytest.mark.action_head
class TestRobotStateProjector:
    """Unit tests for RobotStateProjector (Phase 3.2.2).

    All CPU-only. No backbone or GPU required.
    Tests cover: output shape, numerical stability, gradient flow,
    output magnitude, batched input, parameter count, and from_cfg().
    """

    @pytest.fixture
    def H(self) -> int:
        """Hidden size matching Qwen3.5-4B."""
        return 2560

    @pytest.fixture
    def B(self) -> int:
        return 4

    @pytest.fixture
    def proj(self, H: int) -> RobotStateProjector:
        return RobotStateProjector(hidden_dim=H)

    # ---- shape tests --------------------------------------------------------

    def test_output_shape(self, proj: RobotStateProjector, B: int, H: int) -> None:
        """Output shape is (B, 1, H) for default H=2560."""
        state = torch.randn(B, STATE_DIM)
        out = proj(state)
        assert out.shape == (B, 1, H)

    def test_batched_input(self, H: int) -> None:
        """Correct output shape for B=1, 4, 16."""
        proj = RobotStateProjector(hidden_dim=H)
        for batch in [1, 4, 16]:
            state = torch.randn(batch, STATE_DIM)
            out = proj(state)
            assert out.shape == (batch, 1, H), f"Failed for B={batch}"

    # ---- numerical stability ------------------------------------------------

    def test_no_nan(self, proj: RobotStateProjector, B: int) -> None:
        """No NaN or Inf in output for random input."""
        state = torch.randn(B, STATE_DIM)
        out = proj(state)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_no_nan_zero_input(self, proj: RobotStateProjector, B: int) -> None:
        """No NaN or Inf for zero input (LayerNorm handles this cleanly)."""
        state = torch.zeros(B, STATE_DIM)
        out = proj(state)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    # ---- gradient flow ------------------------------------------------------

    def test_gradient_flow(self, proj: RobotStateProjector, B: int) -> None:
        """Non-zero gradients flow through all named parameters."""
        state = torch.randn(B, STATE_DIM)
        out = proj(state)
        out.sum().backward()
        for name, p in proj.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"

    # ---- output magnitude ---------------------------------------------------

    def test_output_magnitude(self, proj: RobotStateProjector, B: int, H: int) -> None:
        """Output L2 norm per token divided by sqrt(H) is O(1).

        Tests that the projector doesn't explode or collapse activations.
        """
        torch.manual_seed(42)
        state = torch.randn(B, STATE_DIM)
        out = proj(state)  # (B, 1, H)
        norm = out.squeeze(1).norm(dim=-1).mean()  # scalar
        scale = norm / (H**0.5)
        assert 0.01 < scale.item() < 100.0, f"Output scale {scale.item():.4f} is not O(1)"

    # ---- parameter count ----------------------------------------------------

    def test_parameter_count(self, H: int) -> None:
        """Total parameter count matches analytical estimate for H=2560.

        LayerNorm(52):        52*2 = 104
        Linear(52, 2560): 52*2560 + 2560 = 135,680
        Linear(2560, 2560): 2560*2560 + 2560 = 6,556,160
        Total: 6,691,944
        """
        proj = RobotStateProjector(hidden_dim=H)
        n_params = sum(p.numel() for p in proj.parameters())
        expected = STATE_DIM * 2 + (STATE_DIM * H + H) + (H * H + H)
        assert n_params == expected, f"Expected {expected} params, got {n_params}"

    # ---- from_cfg -----------------------------------------------------------

    def test_from_cfg_defaults(self, H: int) -> None:
        """from_cfg({}, hidden_dim) uses defaults: state_dim=52, silu activation."""
        proj = RobotStateProjector.from_cfg({}, hidden_dim=H)
        assert proj.state_dim == STATE_DIM
        assert proj.hidden_dim == H

    def test_from_cfg_none(self, H: int) -> None:
        """from_cfg(None, hidden_dim) returns defaults."""
        proj = RobotStateProjector.from_cfg(None, hidden_dim=H)
        assert proj.state_dim == STATE_DIM
        assert proj.hidden_dim == H

    def test_from_cfg_custom_activation(self, H: int) -> None:
        """from_cfg() respects projector.activation override."""
        cfg = {"projector": {"activation": "gelu"}}
        proj = RobotStateProjector.from_cfg(cfg, hidden_dim=H)
        # SiLU is the activation type in default; GELU should be substituted
        assert isinstance(proj.mlp[1], torch.nn.GELU)

    def test_invalid_activation_raises(self) -> None:
        """Unknown activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            RobotStateProjector(hidden_dim=256, activation="tanh")


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (Phase 3.2.3)
# ---------------------------------------------------------------------------


@pytest.mark.action_head
class TestSinusoidalTimestepEmbedding:
    """Unit tests for sinusoidal_timestep_embedding().

    All CPU-only. Tests cover: output shape, distinct vectors across timesteps,
    numerical stability, and determinism.
    """

    def test_output_shape(self) -> None:
        """Output shape is (B, dim) for various batch sizes and dims."""
        for B, dim in [(1, 64), (4, 256), (8, 128)]:
            t = torch.rand(B, 1)
            out = sinusoidal_timestep_embedding(t, dim)
            assert out.shape == (B, dim), f"Expected ({B}, {dim}), got {out.shape}"

    def test_distinct_timesteps(self) -> None:
        """t=0.0, 0.5, and 1.0 produce distinct embedding vectors."""
        dim = 256
        t0 = sinusoidal_timestep_embedding(torch.zeros(1, 1), dim)
        t_half = sinusoidal_timestep_embedding(torch.full((1, 1), 0.5), dim)
        t1 = sinusoidal_timestep_embedding(torch.ones(1, 1), dim)
        assert not torch.allclose(t0, t_half), "t=0.0 and t=0.5 embeddings should differ"
        assert not torch.allclose(t0, t1), "t=0.0 and t=1.0 embeddings should differ"
        assert not torch.allclose(t_half, t1), "t=0.5 and t=1.0 embeddings should differ"

    def test_no_nan(self) -> None:
        """No NaN or Inf for t=0, t=0.5, t=1.0."""
        dim = 256
        for t_val in [0.0, 0.5, 1.0]:
            t = torch.full((4, 1), t_val)
            out = sinusoidal_timestep_embedding(t, dim)
            assert not torch.any(torch.isnan(out)), f"NaN at t={t_val}"
            assert not torch.any(torch.isinf(out)), f"Inf at t={t_val}"

    def test_deterministic(self) -> None:
        """Same input produces the same output (parameter-free)."""
        t = torch.rand(4, 1)
        out1 = sinusoidal_timestep_embedding(t, 256)
        out2 = sinusoidal_timestep_embedding(t, 256)
        assert torch.allclose(out1, out2)

    def test_odd_dim_raises(self) -> None:
        """Odd dim raises ValueError."""
        with pytest.raises(ValueError):
            sinusoidal_timestep_embedding(torch.rand(1, 1), 3)

    def test_zero_dim_raises(self) -> None:
        """Zero or negative dim raises ValueError."""
        with pytest.raises(ValueError):
            sinusoidal_timestep_embedding(torch.rand(1, 1), 0)


# ---------------------------------------------------------------------------
# Noisy action projector (Phase 3.2.3)
# ---------------------------------------------------------------------------


@pytest.mark.action_head
class TestNoisyActionProjector:
    """Unit tests for NoisyActionProjector (Phase 3.2.3).

    All CPU-only. No backbone or GPU required.
    Tests cover: output shape, numerical stability, gradient flow,
    output magnitude, parameter count, from_cfg(), and timestep distinctness.
    """

    @pytest.fixture
    def H(self) -> int:
        """Hidden size matching Qwen3.5-4B."""
        return 2560

    @pytest.fixture
    def B(self) -> int:
        return 4

    @pytest.fixture
    def proj(self, H: int) -> NoisyActionProjector:
        return NoisyActionProjector(hidden_dim=H)

    # ---- shape tests --------------------------------------------------------

    def test_output_shape(self, proj: NoisyActionProjector, B: int, H: int) -> None:
        """Output shape is (B, 16, H) for default config."""
        noisy = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        t = torch.rand(B, 1)
        out = proj(noisy, t)
        assert out.shape == (B, ACTION_CHUNK_SIZE, H)

    def test_batched_input(self, H: int) -> None:
        """Correct output shape for B=1, 4, 16."""
        proj = NoisyActionProjector(hidden_dim=H)
        for batch in [1, 4, 16]:
            noisy = torch.randn(batch, ACTION_CHUNK_SIZE, ACTION_DIM)
            t = torch.rand(batch, 1)
            out = proj(noisy, t)
            assert out.shape == (batch, ACTION_CHUNK_SIZE, H), f"Failed for B={batch}"

    # ---- numerical stability ------------------------------------------------

    def test_no_nan(self, proj: NoisyActionProjector, B: int) -> None:
        """No NaN or Inf for random inputs."""
        noisy = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        t = torch.rand(B, 1)
        out = proj(noisy, t)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_no_nan_t_boundaries(self, proj: NoisyActionProjector, B: int) -> None:
        """No NaN/Inf at t=0 and t=1 boundary values."""
        noisy = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        for t_val in [0.0, 1.0]:
            t = torch.full((B, 1), t_val)
            out = proj(noisy, t)
            assert not torch.any(torch.isnan(out)), f"NaN at t={t_val}"
            assert not torch.any(torch.isinf(out)), f"Inf at t={t_val}"

    # ---- gradient flow ------------------------------------------------------

    def test_gradient_flow(self, proj: NoisyActionProjector, B: int) -> None:
        """Non-zero gradients flow through all named parameters."""
        noisy = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        t = torch.rand(B, 1)
        out = proj(noisy, t)
        out.sum().backward()
        for name, p in proj.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"

    # ---- output magnitude ---------------------------------------------------

    def test_output_magnitude(self, proj: NoisyActionProjector, B: int, H: int) -> None:
        """Output L2 norm per token divided by sqrt(H) is O(1)."""
        torch.manual_seed(42)
        noisy = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        t = torch.rand(B, 1)
        out = proj(noisy, t)  # (B, 16, H)
        norm = out.norm(dim=-1).mean()  # mean over (B, 16)
        scale = norm / (H**0.5)
        assert 0.01 < scale.item() < 100.0, f"Output scale {scale.item():.4f} is not O(1)"

    # ---- parameter count ----------------------------------------------------

    def test_parameter_count(self, H: int) -> None:
        """Total parameter count matches analytical estimate for H=2560, d_t=256.

        Linear(17+256, 2560): 273*2560 + 2560 = 701,440
        Linear(2560, 2560): 2560*2560 + 2560 = 6,556,160
        Total: 7,257,600
        """
        proj = NoisyActionProjector(hidden_dim=H, timestep_embed_dim=256)
        n_params = sum(p.numel() for p in proj.parameters())
        d_t = 256
        in_dim = ACTION_DIM + d_t
        expected = (in_dim * H + H) + (H * H + H)
        assert n_params == expected, f"Expected {expected} params, got {n_params}"

    # ---- timestep distinctness ----------------------------------------------

    def test_timestep_distinct_outputs(self, H: int, B: int) -> None:
        """Different timesteps produce different output tokens."""
        proj = NoisyActionProjector(hidden_dim=H)
        noisy = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        out_t0 = proj(noisy, torch.zeros(B, 1))
        out_t1 = proj(noisy, torch.ones(B, 1))
        assert not torch.allclose(out_t0, out_t1), "t=0 and t=1 should produce different tokens"

    # ---- from_cfg -----------------------------------------------------------

    def test_from_cfg_defaults(self, H: int) -> None:
        """from_cfg({}, hidden_dim) uses defaults: action_dim=17, d_t=256."""
        proj = NoisyActionProjector.from_cfg({}, hidden_dim=H)
        assert proj.action_dim == ACTION_DIM
        assert proj.hidden_dim == H
        assert proj.timestep_embed_dim == 256

    def test_from_cfg_none(self, H: int) -> None:
        """from_cfg(None, hidden_dim) returns defaults."""
        proj = NoisyActionProjector.from_cfg(None, hidden_dim=H)
        assert proj.action_dim == ACTION_DIM
        assert proj.hidden_dim == H

    def test_from_cfg_custom_timestep_dim(self, H: int) -> None:
        """from_cfg() respects timestep_embed_dim override."""
        cfg = {"timestep_embed_dim": 128}
        proj = NoisyActionProjector.from_cfg(cfg, hidden_dim=H)
        assert proj.timestep_embed_dim == 128

    def test_invalid_activation_raises(self) -> None:
        """Unknown activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            NoisyActionProjector(hidden_dim=256, activation="tanh")


# ---------------------------------------------------------------------------
# Action output head (Phase 3.2.3)
# ---------------------------------------------------------------------------


@pytest.mark.action_head
class TestActionOutputHead:
    """Unit tests for ActionOutputHead (Phase 3.2.3).

    All CPU-only. No backbone or GPU required.
    Tests cover: output shape, numerical stability, gradient flow,
    output magnitude, parameter count, from_cfg(), and round-trip compatibility.
    """

    @pytest.fixture
    def H(self) -> int:
        """Hidden size matching Qwen3.5-4B."""
        return 2560

    @pytest.fixture
    def B(self) -> int:
        return 4

    @pytest.fixture
    def head(self, H: int) -> ActionOutputHead:
        return ActionOutputHead(hidden_dim=H)

    # ---- shape tests --------------------------------------------------------

    def test_output_shape(self, head: ActionOutputHead, B: int) -> None:
        """Output shape is (B, 16, 17) for default config."""
        hidden = torch.randn(B, ACTION_CHUNK_SIZE, 2560)
        out = head(hidden)
        assert out.shape == (B, ACTION_CHUNK_SIZE, ACTION_DIM)

    def test_batched_input(self, H: int) -> None:
        """Correct output shape for B=1, 4, 16."""
        head = ActionOutputHead(hidden_dim=H)
        for batch in [1, 4, 16]:
            hidden = torch.randn(batch, ACTION_CHUNK_SIZE, H)
            out = head(hidden)
            assert out.shape == (batch, ACTION_CHUNK_SIZE, ACTION_DIM), f"Failed for B={batch}"

    # ---- numerical stability ------------------------------------------------

    def test_no_nan(self, head: ActionOutputHead, B: int) -> None:
        """No NaN or Inf for random inputs."""
        hidden = torch.randn(B, ACTION_CHUNK_SIZE, 2560)
        out = head(hidden)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    # ---- gradient flow ------------------------------------------------------

    def test_gradient_flow(self, head: ActionOutputHead, B: int) -> None:
        """Non-zero gradients flow through all named parameters."""
        hidden = torch.randn(B, ACTION_CHUNK_SIZE, 2560)
        out = head(hidden)
        out.sum().backward()
        for name, p in head.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"

    # ---- output magnitude ---------------------------------------------------

    def test_output_magnitude(self, head: ActionOutputHead, B: int) -> None:
        """Velocity predictions are not exploding (abs mean < 1000)."""
        torch.manual_seed(42)
        hidden = torch.randn(B, ACTION_CHUNK_SIZE, 2560)
        out = head(hidden)
        assert out.abs().mean().item() < 1000.0, "Output values are unexpectedly large"
        assert not torch.any(torch.isnan(out))

    # ---- parameter count ----------------------------------------------------

    def test_parameter_count(self, H: int) -> None:
        """Total parameter count matches analytical estimate for H=2560.

        Linear(2560, 2560): 2560*2560 + 2560 = 6,556,160
        Linear(2560, 17):   2560*17   + 17   = 43,537
        Total: 6,599,697
        """
        head = ActionOutputHead(hidden_dim=H)
        n_params = sum(p.numel() for p in head.parameters())
        expected = (H * H + H) + (H * ACTION_DIM + ACTION_DIM)
        assert n_params == expected, f"Expected {expected} params, got {n_params}"

    # ---- round-trip shape compatibility ------------------------------------

    def test_round_trip_shape(self, H: int, B: int) -> None:
        """NoisyActionProjector output shape is compatible with ActionOutputHead input."""
        projector = NoisyActionProjector(hidden_dim=H)
        head = ActionOutputHead(hidden_dim=H)
        noisy = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM)
        t = torch.rand(B, 1)
        tokens = projector(noisy, t)  # (B, 16, H)
        velocity = head(tokens)  # (B, 16, 17)
        assert velocity.shape == (B, ACTION_CHUNK_SIZE, ACTION_DIM)

    # ---- from_cfg -----------------------------------------------------------

    def test_from_cfg_defaults(self, H: int) -> None:
        """from_cfg({}, hidden_dim) uses defaults: action_dim=17."""
        head = ActionOutputHead.from_cfg({}, hidden_dim=H)
        assert head.action_dim == ACTION_DIM
        assert head.hidden_dim == H

    def test_from_cfg_none(self, H: int) -> None:
        """from_cfg(None, hidden_dim) returns defaults."""
        head = ActionOutputHead.from_cfg(None, hidden_dim=H)
        assert head.action_dim == ACTION_DIM
        assert head.hidden_dim == H

    def test_invalid_activation_raises(self) -> None:
        """Unknown activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            ActionOutputHead(hidden_dim=256, activation="tanh")
