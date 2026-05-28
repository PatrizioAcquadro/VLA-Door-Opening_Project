# Phase 4.1 — Loss Function Implementation (3 days)

**Goal:** Implement and validate standalone, production-quality loss modules — autoregressive cross-entropy for text, conditional flow matching MSE for actions, and a configurable combined training objective — that replace Phase 3.2's inline loss computation in `VLAModel.forward()` with testable, monitored, configurable components verified on real Phase 2.3 training data, so that Phase 4.2 can integrate the full training loop with known, profiled loss behavior.

**Fixed upstream decisions (from 1.1–1.2 + 2.1–2.3 + 3.1–3.2):**
- **Action space:** Frozen 17-D `[Δq_spine(1), Δq_left_arm(7), Δq_right_arm(7), gripper_left(1), gripper_right(1)]`, arm actions normalized `[-1, 1]`, gripper actions absolute `[0, 1]`
- **Robot state:** Frozen 52-D normalized vector `[q(15), q_dot(15), gripper(2), left_ee_pos(3), left_ee_quat(4), right_ee_pos(3), right_ee_quat(4), left_ee_vel(3), right_ee_vel(3)]`
- **VLM backbone:** Qwen3.5-4B loaded and profiled in Phase 3.1 — provides `get_hidden_states()` → `(B, seq_len, hidden_size)`, `hidden_size` property, LM head with `vocab_size` output dimension
- **Action head:** Phase 3.2 `FlowMatchingModule` (CFM: interpolation, velocity, ODE solver), `RobotStateProjector`, `NoisyActionProjector`, `ActionOutputHead`, `VLAModel` (Transfusion-style dual loss + multi-step denoising inference)
- **Token type mask:** Per-position label from Phase 3.2.0 — `TEXT`, `IMAGE`, `STATE`, `ACTION` — routes each position to the correct loss
- **Chunk contract:** `chunk_size=16`, `action_dim=17`, 1 token per action step, binary `chunk_mask` for padded chunks
- **Batch format:** Phase 3.2.4 batch dict with keys `input_ids`, `pixel_values`, `attention_mask`, `robot_states`, `action_chunks`, `chunk_masks`, `token_type_ids`, `text_labels`
- **Tracking infrastructure:** `tracking/metrics.py:extract_loss_components()` handles dict-based loss with `loss/ar` and `loss/fm` keys; `tracking/experiment.py:log_training_step()` accepts `loss_ar` and `loss_fm` params
- **Loss config baseline:** `configs/model/action_head.yaml` defines `loss.lambda_text: 1.0`, `loss.lambda_action: 1.0`
- **Training data:** Phase 2.3 reference-based JSONL sequences with `action_ref` → continuous 17-D float32 actions in HDF5; ~200 steps per single-placement episode

**Key Phase 4.1 stance:**
- **Build on Phase 3.2, not beside it.** Phase 3.2 implements loss computation inline in `VLAModel.forward()`, verified with synthetic tensors. Phase 4.1 extracts that inline code into standalone, testable, configurable loss modules with production monitoring — then plugs them back into `VLAModel.forward()`. The mathematical formulations do not change; what changes is the software engineering quality, testability, and observability of the loss computation.
- **No EO-1 code in the local repository.** Investigation confirmed that this repository contains no EO-1 reference implementation code — no cloned directories, no imports, no submodules. All loss modules are therefore newly implemented following the EO-1/Pi-0 architectural pattern (Transfusion dual loss with conditional flow matching) as designed in Phase 3.2. If external code reuse from the public EO-1 open-source implementation is permitted in a future phase, the loss modules could be cross-checked or partially replaced; for now, the "EO-1 reuse first" strategy applies to architectural decisions, not to code reuse within this repo.
- **Fixed-weight balancing as default.** The Pi-0/EO-1 architecture uses a fixed weighted sum of text and action losses. Phase 4.1 implements this as the default strategy, with optional dynamic balancing (uncertainty weighting) available as a config toggle for experimentation. No speculative complexity.
- **Real-data validation.** Phase 3.2 verifies loss computation on synthetic tensors with correct shapes. Phase 4.1 closes the gap to production by verifying on actual Phase 2.3 training data — checking loss magnitudes, gradient flow, and optimization convergence on real data distributions.

**Critical gap this phase closes:**
Phase 3.2 delivers a `VLAModel` with inline loss computation that returns `{total_loss, text_loss, action_loss}` — sufficient for architecture validation with synthetic tensors. But this inline approach has three production gaps: (1) the loss functions are not independently testable or configurable without instantiating the full VLA model; (2) there are no training diagnostics — no perplexity, no per-token accuracy, no per-joint action error breakdown, no loss scale monitoring; (3) the loss has never been computed on real training data, so scale mismatches, masking bugs, and numerical instabilities remain undetected. Without Phase 4.1, the Phase 4.2 training loop would operate blind on its first real training run — an expensive way to discover loss bugs.

---

## 4.1.0) Loss Contract & Token-Type Masking Specification

### What we will do
Codify the Phase 3.2.0 token-type routing rules and the loss interface contracts into frozen data structures — a `TokenType` enum, a `LossOutput` dataclass, and shape verification utilities — that every downstream loss module and the `VLAModel` integration depend on. Define the mapping between loss module outputs and the existing tracking infrastructure.

### Why this matters
Phase 3.2 defines token types and loss routing in prose and pseudocode. Phase 4.1 must turn this into enforceable code contracts before implementing any loss module. Without a frozen `LossOutput` type, each loss module could return a different dict structure, breaking the combined loss and the tracking pipeline. Without `TokenType` as an enum, the integer constants used for masking would be scattered across multiple files as magic numbers. Getting the contract right first prevents cascading integration bugs in 4.1.1–4.1.4.

### Design

**`TokenType` enum:**

| Name | Value | Loss applied | Rationale |
|---|---|---|---|
| `TEXT` | 0 | AR cross-entropy (next-token prediction) | Language grounding; backbone LM head outputs logits |
| `IMAGE` | 1 | None | Vision tokens produced by backbone's internal vision encoder; no supervised target |
| `STATE` | 2 | None | Conditioning input only; robot state projected into hidden space for attention context |
| `ACTION` | 3 | FM velocity MSE | Action generation objective; predicted velocity vs target velocity from CFM |

**`LossOutput` dataclass:**

```python
@dataclass
class LossOutput:
    loss: torch.Tensor          # Scalar loss value (differentiable)
    metrics: dict[str, float]   # Detached monitoring metrics (non-differentiable)
```

All loss modules return `LossOutput`. The `loss` field is the value that participates in backpropagation. The `metrics` dict contains diagnostic quantities (perplexity, accuracy, per-joint MSE, etc.) that are `.detach().item()`-ed at creation time. This separation ensures that monitoring never introduces unintended gradient paths.

**Tracking infrastructure mapping:**

| `LossOutput` field | `extract_loss_components()` key | W&B panel |
|---|---|---|
| `VLACombinedLoss.loss` | `loss/total` | Training / Total Loss |
| `VLATextLoss.loss` | `loss/ar` | Training / AR Text Loss |
| `VLAActionLoss.loss` | `loss/fm` | Training / FM Action Loss |
| `VLATextLoss.metrics["perplexity"]` | `loss/text_perplexity` | Training / Text Perplexity |
| `VLATextLoss.metrics["accuracy_top1"]` | `loss/text_accuracy` | Training / Text Accuracy |
| `VLAActionLoss.metrics["per_joint_mse"]` | `loss/action_per_joint_mse` | Training / Per-Joint MSE |

The `VLACombinedLoss` module is responsible for assembling the full metrics dict by merging sub-loss metrics with prefixed keys and passing them to `log_training_step()` via the `extra_metrics` parameter. The `loss_ar` and `loss_fm` positional parameters map directly to the text and action loss scalars.

**Shape verification utilities:**

- `verify_text_loss_inputs(logits, labels)` — asserts `logits.shape == (B, S, V)`, `labels.shape == (B, S)`, `labels.dtype == torch.long`
- `verify_action_loss_inputs(pred_v, target_v, mask)` — asserts matching `(B, T, 17)` shapes, mask `(B, T)` with values in `{0, 1}`
- These are `assert`-based debug helpers (compiled away with `python -O`), not runtime validators

**Config extension** — extend `configs/model/action_head.yaml` `loss:` section:

```yaml
loss:
  lambda_text: 1.0
  lambda_action: 1.0
  balancing_strategy: "fixed"       # fixed | normalized | uncertainty
  ema_decay: 0.999                  # for "normalized" strategy
  text:
    ignore_index: -100
    label_smoothing: 0.0
  action:
    reduction: "mean"               # mean over valid positions and dimensions
```

### Execution checklist
- Implement `TokenType` enum in `models/losses.py`
- Implement `LossOutput` frozen dataclass in `models/losses.py`
- Implement `verify_text_loss_inputs()` and `verify_action_loss_inputs()` shape checkers
- Extend `configs/model/action_head.yaml` with the full loss config section
- Verify config parses via Hydra compose without error
- Verify existing configs (`model=base`, `model=large`, `model=vlm`, `model=vla`) are unaffected
- Unit test: `LossOutput` construction and metric access
- Unit test: `TokenType` enum values match Phase 3.2.0 contract
- Unit test: shape verification rejects wrong shapes and accepts correct shapes

### Milestone (minimum success criteria)
- `TokenType` enum, `LossOutput` dataclass, and shape verification utilities exist in `models/losses.py`. Extended loss config parses. All existing configs unaffected. 5 unit tests pass.

---

## 4.1.1) Autoregressive Text Loss Module

### What we will do
Implement `VLATextLoss` — a standalone `nn.Module` that computes cross-entropy loss on text token positions with causal shift, ignore masking, optional label smoothing, and training diagnostics (perplexity, top-1 accuracy). This module replaces the inline `F.cross_entropy(...)` call in `VLAModel.forward()` from Phase 3.2.4.

### Why this matters
The autoregressive text loss grounds the model's language understanding — it is the signal that teaches the VLA to interpret task descriptions, narrations, and manipulation instructions. A correct implementation requires careful handling of the causal shift (predict next token from current), the ignore mask (-100 for non-text positions, BOS tokens, and padding), and reduction over variable-length text sequences. The perplexity and accuracy metrics are essential for diagnosing whether the model is learning language grounding or merely memorizing surface patterns. Without these diagnostics, the Phase 4.2 training loop would have no visibility into the language branch of the dual objective.

### Design

**`VLATextLoss(nn.Module)`:**

```python
class VLATextLoss(nn.Module):
    """Autoregressive cross-entropy loss for text token positions.

    Handles causal shift internally: position i predicts token i+1.
    Respects ignore_index for non-text positions, padding, and BOS tokens.
    Returns LossOutput with perplexity and top-1 accuracy metrics.
    """

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        ...

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> LossOutput:
        """
        Args:
            logits: (B, S, vocab_size) — full-sequence LM head output (all positions,
                    not compacted text-only spans)
            labels: (B, S) — next-token target IDs for the full sequence;
                    non-text positions (IMAGE, STATE, ACTION), BOS, and padding
                    are set to -100 (ignore_index)

        Returns:
            LossOutput with:
                loss: scalar cross-entropy (differentiable)
                metrics: {"perplexity": float, "accuracy_top1": float, "n_valid_tokens": int}
        """
```

**Loss-routing approach — full-sequence logits with ignore-index masking:**

The text loss is computed on full-sequence logits, **not** on compacted text-only spans. `VLAModel.forward()` passes the LM head output at all sequence positions `(B, S, V)` and a label tensor `(B, S)` where non-text positions (IMAGE, STATE, ACTION), BOS tokens, and padding are set to `ignore_index=-100`. The causal shift then operates on the original sequence positions, preserving the autoregressive structure.

This design is chosen over compacting text-only spans because compaction would concatenate disjoint text spans into a contiguous tensor, and the subsequent causal shift would create **cross-span prediction bugs** — the last token of one text span would be trained to predict the first token of the next text span, even though those tokens are not adjacent in the original interleaved sequence. With full-sequence logits and ignore-index masking, the causal shift at non-text boundaries simply produces ignored positions, which is correct.

**Causal shift:**
```python
# Position i predicts token i+1 — operates on FULL sequence, not compacted spans
shift_logits = logits[:, :-1, :].contiguous()   # (B, S-1, V)
shift_labels = labels[:, 1:].contiguous()        # (B, S-1)
# Non-text positions in shift_labels are -100 → excluded by F.cross_entropy
```

The shift is applied inside `VLATextLoss`, not in `VLAModel`. This keeps the loss module self-contained and matches the standard HuggingFace language model loss pattern. `VLAModel.forward()` passes the raw (unshifted) full-sequence logits and the full-sequence labels (with -100 at all non-text positions).

**Cross-entropy computation:**
```python
loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1),
    ignore_index=self.ignore_index,
    label_smoothing=self.label_smoothing,
    reduction="mean",
)
```

`reduction="mean"` averages over non-ignored tokens only (PyTorch's `F.cross_entropy` with `ignore_index` excludes ignored positions from both numerator and denominator). This produces a scale-invariant loss that does not depend on sequence length.

**Perplexity:**
```python
perplexity = torch.exp(loss).item()
```

Perplexity = exp(cross-entropy). For a randomly initialized model over a 150K vocabulary, expected initial perplexity ≈ 150K ≈ exp(11.9). A converging model should show perplexity decreasing toward single digits on the manipulation domain vocabulary.

**Top-1 accuracy:**
```python
valid_mask = shift_labels != self.ignore_index
preds = shift_logits.argmax(dim=-1)
correct = (preds == shift_labels) & valid_mask
accuracy = correct.sum().float() / valid_mask.sum().float()
```

Accuracy on non-ignored text tokens. Random initialization gives ~1/vocab_size ≈ 0.0007%. A converging model should reach >50% on the relatively constrained manipulation vocabulary within the first few thousand steps.

**All-ignored edge case:**
If all positions have `label == -100` (e.g., a batch with no text tokens), this is handled with an explicit early-return guard rather than relying on framework conventions:
```python
valid_mask = shift_labels != self.ignore_index
n_valid = valid_mask.sum()
if n_valid == 0:
    return LossOutput(
        loss=torch.tensor(0.0, device=logits.device, requires_grad=True),
        metrics={"perplexity": 1.0, "accuracy_top1": 0.0, "n_valid_tokens": 0},
    )
```
This avoids dependence on PyTorch's `F.cross_entropy` behavior for fully-ignored batches (which returns 0.0 by current convention but is not a documented guarantee). The `requires_grad=True` on the zero tensor ensures the computational graph remains valid for backward pass even when text loss contributes nothing.

**Parameter count:** Zero learnable parameters — `VLATextLoss` is a pure computation module. All parameters are in the backbone LM head and the backbone transformer.

### Execution checklist
- Implement `VLATextLoss` in `models/losses.py` with causal shift, cross-entropy, perplexity, and accuracy
- Verify causal shift: with known logits and labels, the predicted token at position i matches label at position i+1
- Unit test: correct cross-entropy value on a hand-computed example (3-token sequence, vocab_size=4, known logits and labels)
- Unit test: ignore_index respected — positions with -100 labels do not contribute to loss
- Unit test: perplexity = exp(loss) for known loss values
- Unit test: top-1 accuracy computation on known predictions
- Unit test: label_smoothing > 0 produces a lower cross-entropy than label_smoothing = 0 on identical inputs
- Unit test: all-ignored sequence returns loss=0.0, perplexity=1.0, accuracy=0.0, no NaN
- Unit test: batch dimension — loss is consistent across batch sizes (same per-sample average)
- Unit test: gradient flows through loss.loss to logits (non-zero grad after backward)

### Milestone (minimum success criteria)
- `VLATextLoss` returns correct cross-entropy, perplexity, and accuracy on synthetic text logits. Causal shift is verified. All-ignored sequences produce zero loss without NaN. Label smoothing is functional. 8 unit tests pass. Gradient flows to logits.

---

## 4.1.2) Flow Matching Action Loss Module

### What we will do
Implement `VLAActionLoss` — a standalone `nn.Module` that computes masked MSE loss between predicted and target velocity vectors at action token positions, with per-joint error breakdown and velocity norm monitoring. This module replaces the inline call to `self.flow_matching.loss(...)` in `VLAModel.forward()` from Phase 3.2.4.

### Why this matters
The flow matching action loss is the signal that teaches the model to generate accurate robot actions. The masked MSE must correctly exclude padded chunk positions (the last chunk of a variable-length episode is zero-padded to `chunk_size=16`). Per-joint error breakdown reveals whether specific joints (e.g., grippers, wrist) are harder to predict, guiding potential per-joint weighting or data augmentation. The velocity norm diagnostic detects mode collapse (all predictions near zero) or explosion (unreasonably large velocities). Without these diagnostics, the Phase 4.2 training loop cannot distinguish between a model that predicts good actions and one that has converged to a trivial solution.

### Design

**Relationship to `FlowMatchingModule.loss()`:**

Phase 3.2.1 defines `FlowMatchingModule.loss(pred_velocity, target_velocity, mask) → scalar`. `VLAActionLoss` is **not** a wrapper around `FlowMatchingModule.loss()` — it is an independent implementation that computes the same masked MSE plus additional monitoring metrics. `FlowMatchingModule.loss()` remains available for standalone flow matching unit tests and for use cases where the full monitoring overhead is unnecessary. There is no circular dependency; both modules independently implement masked MSE (a 5-line computation) with different return types.

**`VLAActionLoss(nn.Module)`:**

```python
class VLAActionLoss(nn.Module):
    """Flow matching velocity MSE loss for action token positions.

    Computes masked MSE between predicted and target velocity fields.
    Provides per-joint error breakdown and velocity norm diagnostics.
    """

    def __init__(self, action_dim: int = 17):
        ...

    def forward(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
    ) -> LossOutput:
        """
        Args:
            pred_velocity: (B, T, action_dim) — predicted velocity from ActionOutputHead
            target_velocity: (B, T, action_dim) — target velocity from FlowMatchingModule
            chunk_mask: (B, T) — 1 for valid action steps, 0 for padded positions.
                        None means all positions are valid.

        Returns:
            LossOutput with:
                loss: scalar masked MSE (differentiable)
                metrics: {"per_joint_mse": list[float] (17), "mean_pred_velocity_norm": float,
                          "mean_target_velocity_norm": float, "mask_fraction_valid": float}
        """
```

**Masked MSE computation:**
```python
sq_error = (pred_velocity - target_velocity) ** 2    # (B, T, action_dim)

if chunk_mask is not None:
    mask = chunk_mask.unsqueeze(-1)                   # (B, T, 1)
    n_valid = mask.sum()
    if n_valid == 0:
        # All positions masked — explicit early return, no division by zero
        return LossOutput(
            loss=torch.tensor(0.0, device=pred_velocity.device, requires_grad=True),
            metrics={
                "per_joint_mse": [0.0] * self.action_dim,
                "mean_pred_velocity_norm": 0.0,
                "mean_target_velocity_norm": 0.0,
                "mask_fraction_valid": 0.0,
            },
        )
    loss = (sq_error * mask).sum() / (n_valid * action_dim)
else:
    loss = sq_error.mean()
```

The all-masked guard is explicit code logic rather than a reliance on framework behavior for zero-denominator edge cases. The `requires_grad=True` on the zero tensor keeps the computational graph valid. The reduction divides by `n_valid * action_dim` — averaging over both valid positions **and** action dimensions. This produces a loss value that is independent of chunk size and batch size.

**Per-joint MSE:**
```python
if chunk_mask is not None:
    per_joint = (sq_error * mask).sum(dim=(0, 1)) / n_valid       # (action_dim,)
else:
    per_joint = sq_error.mean(dim=(0, 1))                          # (action_dim,)
```

Per-joint MSE is a `(17,)` vector whose element ordering matches the frozen 17-D action space defined in `sim/action_space.py:ARM_ACTUATOR_NAMES` (indices 0–14) and `GRIPPER_ACTUATOR_NAMES` (indices 15–16). Specifically: `[spine_z, left_shoulder_y, left_shoulder_x, left_shoulder_z, left_elbow, left_wrist_z, left_wrist_x, left_gripper_z, right_shoulder_y, right_shoulder_x, right_shoulder_z, right_elbow, right_wrist_z, right_wrist_x, right_gripper_z, left_ezgripper, right_ezgripper]`. Logged as a list in `metrics` and individually to W&B for per-joint monitoring dashboards. The canonical source of truth for this ordering is `ARM_ACTUATOR_NAMES + GRIPPER_ACTUATOR_NAMES` — do not hardcode joint names in the loss module; derive them from the action space constants at W&B logging time.

**Note on joint semantics:** Indices 0–14 are delta-q arm actions (normalized `[-1, 1]`); indices 15–16 are absolute gripper commands (`[0, 1]`). Gripper commands use a different value space than arm deltas, so their MSE scale may differ from arm joints. This is expected and should not be "corrected" by per-joint weighting unless empirically justified.

**Velocity norm diagnostics:**
```python
mean_pred_norm = pred_velocity.norm(dim=-1).mean().item()
mean_target_norm = target_velocity.norm(dim=-1).mean().item()
```

If `mean_pred_norm` → 0 while `mean_target_norm` stays constant, the model is collapsing to zero velocity (predicting no motion). If `mean_pred_norm` ≫ `mean_target_norm`, the model is producing unreasonably large velocities. Both cases indicate training failure.

**Mask fraction diagnostic:**
```python
mask_fraction = chunk_mask.float().mean().item() if chunk_mask is not None else 1.0
```

Tracks the fraction of valid (non-padded) action positions. A sudden drop suggests a data pipeline issue (e.g., all episodes ending with large padded chunks).

**Parameter count:** Zero learnable parameters.

### Execution checklist
- Implement `VLAActionLoss` in `models/losses.py` with masked MSE, per-joint breakdown, and velocity diagnostics
- Unit test: correct MSE value on a hand-computed example (B=1, T=2, action_dim=3, known values)
- Unit test: chunk mask correctly excludes padded positions from loss
- Unit test: all-masked batch returns zero loss without NaN
- Unit test: per-joint MSE breakdown sums to approximately `action_dim * loss` (within float precision)
- Unit test: velocity norm diagnostics return reasonable values for known inputs
- Unit test: mask fraction diagnostic is correct
- Unit test: None mask produces same result as all-ones mask
- Unit test: gradient flows through loss.loss to pred_velocity (non-zero grad after backward)

### Milestone (minimum success criteria)
- `VLAActionLoss` returns correct masked MSE, per-joint breakdown, and velocity diagnostics on synthetic velocity pairs. Masking verified. All-masked edge case handled. 8 unit tests pass. Gradient flows to predicted velocity tensor.

---

## 4.1.3) Combined Training Objective

### What we will do
Implement `VLACombinedLoss` — a module that takes the `LossOutput` from `VLATextLoss` and `VLAActionLoss`, combines them with configurable weighting, assembles the full metrics dict for W&B logging, and returns a single `LossOutput` whose `loss` field is the scalar used for `loss.backward()`. This module replaces the inline `lambda_text * text_loss + lambda_action * action_loss` in `VLAModel.forward()`. Support three balancing strategies: fixed (default), EMA-normalized, and uncertainty weighting.

### Why this matters
The combined training objective determines how the model allocates gradient capacity between language understanding and action generation. With fixed equal weights, one loss can dominate training if its gradient magnitude is orders of magnitude larger — a common failure mode in multi-task learning. EMA normalization addresses this by dynamically rescaling losses to comparable magnitudes. Uncertainty weighting (Kendall et al., 2018) goes further by learning task-dependent uncertainty parameters that balance the losses at the mathematical optimum. Phase 4.1 provides all three strategies behind a config switch so that Phase 4.2 can run controlled experiments.

### Design

**`VLACombinedLoss(nn.Module)`:**

```python
class VLACombinedLoss(nn.Module):
    """Combined VLA training objective with configurable balancing.

    Strategies:
        fixed: total = λ_text * L_text + λ_action * L_action
        normalized: total = λ_text * (L_text/EMA_text) + λ_action * (L_action/EMA_action)
        uncertainty: total = L_text/(2σ²_text) + log(σ_text)
                           + L_action/(2σ²_action) + log(σ_action)
    """

    def __init__(
        self,
        lambda_text: float = 1.0,
        lambda_action: float = 1.0,
        balancing_strategy: str = "fixed",
        ema_decay: float = 0.999,
    ):
        ...

    def forward(
        self,
        text_output: LossOutput,
        action_output: LossOutput,
    ) -> LossOutput:
        """
        Returns:
            LossOutput with:
                loss: scalar combined loss (differentiable)
                metrics: merged dict with prefixed keys from both sub-losses plus:
                    "text_loss": float, "action_loss": float,
                    "text_weight_effective": float, "action_weight_effective": float,
                    "loss_ratio_text_action": float
        """
```

**Fixed strategy (default):**
```python
total = self.lambda_text * text_output.loss + self.lambda_action * action_output.loss
```

This is the direct replacement of Phase 3.2.4's inline computation. With `lambda_text=1.0` and `lambda_action=1.0`, the behavior is identical. The effective weights are the configured lambdas.

**Normalized strategy:**
```python
# Update EMAs (detached, no gradient)
self.ema_text = self.ema_decay * self.ema_text + (1 - self.ema_decay) * text_output.loss.detach()
self.ema_action = self.ema_decay * self.ema_action + (1 - self.ema_decay) * action_output.loss.detach()

# Normalize before weighting
total = (self.lambda_text * text_output.loss / self.ema_text.clamp(min=1e-8)
       + self.lambda_action * action_output.loss / self.ema_action.clamp(min=1e-8))
```

Dividing each loss by its EMA makes the effective contribution scale-invariant: if text loss is 10.0 and action loss is 0.01, both contribute roughly equally after normalization regardless of the absolute scale. The `clamp(min=1e-8)` prevents division by zero during the first few steps when EMAs are near-zero.

**EMA initialization:** Both EMAs are initialized to 1.0, not 0.0. This means the normalization has no effect for the first few hundred steps (when EMA ≈ 1.0) and gradually takes effect as the EMAs converge to the true loss scales. This avoids erratic weighting during early training when loss values are unstable.

**Uncertainty weighting (Kendall et al., 2018):**
```python
# Learnable log-variance parameters (initialized to 0.0 → σ² = 1.0)
self.log_var_text = nn.Parameter(torch.tensor(0.0))
self.log_var_action = nn.Parameter(torch.tensor(0.0))

# Loss = L/(2σ²) + log(σ) = L/(2·exp(log_var)) + 0.5·log_var
total = (text_output.loss / (2 * torch.exp(self.log_var_text)) + 0.5 * self.log_var_text
       + action_output.loss / (2 * torch.exp(self.log_var_action)) + 0.5 * self.log_var_action)
```

The `log_var` parameters are learned end-to-end via gradient descent. A high-loss task gets a large σ² (large `log_var`), which downweights its contribution and adds a `log(σ)` regularizer that prevents σ from growing unboundedly. This is the mathematically principled multi-task weighting from Kendall et al. (2018), "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." The two `log_var` parameters add negligible VRAM (2 scalars).

**Effective weight logging:**

For all strategies, the module computes and logs the effective weight per loss component:
```python
metrics["text_weight_effective"] = ∂(total) / ∂(L_text)  # analytical, not autograd
metrics["action_weight_effective"] = ∂(total) / ∂(L_action)
metrics["loss_ratio_text_action"] = text_loss / action_loss
```

For `fixed`: effective weights = configured lambdas. For `normalized`: effective weights = lambda / EMA. For `uncertainty`: effective weights = 1 / (2 × exp(log_var)).

**Metrics assembly:**

The combined loss module merges metrics from both sub-losses with prefixed keys:
```python
merged = {}
for k, v in text_output.metrics.items():
    merged[f"text/{k}"] = v
for k, v in action_output.metrics.items():
    merged[f"action/{k}"] = v
merged["text_loss"] = text_output.loss.detach().item()
merged["action_loss"] = action_output.loss.detach().item()
merged["text_weight_effective"] = ...
merged["action_weight_effective"] = ...
merged["loss_ratio_text_action"] = ...
```

This `merged` dict is passed to `log_training_step(extra_metrics=merged)`. The `loss_ar` and `loss_fm` positional params use the raw scalar values.

**Parameter count:**
- `fixed` and `normalized`: 0 learnable parameters
- `uncertainty`: 2 learnable parameters (`log_var_text`, `log_var_action`)

### Execution checklist
- Implement `VLACombinedLoss` in `models/losses.py` with all three strategies
- Unit test: `fixed` strategy produces exact weighted sum for known inputs
- Unit test: `fixed` with lambda_text=0 produces total = lambda_action * action_loss (and vice versa)
- Unit test: `normalized` strategy — after many steps, effective weights converge to lambda/EMA
- Unit test: `uncertainty` strategy — `log_var` parameters have gradients and are updated
- Unit test: all metrics from sub-losses are present in merged output with correct prefixes
- Unit test: effective weight logging is correct for each strategy
- Unit test: gradient flows through `total_loss` to both sub-loss inputs
- Verify config parsing: `balancing_strategy: "fixed"`, `"normalized"`, and `"uncertainty"` all accepted
- Verify that `fixed` strategy with default lambdas (1.0, 1.0) produces identical results to Phase 3.2.4's inline computation

### Milestone (minimum success criteria)
- `VLACombinedLoss` correctly combines text and action losses for all three strategies. Fixed strategy is numerically identical to Phase 3.2.4's inline sum. Uncertainty strategy produces learned `log_var` parameters with non-zero gradients. All metrics are assembled and prefixed correctly. 7 unit tests pass.

---

## 4.1.4) Loss Sanity Checks & Validation

### What we will do
Create a comprehensive test suite and standalone validation script that verify all loss modules on sample batches — including synthetic data with known properties and, where available, real Phase 2.3 training data — checking numerical stability, gradient flow, optimization convergence, and production readiness. Integrate the loss modules into `VLAModel.forward()`, replacing the inline loss computation from Phase 3.2.4.

### Why this matters
Individual unit tests verify that each loss module computes the correct math on isolated inputs. Sanity checks verify that the full pipeline — from batch to VLAModel to loss to gradient — works correctly end-to-end. This is the last verification gate before the Phase 4.2 training loop. Key risks at this stage: (1) loss magnitudes are mismatched between text and action branches, causing one to dominate; (2) masking bugs cause gradient to flow to wrong positions; (3) loss does not decrease on a small overfit test, indicating a broken training signal; (4) per-joint action errors reveal that some joints are not being trained. The validation script catches all of these before committing GPU-hours.

### Design

**VLAModel integration:**

Update `VLAModel.forward()` to instantiate and call the new loss modules instead of inline computation:

```python
# Phase 3.2.4 inline (replaced):
#   text_loss = F.cross_entropy(text_logits.view(-1, V), text_labels.view(-1), ignore_index=-100)
#   action_loss = self.flow_matching.loss(velocity_pred, target_velocity, chunk_masks_flat)
#   total_loss = lambda_text * text_loss + lambda_action * action_loss

# Phase 4.1 modular (new):
text_output = self.text_loss(text_logits, text_labels)
action_output = self.action_loss(velocity_pred, target_velocity, chunk_masks_flat)
combined_output = self.combined_loss(text_output, action_output)

return {
    "total_loss": combined_output.loss,
    "text_loss": text_output.loss,
    "action_loss": action_output.loss,
    "metrics": combined_output.metrics,
}
```

The return dict is backward-compatible with Phase 3.2.4's interface (same keys: `total_loss`, `text_loss`, `action_loss`). The `metrics` key is new and carries all monitoring data.

**`VLAModel.__init__()` additions:**
```python
self.text_loss = VLATextLoss(
    ignore_index=cfg.loss.text.ignore_index,
    label_smoothing=cfg.loss.text.label_smoothing,
)
self.action_loss = VLAActionLoss(action_dim=cfg.action_head.action_dim)
self.combined_loss = VLACombinedLoss(
    lambda_text=cfg.loss.lambda_text,
    lambda_action=cfg.loss.lambda_action,
    balancing_strategy=cfg.loss.balancing_strategy,
    ema_decay=cfg.loss.ema_decay,
)
```

**Test file** — `tests/test_losses.py`:

| Test class | Scope | Markers | Count |
|---|---|---|---|
| `TestLossContract` | TokenType enum, LossOutput dataclass, shape verification | (none) | ~5 |
| `TestVLATextLoss` | Cross-entropy, causal shift, perplexity, accuracy, edge cases | (none) | ~8 |
| `TestVLAActionLoss` | Masked MSE, per-joint breakdown, velocity norms, edge cases | (none) | ~8 |
| `TestVLACombinedLoss` | Fixed/normalized/uncertainty strategies, metrics assembly | (none) | ~7 |
| `TestLossIntegration` | Full VLAModel.forward() with loss modules, gradient routing | `vlm`, `gpu`, `slow` | ~5 |

Tests marked `(none)` run on CPU without any external dependencies — they use synthetic tensors. Tests marked `vlm, gpu, slow` require the full VLM backbone and are auto-skipped in CI.

**Validation script** — `scripts/validate_losses.py`:

Following the established `scripts/validate_*.py` pattern, runs 12 sequential checks and produces artifacts to `logs/loss_validation/`.

| Check | Input | Verifies |
|---|---|---|
| 1. Config parsing | `action_head.yaml` loss section | All loss configs parse without error |
| 2. Module instantiation | Config | `VLATextLoss`, `VLAActionLoss`, `VLACombinedLoss` instantiate |
| 3. Text loss correctness | Synthetic logits + labels | Loss matches hand-computed cross-entropy |
| 4. Action loss correctness | Synthetic velocity pairs + mask | Loss matches hand-computed masked MSE |
| 5. Combined loss correctness | Synthetic sub-losses | Weighted sum matches for all three strategies |
| 6. VLAModel integration | Synthetic VLA batch | `forward()` returns correct dict keys with finite values |
| 7. Initial loss magnitude | Synthetic VLA batch | Text loss ≈ log(vocab_size) ± 1.0; action loss ≈ O(1) |
| 8. Gradient norms | Synthetic VLA batch | Action head gradient norms in (0.001, 100); no NaN/Inf |
| 9. Gradient routing | Synthetic VLA batch | Trainable params have non-zero grad; frozen backbone has zero grad |
| 10. Overfit convergence | Synthetic VLA batch × 50 steps | Total loss decreases by ≥ 50% after 50 optimizer steps on a single repeated batch |
| 11. Per-joint action MSE | Synthetic VLA batch | All 17 joints have non-zero MSE (no dead joints) |
| 12. Metrics completeness | Full forward pass | All expected metric keys present in output; all values finite |

**Checks 6–12** require `VLAModel` with backbone (GPU + `transformers`). The script detects GPU availability and runs checks 1–5 unconditionally (CPU-only) and 6–12 conditionally.

**Overfit convergence test (check 10):**

This is the most critical sanity check. A single batch is repeated for 50 optimizer steps with `lr=1e-3` (deliberately high for fast convergence). If the total loss does not decrease by at least 50%, there is a fundamental bug in the loss or gradient pipeline. This test uses a small synthetic batch with known properties — it does not require Phase 2.3 real data.

**Real data verification (optional, if Phase 2.3 data available):**

If Phase 2.3 training data exists at `data/training_sequences/v2.3.0/`, the validation script additionally:
- Loads 5 sample sequences from the interleaved VLA subset
- Verifies that the dataloader produces batches matching the Phase 3.2.4 format
- Runs forward pass and reports loss magnitudes on real data

This optional check bridges the gap between synthetic validation and production training. If Phase 2.3 data is not yet available, the check is skipped with a warning — it does not block Phase 4.1 completion.

**Artifacts:**
- `logs/loss_validation/validation_report.json` — all check results (pass/fail + values)
- `logs/loss_validation/overfit_curve.json` — loss values over 50 steps for convergence check
- `logs/loss_validation/per_joint_mse.json` — 17-element per-joint MSE vector
- `logs/loss_validation/gradient_norms.json` — per-parameter-group gradient norms

### Execution checklist
- Integrate loss modules into `VLAModel.__init__()` and `VLAModel.forward()` — replace inline loss computation
- Verify `VLAModel.forward()` return dict is backward-compatible with Phase 3.2.4 (same keys, same types)
- Write `tests/test_losses.py` with all test classes (~33 tests total)
- Create `scripts/validate_losses.py` with 12 checks and artifact output
- Run CPU-only tests: `pytest tests/test_losses.py -v -m "not vlm and not gpu"`
- Run full suite on GPU: `pytest tests/test_losses.py -v`
- Run validation script: `python scripts/validate_losses.py`
- Verify all 12 checks pass (checks 6–12 on GPU; checks 1–5 on CPU)
- Verify overfit convergence: loss decreases ≥ 50% in 50 steps
- Verify all existing tests (`test_models.py`, `test_vlm_backbone.py`, `test_action_head.py`) pass unchanged
- Save artifacts to `logs/loss_validation/`
- Update `CLAUDE.md` with loss module documentation, commands, and test markers

### Milestone (minimum success criteria)
- All loss modules integrated into `VLAModel.forward()`. Test suite passes: ~33 tests across 5 test classes. Validation script exits with 12/12 checks passed (CPU-only: 5/5; GPU: 7/7). Overfit convergence verified. All existing tests pass unchanged. Artifacts saved to `logs/loss_validation/`.

---

# Downstream Contract with Phase 4.2

Phase 4.2 (Training Loop Integration) will consume Phase 4.1's loss modules as follows:

| Phase 4.2 Need | Phase 4.1 Provider | Interface |
|---|---|---|
| Complete loss computation | `VLAModel.forward(batch)` | Returns `{total_loss, text_loss, action_loss, metrics}` |
| Loss for backpropagation | `total_loss` from return dict | Scalar tensor with grad — call `.backward()` directly |
| Per-component logging | `metrics` dict from return dict | Pass to `log_training_step(extra_metrics=metrics)` |
| AR/FM loss for tracking | `text_loss`, `action_loss` from return dict | Pass as `loss_ar=text_loss.item()`, `loss_fm=action_loss.item()` |
| Loss balancing config | `configs/model/action_head.yaml` `loss:` section | Hydra-composable; switch strategy via config override |
| Overfit sanity check | `scripts/validate_losses.py` check 10 | Confirm training signal before launching full run |
| Initial loss baseline | Validation artifacts | Expected initial text loss ≈ log(V), action loss ≈ O(1) |

**Key design constraint:** The loss modules do not call `loss.backward()` or step the optimizer. They return differentiable loss tensors. The Phase 4.2 training loop owns the backward pass, gradient clipping, optimizer step, and scheduler step. This separation matches Phase 3.2.4's design contract.

**Logging integration:** Phase 4.2's training loop calls:
```python
output = model.forward(batch)
tracker.log_training_step(
    loss=output["total_loss"].item(),
    step=step,
    loss_ar=output["text_loss"].item(),
    loss_fm=output["action_loss"].item(),
    extra_metrics=output["metrics"],
)
```

This is compatible with the existing `tracking/experiment.py:log_training_step()` signature — no changes needed to the tracking infrastructure.

---

# Startup-Grade Outputs (deliverables by end of 4.1)
- **Autoregressive text loss module** — `VLATextLoss` in `models/losses.py` with cross-entropy, causal shift, perplexity, top-1 accuracy, label smoothing, and all-ignored edge case handling
- **Flow matching action loss module** — `VLAActionLoss` in `models/losses.py` with masked MSE, per-joint breakdown (17 joints), velocity norm diagnostics, and mask fraction tracking
- **Combined training objective** — `VLACombinedLoss` in `models/losses.py` with three configurable strategies (fixed, normalized, uncertainty), metrics assembly, and effective weight logging
- **Loss contract** — `TokenType` enum, `LossOutput` dataclass, shape verification utilities — all in `models/losses.py`
- **VLAModel integration** — `VLAModel.forward()` updated to use loss modules; return dict backward-compatible with Phase 3.2.4
- **Test suite** — ~33 tests across 5 test classes in `tests/test_losses.py`, covering unit + integration
- **Validation script** — 12-check `scripts/validate_losses.py` with overfit convergence test and artifact output
- **Extended loss config** — `configs/model/action_head.yaml` `loss:` section with text/action sub-configs and balancing strategy

---

# Files Inventory

### New files (3)

| File | Purpose | EO-1 reuse status |
|---|---|---|
| `models/losses.py` | `TokenType`, `LossOutput`, `VLATextLoss`, `VLAActionLoss`, `VLACombinedLoss` | **Newly implemented** — no EO-1 loss code in local repo; follows Pi-0/EO-1 Transfusion dual-loss pattern |
| `tests/test_losses.py` | Unit + integration tests (~33 tests) | **Newly implemented** — project-specific test suite |
| `scripts/validate_losses.py` | 12-check validation script with artifacts | **Newly implemented** — follows established `scripts/validate_*.py` pattern |

### Modified files (3)

| File | Change |
|---|---|
| `models/vla_model.py` | Replace inline loss computation in `forward()` with `VLATextLoss`, `VLAActionLoss`, `VLACombinedLoss` calls; add loss modules to `__init__()` |
| `configs/model/action_head.yaml` | Extend `loss:` section with `balancing_strategy`, `ema_decay`, `text:` sub-config, `action:` sub-config |
| `CLAUDE.md` | Add loss module documentation (module structure, config keys, test commands, validation commands) |

---

# Phase 4.1 Definition of Done

Phase 4.1 is complete when:
- `TokenType` enum, `LossOutput` dataclass, and shape verification utilities exist in `models/losses.py`.
- `VLATextLoss` correctly computes cross-entropy with causal shift, returns perplexity and top-1 accuracy, handles all-ignored sequences.
- `VLAActionLoss` correctly computes masked MSE, returns per-joint breakdown and velocity diagnostics, handles all-masked sequences.
- `VLACombinedLoss` correctly implements fixed, normalized, and uncertainty balancing strategies. Fixed strategy with default lambdas is numerically identical to Phase 3.2.4's inline computation.
- `VLAModel.forward()` uses the new loss modules and returns a backward-compatible dict with an additional `metrics` key.
- All existing tests (`test_models.py`, `test_vlm_backbone.py`, `test_action_head.py`) continue to pass — zero regressions.
- Loss config in `configs/model/action_head.yaml` parses for all three strategies.
- Test suite passes: ~33 tests across 5 test classes.
- Validation script (`scripts/validate_losses.py`) exits with 12/12 checks passed.
- Overfit convergence verified: total loss decreases ≥ 50% in 50 optimizer steps on a repeated synthetic batch.
- Per-joint action MSE shows non-zero error on all 17 joints (no dead joints).
- All monitoring metrics (perplexity, accuracy, per-joint MSE, velocity norms, effective weights) are finite and logged.
- Artifacts saved to `logs/loss_validation/`.
- `CLAUDE.md` is updated with loss module docs, config keys, and test/validation commands.

---

# Verification Plan

### Local (lab PC or CI, CPU-only)
```bash
# Unit tests — no GPU, no transformers required
pytest tests/test_losses.py -v -m "not vlm and not gpu and not slow"

# Validation checks 1–5 (CPU-only)
python scripts/validate_losses.py --cpu-only
```

### Local (lab PC, GPU)
```bash
# Full test suite including VLAModel integration
pytest tests/test_losses.py -v

# Backward compatibility
pytest tests/test_action_head.py tests/test_vlm_backbone.py tests/test_models.py -v

# Full validation (all 12 checks)
python scripts/validate_losses.py

# Overfit convergence only
python scripts/validate_losses.py --check 10
```

### CI (no changes needed)
Existing CI installs `.[ci]` which excludes `transformers`. Tests marked `vlm`/`gpu`/`slow` are auto-skipped. CPU-only loss tests run in CI. Verify with:
```bash
pytest tests/ -m "not slow and not gpu and not mujoco and not vlm" -v
```
