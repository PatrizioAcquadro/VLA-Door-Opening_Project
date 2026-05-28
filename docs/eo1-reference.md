# EO-1 Repository Reference

Reference for the EO-1 (EmbodiedOne) VLA robot model. Use this for targeted lookups during future phase implementations instead of re-exploring the full repo online.

**Repo**: https://github.com/EO-Robotics/EO1 (archived, read-only since 2025-11-12)
**Paper**: arXiv:2508.21112
**HuggingFace**: `IPEC-COMMUNITY/EO-1-3B` (trained), `IPEC-COMMUNITY/eo1-qwen2_5_vl` (base for fine-tuning)

---

## Architecture Summary

- **Backbone**: Qwen2.5-VL-3B-Instruct (hidden_size=2048, ~3B params)
- **Two output modes** sharing one transformer:
  - **Text generation**: standard autoregressive `.generate()` via HuggingFace GenerationMixin
  - **Action generation**: flow matching iterative denoising via `model.sample_actions()` — 10 steps from Gaussian noise → continuous actions
- **Action chunk**: 50 timesteps, up to 32 action dimensions
- **Training loss**: MSE flow matching — `FM_loss = MSE(u_t, v_t)` where `u_t = noise - actions`
- **Inference VRAM**: ~6–6.5 GB (bf16)

---

## File Layout

```
EO1/
  eo/
    constants.py              # Special tokens, system message
    model/
      configuration_eo1.py    # EO1VisionFlowMatchingConfig (PretrainedConfig)
      modeling_eo1.py         # Main model: forward(), sample_actions()
      modeling_qwen2_5_vl.py  # Modified Qwen2.5-VL backbone
      processing_eo1.py       # EO1VisionProcessor: __call__(), select_action()
    data/                     # Dataset loading
    train/                    # Training logic (Accelerate-based)
  scripts/
    train.py                  # Training entry point
    eval_policy.py            # Synthetic-image policy evaluation
    inference_service.py      # Inference service stub
    chat_template.json        # Jinja2 template for multimodal chat
  tests/
    test_vlm.py               # Manual VLM test (no assertions)
    test_dataset.py           # Empty stub
  getting_started/            # Tutorial notebooks
  experiments/                # Benchmark configs (LIBERO, SimplerEnv, etc.)
  demo_data/
    example1.jpg, example2.png
    refcoco/images/           # 6 COCO images for testing
```

---

## Key Modules

### `eo/model/modeling_eo1.py` — Main Model

`EO1VisionFlowMatchingModel` inherits from Qwen2.5-VL's conditional generation model.

**`forward()` (training)**:
1. `embed_prefix()`: runs vision encoder on pixel_values, merges into token embeddings via masked scatter at image_token positions
2. Samples random timesteps and noise for flow matching
3. Interpolates noisy actions: `x_t = time * noise + (1 - time) * actions`
4. Predicts velocity `v_t` via action head MLP
5. Returns MSE loss: `FM_loss = MSE(u_t, v_t)` where `u_t = noise - actions`

**`sample_actions(num_steps=10)` (inference)**:
1. Processes prefix (images + text + state) once through backbone
2. Starts from `a_t^0 ~ N(0, I)`
3. Each step: embeds noisy actions via `embed_suffix()` (sinusoidal time embedding + MLP), passes through transformer, extracts velocity prediction, updates action estimate with `dt = -1/num_steps`
4. Returns denoised action chunk

**`embed_prefix()`**: extracts image/video features from vision encoder, injects into token embeddings using masked scatter at image token positions.

**`embed_suffix()`**: sinusoidal timestep embedding → MLP → action token embeddings for the noisy action sequence.

**Action head**: 2-layer MLP (`hidden_size → hidden_size → action_dim`).

### `eo/model/processing_eo1.py` — Processor

`EO1VisionProcessor` wraps Qwen2.5-VL's processor.

- `__call__()`: passes images through `image_processor` (Qwen2.5-VL native), produces `pixel_values` and `image_grid_thw`
- `select_action(model, batch)`: full pipeline raw observations → actions
  - `_prepare_robot_inputs(batch)`: normalizes states, constructs chat messages with image/state/action markers
  - Applies chat template → formats multimodal inputs
  - Calls model to generate/sample actions
  - `_process_robot_outputs()`: unnormalizes predicted actions
- Accepts both numpy uint8 arrays AND PIL Images in the same batch
- Placeholder tokens expand: `"<|placeholder|>" * (grid_thw.prod() // merge_length)`
- Vision tokens wrapped in `<|vision_start|>...<|vision_end|>`

### `eo/constants.py` — Special Tokens

```
ACTION_START = "<|action_start|>"
ACTION_END   = "<|action_end|>"
STATE_START  = "<|state_start|>"
STATE_END    = "<|state_end|>"
SYSTEM_MESSAGE = "..."   # default system prompt
```

### `eo/model/configuration_eo1.py` — Config

`EO1VisionFlowMatchingConfig` extends `PretrainedConfig`:
- `action_dim`, `action_chunk_size` (default 50), `num_denoise_steps` (default 10)
- `state_dim`, action/state normalization statistics

---

## Architecture Constants Comparison

| Parameter | EO-1 (Qwen2.5-VL-3B) | VLA-LEGO (Qwen3.5-4B) |
|---|---|---|
| hidden_size | 2048 | 2560 |
| num_hidden_layers | ~28 | 32 |
| vocab_size | ~151,000+ | 248,320 |
| vision encoder | Qwen2.5-VL ViT | Qwen3.5 native vision |
| action_chunk_size | 50 | TBD (Phase 3.2) |
| action_dim | up to 32 | 17 (frozen) |
| denoising steps | 10 | TBD (Phase 3.2) |
| flow matching loss | MSE(u_t, v_t) | TBD (Phase 3.2) |
| action head | 2-layer MLP | TBD (Phase 3.2) |
| inference VRAM | ~6–6.5 GB (bf16) | ~8 GB (bf16) |

---

## Testing & Validation (Sparse)

EO-1 has minimal testing — no pytest, no assertions, no CI:

- `tests/test_vlm.py`: Loads raw Qwen2.5-VL (NOT EO1 model), generates text from COCO image, interactive multi-turn chat. Manual smoke test only.
- `scripts/eval_policy.py`: Creates random 224×224 numpy arrays, runs `processor.select_action()`, prints output. No assertions.
- **Not tested**: parameter count, NaN/Inf, output shapes, memory profiling, determinism, gradient flow.

---

## Phase-Specific Lookup Guide

| VLA-LEGO Phase | What to check in EO-1 |
|---|---|
| **3.1** (backbone) | `modeling_eo1.py` init, `configuration_eo1.py`, `processing_eo1.py` |
| **3.2** (action head) | `modeling_eo1.py`: action head MLP, `embed_suffix()`, `sample_actions()`, `constants.py` special tokens |
| **3.3** (training) | `eo/train/`, `scripts/train.py`, flow matching loss in `forward()` |
| **3.4** (inference) | `sample_actions()`, `select_action()`, KV cache in `test_vlm.py` |

---

## Reusable Patterns

| Pattern | EO-1 Source | Applicability |
|---|---|---|
| Synthetic image testing | `eval_policy.py` | Inference sanity checks when MuJoCo unavailable |
| Raw numpy uint8 input | `processing_eo1.py` | Already adopted in `preprocess_images()` |
| Chat template formatting | `processor.apply_chat_template()` | Already adopted in `preprocess_images()` |
| Flow matching denoising | `sample_actions()` | Phase 3.2 action head |
| Action head MLP | 2-layer `hidden → hidden → action_dim` | Phase 3.2 |
| Action/state special tokens | `<\|action_start\|>`, `<\|state_start\|>` | Phase 3.2 token design |
| Prefix/suffix embedding split | `embed_prefix()` + `embed_suffix()` | Phase 3.2 efficient inference |
| KV cache across turns | `test_vlm.py` past_key_values | Phase 3.3+ multi-step |
