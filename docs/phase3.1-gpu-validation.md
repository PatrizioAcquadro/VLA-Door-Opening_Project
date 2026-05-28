# Phase 3.1 GPU Validation

## 1. Lab PC (RTX 4090) — Completed

**Driver upgrade**: 535 → 570 (CUDA 12.2 → 12.8, matching PyTorch 2.10.0+cu128).

**Fixes applied during GPU validation**:
- Added `torchvision>=0.15.0` to `vlm` + `dev` groups in `pyproject.toml` (required by `Qwen3VLVideoProcessor`)
- Changed `AutoModelForCausalLM` → `AutoModelForImageTextToText` in `load_vlm_backbone()` — the causal LM class loaded text-only `Qwen3_5ForCausalLM` without vision encoder; `AutoModelForImageTextToText` resolves to `Qwen3_5ForConditionalGeneration` which includes `model.model.visual` (334M params)

**Results**: 43/43 tests pass (35 VLM + 8 backward-compat model tests).

```bash
pytest tests/test_vlm_backbone.py tests/test_models.py -v
# 43 passed in 10.62s
```

---

## 2. Gilbreth A100 — Pending

### Prerequisites

1. **Pre-cache weights** (one-time, no GPU needed):
   ```bash
   sbatch infra/gilbreth/job_templates/07_download_vlm_weights.sh
   ```
   Wait for completion (~10 min). Weights land in `$VLA_SCRATCH_ROOT/cache/huggingface/`.

2. **Install VLM deps** (if not already in conda env):
   ```bash
   pip install -e ".[vlm]"
   ```

### Run tests

```bash
# Interactive GPU session
srun --gres=gpu:1 --time=00:30:00 --partition=ai --pty bash

# Inside the session:
source activate vla-lego
cd $PROJECT_DIR

# All VLM tests (19 CPU + 16 GPU = 35)
pytest tests/test_vlm_backbone.py -v

# Or GPU-only subset
pytest tests/test_vlm_backbone.py -v -m "vlm and gpu"
```

### What success looks like

- 35 tests pass, no errors, no skips
- `TestVLMBackboneLoading` (8 tests): bf16 dtype, hidden_size=2560, ~4.54B params, frozen by default
- `TestProcessorFunctions` (8 tests): vision token measurement, single/quad-view preprocessing, text tokenization, context budget

### Troubleshooting

- **`ImportError: torchvision`**: Run `pip install torchvision`
- **`torch_dtype is deprecated`**: Warning only (transformers 5.x prefers `dtype`), does not affect correctness
- **OOM on model load**: Ensure `device_map="auto"` in config. Qwen3.5-4B needs ~8 GB VRAM for bf16 weights
- **Weight download timeout**: Pre-cache with job 07, or set `HF_HOME` to scratch storage

---

## 3. Key metrics from GPU validation

| Metric | Expected range | Test |
|--------|---------------|------|
| Vision tokens per 320x320 image | 130–260 | `test_estimate_vision_tokens` |
| Total params | ~4.54B | `test_backbone_info_param_count` |
| Context budget: remaining for actions | >5000 tokens | `test_context_budget_with_measured_tokens` |

These values are inputs to Phase 3.1.3 (inference sanity check) and Phase 3.2 (action head sizing).
