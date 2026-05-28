# Phase 2.2 — VLM-Based Annotation Pipeline (4 days)

**Goal:** Generate structured language annotations for all 10K trajectory episodes produced in Phase 2.1 — task descriptions, step narrations, and reasoning QA pairs — using a self-hosted VLM, validated against simulation ground truth, and formatted for direct consumption by Phase 2.3's interleaved data construction pipeline.

**Fixed upstream decisions (from 2.1):**
- **Dataset**: 10K HDF5 episodes (`data/demonstrations/v2.1.1/episodes/episode_{seed:06d}.hdf5`), with train/val/test manifests and `dataset_stats.json`
- **Per-episode content**: RGB+Depth+Segmentation (4 views, 320×320), 52-D state, 17-D action, 20 Hz
- **Labels**: 8 waypoint phases (0–7: pre-grasp, approach, close-gripper, lift, transport, lower/insert, release, retract), `grasp_state`, `episode_outcome` ("success" | failure type), `perturbation_type`, `failure_timestep`, `has_recovery` (bool), `recovery_start_step` (int, -1 if none). **Upstream requirement on Phase 2.1**: `has_recovery` and `recovery_start_step` must be stored in the HDF5 `labels/` group so Phase 2.2 can deterministically identify recovery segments without heuristic inference from phase labels.
- **Metadata**: seed, level, brick_types, assembly_result (JSON), goal (JSON), spawn_poses (JSON), control_hz, version, timestamp
- **Camera metadata**: per-camera intrinsics (constant), per-step extrinsics [pos(3) + mat(9)]
- **Views**: overhead, left_wrist_cam, right_wrist_cam, third_person (frozen 4-view contract)
- **Episode structure**: 8-phase pick-and-place sequence, ~200 steps, ~55 MB per file
- **Composition**: 7K success + 2K failure + 1K recovery episodes

**Key Phase 2.2 stance:**
- **Self-hosted VLM**: Qwen3.5-9B on a single A100 80GB. Qwen3.5 (released March 2026) is natively multimodal — vision is built into all model sizes via early fusion training on trillions of multimodal tokens, so no separate "-VL" variant is needed. The 9B dense model outperforms Qwen3-VL and Qwen2.5-VL across reasoning, spatial understanding, and visual benchmarks. It is fully reproducible (no API dependency, pinned weights), fits comfortably on a single A100 (~18 GB bf16 + KV cache), and shares the Qwen model family lineage with EO-1's backbone.
- **Sim-grounded generation**: Simulation metadata (object positions, phase labels, success metrics, brick types, placement targets) serves as the authoritative source of all spatial and temporal facts. The VLM generates natural language conditioned on this ground truth — it does not infer facts from pixels alone. This eliminates the primary VLM failure mode (spatial hallucination) while preserving natural, varied language.
- **Template + VLM hybrid pipeline**: Following EO-Data1.5M's "rule-based cleaning and LLM-based rewriting" approach. Structured facts are extracted from sim ground truth (instant, deterministic). The VLM naturalizes these facts into fluent descriptions and QA pairs. This is 3–5× faster than pure VLM generation and produces factually reliable output.
- **EO-Data1.5M-aligned categories**: Three annotation families — (1) task descriptions + step narrations, (2) reasoning QA (temporal + spatial + physical common sense), (3) keyframe captions. Stored as JSONL sidecars alongside HDF5 episodes.
- **Downstream contract**: Annotations are structured so Phase 2.3 can mechanically assemble them into interleaved vision-text-action sequences, temporal reasoning data, spatial reasoning data, and free chatting format — without additional VLM calls.

**Critical gap this phase closes:**
Phase 2.1 produces physically valid robot demonstrations with actions, states, and images — but no language. VLA training in the EO-1 framework requires language-conditioned data: task instructions that describe *what* the robot is doing, step narrations that segment *how* it proceeds, and reasoning QA that explains *why* decisions are made and *where* objects are. Without these annotations, the trajectory dataset cannot be used for interleaved vision-text-action pre-training.

---

## 2.2.0) Annotation Schema Specification

### What we will do
Define the complete annotation schema — JSON structure, field names, value types, and semantic contracts — for all language annotations that Phase 2.2 will generate. This schema is the single source of truth that the annotation pipeline writes to and Phase 2.3 reads from.

### Why this matters
Getting the schema right before generating annotations for 10K episodes prevents costly rework. The schema must be rich enough for EO-1 training, aligned with EO-Data1.5M's three categories (physical, reasoning, spatial), and structured so Phase 2.3 can consume it without ambiguity.

### Schema design

**Per-episode annotation file**: `episode_{seed:06d}_annotations.jsonl`
One JSON object per line, one file per episode, stored in `data/annotations/v2.2.0/` parallel to the HDF5 directory.

```jsonl
{"type": "task_description", "episode_seed": 0, "level": 1, "text": "...", "brick_types": ["2x2"], "n_placements": 1, "outcome": "success", "has_recovery": false, "failure_step": -1}
{"type": "step_narration", "episode_seed": 0, "phase": 0, "phase_name": "pre_grasp", "step_range": [0, 25], "text": "...", "arm": "left", "brick_type": "2x2", "placement_index": 0, "is_recovery": false}
{"type": "temporal_qa", "episode_seed": 0, "subtype": "planning", "question": "...", "answer": "...", "evidence_steps": [0, 50, 100]}
{"type": "spatial_qa", "episode_seed": 0, "subtype": "object_referencing", "question": "...", "answer": "...", "camera": "overhead", "keyframe_step": 50}
{"type": "physical_qa", "episode_seed": 0, "subtype": "grasp_physics", "question": "...", "answer": "..."}
{"type": "keyframe_caption", "episode_seed": 0, "step": 50, "camera": "overhead", "text": "..."}
```

**Recovery episode example** (1K of the 10K episodes):
```jsonl
{"type": "task_description", "episode_seed": 7500, "level": 1, "text": "The robot attempts to place a 2x2 brick but misaligns the insertion. It detects the failure and re-approaches for a successful retry.", "brick_types": ["2x2"], "n_placements": 1, "outcome": "recovery", "has_recovery": true, "failure_step": 120}
{"type": "step_narration", "episode_seed": 7500, "phase": 5, "phase_name": "lower_insert", "step_range": [100, 130], "text": "...", "arm": "left", "brick_type": "2x2", "placement_index": 0, "is_recovery": false}
{"type": "step_narration", "episode_seed": 7500, "phase": 0, "phase_name": "pre_grasp", "step_range": [135, 155], "text": "...", "arm": "left", "brick_type": "2x2", "placement_index": 0, "is_recovery": true}
```

**Annotation categories** (following EO-Data1.5M):

| Category | Type Key | Count per Episode | VLM Required | Description |
|----------|----------|-------------------|--------------|-------------|
| Task description | `task_description` | 1 | Yes | Full episode summary with brick types, goal, outcome |
| Step narration | `step_narration` | 8 per placement | Light (rephrasing) | One per waypoint phase, arm, gripper state |
| Temporal reasoning QA | `temporal_qa` | 4 | Light (rephrasing) | Planning, sequencing, verification, outcome prediction |
| Spatial reasoning QA | `spatial_qa` | 4 | Light (rephrasing) | Object referencing, trajectory prediction, manipulation planning, multiview |
| Physical common sense QA | `physical_qa` | 2–3 | No (template-only) | Grasp physics, press-fit, stability |
| Keyframe caption | `keyframe_caption` | 3–4 | Yes | Per-view captions at key phase transitions |

**Estimated totals**: ~22–24 annotation records per single-placement episode. ~300K total records for 10K episodes (accounting for multi-placement episodes at higher levels).

**Annotation storage layout:**
```
data/annotations/v2.2.0/
├── episodes/
│   ├── episode_000000_annotations.jsonl
│   └── ...
├── schema.json                # JSON Schema for validation
├── quality_report.json        # per-category pass rates, diversity metrics
├── annotation_stats.json      # distribution of types, lengths, subtypes
└── manifest.json              # maps episode seeds to annotation file paths
```

### Execution checklist
- Define annotation dataclasses in `data/annotations/schema.py` (frozen, typed)
- Write JSON Schema file (`data/annotations/v2.2.0/schema.json`) for format enforcement
- Enumerate all `type` and `subtype` values as string constants
- Write one synthetic example annotation file by hand, validate against schema
- Map each annotation type to Phase 2.3 consumer: document which Phase 2.3 workstream reads which annotation type

### Milestone (minimum success criteria)
- Schema is defined, documented, and validated. A synthetic example annotation file passes JSON Schema validation. Phase 2.3 downstream mapping is explicitly documented.

---

## 2.2.1) VLM Annotation Infrastructure

### What we will do
Set up Qwen3.5-9B on a single A100 80GB, implement the template + VLM hybrid pipeline, build prompt templates for each annotation category, and implement the per-episode annotation function.

### Why this matters
This is the infrastructure all annotation generation depends on. The VLM must be correctly loaded, the hybrid pipeline must reliably combine sim ground truth with VLM naturalization, and the system must process 10K episodes within the 4-day timeline.

### Design

**VLM setup — two-tier deployment:**

| Environment | Model | Precision | VRAM | Purpose |
|-------------|-------|-----------|------|---------|
| **Gilbreth A100 80GB** (production) | `Qwen/Qwen3.5-9B` | bfloat16 | ~22 GB / 80 GB | Full 10K annotation run |
| **Lab PC RTX 4090 24GB** (dev/test) | `Qwen/Qwen3.5-4B` | bfloat16 | ~10 GB / 24 GB | Pipeline development, prompt iteration, single-episode tests |

- Both models are natively multimodal (early fusion) — accept interleaved image+text inputs without a separate vision encoder adapter
- Qwen3.5-9B at bf16 (~22 GB) exceeds the 4090's 24 GB headroom, so local dev uses the 4B variant which fits comfortably and shares the identical architecture and prompt format
- Serving alternative: vLLM (`vllm serve Qwen/Qwen3.5-9B --port 8000`) for higher throughput via continuous batching on Gilbreth
- Inference: sequential per-episode (one VLM session per episode with multiple calls)
- Determinism: fixed seed per episode for reproducible annotations
- Config switch: `configs/annotation/default.yaml` specifies `model_id` — override with `annotation.model_id=Qwen/Qwen3.5-4B` for local dev

**Template + VLM hybrid pipeline** (per episode):
```
1. Load HDF5 metadata + phase labels + state trajectories   [no VLM, instant]
2. Extract keyframe images at phase transitions               [no VLM, instant]
3. Extract sim ground truth context:                           [no VLM, instant]
   - Brick types, positions (from spawn_poses + data.xpos)
   - Target positions (from goal JSON)
   - Phase boundaries (from labels/phase)
   - EE trajectory (from state vectors)
   - Outcome + failure info (from labels)
4. Generate template drafts for all annotation types:          [no VLM, instant]
   - Fill templates with concrete sim facts
5. VLM naturalization pass:                                    [VLM, ~10–15s/episode]
   - Task description: VLM generates from images + facts
   - Step narrations: VLM rephrases 8 template drafts (batched in one call)
   - QA pairs: VLM rephrases template-generated Q&A (batched)
   - Keyframe captions: VLM generates from keyframe images
6. Parse + validate VLM output against schema                  [no VLM, instant]
7. Retry on parse failure (max 2 retries, simplified prompt)   [VLM, rare]
8. Write validated JSONL                                       [no VLM, instant]
```

**Throughput estimate** (template + VLM hybrid):
- Per episode: ~10–15 seconds total VLM time (batched calls, short structured outputs)
- 10K episodes: ~28–42 hours on 1 A100
- With 2 A100s (1 Gilbreth node, data-parallel by seed range): ~14–21 hours
- Fits comfortably in 4-day timeline with margin for validation and cleaning

**Keyframe selection:**
- Phase transitions: first frame of each waypoint phase (from `labels/phase` change points)
- Episode bookends: first frame, last frame
- Insertion moment: frame where Z engagement occurs (from state trajectory)
- Total: ~10–12 keyframes per episode, deterministic from phase labels
- Primary view: overhead (best spatial context); secondary: active wrist cam

### Execution checklist
- Add `transformers>=4.49.0`, `accelerate>=0.30.0` to `pyproject.toml` (`annotation` dependency group); optionally `vllm>=0.7.0` for continuous-batching serving
- `AnnotationModel` class in `data/annotations/model.py`: loads VLM, wraps inference with structured JSON output parsing, retry logic
- Prompt templates in `configs/annotation/prompts.yaml`: one section per annotation type, with system prompt, ground-truth template, and generation instruction
- `KeyframeSelector` in `data/annotations/keyframes.py`: extracts keyframe indices and images from HDF5
- `GroundTruthExtractor` in `data/annotations/ground_truth.py`: extracts all sim facts needed for template generation
- `annotate_episode(seed, hdf5_path, model, config) → Path` end-to-end function producing validated JSONL
- Standalone test script: `scripts/annotate_single_episode.py` — annotate one episode, print all outputs
- Throughput benchmark: measure seconds/episode, project total runtime

### Milestone (minimum success criteria)
- Qwen3.5-9B loads on single A100, generates all 6 annotation types for one episode, all outputs pass schema validation. Measured throughput ≤ 15 seconds/episode.

---

## 2.2.2) Task Description & Step Narration Generation

### What we will do
Generate two annotation categories across all 10K episodes: (1) one episode-level task description summarizing the full manipulation sequence, and (2) per-phase step narrations describing each waypoint phase in the pick-and-place sequence.

### Why this matters
Task descriptions provide the high-level instruction that conditions the VLA policy ("Pick up the red 2×4 brick and place it on the baseplate at row 3, column 2"). Step narrations provide fine-grained temporal segmentation that Phase 2.3 will use to construct interleaved vision-text-action sequences with language aligned to specific timestep ranges.

### Design

**Task description generation:**
- Input: overhead keyframes (start, mid, end) + sim context (brick types, target positions, outcome)
- VLM prompt structure:
  ```
  System: You are an expert robot operation narrator for a bimanual-capable robot executing single-arm demonstrations.

  Context:
  - Brick types: {brick_types}
  - Goal: place {n_placements} brick(s) at baseplate positions {target_positions}
  - Active arm: {arm} (selected by workspace position)
  - Outcome: {outcome}

  [3 overhead keyframe images]

  Write a natural 2–4 sentence task description. Be specific about brick types,
  target locations, and the manipulation sequence. Output JSON: {"text": "..."}
  ```
- Temperature: 0.7 (variety across episodes)
- Outcome-dependent: success/failure/recovery episodes get different prompt variants

**Step narration generation:**
- Input: per-phase sim facts (phase name, arm, gripper transition, EE displacement, duration)
- Template draft (no VLM): "The {arm} arm {action_verb} the {brick_type} brick, moving the gripper from ({x0:.2f}, {y0:.2f}, {z0:.2f}) to ({x1:.2f}, {y1:.2f}, {z1:.2f}) over {duration:.1f}s."
- VLM rephrasing: send all 8 drafts in one call, ask VLM to rephrase each into natural language while preserving facts
- Temperature: 0.3 (precision over variety for step-level descriptions)

**Recovery episode handling:**
Recovery episodes contain TWO phase sequences: phases 0–7 for the initial (failed) attempt, then phases 0–7 again for the recovery retry. The `is_recovery` field on each `step_narration` disambiguates them. The pipeline uses `recovery_start_step` from upstream labels to split the phase timeline:
- Steps `[0, recovery_start_step)`: narrated with `is_recovery: false`, failure-aware language
- Steps `[recovery_start_step, end]`: narrated with `is_recovery: true`, retry-aware language
- Keyframe captions include the failure moment (`failure_timestep`) and recovery start (`recovery_start_step`)

**Variation strategy:**
- Vocabulary pool: randomize action verbs per episode ("grasp"/"pick up"/"grab", "place"/"position"/"set down")
- Sentence structure rotation: declarative, progressive, passive voice (cycled by seed % 3)
- Failure episodes: prompt explicitly requires mentioning the failure mode
- Recovery episodes: prompt distinguishes initial attempt ("the robot attempted to...") from retry ("the robot re-approaches...")

### Execution checklist
- `generate_task_descriptions(seeds, hdf5_dir, model, config)` → writes `task_description` records to JSONL
- `generate_step_narrations(seeds, hdf5_dir, model, config)` → writes `step_narration` records
- Factual consistency spot-check: 50 random episodes — verify brick types, positions, outcomes match sim metadata
- Coverage check: every episode has exactly 1 `task_description` and ≥ 8 `step_narration` records
- Failure/recovery episodes: verify annotations mention failure mode and recovery attempt
- Multi-placement episodes (Level 2–3): verify step narrations cover all placements (8 × n_placements)

### Milestone (minimum success criteria)
- All 10K episodes have task descriptions and step narrations. ≥ 95% pass schema validation. 50-episode spot-check confirms factual consistency with sim metadata.

---

## 2.2.3) Reasoning QA Generation

### What we will do
Generate three categories of QA pairs for all 10K episodes — temporal reasoning, spatial reasoning, and physical common sense — following EO-Data1.5M's annotation categories. These are the core annotations that give the VLA model reasoning capability beyond pure action cloning.

### Why this matters
EO-Data1.5M's key insight is that interleaving reasoning QA with robot control data teaches the model to plan, verify, and ground actions in spatial understanding. Temporal QA teaches task planning and sequencing. Spatial QA teaches geometric reasoning from camera views. Physical QA teaches intuition about forces, contact, and stability. Without these, the model learns only to mimic actions without understanding why.

### Design

**Temporal reasoning QA** (4 subtypes, 1 each per episode):

| Subtype | Question Pattern | Answer Source |
|---------|-----------------|---------------|
| `planning` | "What steps are needed to {goal}?" | Phase sequence + goal JSON |
| `sequencing` | "What should the robot do after {current_phase}?" | Next phase in sequence |
| `verification` | "Has the robot successfully {last_action}?" | assembly_result + check_placement values |
| `outcome_prediction` | "Will the current approach succeed?" | episode_outcome + position error |

**Spatial reasoning QA** (4 subtypes, 1 each per episode):

| Subtype | Question Pattern | Answer Source |
|---------|-----------------|---------------|
| `object_referencing` | "Where is the {brick_type} brick?" | spawn_poses, data.xpos projected to workspace description |
| `trajectory_prediction` | "Where will the gripper move next?" | Next waypoint position from EE state trajectory |
| `manipulation_planning` | "Which arm should pick up this brick?" | Arm selection logic (Y > 0 → left) + brick position |
| `multiview` | "From the {camera} view, describe the gripper position relative to the target." | EE pos vs target pos in camera frame |

**Physical common sense QA** (2–3 per episode, template-only — no VLM needed):

| Subtype | Question | Answer |
|---------|----------|--------|
| `grasp_physics` | "Why does the robot close the gripper gradually?" | "Gradual closing ensures stable contact with the brick without displacing it from the grasp position." |
| `press_fit` | "Why does the robot press downward after positioning the brick?" | "Downward force engages the brick's tubes onto the baseplate studs, creating a friction-based press-fit connection that holds the brick in place." |
| `stability` | "What would happen if the gripper opened during transport?" | "The brick would fall due to gravity, missing the target position and requiring a recovery attempt." |

Physical QA uses a fixed pool of ~15 question-answer pairs, selected by episode context (brick type, failure mode). These are domain-constant facts that don't need VLM generation.

**Generation pipeline:**
1. Extract sim ground truth context from HDF5 metadata and state trajectory
2. Fill QA templates with concrete values (positions, brick types, phases, outcomes)
3. For temporal and spatial QA: send batched template drafts to VLM for natural rephrasing
4. For physical QA: select from pre-written pool (no VLM call)
5. Validate all QA pairs: answers must reference correct facts from sim ground truth

### Execution checklist
- QA templates in `configs/annotation/qa_templates.yaml` — one section per subtype with question patterns and answer templates
- Physical QA pool in `configs/annotation/physical_qa_pool.yaml` — 15 pre-written Q&A pairs with selection criteria
- `generate_temporal_qa(seeds, hdf5_dir, model, config)` → `temporal_qa` records
- `generate_spatial_qa(seeds, hdf5_dir, model, config)` → `spatial_qa` records
- `generate_physical_qa(seeds, hdf5_dir, config)` → `physical_qa` records (no VLM)
- Factual consistency check: verify numerical claims in answers match sim ground truth (positions ±5 mm, correct brick types, correct outcome)
- Coverage: every episode has ≥ 10 QA pairs (4 temporal + 4 spatial + 2–3 physical)
- Variety: no two consecutive episodes have identical QA text for same subtype

### Milestone (minimum success criteria)
- All 10K episodes have ≥ 10 QA pairs across all three categories. ≥ 95% pass schema validation. Factual consistency check passes for ≥ 98% of verifiable claims.

---

## 2.2.4) Annotation Quality Validation

### What we will do
Run a comprehensive quality assurance pipeline on all ~300K annotations: automated format checks, factual consistency validation against sim ground truth, diversity analysis, and rule-based cleaning with LLM rewriting for flagged entries. This follows EO-Data1.5M's documented "rule-based cleaning and LLM-based rewriting for format uniformity" step.

### Why this matters
One corrupted or hallucinated annotation propagated through Phase 2.3 into training data silently degrades the policy. Automated QA against sim ground truth is a unique advantage of synthetic data — we have perfect labels to validate against. This step ensures the annotation dataset is correct, consistent, and useful before downstream consumption.

### Validation pipeline

**Level 1 — Format validation (automated, no VLM):**
- JSON Schema validation: every record matches the schema from 2.2.0
- Required fields present with correct types and valid enum values
- No empty `text` fields, no truncated JSON, no null values
- Episode coverage: every episode has exactly 1 `task_description`, ≥ 8 `step_narration`, ≥ 10 QA records, ≥ 3 `keyframe_caption`

**Level 2 — Factual consistency (automated, sim ground truth comparison):**
- Brick types mentioned in text match HDF5 `metadata/brick_types`
- Arm references ("left"/"right") match the arm selection for each placement
- Outcome descriptions ("success"/"failure") match `labels/episode_outcome`
- Spatial references: extract mentioned coordinates, verify within 2 cm of sim `xpos` values
- Phase ordering: step narrations reference phases in correct temporal sequence
- Failure episodes: annotations must not claim success; must mention failure mode
- Recovery episodes: `task_description.has_recovery` must match upstream `labels/has_recovery`; step narrations must contain both `is_recovery: false` and `is_recovery: true` segments; failure moment and recovery start must be referenced
- QA factual grounding: verify answers contain information derivable from sim state

**Level 3 — Diversity and quality metrics (automated):**
- Vocabulary diversity: unique unigram ratio across task descriptions (target ≥ 0.4)
- Sentence length: flag outliers (< 5 words or > 150 words per field)
- Cross-episode repetition: exact-match deduplication (flag if > 5% identical text for same annotation type)
- QA answer length: answers must be ≥ 10 words (reject trivially short)
- Question variety: questions for same subtype should not be identical across > 10% of episodes

**Level 4 — Cleaning and rewriting (rule-based + VLM):**
- Rule-based fixes: standardize brick names ("2×4 brick" not "two by four block"), normalize arm names ("left arm" not "L arm"), fix JSON formatting
- LLM rewriting: for annotations flagged in Level 3 (low quality, too short, high repetition), re-generate with modified prompt (higher temperature, explicit diversity instruction, different vocabulary hints)
- Re-validate rewritten annotations through Level 1 + Level 2

### Quality report output
```
data/annotations/v2.2.0/
├── episodes/                          # annotation JSONL files
├── schema.json                        # JSON Schema
├── quality_report.json                # per-level pass rates, per-category metrics
│   ├── level_1_format: {pass_rate, failures_by_type}
│   ├── level_2_factual: {pass_rate, failures_by_check}
│   ├── level_3_diversity: {vocab_diversity, repetition_rate, length_stats}
│   └── level_4_cleaning: {n_rewritten, n_rule_fixed}
├── annotation_stats.json              # type counts, length distributions, subtype coverage
├── manifest.json                      # episode_seed → annotation_file mapping
└── flagged_episodes.json              # episodes with unresolved issues (target: 0)
```

### Execution checklist
- `validate_annotations(annotation_dir, hdf5_dir)` runs Level 1–3 checks on all JSONL files
- `clean_annotations(annotation_dir)` applies rule-based normalization
- `rewrite_flagged(annotation_dir, model, config)` re-generates low-quality annotations via VLM
- Quality report generation: automated, produces `quality_report.json` + `annotation_stats.json`
- All annotations pass Level 1 (format) after cleaning
- ≥ 98% pass Level 2 (factual consistency)
- Level 3 diversity metrics meet thresholds (vocab diversity ≥ 0.4, repetition < 5%)
- Add validation commands to `CLAUDE.md`

### Milestone (minimum success criteria)
- 100% of annotations pass Level 1 format validation. ≥ 98% pass Level 2 factual consistency. Vocabulary diversity ≥ 0.4. Quality report generated with full metrics. Zero flagged episodes remaining.

---

# Downstream Contract with Phase 2.3

Phase 2.3 (Interleaved Data Construction) will consume Phase 2.2 annotations as follows:

| Phase 2.3 Workstream | Phase 2.2 Annotation Source | Assembly Method |
|----------------------|---------------------------|----------------|
| Interleaved sequence construction | `task_description` + `step_narration` + `keyframe_caption` | Interleave text tokens with image tokens and action tokens at matching timesteps using `step_range` alignment |
| Temporal reasoning data | `temporal_qa` (planning, sequencing, verification, outcome_prediction) | Pair questions with episode video clips; answers reference phase boundaries via `evidence_steps` |
| Spatial reasoning data | `spatial_qa` (object_referencing, trajectory_prediction, manipulation_planning, multiview) | Pair questions with multi-view keyframe images via `camera` and `keyframe_step` fields |
| Free chatting format | All QA types combined | Wrap Q&A pairs as conversational turns, interleave with images |

**Key design constraint**: Phase 2.2 annotations include explicit temporal anchors (`step_range`, `evidence_steps`, `keyframe_step`) and camera references (`camera`) so Phase 2.3 can mechanically align text with the correct images and actions — no heuristic matching required.

---

# Startup-Grade Outputs (deliverables by end of 2.2)
- **Annotation schema** — frozen JSONL schema with 6 annotation types, JSON Schema validation, documented downstream mapping
- **VLM annotation infrastructure** — self-hosted Qwen3.5-9B on A100, template + VLM hybrid pipeline, prompt templates in Hydra configs
- **~300K language annotations** — task descriptions, step narrations, temporal/spatial/physical QA, keyframe captions for all 10K episodes
- **4-level quality validation pipeline** — format, factual consistency (vs sim ground truth), diversity, cleaning + rewriting
- **Quality report** — per-level pass rates, coverage metrics, diversity analysis, zero unresolved flags

---

# Phase 2.2 Definition of Done
Phase 2.2 is complete when:
- The annotation schema is defined, frozen, and validated with JSON Schema.
- Qwen3.5-9B runs on a single A100 and produces all 6 annotation types via the template + VLM hybrid pipeline.
- All 10K episodes have task descriptions, step narrations, and ≥ 10 reasoning QA pairs each.
- 100% of annotations pass format validation; ≥ 98% pass factual consistency checks against sim ground truth.
- Vocabulary diversity ≥ 0.4; cross-episode repetition < 5%.
- Quality report exists with coverage, diversity, and consistency metrics.
- Annotations are stored as JSONL sidecars with manifest, ready for Phase 2.3 consumption.
- Downstream contract with Phase 2.3 is explicitly documented and verified.
