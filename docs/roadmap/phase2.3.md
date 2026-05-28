# Phase 2.3 — Interleaved Data Construction (5 days)

**Goal:** Transform the 10K trajectory episodes (Phase 2.1) and ~300K language annotations (Phase 2.2) into structured training sequences for EO-1-style VLA co-training — producing four distinct data subsets (interleaved VLA, temporal reasoning, spatial reasoning, free chatting) that a training dataloader can directly consume without additional VLM calls or manual curation.

**Fixed upstream decisions (from 2.1 + 2.2):**
- **Trajectory dataset**: 10K HDF5 episodes in `data/demonstrations/v2.1.1/episodes/`, with train/val/test manifests (80/10/10) and `dataset_stats.json`
- **Per-episode content**: RGB+Depth+Segmentation (4 views, 320×320), 52-D state, 17-D action, 20 Hz, ~200 steps, ~55 MB per file
- **Labels**: 8 waypoint phases (0–7), `grasp_state`, `episode_outcome`, `perturbation_type`, `has_recovery`, `recovery_start_step`
- **Composition**: 7K success + 2K failure + 1K recovery episodes
- **Annotations**: ~300K JSONL records in `data/annotations/v2.2.0/episodes/`, 6 types: `task_description`, `step_narration`, `temporal_qa`, `spatial_qa`, `physical_qa`, `keyframe_caption`
- **Temporal anchors**: `step_narration.step_range`, `temporal_qa.evidence_steps`, `spatial_qa.keyframe_step`, `keyframe_caption.step` — Phase 2.3 uses these to mechanically align text with images and actions
- **Camera anchors**: `spatial_qa.camera`, `keyframe_caption.camera` — Phase 2.3 uses these to select specific views
- **Schema**: validated via `data/annotations/v2.2.0/schema.json`, quality-assured by Phase 2.2.4

**Key Phase 2.3 stance:**
- **Reference-based sequences**: Each training sequence is a JSONL record containing inline text content plus pointers into existing HDF5 episode data (episode seed, step indices, camera names). This avoids duplicating 550 GB of images, decouples data construction from model-specific tokenization, and lets the training dataloader resolve references at load time.
- **Phase-aligned image subsampling**: For interleaved VLA sequences, one observation (2 views: overhead + active wrist camera) per waypoint phase boundary. This produces 16 images per single-placement episode — manageable for an 8K context window while preserving temporal coverage of the full manipulation.
- **Continuous action references**: Sequences specify which action timestep ranges belong to each segment. Discretization into action tokens (for autoregressive loss) and continuous regression (for flow matching loss) are training-time concerns handled in Phase 3.
- **Deterministic construction**: All sequence assembly is deterministic from episode seed. No randomness is introduced in Phase 2.3 — the same upstream data always produces the same training sequences.

**Critical gap this phase closes:**
Phase 2.1 produces trajectories (states, actions, images) and Phase 2.2 produces language annotations (descriptions, narrations, QA pairs) — but these exist as separate, unstructured artifacts. VLA training in the EO-1 framework requires *interleaved* sequences where vision tokens, language tokens, and action tokens share a single context window with explicit temporal alignment. Without Phase 2.3, the training pipeline has no way to combine modalities into the format the model actually consumes.

---

## 2.3.0) Sequence Schema & Conventions

### What we will do
Define the output format, image selection strategy, and data layout for all four training data subsets. This schema is the contract between Phase 2.3 (data construction) and Phase 3 (training dataloader).

### Why this matters
The training dataloader must know exactly how to interpret each JSONL record — which fields contain text, which reference HDF5 data, and how to assemble them into model input. Defining this schema upfront prevents costly rework when implementing the four subset builders (2.3.1–2.3.4) and ensures the training pipeline can be developed in parallel.

### Design

**Output format: reference-based JSONL**

Each training sequence is one JSON line containing:
- **Inline text**: actual text strings (task descriptions, narrations, QA text) — ready for tokenization
- **Image references**: `{"episode_seed": int, "step": int, "camera": str}` — resolved by the dataloader to load from HDF5
- **Action/state references**: `{"episode_seed": int, "step_range": [int, int]}` — resolved to load continuous vectors from HDF5
- **Metadata**: sequence type, episode seed, split, outcome label

The dataloader's job (Phase 3) is: (1) load referenced images from HDF5, (2) tokenize text with the model's tokenizer, (3) load action/state arrays from HDF5, (4) apply any action discretization, (5) assemble the interleaved token sequence.

**Image reference convention:**
- `step` is a 0-indexed timestep into the episode's image arrays
- `camera` is one of the 4 frozen view names: `overhead`, `left_wrist_cam`, `right_wrist_cam`, `third_person`
- For VLA sequences, each observation includes 2 views: `overhead` (workspace context) + the active wrist camera (`left_wrist_cam` if `arm=="left"`, `right_wrist_cam` if `arm=="right"`)
- For QA sequences, the camera is specified by the annotation's `camera` field

**Action/state reference convention:**
- `step_range` is `[start, end)` (exclusive end), matching Python slice semantics
- References resolve to `action/normalized[start:end]` and `state/normalized[start:end]` in HDF5
- The dataloader reads the continuous 17-D actions and 52-D states; any discretization happens at training time

**Train/val/test split rule:**
- Sequences inherit the split of their source episode (from Phase 2.1 manifests)
- Episode seed in train manifest → all derived sequences go in `train.jsonl`
- No cross-split contamination: a sequence never references data from a different split's episode

**Data layout:**
```
data/training_sequences/v2.3.0/
├── interleaved/              # Interleaved VLA sequences
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── temporal/                 # Temporal reasoning QA
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── spatial/                  # Spatial reasoning QA
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── free_chat/                # Conversational format
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── schema.json               # JSON Schema for all 4 sequence types
├── manifest.json             # Per-subset sample counts, split sizes, version, upstream refs
└── stats.json                # Per-subset statistics (lengths, token estimates, image counts)
```

**Module layout:**
```
data/sequences/
├── __init__.py               # Exports build functions and dataclasses
├── schema.py                 # Frozen dataclasses for all 4 sequence types
├── interleaved.py            # Interleaved VLA builder
├── temporal.py               # Temporal reasoning builder
├── spatial.py                # Spatial reasoning builder
├── free_chat.py              # Free chatting builder
├── validation.py             # Cross-reference validation
└── stats.py                  # Dataset statistics computation
```

**Config:** `configs/data/sequences.yaml`
```yaml
version: "2.3.0"
output_dir: ${paths.data}/training_sequences/v2.3.0
hdf5_dir: ${paths.data}/demonstrations/v2.1.1/episodes
annotation_dir: ${paths.data}/annotations/v2.2.0/episodes

interleaved:
  views_per_observation: ["overhead", "active_wrist"]  # "active_wrist" resolved by arm
  include_keyframe_captions: true
  include_outcome_text: true

temporal:
  views_per_evidence: ["overhead"]          # View for evidence_steps images
  max_evidence_images: 5

spatial:
  include_secondary_view: false             # Only use the annotated camera

free_chat:
  conversations_per_episode: 2
  turns_per_conversation: [3, 5]            # min, max QA turns
  scene_views: ["overhead"]                 # Context images
  scene_steps: [0, -1]                      # First and last frame
```

### Execution checklist
- Define frozen dataclasses in `data/sequences/schema.py`: `ImageRef`, `ActionRef`, `StateRef`, `VLASegment`, `InterleavedVLASequence`, `TemporalQASequence`, `SpatialQASequence`, `FreeChatSequence`
- Write JSON Schema file (`data/training_sequences/v2.3.0/schema.json`) covering all 4 types
- Write one synthetic example per type by hand, validate against schema
- Create `configs/data/sequences.yaml` with all configurable parameters
- Document the dataloader contract: what the training pipeline must do to consume each sequence type

### Milestone (minimum success criteria)
- Schema is defined, documented, and validated with JSON Schema. One hand-written example per sequence type passes validation. Config file exists with all parameters. Dataloader contract is documented.

---

## 2.3.1) Interleaved VLA Sequence Construction

### What we will do
Construct the core VLA training data: interleaved vision-text-action sequences where each episode is represented as a temporally ordered sequence of observation images, step narrations, and action references — all aligned by the `step_range` anchors from Phase 2.2 annotations.

### Why this matters
This is the primary data that teaches the VLA model to execute manipulation tasks. EO-1's key design is co-training on interleaved sequences where language tokens and action tokens share the same context window. The model learns to read a task instruction, observe the scene, generate (or consume) a natural language plan for the current phase, and then produce the corresponding actions. Without this interleaved structure, the model cannot learn the vision-language-action reasoning loop.

### Design

**Sequence structure per episode:**

Each interleaved VLA sequence follows this pattern:
```
[task_description] → [obs₀][narration₀][actions₀] → [obs₁][narration₁][actions₁] → ... → [obs₇][narration₇][actions₇] → [outcome_text]
```

Where:
- `task_description`: the full episode task description text (from `task_description` annotation)
- `obs_k`: image references at the first step of phase `k` (from `step_narration.step_range[0]`)
- `narration_k`: the step narration text for phase `k` (from `step_narration.text`)
- `actions_k`: action reference for the step range of phase `k` (from `step_narration.step_range`)
- `outcome_text`: optional short outcome string ("Assembly successful." / "Insertion failed: misalignment.")

The model processes this left-to-right: see instruction → see scene → read plan → execute actions → see result → read plan → execute actions → ... → see outcome.

**Concrete JSONL format:**

```jsonl
{"type": "interleaved_vla", "episode_seed": 0, "level": 1, "outcome": "success", "task_description": "Pick up the red 2x4 brick and place it on the baseplate at row 3, column 2.", "segments": [{"phase": 0, "phase_name": "pre_grasp", "placement_index": 0, "is_recovery": false, "observation": [{"episode_seed": 0, "step": 0, "camera": "overhead"}, {"episode_seed": 0, "step": 0, "camera": "left_wrist_cam"}], "narration": "The left arm moves to hover above the 2x4 brick.", "action_ref": {"episode_seed": 0, "step_range": [0, 25]}, "state_ref": {"episode_seed": 0, "step_range": [0, 25]}}, {"phase": 1, "phase_name": "approach", "placement_index": 0, "is_recovery": false, "observation": [{"episode_seed": 0, "step": 25, "camera": "overhead"}, {"episode_seed": 0, "step": 25, "camera": "left_wrist_cam"}], "narration": "The arm descends toward the brick for grasping.", "action_ref": {"episode_seed": 0, "step_range": [25, 50]}, "state_ref": {"episode_seed": 0, "step_range": [25, 50]}}], "keyframe_captions": [{"step": 0, "camera": "overhead", "caption": "The robot's left arm is positioned above the workspace..."}], "outcome_text": "Assembly successful."}
```

**View selection per segment:**
- **Overhead** (always): workspace-level context for spatial reasoning
- **Active wrist camera**: `left_wrist_cam` if `step_narration.arm == "left"`, `right_wrist_cam` if `"right"` — close-up of manipulation contact
- Total: 2 images per segment, 16 images per single-placement episode (8 phases × 2 views)

**Multi-placement episodes:**
- One sequence per episode (not per placement) — preserves multi-step planning context
- Level 2–3 episodes have 8 × N segments (N placements), with `placement_index` on each segment
- Task description covers the full assembly goal; narrations cover each placement's phases

**Recovery episodes:**
- Single sequence containing both the failed attempt and the recovery retry
- Segments from the failed attempt have `is_recovery: false`; retry segments have `is_recovery: true`
- The narration text (from Phase 2.2) already reflects failure/recovery language
- A `failure_step` field on the sequence marks where the initial attempt failed

**Token budget estimate (single-placement success episode):**
- Task description: ~40 tokens
- 8 narrations × ~25 tokens: ~200 tokens
- 16 images × ~256 vision tokens: ~4,096 tokens (Qwen VL with 14×14 patches)
- Outcome text: ~5 tokens
- Total text + vision: ~4,341 tokens — fits in 8K context with room for action tokens
- Action tokens depend on Phase 3 discretization (~200 steps × 17D)

**Estimated counts:**
- 10K episodes → 10K interleaved VLA sequences
- Train: ~8K, Val: ~1K, Test: ~1K (following Phase 2.1 splits)

### Construction algorithm
```
For each episode in manifest:
  1. Load annotation JSONL for this episode
  2. Extract the task_description record
  3. Collect all step_narration records, sorted by (placement_index, phase)
  4. For each step_narration:
     - Create segment with observation refs at step_range[0]
     - Set cameras = [overhead, active_wrist_cam(arm)]
     - Set narration = step_narration.text
     - Set action_ref and state_ref from step_range
  5. Collect keyframe_caption records
  6. Determine outcome_text from task_description.outcome
  7. Emit one InterleavedVLASequence record
```

### Execution checklist
- `build_interleaved_vla(hdf5_dir, annotation_dir, output_dir, config)` processes all episodes
- Determinism: same inputs → identical JSONL output (sorted by seed)
- Coverage: every episode has exactly 1 interleaved VLA sequence
- Cross-reference validation: every `step` in observation refs exists in the HDF5 file's image array range; every `step_range` is within bounds
- Phase ordering: segments are in correct temporal order (monotonically increasing step ranges)
- Multi-placement: Level 2–3 episodes produce sequences with > 8 segments
- Recovery: recovery sequences contain both `is_recovery: false` and `is_recovery: true` segments
- Failure: failure episodes have `outcome != "success"` and `outcome_text` describes the failure
- Standalone test: build sequences for 10 episodes, inspect JSONL output

### Milestone (minimum success criteria)
- All 10K episodes produce valid interleaved VLA sequences. Every sequence passes JSON Schema validation. Cross-reference checks confirm all image/action/state references resolve to valid HDF5 data. Phase ordering is correct for 100% of sequences.

---

## 2.3.2) Temporal Reasoning Subset

### What we will do
Transform all `temporal_qa` annotations into training sequences for temporal reasoning — pairing each question-answer pair with episode context images at the annotated `evidence_steps`, producing a visual QA format that teaches the model to plan, sequence, verify, and predict outcomes of manipulation tasks.

### Why this matters
Temporal reasoning is what separates a reactive action-cloner from a planning agent. These sequences teach the model to reason about task ordering ("what step comes next?"), verify execution state ("has the robot successfully grasped the brick?"), and predict outcomes ("will this approach succeed?"). EO-Data1.5M dedicates a significant fraction of training data to temporal QA for exactly this reason.

### Design

**Sequence structure per QA pair:**
```
[evidence_image₀] [evidence_image₁] ... [evidence_imageₙ] [question] [answer]
```

The model sees a temporal progression of images from the episode (at the annotated `evidence_steps`), reads the question, and must produce the answer. This is a standard visual QA format.

**Concrete JSONL format:**
```jsonl
{"type": "temporal_qa", "episode_seed": 0, "subtype": "planning", "image_refs": [{"episode_seed": 0, "step": 0, "camera": "overhead"}, {"episode_seed": 0, "step": 50, "camera": "overhead"}, {"episode_seed": 0, "step": 100, "camera": "overhead"}], "question": "What steps are needed to place the 2x4 brick on the baseplate at row 3?", "answer": "The robot needs to: 1) Move the left arm above the brick, 2) Lower the gripper and grasp the brick, 3) Lift and transport the brick to the target position, 4) Lower the brick onto the baseplate studs and press down, 5) Release the gripper and retract."}
```

**Image selection:**
- Use the `evidence_steps` field from the `temporal_qa` annotation (set by Phase 2.2)
- All evidence images use the `overhead` camera (best spatial context for temporal progression)
- Cap at `max_evidence_images` (default 5) to limit sequence length

**Subtypes (all 4 per episode):**
| Subtype | Image Context | Reasoning Skill |
|---------|--------------|-----------------|
| `planning` | Start + mid + end frames | Multi-step task decomposition |
| `sequencing` | Current phase + next phase frames | Temporal ordering |
| `verification` | Pre-action + post-action frames | Success/failure assessment |
| `outcome_prediction` | Current state frames | Predictive reasoning |

**Token budget estimate:**
- 3–5 images × ~256 tokens: ~768–1,280 vision tokens
- Question: ~20–40 tokens
- Answer: ~30–80 tokens
- Total: ~820–1,400 tokens per sequence — very compact

**Estimated counts:**
- 4 temporal QA per episode × 10K episodes = 40K temporal reasoning sequences
- Train: ~32K, Val: ~4K, Test: ~4K

### Execution checklist
- `build_temporal_qa(annotation_dir, output_dir, config)` processes all temporal_qa annotations
- Each `temporal_qa` annotation produces exactly 1 temporal reasoning sequence
- Image refs derived from `evidence_steps` with `overhead` camera
- Cross-reference: all `evidence_steps` values within episode length bounds
- Coverage: 4 subtypes × 10K episodes = 40K sequences
- Standalone test: build sequences for 10 episodes, verify image refs resolve

### Milestone (minimum success criteria)
- 40K temporal reasoning sequences produced, all passing schema validation. Cross-reference checks confirm all image references are within bounds. All 4 subtypes are represented.

---

## 2.3.3) Spatial Reasoning Subset

### What we will do
Transform all `spatial_qa` annotations into training sequences for spatial reasoning — pairing each question-answer pair with the specific camera view and keyframe specified in the annotation, producing visual QA that teaches the model to locate objects, predict trajectories, plan grasps, and reason across viewpoints.

### Why this matters
Spatial reasoning grounds the model's language understanding in the physical scene. Without it, the model cannot answer "where is the brick?" or "which arm should pick this up?" — capabilities essential for real-world deployment where task instructions reference spatial concepts. EO-Data1.5M's spatial QA subset specifically targets this capability gap.

### Design

**Sequence structure per QA pair:**
```
[keyframe_image] [question] [answer]
```

Each spatial QA pair is grounded in a specific camera view at a specific timestep, as annotated by Phase 2.2. The model sees one (or optionally two) images and must reason about spatial relationships.

**Concrete JSONL format:**
```jsonl
{"type": "spatial_qa", "episode_seed": 0, "subtype": "object_referencing", "image_refs": [{"episode_seed": 0, "step": 50, "camera": "overhead"}], "question": "Where is the 2x4 brick relative to the baseplate?", "answer": "The 2x4 brick is positioned approximately 10 cm to the left of the baseplate center, resting on the table surface with its long axis parallel to the baseplate edge."}
```

**Image selection:**
- Primary: the camera and step specified by `spatial_qa.camera` and `spatial_qa.keyframe_step`
- Optional secondary view (configurable, default off): add the `overhead` view if the primary is a wrist camera, for complementary spatial context

**Subtypes (all 4 per episode):**
| Subtype | Primary View | Reasoning Skill |
|---------|-------------|-----------------|
| `object_referencing` | `overhead` | Object localization and relational description |
| `trajectory_prediction` | Active wrist cam | End-effector path prediction |
| `manipulation_planning` | `overhead` | Arm selection and approach strategy |
| `multiview` | Varies (annotated) | Cross-view spatial correspondence |

**Token budget estimate:**
- 1–2 images × ~256 tokens: ~256–512 vision tokens
- Question: ~15–30 tokens
- Answer: ~30–60 tokens
- Total: ~300–600 tokens per sequence — very compact

**Estimated counts:**
- 4 spatial QA per episode × 10K episodes = 40K spatial reasoning sequences
- Train: ~32K, Val: ~4K, Test: ~4K

### Execution checklist
- `build_spatial_qa(annotation_dir, output_dir, config)` processes all spatial_qa annotations
- Each `spatial_qa` annotation produces exactly 1 spatial reasoning sequence
- Primary image ref from `camera` + `keyframe_step`; optional secondary per config
- Cross-reference: `keyframe_step` within episode length, `camera` is one of the 4 frozen view names
- Coverage: 4 subtypes × 10K episodes = 40K sequences
- Standalone test: build sequences for 10 episodes, verify camera/step refs

### Milestone (minimum success criteria)
- 40K spatial reasoning sequences produced, all passing schema validation. Cross-reference checks confirm all image references are valid. All 4 subtypes are represented.

---

## 2.3.4) Free Chatting Subset

### What we will do
Compose all QA annotations (temporal, spatial, physical) from each episode into multi-turn conversations with scene images, producing a conversational VLM training format that maintains the model's ability to discuss robot scenes and manipulation in free-form dialogue.

### Why this matters
VLA co-training risks degrading the base VLM's conversational ability — a phenomenon called "catastrophic forgetting." The free chatting subset counteracts this by providing conversation-format training data within the robot manipulation domain. This teaches the model to answer open-ended questions about what it sees and does, maintaining the general reasoning capabilities that make VLMs useful beyond pure action prediction.

### Design

**Sequence structure per conversation:**
```
[scene_image₀] [scene_image₁] [user_turn₀] [assistant_turn₀] [user_turn₁] [assistant_turn₁] ... [user_turnₙ] [assistant_turnₙ]
```

Images are shown first (scene context), then the conversation unfolds as alternating user questions and assistant answers.

**Concrete JSONL format:**
```jsonl
{"type": "free_chat", "episode_seed": 0, "image_refs": [{"episode_seed": 0, "step": 0, "camera": "overhead"}, {"episode_seed": 0, "step": 100, "camera": "overhead"}], "turns": [{"role": "user", "text": "What is the robot doing in this scene?"}, {"role": "assistant", "text": "The robot is using its left arm to pick up a red 2x4 LEGO brick from the table surface. The brick is positioned to the left of the baseplate."}, {"role": "user", "text": "What would happen if the gripper opened during transport?"}, {"role": "assistant", "text": "The brick would fall due to gravity, missing the target position and requiring a recovery attempt."}, {"role": "user", "text": "Where should the brick be placed?"}, {"role": "assistant", "text": "The brick should be placed on the baseplate at row 3, column 2, aligned with the stud grid for a secure press-fit connection."}]}
```

**Conversation composition strategy:**
- Each episode produces 2–3 conversations (configurable)
- Each conversation has 3–5 QA turns (configurable)
- Turns are selected from the episode's QA annotations with type mixing:
  - Conversation 1: `task_description` paraphrase + `spatial_qa` + `physical_qa` (scene understanding focus)
  - Conversation 2: `temporal_qa` + `spatial_qa` (planning + reasoning focus)
  - Conversation 3 (if sufficient QA): remaining `physical_qa` + `temporal_qa` (physics + verification focus)
- QA pairs are reworded into conversational style:
  - Question text used as `user` turn (natural questions, already varied by Phase 2.2)
  - Answer text used as `assistant` turn
- `physical_qa` pairs (template-only, no VLM) provide stable factual anchors in conversations

**Scene image selection:**
- 2 context images per conversation: first frame and a mid-episode frame (from overhead camera)
- Step indices: `[0, episode_length // 2]` (deterministic from episode length)

**Turn ordering within a conversation:**
- Start with a broad question (task description or object referencing) to establish context
- Follow with more specific questions (trajectory, physics, verification)
- Ordering is deterministic: sorted by annotation `subtype` priority within each conversation template

**Handling the task_description annotation as a conversational turn:**
- The episode's `task_description.text` is adapted into a Q&A turn:
  - User: "What is the robot doing in this scene?" (or a variant selected by `seed % 5`)
  - Assistant: the `task_description.text` content
- This provides the opening context for the first conversation

**Token budget estimate:**
- 2 images × ~256 tokens: ~512 vision tokens
- 3–5 turns × ~50 tokens per turn: ~150–250 text tokens
- Total: ~660–760 tokens per conversation — very compact

**Estimated counts:**
- 2–3 conversations per episode × 10K episodes = 20K–30K free chat sequences
- Train: ~16K–24K, Val: ~2K–3K, Test: ~2K–3K

### Execution checklist
- `build_free_chat(annotation_dir, output_dir, config)` processes all episodes
- Deterministic conversation composition: same episode seed → identical conversations
- Turn variety: no two conversations from the same episode have identical turn sequences
- QA coverage: every non-duplicate QA annotation is used in at least one conversation
- Role alternation: strict `user` / `assistant` alternation, starting with `user`
- Scene images: valid step indices within episode length
- Standalone test: build conversations for 10 episodes, inspect dialogue quality

### Milestone (minimum success criteria)
- 20K–30K free chat sequences produced, all passing schema validation. Every conversation has 3–5 turns with strict role alternation. Scene image references are valid. QA annotations are consumed without waste.

---

## 2.3.5) Dataset Assembly & Validation

### What we will do
Combine all four subsets into a unified training dataset with cross-reference validation, compute comprehensive statistics, produce final manifests, and generate a summary report confirming the dataset is ready for Phase 3 training.

### Why this matters
This is the quality gate before training begins. A single malformed reference, missing image, or split contamination propagated into training will waste GPU-hours and silently degrade policy quality. Comprehensive validation against the upstream HDF5 and annotation data ensures every training sequence can actually be loaded.

### Validation pipeline

**Level 1 — Schema validation (all sequences):**
- Every JSONL record matches the JSON Schema from 2.3.0
- Required fields present with correct types
- `type` field matches the subset directory (e.g., `interleaved/` → `type == "interleaved_vla"`)
- No empty text fields, no null values, no truncated JSON

**Level 2 — Cross-reference validation (all sequences):**
- Every `episode_seed` in image/action/state refs exists in Phase 2.1 HDF5 manifests
- Every `step` index is within `[0, n_timesteps)` for that episode
- Every `step_range` satisfies `0 <= start < end <= n_timesteps`
- Every `camera` name is one of the 4 frozen views
- Every annotation-derived text field can be traced to a Phase 2.2 annotation record (spot-check 1% of sequences)

**Level 3 — Split consistency:**
- No episode seed appears in more than one split
- Interleaved, temporal, spatial, and free_chat splits are mutually consistent (episode X is in train for ALL subsets or NONE)
- Total episode coverage: every episode in Phase 2.1 manifests is represented in all 4 subsets

**Level 4 — Statistical sanity:**
- Per-subset sample counts match expected ranges (within ±5%):
  - Interleaved: ~10K
  - Temporal: ~40K
  - Spatial: ~40K
  - Free chat: ~20K–30K
- Segment count distribution for interleaved: 8 segments for single-placement, 16+ for multi-placement
- Image reference distribution: overhead appears in ~100% of sequences; wrist cameras appear in interleaved sequences
- No degenerate sequences: every interleaved sequence has ≥ 8 segments, every QA has non-empty question AND answer, every conversation has ≥ 3 turns

### Dataset statistics (`stats.json`)
```json
{
  "version": "2.3.0",
  "total_sequences": 110000,
  "subsets": {
    "interleaved_vla": {
      "train": 8000, "val": 1000, "test": 1000,
      "avg_segments": 8.5, "avg_images": 17.0, "avg_text_tokens_est": 240
    },
    "temporal_qa": {
      "train": 32000, "val": 4000, "test": 4000,
      "avg_images": 3.2, "avg_text_tokens_est": 85
    },
    "spatial_qa": {
      "train": 32000, "val": 4000, "test": 4000,
      "avg_images": 1.1, "avg_text_tokens_est": 70
    },
    "free_chat": {
      "train": 20000, "val": 2500, "test": 2500,
      "avg_turns": 3.8, "avg_images": 2.0, "avg_text_tokens_est": 210
    }
  },
  "unique_episodes": 10000,
  "image_refs_total": 200000,
  "upstream_hdf5_version": "2.1.1",
  "upstream_annotation_version": "2.2.0"
}
```

### Summary report
- Per-level validation pass rates (target: 100% for Levels 1–3)
- Statistical distributions (segment counts, token estimates, image counts)
- Comparison to expected counts
- Any anomalies or warnings

### Execution checklist
- `validate_sequences(sequence_dir, hdf5_dir, annotation_dir)` runs Level 1–4 checks
- `compute_stats(sequence_dir)` produces `stats.json`
- `generate_manifest(sequence_dir)` produces `manifest.json` with per-subset paths and counts
- Level 1: 100% pass rate
- Level 2: 100% pass rate (every reference resolves)
- Level 3: 100% split consistency
- Level 4: all counts within expected ranges, no degenerate sequences
- Add build and validation commands to `CLAUDE.md`
- Standalone validation script: `scripts/validate_training_sequences.py`
- Build script: `scripts/build_training_sequences.py`

### Milestone (minimum success criteria)
- 100% of sequences pass schema validation (Level 1). 100% of cross-references resolve to valid HDF5 data (Level 2). Splits are consistent across all 4 subsets (Level 3). Statistical counts are within expected ranges (Level 4). `stats.json` and `manifest.json` exist. Validation script exits cleanly.

---

# Startup-Grade Outputs (deliverables by end of 2.3)
- **Sequence schema** — frozen JSON Schema for 4 sequence types, with dataclass definitions and dataloader contract
- **~10K interleaved VLA sequences** — core training data with phase-aligned vision-text-action interleaving, covering success + failure + recovery episodes
- **~40K temporal reasoning sequences** — temporal QA grounded in episode evidence frames, covering planning, sequencing, verification, and outcome prediction
- **~40K spatial reasoning sequences** — spatial QA grounded in specific camera views, covering object referencing, trajectory prediction, manipulation planning, and multi-view reasoning
- **~20K–30K free chatting sequences** — multi-turn conversational format mixing all QA types, preserving VLM conversation capability
- **Cross-reference validated dataset** — every image/action/state reference verified against upstream HDF5 data
- **Dataset statistics and manifest** — sample counts, token estimates, image distributions, split consistency report

---

# Phase 2.3 Definition of Done
Phase 2.3 is complete when:
- The sequence schema is defined, frozen, and validated with JSON Schema for all 4 types.
- All 10K episodes produce interleaved VLA sequences with phase-aligned observation-narration-action segments.
- 40K temporal reasoning and 40K spatial reasoning sequences are constructed from Phase 2.2 QA annotations.
- 20K–30K free chatting sequences are composed from mixed QA types in conversational format.
- 100% of sequences pass schema validation and cross-reference checks against upstream HDF5 data.
- Train/val/test splits are consistent across all subsets and aligned with Phase 2.1 splits.
- `manifest.json` and `stats.json` exist with complete metadata.
- Build and validation scripts are added to `CLAUDE.md`.
- The training dataloader (Phase 3) can consume any sequence type by resolving references to load images, text, and actions.
