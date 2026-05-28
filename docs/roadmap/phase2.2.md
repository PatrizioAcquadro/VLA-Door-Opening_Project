# Phase 2.2 - Door-Opening Language and Annotation Pipeline

**Goal:** Generate grounded language for door-opening demonstrations without inventing facts that are not present in simulation metadata.

## Source of Truth

Simulation metadata is authoritative for:

- handle type
- latch state
- initial and final door angle
- target angle
- robot arm used
- contact/slip events
- failure or recovery reason
- phase boundaries

Language generation must condition on these fields rather than inferring physical facts from images alone.

## Annotation Types

### Task Description

One high-level instruction or summary per episode.

Example:

```json
{
  "type": "task_description",
  "episode_seed": 0,
  "text": "Open the lever-handle door to about 60 degrees using the right arm.",
  "handle_type": "lever",
  "target_door_angle_deg": 60,
  "latch_enabled": true,
  "outcome": "success"
}
```

### Step Narration

Short text aligned to controller phases:

- move_to_handle
- approach_handle
- establish_contact
- release_latch
- open_door
- stabilize
- release_and_retract
- recovery

### Physical QA

Questions and answers grounded in metadata:

- "Is the latch released?"
- "What is the current door angle?"
- "Which arm is contacting the handle?"
- "Why did the first pull fail?"
- "What should the robot do next?"

## Consistency Checks

- angles in text match metadata within tolerance
- handle type is correct
- latch state is correct
- recovery text only appears when recovery is labeled
- no references to unrelated object-assembly tasks

## Outputs

Recommended files:

- `annotations/task_descriptions.jsonl`
- `annotations/step_narrations.jsonl`
- `annotations/physical_qa.jsonl`
- `annotations/validation_report.json`

## Acceptance

- 100% of generated text is traceable to metadata fields.
- Random spot checks show no unsupported spatial or physical claims.
- The annotation schema can be joined with Phase 2.1 episodes by seed and step range.
