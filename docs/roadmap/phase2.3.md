# Phase 2.3 - Door-Opening Interleaved VLA Sequences

**Goal:** Convert door-opening demonstrations and annotations into interleaved vision-text-state-action sequences for VLA training.

## Sequence Unit

Each training example should bind:

- language instruction or narration
- one or more camera observations
- current 52-D Alex robot state
- door state scalars
- future 17-D action chunk
- token type labels for loss routing

## Example Schema

```json
{
  "type": "interleaved_vla",
  "episode_seed": 0,
  "phase_name": "open_door",
  "instruction": "Pull the lever-handle door open to about 60 degrees.",
  "observations": [
    {"camera": "overhead", "step": 120},
    {"camera": "handle_closeup", "step": 120}
  ],
  "state_ref": {"step": 120},
  "door_state": {
    "angle_rad": 0.42,
    "angular_velocity_rad_s": 0.15,
    "latch_released": true
  },
  "action_ref": {"step_range": [120, 136]},
  "target_door_angle_rad": 1.0472
}
```

## Action Chunking

Use the frozen Phase 3.2 action contract:

- chunk size: 16 steps
- action dim: 17
- policy rate: 20 Hz
- chunk duration: 0.8 s

The sequence builder should use `models.action_head.chunk_actions()` and preserve masks for short final chunks.

## Loss Routing

Use token types consistently:

- text tokens: autoregressive text loss
- image tokens: context only
- state tokens: context only
- action tokens: flow matching loss

## Dataset Splits

Split by episode seed, not by individual timestep, so adjacent action chunks from the same rollout do not leak across train and validation.

Recommended first split:

- train: 80%
- validation: 10%
- test: 10%

## Acceptance

- sequence construction is deterministic from episode data
- action chunks align exactly with recorded robot actions
- observations and state come from the same timestep
- labels use `-100` for ignored text positions
- door metrics remain available for evaluation-time grouping
