# Phase 2.1 - Door-Opening Demonstration Dataset

**Goal:** Generate physically valid Alex door-opening demonstrations that produce synchronized observations, robot state, continuous 17-D actions, and door task metrics.

## Starting Point

Use the active door scene and task config:

- `sim/assets/scenes/alex_door_workspace.xml`
- `configs/sim/door.yaml`
- `scripts/validate_door_scene.py`

The dataset must represent actions emitted through the same control path used for training and inference. Avoid demonstrations that teleport the object or bypass `SimRunner`.

## Demonstration Controller

Implement a scripted baseline with these phases:

1. reset scene and sample target angle
2. move to a pre-handle pose
3. approach handle
4. establish handle contact
5. release latch if enabled
6. pull or push through the hinge arc
7. stabilize at target angle
8. release and retract

Each phase should log phase IDs, timing, success/failure reason, and relevant scalar metrics.

## Dataset Schema

Recommended episode layout:

```text
episode/
  metadata/
    seed
    scene_variant
    handle_type
    latch_enabled
    initial_door_angle
    target_door_angle
    final_door_angle
    success
    failure_reason
  observations/
    overhead_rgb
    third_person_rgb
    handle_closeup_rgb
    depth_optional
    segmentation_optional
  state/
    robot_state_52d
    door_angle
    door_angular_velocity
    handle_contact
  actions/
    alex_action_17d
  phases/
    phase_id
    phase_name
    step_range
```

## Generation Waves

Wave 1:

- free-hinge lever door
- fixed handle type
- target angles from 45 to 70 degrees
- goal: prove stable logging and basic success metrics

Wave 2:

- latch proxy enabled
- varied initial door angles
- varied target angles
- recovery from handle slip

Wave 3:

- handle variants
- push and pull doors
- camera and lighting randomization
- broader failure/recovery coverage

## Acceptance

- Demonstrations are deterministic by seed.
- Every timestep contains image, robot state, action, and door metric records.
- No episode uses object teleportation after reset.
- Success is judged by door metrics, not only by controller completion.
- Failure cases are labeled with concrete reasons.

## Validation

```bash
python scripts/validate_door_scene.py
python scripts/validate_action_space.py
python scripts/validate_robot_state.py
python scripts/validate_cameras.py
```
