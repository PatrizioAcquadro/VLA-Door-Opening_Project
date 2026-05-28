# Viewer Debug Checklist

Quick reference for visually inspecting MuJoCo scenes with `vla-viewer`.

## Launch Commands

```bash
vla-viewer sim/assets/scenes/test_scene.xml
vla-viewer sim/assets/scenes/test_scene.xml --show-contacts --show-joints
vla-viewer sim/assets/scenes/alex_door_workspace.xml --show-contacts --show-joints
vla-viewer sim/assets/scenes/alex_door_workspace.xml --camera handle_closeup
```

## Automated Preflight Checks

The viewer checks basic scene health before opening:

| Check | What it verifies |
|-------|------------------|
| `gravity` | Gravity is nonzero and points down |
| `ground_plane` | At least one plane geom exists |
| `geom_summary` | Collision and visual geoms are present |

## Test Scene Walkthrough

Use `sim/assets/scenes/test_scene.xml` to verify the viewer itself.

1. Launch `vla-viewer sim/assets/scenes/test_scene.xml`.
2. Press Space to unpause.
3. Confirm the box falls downward, contacts the ground, and settles.
4. Enable contact points and contact forces.
5. Double-click the box and use Ctrl + right-drag to perturb it.
6. Confirm reset and reload work.

## Door Scene Walkthrough

Use `sim/assets/scenes/alex_door_workspace.xml` for the active task.

### Door Geometry

- [ ] Hinged door panel is visible near Alex.
- [ ] Door frame and hinge side are visually clear.
- [ ] Lever handle is visible and contact-enabled.
- [ ] Handle is reachable from the Alex upper-body workspace.
- [ ] Door starts near the closed angle.

### Joint Debugging

Launch:

```bash
vla-viewer sim/assets/scenes/alex_door_workspace.xml --show-joints
```

Check:

- [ ] `door_hinge` joint axis is vertical.
- [ ] Joint limit allows at least 1 radian of opening.
- [ ] Joint visualization is located on the hinge edge, not at the panel center.
- [ ] No unexpected freejoint exists on the door panel.

### Contact Debugging

Launch:

```bash
vla-viewer sim/assets/scenes/alex_door_workspace.xml --show-contacts
```

Check:

- [ ] Contact points appear on the handle when touched.
- [ ] Contact force arrows are stable and not explosive.
- [ ] Door frame is not initially penetrating the panel.
- [ ] Alex fingers and handle geoms are both collision-enabled.

### Camera Debugging

Check named cameras:

```bash
vla-viewer sim/assets/scenes/alex_door_workspace.xml --camera overhead
vla-viewer sim/assets/scenes/alex_door_workspace.xml --camera third_person
vla-viewer sim/assets/scenes/alex_door_workspace.xml --camera handle_closeup
```

Acceptance:

- [ ] `overhead` shows robot and door workspace.
- [ ] `third_person` shows the approach direction and door swing.
- [ ] `handle_closeup` clearly frames the handle.
- [ ] No camera clips through the robot, door, or floor.

## Robot Scene Checklist

- [ ] Robot settles without falling through the floor.
- [ ] Joint axes align with expected arm and wrist motion.
- [ ] Collision geoms roughly match visual meshes.
- [ ] Hands have collision geoms near the contact surfaces.
- [ ] No spurious contacts occur at rest.
- [ ] Timestep and solver stats stay stable in the Info panel.

## Mouse Reference

| Action | Effect |
|--------|--------|
| Right-click + drag | Rotate view |
| Middle-click + drag | Pan view |
| Scroll wheel | Zoom |
| Double-click body | Select body |
| Ctrl + right-click + drag | Apply force |
| Ctrl + left-click + drag | Apply torque |

## Validation Before Viewer Debugging

Run the automated checks first:

```bash
python scripts/validate_door_scene.py
python scripts/validate_assets.py
python scripts/validate_mujoco.py
```
