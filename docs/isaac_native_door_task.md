# Isaac Native Hallway Door Task

This phase extends the first Isaac backend slice without replacing MuJoCo or the
frozen VLA contracts. The live task composes the real Desktop HallwayScene:

```text
/home/pacquadr/Desktop/HallwayScene/Hallway.usdc
```

It also opens and inspects:

```text
/home/pacquadr/Desktop/HallwayScene/Objects/DoorObject_v2.usda
```

`DoorObject_v2.usda` contains a door, frame, collision meshes, and a
`PhysicsRevoluteJoint`. The task uses its doorway transform as the documented
pose source, then builds `/World/IsaacNativeDoorTask` from Isaac-native rigid
bodies so the proof has stable programmatic access to the fixed frame, dynamic
panel, revolute hinge, handle collision, handle-region contact readings, door
angle, latch proxy, and cameras.

Run the native proof on the lab PC:

```bash
python3 scripts/run_isaac_native_door_proof.py
```

Evidence is written under:

```text
logs/isaac_native_door/<timestamp>/
```

The proof records HallwayScene open status, selected prim paths, hinge axis and
limits, angle samples, target-angle error samples, latch transitions, handle
contact readings, live camera frame checks, and the recorded episode artifact.

Current live camera views are `overhead` and `third_person`. `left_wrist_cam`
and `right_wrist_cam` remain frozen VLA contract names, but this phase has no
live Alex articulation, so wrist views raise an explicit runtime-unavailable
error instead of returning placeholder frames.
