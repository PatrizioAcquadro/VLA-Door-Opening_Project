# Alex V1 Upper-Body Model — Provenance

## Source Repository
- **URL**: https://github.com/ihmcrobotics/ihmc-alex-sdk
- **Pinned commit**: `be25a395e35238bc6385a58bcc50aa047d936a25`
- **Date of extraction**: 2026-02-27

## Source File
`alex-models/alex_V1_description/mjcf/alex_v1_full_body_mjx.xml`

## Meshes Copied (from `alex-models/alex_V1_description/meshes/`)
- `torso.obj`, `ALX02_01_A02_HeadLink.obj`, `legs/Pelvis.obj`
- `cycloidal_arm/`: Left/Right ShoulderPitch, ShoulderRoll_longBicep, ShoulderYaw, ELBOW_PITCH_413mm, WristYaw, WristRoll, nub.obj

## EZGripper STL Meshes (Phase 1.1.3)
- **Source**: SAKE Robotics EZGripper Gen2 ROS package (`sake_ezgripper` / `ezgripper_driver`)
- **Files**: `SAKE_Palm_Dual_Gen2.stl`, `SAKE_Finger_L1_Gen2.stl`, `SAKE_Finger_L2_Gen2.stl`
- **Adapter URDF reference**: `ihmc-alex-sdk` commit `be25a395` files `alex_v1.leftEZGripperAdapter.urdf` / `alex_v1.rightEZGripperAdapter.urdf`
- **Adapter transform**: `euler="3.14159 1.5708 0"` (from URDF); verified correct via kinematics validation (FK symmetry 0.0000 cm error, GO verdict)

## Modifications Applied
1. **Stripped lower body**: Removed 12 leg joints and all leg bodies/meshes/actuators/sensors
2. **Fixed base**: Removed freejoint from pelvis (static base at z=1.0m)
3. **Fixed neck**: Removed NECK_Z and NECK_Y joints (head is static)
4. **Added GRIPPER_Z joints**: LEFT_GRIPPER_Z and RIGHT_GRIPPER_Z (exist in URDF, absent from source MJCF)
5. **Added collision geoms**: Capsule/box/sphere primitives for all upper-body links (group 3)
6. **Added EE sites**: left_ee_site, right_ee_site at gripper tips (group 4)
7. **Added robot camera**: robot_cam on head body
8. **Added self-collision exclusions**: Adjacent body pairs excluded
9. **Fixed inertia format**: Converted fullinertia from SDK row-major format to MuJoCo format (Ixx,Iyy,Izz,Ixy,Ixz,Iyz); enabled balanceinertia for triangle inequality compliance
10. **Removed meshdir**: Mesh paths are explicit relative paths (meshes/filename.obj) for include compatibility

## Phase 1.1.2 Modifications
11. **Integrator**: Changed from Euler to `implicitfast` (unconditionally stable)
12. **Solver iterations**: Increased from 3 to 50; ls_iterations from 5 to 10
13. **Energy tracking**: Enabled `energy` flag
14. **Armature on spine**: Added `armature="0.01"` to spine_z (was missing)
15. **Per-joint damping**: Tiered damping/frictionloss (proximal 2.0/0.3, mid 1.5/0.2, distal 0.5/0.1, gripper 0.3/0.05)
16. **ctrlrange**: Changed `inheritrange` from 2 to 1 (actuators clamped to joint range)
17. **Contact solver params**: Added `solref="0.005 1.0" solimp="0.9 0.95 0.001 0.5 2"` to collision geoms
18. **Rest keyframe**: Added `rest` keyframe to scene (shoulders abducted, elbows bent)

## Phase 1.1.3 Modifications (EZGripper Integration)
19. **EZGripper palm bodies**: Added `left_ezgripper_palm` / `right_ezgripper_palm` bodies with adapter transform from IHMC URDF (`euler="3.14159 1.5708 0"`)
20. **EZGripper finger bodies**: 4 finger link bodies per hand (finger1_l1, finger1_l2, finger2_l1, finger2_l2), each with inertial, visual mesh, and collision box
21. **EZGripper knuckle joints**: 4 joints per hand (`knuckle_palm_l1_1/2`, `knuckle_l1_l2_1/2`), range [0, 1.94] rad, armature 0.005, damping 0.2
22. **Equality constraints**: 6 constraints (3 per hand) coupling all finger joints to primary `knuckle_palm_l1_1` via 1:1 polynomial (`polycoef="0 1 0 0 0"`)
23. **EZGripper actuators**: `left_ezgripper` / `right_ezgripper` position actuators on primary knuckle joint, `forcerange="-8 8"`, `kp=10`
24. **Tool frame sites**: `left_tool_frame` / `right_tool_frame` on palm body at `pos="0.073 0 0"` (frozen reference for state contract)
25. **Tool frame sensors**: `framepos` + `framequat` sensors for both tool frame sites
26. **EZGripper collision defaults**: `class="alex/ezgripper/collision"` with `friction="1.5 0.02 0.01"` for non-slip grasps
27. **Self-collision exclusions**: 6 per hand (palm↔gripper, palm↔fingers, finger L1↔L2, finger1↔finger2)
28. **STL mesh assets**: 3 shared meshes (`ezgripper_palm`, `ezgripper_finger_l1`, `ezgripper_finger_l2`) from SAKE Gen2
