# assets/x1_29dof provenance

source asset : `/Users/yumx/code/X1_29_AMP/X1_29DOF`
source xml   : `mjcf/robot/xyber_x1/xyber_x1_serial.xml` (sha256 `f3ce939cbbdf5ef64871a1711b188bd11fb6ea8bfc2a8477ba81757e00a2ba19`)
vendored xml : `assets/x1_29dof/x1_29dof.xml` (sha256 `e8afe91024178b4b3b3f95bdec2f91b975c2125050ca404a0989f1d7bcc1b16a`)
meshes       : 30 STL referenced by the xml, copied under the referenced name

## Changes applied by tools/vendor_x1_assets.py

1. `<compiler meshdir>`: `../meshes` -> `meshes` (flat layout).
3. geom names added (collision selection / domain randomization by name):

   - base_link: unnamed <geom> -> name="base_visual"
   - base_link: unnamed <geom> -> name="base_collision"
   - lumbar_yaw_link: unnamed <geom> -> name="lumbar_yaw_visual"
   - lumbar_roll_link: unnamed <geom> -> name="lumbar_roll_visual"
   - lumbar_pitch_link: unnamed <geom> -> name="lumbar_pitch_visual"
   - lumbar_pitch_link: unnamed <geom> -> name="lumbar_pitch_collision"
   - left_shoulder_pitch_link: unnamed <geom> -> name="left_shoulder_pitch_visual"
   - left_shoulder_roll_link: unnamed <geom> -> name="left_shoulder_roll_visual"
   - left_shoulder_yaw_link: unnamed <geom> -> name="left_shoulder_yaw_visual"
   - left_elbow_pitch_link: unnamed <geom> -> name="left_elbow_pitch_visual"
   - left_elbow_yaw_link: unnamed <geom> -> name="left_elbow_yaw_visual"
   - left_wrist_pitch_link: unnamed <geom> -> name="left_wrist_pitch_visual"
   - left_wrist_roll_link: unnamed <geom> -> name="left_wrist_roll_visual"
   - right_shoulder_pitch_link: unnamed <geom> -> name="right_shoulder_pitch_visual"
   - right_shoulder_roll_link: unnamed <geom> -> name="right_shoulder_roll_visual"
   - right_shoulder_yaw_link: unnamed <geom> -> name="right_shoulder_yaw_visual"
   - right_elbow_pitch_link: unnamed <geom> -> name="right_elbow_pitch_visual"
   - right_elbow_yaw_link: unnamed <geom> -> name="right_elbow_yaw_visual"
   - right_wrist_pitch_link: unnamed <geom> -> name="right_wrist_pitch_visual"
   - right_wrist_roll_link: unnamed <geom> -> name="right_wrist_roll_visual"
   - left_hip_pitch_link: unnamed <geom> -> name="left_hip_pitch_visual"
   - left_hip_roll_link: unnamed <geom> -> name="left_hip_roll_visual"
   - left_hip_yaw_link: unnamed <geom> -> name="left_hip_yaw_visual"
   - left_knee_pitch_link: unnamed <geom> -> name="left_knee_pitch_visual"
   - left_ankle_pitch_link: unnamed <geom> -> name="left_ankle_pitch_visual"
   - left_ankle_roll_link: unnamed <geom> -> name="left_foot_1_marker"
   - left_ankle_roll_link: unnamed <geom> -> name="left_foot_1_collision"
   - left_ankle_roll_link: unnamed <geom> -> name="left_foot_2_collision"
   - left_ankle_roll_link: unnamed <geom> -> name="left_foot_3_collision"
   - left_ankle_roll_link: unnamed <geom> -> name="left_foot_4_collision"
   - left_ankle_roll_link: unnamed <geom> -> name="left_foot_2_marker"
   - left_ankle_roll_link: unnamed <geom> -> name="left_foot_3_marker"
   - left_ankle_roll_link: unnamed <geom> -> name="left_foot_4_marker"
   - left_ankle_roll_link: unnamed <geom> -> name="left_foot_5_marker"
   - right_hip_pitch_link: unnamed <geom> -> name="right_hip_pitch_visual"
   - right_hip_roll_link: unnamed <geom> -> name="right_hip_roll_visual"
   - right_hip_yaw_link: unnamed <geom> -> name="right_hip_yaw_visual"
   - right_knee_pitch_link: unnamed <geom> -> name="right_knee_pitch_visual"
   - right_ankle_pitch_link: unnamed <geom> -> name="right_ankle_pitch_visual"
   - right_ankle_roll_link: unnamed <geom> -> name="right_foot_1_marker"
   - right_ankle_roll_link: unnamed <geom> -> name="right_foot_1_collision"
   - right_ankle_roll_link: unnamed <geom> -> name="right_foot_2_collision"
   - right_ankle_roll_link: unnamed <geom> -> name="right_foot_3_collision"
   - right_ankle_roll_link: unnamed <geom> -> name="right_foot_4_collision"
   - right_ankle_roll_link: unnamed <geom> -> name="right_foot_2_marker"
   - right_ankle_roll_link: unnamed <geom> -> name="right_foot_3_marker"
   - right_ankle_roll_link: unnamed <geom> -> name="right_foot_4_marker"
   - right_ankle_roll_link: unnamed <geom> -> name="right_foot_5_marker"

Everything else (body poses, inertias, joint axes/limits, damping, friction loss, actuators, contact
parameters) is byte-identical to the source XML.

`--check` recomputes the same transformation from the source and diffs it against the vendored copy.
