"""X1 29-DoF humanoid asset for mjlab 1.6.0 (SMP pipeline).

Ported from super_mimic_F1 allmimic/robots/x1_29dof.py (written against
mjlab 1.2.0; the EntityCfg/EntityArticulationInfoCfg/BuiltinPositionActuatorCfg/
CollisionCfg surface used here is unchanged in 1.6.0 -- verified against
mjlab v1.6.0 asset_zoo unitree_g1/g1_constants.py).

PD gains, armature and effort limits are the lab-identified values from
~/code/X1_29_AMP/x1_armature_pd.md ("方案 B": torque identification +
on-robot walk-log fits, kd capped at 8 by the MIT firmware).
Foot contact geoms get condim 3 + solref (0.02, 1.0) -- the asset's 5 ms
solref is for its own 1 kHz scene and blows up at the 200 Hz physics step
(measured |qd| -> 1e6 rad/s; see allmimic robots/x1_29dof.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco

X1_SMP_ROOT = Path(__file__).resolve().parent.parent.parent
ASSET_DIR = X1_SMP_ROOT / 'assets' / 'x1_29dof'
X1_XML = ASSET_DIR / 'x1_29dof.xml'
MESH_DIR = ASSET_DIR / 'meshes'

# MuJoCo model order (depth-first); equals qpos[7:] of the motion library and
# scripts/x1_data_lib.X1_JOINT_ORDER.
X1_JOINT_ORDER = (
    'lumbar_yaw_joint', 'lumbar_roll_joint', 'lumbar_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_pitch_joint', 'left_elbow_yaw_joint',
    'left_wrist_pitch_joint', 'left_wrist_roll_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint', 'right_elbow_yaw_joint',
    'right_wrist_pitch_joint', 'right_wrist_roll_joint',
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
)
assert len(X1_JOINT_ORDER) == 29

# kp [Nm/rad], kd [Nms/rad], armature [kg m^2], effort [Nm], frictionloss [Nm]
# verbatim from allmimic/robots/x1_29dof.py X1_JOINT_PARAMS (lab-identified)
X1_JOINT_PARAMS = {
    'lumbar_yaw_joint': dict(kp=90.0, kd=8.0, armature=0.022, effort=180.0, frictionloss=3.0),
    'lumbar_roll_joint': dict(kp=70.0, kd=8.0, armature=0.021, effort=180.0, frictionloss=3.0),
    'lumbar_pitch_joint': dict(kp=100.0, kd=8.0, armature=0.012, effort=180.0, frictionloss=4.0),
    'left_hip_pitch_joint': dict(kp=65.0, kd=8.0, armature=0.208, effort=180.0, frictionloss=4.0),
    'right_hip_pitch_joint': dict(kp=65.0, kd=8.0, armature=0.208, effort=180.0, frictionloss=4.0),
    'left_hip_roll_joint': dict(kp=60.0, kd=8.0, armature=0.25, effort=180.0, frictionloss=3.0),
    'right_hip_roll_joint': dict(kp=60.0, kd=8.0, armature=0.25, effort=180.0, frictionloss=3.0),
    'left_hip_yaw_joint': dict(kp=100.0, kd=3.3, armature=0.0148, effort=180.0, frictionloss=3.0),
    'right_hip_yaw_joint': dict(kp=100.0, kd=3.3, armature=0.0148, effort=180.0, frictionloss=3.0),
    'left_knee_pitch_joint': dict(kp=85.0, kd=8.0, armature=0.273, effort=180.0, frictionloss=4.0),
    'right_knee_pitch_joint': dict(kp=85.0, kd=8.0, armature=0.273, effort=180.0, frictionloss=4.0),
    'left_ankle_pitch_joint': dict(kp=50.0, kd=4.0, armature=0.15, effort=80.0, frictionloss=1.5),
    'right_ankle_pitch_joint': dict(kp=50.0, kd=4.0, armature=0.15, effort=80.0, frictionloss=1.5),
    'left_ankle_roll_joint': dict(kp=50.0, kd=2.5, armature=0.035, effort=80.0, frictionloss=1.5),
    'right_ankle_roll_joint': dict(kp=50.0, kd=2.5, armature=0.035, effort=80.0, frictionloss=1.5),
    'left_shoulder_pitch_joint': dict(kp=40.0, kd=4.0, armature=0.02, effort=27.0, frictionloss=0.5),
    'right_shoulder_pitch_joint': dict(kp=40.0, kd=4.0, armature=0.02, effort=27.0, frictionloss=0.5),
    'left_shoulder_roll_joint': dict(kp=40.0, kd=4.0, armature=0.013, effort=27.0, frictionloss=0.5),
    'right_shoulder_roll_joint': dict(kp=40.0, kd=4.0, armature=0.013, effort=27.0, frictionloss=0.5),
    'left_shoulder_yaw_joint': dict(kp=40.0, kd=2.0, armature=0.01, effort=27.0, frictionloss=0.5),
    'right_shoulder_yaw_joint': dict(kp=40.0, kd=2.0, armature=0.01, effort=27.0, frictionloss=0.5),
    'left_elbow_pitch_joint': dict(kp=30.0, kd=1.9, armature=0.01, effort=27.0, frictionloss=0.5),
    'right_elbow_pitch_joint': dict(kp=30.0, kd=1.9, armature=0.01, effort=27.0, frictionloss=0.5),
    'left_elbow_yaw_joint': dict(kp=20.0, kd=1.0, armature=0.01, effort=27.0, frictionloss=0.5),
    'right_elbow_yaw_joint': dict(kp=20.0, kd=1.0, armature=0.01, effort=27.0, frictionloss=0.5),
    'left_wrist_pitch_joint': dict(kp=15.0, kd=1.0, armature=0.01, effort=10.0, frictionloss=0.2),
    'right_wrist_pitch_joint': dict(kp=15.0, kd=1.0, armature=0.01, effort=10.0, frictionloss=0.2),
    'left_wrist_roll_joint': dict(kp=15.0, kd=1.0, armature=0.01, effort=10.0, frictionloss=0.2),
    'right_wrist_roll_joint': dict(kp=15.0, kd=1.0, armature=0.01, effort=10.0, frictionloss=0.2),
}

# Bodies named like the G1 set used by the SMP feature builder.
X1_ROOT_BODY = 'base_link'                 # G1: pelvis
X1_TORSO_BODY = 'lumbar_pitch_link'        # G1: torso_link (viewer + COM DR proxy)
X1_FOOT_GEOMS = tuple(f'{s}_foot_{k}_collision' for s in ('left', 'right') for k in (1, 2, 3, 4))
X1_FOOT_GEOM_RE = '^(' + '|'.join(X1_FOOT_GEOMS) + ')$'
X1_FOOT_SOLREF = (0.02, 1.0)
X1_FOOT_CONDIM = 3
X1_FOOT_PRIORITY = 1
X1_FOOT_FRICTION = (0.6,)
X1_INIT_HEIGHT = 0.6275


def x1_spec():
    """MjSpec of the X1 (no actuators -- mjlab adds position servos)."""
    spec = mujoco.MjSpec.from_file(str(X1_XML))
    spec.assets = {f'{spec.meshdir}/{p.name}': p.read_bytes()
                   for p in sorted(MESH_DIR.iterdir()) if p.is_file()}
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def x1_actuator_cfgs():
    from mjlab.actuator import BuiltinPositionActuatorCfg

    groups: dict[tuple, list[str]] = {}
    for name in X1_JOINT_ORDER:
        p = X1_JOINT_PARAMS[name]
        key = (p['kp'], p['kd'], p['effort'], p['armature'], p['frictionloss'])
        groups.setdefault(key, []).append(name)
    return tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=tuple(names),
            stiffness=kp, damping=kd, effort_limit=effort,
            armature=armature, frictionloss=frictionloss,
        )
        for (kp, kd, effort, armature, frictionloss), names in groups.items()
    )


def x1_collision_cfgs():
    from mjlab.utils.spec_config import CollisionCfg

    foot = X1_FOOT_GEOM_RE
    return (
        CollisionCfg(
            geom_names_expr=('.*_collision',),
            contype=1,
            conaffinity=1,
            condim={foot: X1_FOOT_CONDIM, '.*_collision': 1},
            priority={foot: X1_FOOT_PRIORITY},
            friction={foot: X1_FOOT_FRICTION},
            solref={foot: X1_FOOT_SOLREF},
        ),
    )


def get_x1_robot_cfg():
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    articulation = EntityArticulationInfoCfg(
        actuators=x1_actuator_cfgs(), soft_joint_pos_limit_factor=0.9)
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, X1_INIT_HEIGHT),
            joint_pos={n: 0.0 for n in X1_JOINT_ORDER},
            joint_vel={'.*': 0.0},
        ),
        collisions=x1_collision_cfgs(),
        spec_fn=x1_spec,
        articulation=articulation,
    )


def x1_action_scale() -> dict[str, float]:
    """Per-joint action scale, same formula as mjlab's G1: 0.25 * effort/kp."""
    return {n: 0.25 * p['effort'] / p['kp'] for n, p in X1_JOINT_PARAMS.items()}


def joint_names_match(names) -> None:
    got = [re.sub(r'^.*/', '', n) for n in names]
    if got != list(X1_JOINT_ORDER):
        raise RuntimeError(f'X1 joint order mismatch:\n  model: {got}\n  config: {list(X1_JOINT_ORDER)}')
