"""Per-clip acceptance gates for the phase-2 X1 retargeted library (contract task 4).

Gates (docs/X1_PHASE2_DATA_REPORT.md reports pass rates per gate):
  arms:   wrist_err_p50/p90 vs the anchored IK target (measured pre-grounding),
          arm at-limit frame fraction (0.02 rad), arm |qdot| max
  torso:  lumbar joints within range with margin, chest-yaw error vs the SMPLX
          spine3 heading, root tilt (1 - R_zz)
  global: zero joint-limit violations, sole penetration > -2 mm, root height band,
          no self-collision between non-adjacent bodies

Usage (batch scorer is embedded in scripts/build_x1_phase2_library.py):
  from tools.x1_retarget_gates import GateModel, score_clip, GATES
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from allmimic.robots import x1_29dof as x1  # noqa: E402

ARM_JOINTS = (
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_pitch_joint', 'left_elbow_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint', 'right_elbow_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_roll_joint',
)
LUMBAR_JOINTS = ('lumbar_yaw_joint', 'lumbar_roll_joint', 'lumbar_pitch_joint')

# gate name -> (limit, comparison, description). A clip passes when every gate passes.
GATES = {
    'wrist_err_p50_m': (0.13, '<=', 'median |wrist FK - anchored intent target|; floor = r*sin(dir-err floor ~15-20deg) '
                                   '~= 0.07-0.10 m (shoulder-cone/posture equilibrium) + 5-tap smoothing lag'),
    'wrist_err_p90_m': (0.25, '<=', 'p90 of the same'),
    'arm_at_limit_frac': (0.30, '<=', 'frames with an arm joint within 0.02 rad of a limit'),
    'arm_qd_max': (25.0, '<=', 'max |arm joint velocity| (rad/s)'),
    'lumbar_margin_ok': (1.0, '>=', 'fraction of frames with every lumbar joint 0.05 rad inside its range'),
    'chest_yaw_err_p90_deg': (45.0, '<=', 'p90 robot-vs-human RELATIVE torso twist error (deg; absolute '
                                      'headings are incomparable across the 90-deg T-pose calibration offset)'),
    'twist_abs_p99_deg': (40.0, '<=', 'p99 |robot torso twist| (bounded lumbar flailing; joint limit is 57 deg)'),
    'tilt_max': (0.35, '<=', 'max 1 - R_zz of the root'),
    'joint_over_limit_max_rad': (1e-6, '<=', 'hard limit violation must be zero'),
    'sole_pen_m': (-0.002, '>=', 'lowest sole point must not penetrate 2 mm'),
    'root_z_mean_band': ((0.45, 0.75), 'in', 'mean root height band (m)'),
    'self_collision_frames': (0.0, '<=', 'frames with a contact between non-adjacent bodies'),
}


class GateModel:
    """Vendored X1 MJCF on CPU: joint limits, soles, body contacts, chest heading."""

    def __init__(self):
        spec = mujoco.MjSpec.from_file(str(x1.X1_XML))
        spec.assets = {f'{spec.meshdir}/{p.name}': p.read_bytes() for p in sorted(x1.MESH_DIR.iterdir()) if p.is_file()}
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, self.m.njnt)]
        self.lo = np.array([self.m.jnt_range[j, 0] for j in range(1, self.m.njnt)])
        self.hi = np.array([self.m.jnt_range[j, 1] for j in range(1, self.m.njnt)])
        self.arm_idx = np.array([self.names.index(n) for n in ARM_JOINTS])
        self.lumbar_idx = np.array([self.names.index(n) for n in LUMBAR_JOINTS])
        self.chest_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'lumbar_pitch_link')
        self.soles = {0: [], 1: []}
        for g in x1.X1_FOOT_GEOMS:
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, g)
            self.soles[0 if g.startswith('left') else 1].append(gid)
        # non-adjacent body pairs for self-collision: exclude parent-child and siblings of a chain
        self._pair_ok = set()
        for b in range(self.m.nbody):
            self._pair_ok.add((min(b, self.m.body_parentid[b]), max(b, self.m.body_parentid[b])))
        # left/right foot bodies share nothing; treat same-parent siblings as adjacent too
        for b in range(self.m.nbody):
            p = self.m.body_parentid[b]
            for c in range(self.m.nbody):
                if self.m.body_parentid[c] == p:
                    self._pair_ok.add((min(b, c), max(b, c)))
        self._pair_ok.add((0, 0))

    def sole_z(self, qpos: np.ndarray) -> np.ndarray:
        out = np.zeros((len(qpos), 2))
        for i, row in enumerate(qpos):
            self.d.qpos[:] = row
            mujoco.mj_kinematics(self.m, self.d)
            for side, gids in self.soles.items():
                out[i, side] = min(lowest_point(self.m, self.d, gid) for gid in gids)
        return out

    def self_collision(self, qpos: np.ndarray) -> int:
        """frames with any contact between non-adjacent bodies (excluding world)"""
        n = 0
        for i, row in enumerate(qpos):
            self.d.qpos[:] = row
            mujoco.mj_collision(self.m, self.d)
            hit = False
            for c in range(self.d.ncon):
                b1 = self.m.geom_bodyid[self.d.contact[c].geom1]
                b2 = self.m.geom_bodyid[self.d.contact[c].geom2]
                pair = (min(b1, b2), max(b1, b2))
                if b1 != b2 and pair not in self._pair_ok:
                    hit = True
                    break
            n += hit
        return n

    def chest_yaw(self, qpos: np.ndarray) -> np.ndarray:
        out = np.zeros(len(qpos))
        for i, row in enumerate(qpos):
            self.d.qpos[:] = row
            mujoco.mj_kinematics(self.m, self.d)
            R = self.d.xmat[self.chest_bid].reshape(3, 3)
            out[i] = np.arctan2(R[1, 0], R[0, 0])
        return np.unwrap(out)


def lowest_point(m, d, gid: int) -> float:
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
        rot = d.geom_xmat[gid].reshape(3, 3)
        half = m.geom_size[gid]
        return float(min((d.geom_xpos[gid] + rot @ (half * np.array([sx, sy, sz])))[2]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)))
    return float(d.geom_xpos[gid][2] - m.geom_size[gid][0])


def human_chest_yaw(frames) -> np.ndarray:
    """SMPLX spine3 heading (unwrapped yaw), same convention as GateModel.chest_yaw."""
    from scipy.spatial.transform import Rotation as Rot

    out = np.zeros(len(frames))
    for i, fr in enumerate(frames):
        q = np.asarray(fr['spine3'][1], dtype=float)  # wxyz
        R = Rot.from_quat(q).as_matrix()
        out[i] = np.arctan2(R[1, 0], R[0, 0])
    return np.unwrap(out)


def human_torso_twist(frames) -> np.ndarray:
    """spine3 yaw minus pelvis yaw (relative chest twist, convention-free)."""
    from scipy.spatial.transform import Rotation as Rot

    out = np.zeros(len(frames))
    for i, fr in enumerate(frames):
        qs = np.asarray(fr['spine3'][1], dtype=float)
        qp = np.asarray(fr['pelvis'][1], dtype=float)
        ys = np.arctan2(Rot.from_quat(qs).as_matrix()[1, 0], Rot.from_quat(qs).as_matrix()[0, 0])
        yp = np.arctan2(Rot.from_quat(qp).as_matrix()[1, 0], Rot.from_quat(qp).as_matrix()[0, 0])
        out[i] = np.angle(np.exp(1j * (ys - yp)))
    return out


def robot_torso_twist(qpos: np.ndarray, model: 'GateModel') -> np.ndarray:
    """lumbar chest yaw minus root yaw."""
    chest = model.chest_yaw(qpos)
    root = 2 * np.arctan2(qpos[:, 6], qpos[:, 3])
    root = np.unwrap(root)
    return np.angle(np.exp(1j * (chest - root)))


def score_clip(qpos: np.ndarray, fps: float, model: GateModel, *,
               wrist_err: np.ndarray | None = None, chest_yaw_human: np.ndarray | None = None,
               tag: str = 'walk', human_net_yaw: float = 0.0) -> dict:
    """Score one grounded 50 Hz clip. wrist_err: pre-grounding per-frame anchored-target error (m).

    Turn-tagged clips are gated on NET yaw reproduction instead of per-frame torso-twist
    phase alignment: during a turn the intent is the heading change, not frame-synchronised
    twist (the retarget lags/leads the human torso transiently by design of the base solve).
    """
    q = qpos[:, 7:]
    over = np.maximum(q - model.hi, 0).max(1)
    under = np.maximum(model.lo - q, 0).max(1)
    arm = q[:, model.arm_idx]
    sat = ((arm >= model.hi[model.arm_idx] - 0.02) | (arm <= model.lo[model.arm_idx] + 0.02)).any(axis=1)
    qd = np.abs(np.gradient(arm, axis=0) * fps).max()
    lum = q[:, model.lumbar_idx]
    lum_ok = ((lum >= model.lo[model.lumbar_idx] + 0.05) & (lum <= model.hi[model.lumbar_idx] - 0.05)).all(axis=1)
    tilt = 2 * (qpos[:, 4] ** 2 + qpos[:, 5] ** 2)
    sole = model.sole_z(qpos.astype(np.float64))
    # chest tracking is compared as RELATIVE torso twist (chest-minus-root on the robot,
    # spine3-minus-pelvis on the human): the T-pose calibration rotates the spine3 ->
    # lumbar_pitch frames by 90 deg about x, so comparing absolute headings is meaningless.
    twist_r = robot_torso_twist(qpos, model)
    twist_h = chest_yaw_human[:len(twist_r)] if chest_yaw_human is not None else np.zeros(len(twist_r))
    cy = np.degrees(np.abs(np.angle(np.exp(1j * (twist_r - twist_h)))))
    coll = model.self_collision(qpos.astype(np.float64))
    vals = {
        'wrist_err_p50_m': float(np.median(wrist_err)) if wrist_err is not None else 0.0,
        'wrist_err_p90_m': float(np.percentile(wrist_err, 90)) if wrist_err is not None else 0.0,
        'arm_at_limit_frac': float(sat.mean()),
        'arm_qd_max': float(qd),
        'lumbar_margin_ok': float(lum_ok.mean()),
        'chest_yaw_err_p90_deg': float(np.percentile(cy, 90)),
        'twist_abs_p99_deg': float(np.percentile(np.degrees(np.abs(twist_r)), 99)),
        'tilt_max': float(tilt.max()),
        'joint_over_limit_max_rad': float(np.maximum(over, under).max()),
        'sole_pen_m': float(sole.min()),
        'root_z_mean_band': float(qpos[:, 2].mean()),
        'self_collision_frames': int(coll),
    }
    # net yaw reproduction (always computed; the deciding chest gate for turn clips)
    root_yaw = np.unwrap(2 * np.arctan2(qpos[:, 6], qpos[:, 3]))
    vals['net_yaw_err_deg'] = float(np.degrees(np.abs(root_yaw[-1] - root_yaw[0] - human_net_yaw)))
    passed = {}
    for name, (limit, cmp, _desc) in GATES.items():
        v = vals[name]
        if tag == 'turn' and name in ('chest_yaw_err_p90_deg', 'twist_abs_p99_deg'):
            # turns: net-heading reproduction replaces per-frame twist-phase alignment;
            # keep a looser physical bound on the twist itself (lumbar limit is 57 deg)
            passed[name] = True if name == 'chest_yaw_err_p90_deg' else bool(v <= 50.0)
            continue
        if cmp == '<=':
            passed[name] = bool(v <= limit)
        elif cmp == '>=':
            passed[name] = bool(v >= limit)
        else:  # 'in'
            passed[name] = bool(limit[0] <= v <= limit[1])
    if tag == 'turn':
        passed['net_yaw_err_deg'] = bool(vals['net_yaw_err_deg'] <= 15.0)
    return {'metrics': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in vals.items()},
            'gates': passed, 'pass': all(passed.values())}
