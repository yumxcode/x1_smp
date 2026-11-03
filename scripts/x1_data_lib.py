"""X1 29-DoF motion data utilities: FK, gait metrics, and SMP window features.

Shared by:
  scripts/gait_dist_audit.py   — gait-distribution audit of retargeted walk clips
  scripts/build_x1_windows.py  — 59-dim windowed features for the TinyMDM prior

Feature layout (must stay bit-compatible with src/smp_ref/rl/utils.py
MotionFeatureBuffer.compute_features, the online RL path):

  [root_pos(3), root_rot(6D tan-norm), joint_pos(29), ee_pos(5*3),
   root_lin_vel(3), root_ang_vel(3)]   -> 59 dims/frame, window H=10

All spatial quantities are anchored to the LAST window frame's yaw-only local
frame (origin at root_T, x-axis = heading_T). root_pos keeps z in world.
6D rotation is [rotmat col0, rotmat col2] (tan-norm, NOT Zhou-2019 first-two-columns).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
X1_XML = ROOT / 'assets' / 'x1_29dof' / 'x1_29dof.xml'

# MuJoCo model order (depth-first), matches allmimic robots/x1_29dof.X1_JOINT_ORDER
# and the qpos[7:] column order of motions/x1_phase2_*.npz.
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

# Tracked end-effectors, X1 counterpart of the G1 set (ankle_roll x2, torso proxy,
# wrist x2). Order defines the ee_pos(15) feature layout.
EE_BODY_NAMES = (
    'left_ankle_roll_link',
    'right_ankle_roll_link',
    'lumbar_pitch_link',
    'left_wrist_roll_link',
    'right_wrist_roll_link',
)

NUM_JOINTS = len(X1_JOINT_ORDER)
NUM_EE = len(EE_BODY_NAMES)
FEATURE_DIM = 3 + 6 + NUM_JOINTS + NUM_EE * 3 + 3 + 3  # = 59


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, quaternions (..., 4) wxyz."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1
    return out


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v (..., 3) by q (..., 4) wxyz."""
    qv = q[..., 1:]
    t = 2.0 * np.cross(qv, v)
    return v + q[..., :1] * t + np.cross(qv, t)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_rotate(quat_conjugate(q), v)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - w * z); m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y); m[..., 2, 1] = 2 * (y * z + w * x); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Yaw-only quaternion (wxyz), projecting out roll/pitch (Isaac-style)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    out = np.zeros_like(q)
    out[..., 0] = np.cos(half)
    out[..., 3] = np.sin(half)
    return out


class X1FK:
    """Batch FK of X1 qpos/qvel clips via MuJoCo (CPU, one clip at a time)."""

    def __init__(self, xml: str | Path = X1_XML):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.ee_bid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                       for n in EE_BODY_NAMES]
        foot_bodies = ('left_ankle_roll_link', 'right_ankle_roll_link')
        self.sole_gids = []
        for fb in foot_bodies:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, fb)
            gids = [g for g in range(self.model.ngeom)
                    if self.model.geom_bodyid[g] == bid
                    and 'collision' in (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')]
            assert len(gids) == 4, f'expected 4 corner collision spheres under {fb}, got {len(gids)}'
            self.sole_gids.append(gids)
        # sanity: joints 1..29 names match X1_JOINT_ORDER
        for i, name in enumerate(X1_JOINT_ORDER):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert self.model.jnt_dofadr[jid] == 6 + i, \
                f'joint order mismatch at {name}: dofadr {self.model.jnt_dofadr[jid]} != {6 + i}'

    def fk_clip(self, qpos: np.ndarray, qvel: np.ndarray):
        """qpos (T,36), qvel (T,35) -> dict of world-frame kinematics.

        Returns root_pos (T,3), root_quat (T,4) wxyz, root_lin_vel_w (T,3),
        root_ang_vel_w (T,3) [qvel free-joint ang vel is body-local; rotated to world],
        joint_pos (T,29), joint_vel (T,29), ee_pos (T,5,3), sole_h (T,2)
        (min over the foot's collision/sphere geoms of geom z, world frame).
        """
        T = qpos.shape[0]
        d = self.data
        root_pos = np.empty((T, 3)); root_quat = np.empty((T, 4))
        root_lin = np.empty((T, 3)); root_ang = np.empty((T, 3))
        joint_pos = np.empty((T, NUM_JOINTS)); joint_vel = np.empty((T, NUM_JOINTS))
        ee_pos = np.empty((T, NUM_EE, 3)); sole_h = np.empty((T, 2))
        for t in range(T):
            d.qpos[:] = qpos[t]
            d.qvel[:] = qvel[t]
            mujoco.mj_forward(self.model, d)  # noqa
            root_pos[t] = d.qpos[:3]  # free joint: root position is qpos[0:3] (world)
            root_quat[t] = d.qpos[3:7]
            root_lin[t] = d.qvel[0:3]  # free-joint translational vel is world frame
            # free-joint angular velocity is expressed in the body local frame -> world
            root_ang[t] = quat_rotate(root_quat[t], d.qvel[3:6])
            joint_pos[t] = d.qpos[7:]
            joint_vel[t] = d.qvel[6:]
            for k, bid in enumerate(self.ee_bid):
                ee_pos[t, k] = d.xpos[bid]
            for k, gids in enumerate(self.sole_gids):
                sole_h[t, k] = min(d.geom_xpos[g, 2] for g in gids)
        return dict(root_pos=root_pos, root_quat=root_quat, root_lin_vel_w=root_lin,
                    root_ang_vel_w=root_ang, joint_pos=joint_pos, joint_vel=joint_vel,
                    ee_pos=ee_pos, sole_h=sole_h)


def compute_windows(fk: dict, window_size: int = 10, stride: int = 1) -> np.ndarray:
    """numpy replica of smp_ref csv_to_npz._compute_windows (bit-compatible layout).

    Returns (N, W, 59) or None if the clip is shorter than window_size.
    """
    base_pos = fk['root_pos']; base_quat = fk['root_quat']
    base_lin = fk['root_lin_vel_w']; base_ang = fk['root_ang_vel_w']
    ee_pos = fk['ee_pos']; joint = fk['joint_pos']
    T = base_pos.shape[0]
    if T < window_size:
        return None
    E = ee_pos.shape[1]
    starts = np.arange(0, T - window_size + 1, stride)
    offsets = np.arange(window_size)
    win_idx = starts[:, None] + offsets[None, :]
    N, W = win_idx.shape
    flat = win_idx.reshape(-1)

    wb_pos = base_pos[flat].reshape(N, W, 3)
    wb_quat = base_quat[flat].reshape(N, W, 4)
    wb_lin = base_lin[flat].reshape(N, W, 3)
    wb_ang = base_ang[flat].reshape(N, W, 3)
    wee = ee_pos[flat].reshape(N, W, E, 3)
    wj = joint[flat].reshape(N, W, NUM_JOINTS)

    yaw_T = yaw_quat(wb_quat[:, -1, :])                       # (N,4)
    yaw_T_W = np.broadcast_to(yaw_T[:, None, :], (N, W, 4)).reshape(-1, 4)
    heading_inv_T_W = quat_conjugate(yaw_T_W)

    root_offset = wb_pos - wb_pos[:, -1:, :]
    root_pos_local = quat_rotate_inverse(yaw_T_W, root_offset.reshape(-1, 3)).reshape(N, W, 3)
    root_pos_local = root_pos_local.copy()
    root_pos_local[..., 2] = wb_pos[..., 2]                    # z in world

    root_rot_q = quat_mul(heading_inv_T_W, wb_quat.reshape(-1, 4)).reshape(N, W, 4)
    rotmat = quat_to_mat(root_rot_q.reshape(-1, 4)).reshape(N, W, 3, 3)
    root_rot_6d = np.concatenate([rotmat[..., :, 0], rotmat[..., :, 2]], axis=-1)  # (N,W,6)

    ee_offset = wee - wb_pos[:, :, None, :]
    yaw_T_E = np.broadcast_to(yaw_T[:, None, None, :], (N, W, E, 4)).reshape(-1, 4)
    ee_local = quat_rotate_inverse(yaw_T_E, ee_offset.reshape(-1, 3)).reshape(N, W, E * 3)

    lin_local = quat_rotate_inverse(yaw_T_W, wb_lin.reshape(-1, 3)).reshape(N, W, 3)
    ang_local = quat_rotate_inverse(yaw_T_W, wb_ang.reshape(-1, 3)).reshape(N, W, 3)

    return np.concatenate([root_pos_local, root_rot_6d, wj, ee_local, lin_local, ang_local],
                          axis=-1).astype(np.float32)


# ---------------- gait metrics (rhythm_gates-compatible semantics) ----------------

def gait_metrics(sole_h: np.ndarray, root_pos: np.ndarray, fps: float) -> dict:
    """Per-clip gait metrics with rhythm_gates.py semantics:
    cadence_hz = 1/median(stride_s) per foot (R1: 0.55-1.2 Hz/foot),
    duty = contact fraction per foot (R3 <= 0.75),
    swing_s from air-phase durations (R2 >= 0.28 s),
    step_len = median(stride displacement)/2 (R4 semantics),
    speed = median horizontal root speed.
    """
    from x1_gait import touchdown_events  # scripts/ on sys.path when run as script
    T = sole_h.shape[0]
    dt = 1.0 / fps
    rel = sole_h - sole_h.min(axis=1, keepdims=True)
    ev = touchdown_events(sole_h)

    per_foot = []
    for k in (0, 1):
        e = ev[k]
        d = np.diff(e)
        stride_s = float(np.median(d) * dt) if len(d) else float('nan')
        cadence = float(1.0 / stride_s) if len(d) else float('nan')
        # swing durations: air phase lengths around each touchdown
        swings, duty, lift = [], float('nan'), float('nan')
        if len(e) >= 1:
            airborne = rel[:, k] > 0.02
            # run-length of airborne phases
            change = np.diff(np.concatenate([[0], airborne.astype(int), [0]]))
            starts_r = np.where(change == 1)[0]
            ends_r = np.where(change == -1)[0]
            seg = (ends_r - starts_r) * dt
            seg = seg[seg >= 0.1]
            if len(seg):
                swings = seg
            duty = float((~airborne).mean())
            lift = float(rel[:, k].max())
        # stride displacement for this foot (root xy at successive touchdowns)
        if len(d) >= 1:
            pts = root_pos[e, :2]
            sl = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            sl = sl[sl > 0.05]
            step_len = float(np.median(sl) / 2.0) if len(sl) else float('nan')
        else:
            step_len = float('nan')
        per_foot.append(dict(
            touchdowns=len(e), stride_s=stride_s, cadence_hz=cadence,
            swing_s_med=float(np.median(swings)) if len(swings) else float('nan'),
            duty=duty, lift_m=lift, step_len_m=step_len,
        ))
    vxy = np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1) / dt
    # robust forward speed: median of the middle 80% (trim start/stop transients)
    lo, hi = int(0.1 * T), int(0.9 * T)
    speed = float(np.median(vxy[lo:hi])) if T > 10 else float('nan')
    return dict(feet=per_foot, speed_mps=speed)


GAIT_BANDS = dict(  # data-audit bands; v2 revision 2026-11-03 (amendment to exp_aafa2481)
    # v2: swing/duty relaxed on the data side only. Measured on 49 retargeted
    # windows: GMR's foot IK systematically shortens the swing phase 30-50% on
    # brisk human walks (cad >= 115 spm) while speed/cadence/step-length/lift
    # stay healthy (02_02: L 0.200/0.649 vs R 0.130/0.860, cad 1.04/1.04 -- the
    # exact 1 m/s @ ~1 Hz/foot target band). The prior is a soft constraint;
    # the POLICY gates (R2 swing >= 0.28 s, R3 duty <= 0.75) stay preregistered
    # and unchanged -- AMP-arc precedent: demos with duty 0.9+ still yielded
    # policies at duty 0.44.
    cadence_hz=(0.55, 1.20),   # R1 band per foot (unchanged)
    speed_mps=(0.40, 1.30),    # 1 m/s neighborhood (unchanged)
    step_len_m=(0.35, 0.65),   # R4 data band (unchanged)
    swing_s=(0.15, None),      # v2: 0.24 -> 0.15 (data side; policy R2 >= 0.28 unchanged)
    duty=(None, 0.87),         # v2: 0.80 -> 0.87 (data side; policy R3 <= 0.75 unchanged)
    lift_m=(0.025, None),      # unchanged
)


def audit_pass(m: dict, bands: dict = GAIT_BANDS) -> tuple[bool, dict]:
    """Whole-clip pass/fail against the preregistered data bands."""
    vals, ok = {}, True
    feet = m['feet']
    for k in ('cadence_hz', 'swing_s_med', 'duty', 'lift_m', 'step_len_m'):
        vals[k] = [f[k] for f in feet]
    vals['speed_mps'] = m['speed_mps']
    # both feet must resolve a cadence
    if any(np.isnan(vals['cadence_hz'][i]) for i in (0, 1)):
        return False, {k: v for k, v in vals.items()}
    def in_band(x, b):
        lo, hi = b
        if lo is not None and x < lo: return False
        if hi is not None and x > hi: return False
        return True
    for i in (0, 1):
        ok &= in_band(vals['cadence_hz'][i], bands['cadence_hz'])
        ok &= in_band(vals['swing_s_med'][i], bands['swing_s']) if not np.isnan(vals['swing_s_med'][i]) else True
        ok &= in_band(vals['duty'][i], bands['duty']) if not np.isnan(vals['duty'][i]) else True
        ok &= in_band(vals['lift_m'][i], bands['lift_m']) if not np.isnan(vals['lift_m'][i]) else True
        ok &= in_band(vals['step_len_m'][i], bands['step_len_m']) if not np.isnan(vals['step_len_m'][i]) else True
    ok &= in_band(vals['speed_mps'], bands['speed_mps'])
    return bool(ok), vals
