"""Build the allmimic X1 motion library: AMASS/SMPLX -> GMR retarget -> 50 Hz npz.

The retargeting is GMR's (same LAFAN1/SMPLX -> robot IK that produced the G1 data), with the X1 registered as
`tgt_robot="x1"` in the local GMR checkout. Everything after the per-frame qpos is this repo's own conversion, so
the output has exactly the contract of scripts/prepare_motions.py:

  qpos  (N, 36): root pos, root quat (w, x, y, z), 29 joint pos (rad) -- allmimic.robots.x1_29dof.X1_JOINT_ORDER
  qvel  (N, 35): root lin vel (world), root ang vel (world), joint vel
  starts, lengths (C,), names (C,), fps

Two X1-specific steps:
  * ik config: GMR ships smplx_to_x1.json whose root (pelvis) scale 1.2009 is calibrated for a ~1.8 m robot, so
    its IK puts the X1's root at the *human's* pelvis height (measured: 0.93-0.98 m raw, feet 31 cm off the
    floor). motions/x1_ik_config.json is that file with pelvis/spine3 scaled by 0.6578 (= X1 root height
    0.6275 / human pelvis 0.954) so the IK solves around the X1's own height.
  * grounding: the soles are then placed on the floor -- per frame by default (lowest sole point = 0), which keeps
    the stance foot planted for walking clips; --ground per-clip subtracts one offset for the whole clip, which
    preserves a flight phase but leaves the feet up to a few cm off the floor.
  * validation: joint-limit violation, root height, sole penetration, root orientation and foot cadence are
    measured on the vendored model (MuJoCo CPU FK) before a clip is accepted.

  PYTHONPATH=. python scripts/build_x1_motions.py --limit 2 --out /tmp/x1_pilot.npz --report /tmp/x1_pilot.json
  PYTHONPATH=. python scripts/build_x1_motions.py --out motions/x1_cmu_walk.npz --report motions/x1_cmu_walk.meta.json

Needs the GMR venv (`general_motion_retargeting` importable): python from ~/code/X1_29_AMP/_gmr_local/venv.
Only the final npz + the JSON report are kept; nothing per-clip is written to disk.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from allmimic.robots import x1_29dof as x1  # noqa: E402
from scripts.prepare_motions import resample, velocities  # noqa: E402
from tools.x1_gait import gait_summary  # noqa: E402

DEFAULT_AMASS = Path.home() / 'code' / 'X1_29_AMP' / 'data' / 'amass' / 'CMU'
# smplx.create() appends '/smplx/SMPLX_<gender>.npz' itself, so this is the PARENT of smplx/. The AMASS CMU clips
# are gender=MALE and data/amass/smplx only ships SMPLX_NEUTRAL.npz, so the default is the GMR checkout's copy
# (all three genders, the layout roboparty_train/run_gmr_retarget.py creates).
DEFAULT_SMPLX = Path.home() / 'code' / 'X1_29_AMP' / '_gmr_local' / 'GMR' / 'assets' / 'body_models'
DEFAULT_IK = Path(__file__).resolve().parent.parent / 'motions' / 'x1_ik_config.json'
FPS_OUT = 50.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class X1Model:
    """the vendored X1 MJCF on CPU: joint limits + foot-sole forward kinematics for validation"""

    def __init__(self):
        spec = mujoco.MjSpec.from_file(str(x1.X1_XML))
        spec.assets = {f'{spec.meshdir}/{p.name}': p.read_bytes() for p in sorted(x1.MESH_DIR.iterdir()) if p.is_file()}
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, self.m.njnt)]
        assert tuple(self.names) == x1.X1_JOINT_ORDER, 'vendored model joint order != X1_JOINT_ORDER'
        self.lo = np.array([self.m.jnt_range[j, 0] for j in range(1, self.m.njnt)])
        self.hi = np.array([self.m.jnt_range[j, 1] for j in range(1, self.m.njnt)])
        # per foot, the contact geoms that belong to it (4 spheres, or a sole box if the asset ever changes again)
        self.soles = {0: [], 1: []}
        for g in x1.X1_FOOT_GEOMS:
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, g)
            self.soles[0 if g.startswith('left') else 1].append(gid)

    def sole_z(self, qpos: np.ndarray) -> np.ndarray:
        """(N, 2) lowest sole point per foot, with the root at the clip's own height"""
        out = np.zeros((len(qpos), 2))
        for i, row in enumerate(qpos):
            self.d.qpos[:] = row
            mujoco.mj_kinematics(self.m, self.d)
            for side, gids in self.soles.items():
                out[i, side] = min(lowest_point(self.m, self.d, gid) for gid in gids)
        return out


def lowest_point(m, d, gid: int) -> float:
    """world z of the lowest point of a geom (box: its 8 corners; sphere/capsule: centre - radius)"""
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
        rot = d.geom_xmat[gid].reshape(3, 3)
        half = m.geom_size[gid]
        return float(min((d.geom_xpos[gid] + rot @ (half * np.array([sx, sy, sz])))[2]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)))
    return float(d.geom_xpos[gid][2] - m.geom_size[gid][0])


def retargeter(human_height: float):
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    from tools.x1_arm_fix import make_retargeter

    # AnchoredRetarget: wrist targets re-anchored on the robot's live shoulder FK
    # (phase-2 arm fix; see tools/x1_arm_fix.py for the measured rationale).
    return make_retargeter(GMR)(src_human='smplx', tgt_robot='x1',
                                actual_human_height=human_height, verbose=False)


def use_ik_config(path: Path) -> None:
    """point GMR's smplx->x1 lookup at our calibrated copy (pelvis scale 0.6578x the stock one)"""
    from general_motion_retargeting.params import IK_CONFIG_DICT

    IK_CONFIG_DICT['smplx']['x1'] = str(path)


def retarget_clip(path: Path, smplx_dir: Path, make_retargeter, last: dict,
                  passes: int = 2, arm_mode: str = 'refine') -> tuple[np.ndarray, float]:
    """(N, 36) qpos in wxyz order, at the clip's own fps.

    GMR's IK integrates from the model's current configuration, so a single pass starts the clip from q=0 and the
    first frame is under-converged (measured: a 0.5-1.0 rad jump into frame 1 on 07_01/07_02, i.e. a 30-50 rad/s
    joint velocity spike at frame 0). A second pass starts from the converged end pose of the first.
    """
    from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast, load_smplx_file

    smplx_data, body_model, smplx_output, height = load_smplx_file(str(path), str(smplx_dir))
    src_fps = float(smplx_data['mocap_frame_rate'].item())
    frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=src_fps)
    if last.get('r') is None or abs(last.get('h', 0.0) - height) > 0.01:
        last['r'] = make_retargeter(height)
        last['h'] = height
    qpos = None
    for _ in range(max(1, passes)):
        qpos = np.array([last['r'].retarget(f).copy() for f in frames], dtype=np.float64)
    if arm_mode == 'refine':
        from tools.x1_arm_fix import arm_refine

        qpos = arm_refine(last['r'], qpos, frames)
    # no quaternion reorder: GMR returns MuJoCo qpos, whose free-joint quat is already wxyz -- the order the allmimic
    # contract uses (scripts/prepare_motions.py converts the G1 CSVs from xyzw *because* that format is xyzw).
    # Verified against MuJoCo FK: reading qpos[3:7] as wxyz gives 1 - R_zz = 1e-4 for an upright walking frame,
    # reading it as xyzw gives 0.065.
    return qpos, src_fps


def metrics(qpos: np.ndarray, qvel: np.ndarray, model: X1Model, sole: np.ndarray, fps: float) -> dict:
    q = qpos[:, 7:]
    over = np.clip(q - model.hi, 0, None)
    under = np.clip(model.lo - q, 0, None)
    worst = np.argmax(np.maximum(over, under).max(0))
    tilt = 2 * (qpos[:, 4] ** 2 + qpos[:, 5] ** 2)  # 1 - R_zz from the wxyz quat
    # gait from the sole heights, one implementation shared with tools/x1_motion_audit.py (tools/x1_gait.py)
    gait = gait_summary(sole, fps)
    yaw = 2 * np.arctan2(qpos[:, 6], qpos[:, 3])
    return {
        'frames': int(len(qpos)),
        'seconds': round(len(qpos) / fps, 2),
        'root_z': [round(float(qpos[:, 2].min()), 4), round(float(qpos[:, 2].mean()), 4), round(float(qpos[:, 2].max()), 4)],
        'sole_min': round(float(sole.min()), 4),
        'sole_min_per_foot': [round(float(sole[:, k].min()), 4) for k in (0, 1)],
        'joint_over_limit_max_rad': round(float(np.maximum(over, under).max()), 4),
        'joint_over_limit_worst': f'{model.names[worst]}: {float(q[:, worst][np.argmax(np.maximum(over, under)[:, worst])]):+.3f} '
                                  f'(limit {model.lo[worst]:+.2f}..{model.hi[worst]:+.2f})',
        'joint_over_limit_n': int((np.maximum(over, under) > 0.02).sum()),
        # share of frames within 0.02 rad of any joint limit, and the joints that are there most often: the X1's arms
        # are far shorter than a human's, so the IK saturates the shoulder/elbow limits on some clips
        'at_limit_frame_frac': round(float(((q >= model.hi - 0.02) | (q <= model.lo + 0.02)).any(axis=1).mean()), 3),
        'at_limit_joints': sorted({model.names[j] for j in np.flatnonzero(((q >= model.hi - 0.02) | (q <= model.lo + 0.02)).mean(0) > 0.25)}),
        'tilt_max': round(float(tilt.max()), 4),
        'yaw_start_deg': round(float(np.degrees(yaw[0])), 1),
        # net heading change over the clip (NOT drift: 14 of the 41 CMU clips turn 45-150 deg, and this is how much
        # of that the retarget reproduces -- tools/x1_motion_audit.py compares it against the SMPLX source)
        'yaw_net_change_deg': round(float(np.degrees(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0])), 1),
        'qd_max': round(float(np.abs(qvel[:, 6:]).max()), 2),
        'q_mean': [round(float(v), 3) for v in q.mean(0)],
        'root_vel_max': round(float(np.abs(qvel[:, :3]).max()), 2),
        'contact_frac': gait['other_foot_contact_frac'],
        # gait, verbatim from tools/x1_gait.py: touchdown counts, cadence/stride per foot, alternation, and whether
        # the clip is interpretable as walking at all (both feet, stride 0.5-1.5 s, alternation mostly 1)
        'touchdowns': gait['touchdowns'],
        'cadence_hz': [c for c in gait['cadence_hz'] if c is not None],
        'stride_s': gait['stride_s'],
        'alternation_between_left_td': gait['alternation_between_left_td'],
        'gait_resolved': gait['gait_resolved'],
        'walk_like': gait['walk_like'],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', type=Path, default=DEFAULT_AMASS, help='AMASS npz directory (searched recursively)')
    ap.add_argument('--smplx', type=Path, default=DEFAULT_SMPLX,
                    help='SMPLX body model root (the directory that contains smplx/SMPLX_<gender>.npz)')
    ap.add_argument('--out', type=Path, default=Path('motions/x1_cmu_walk.npz'))
    ap.add_argument('--report', type=Path, default=Path('motions/x1_cmu_walk.meta.json'))
    ap.add_argument('--limit', type=int, default=0, help='only the first N clips (pilot run)')
    ap.add_argument('--fps-out', type=float, default=FPS_OUT)
    ap.add_argument('--ik-config', type=Path, default=DEFAULT_IK,
                    help='smplx->x1 IK config (default: the calibrated motions/x1_ik_config.json)')
    ap.add_argument('--ground', choices=('per-frame', 'per-clip'), default='per-frame',
                    help='per-frame: the lowest sole point is on the floor every frame (walking); '
                         'per-clip: one offset, keeps a flight phase')
    ap.add_argument('--ik-passes', type=int, default=2, help='GMR IK passes over each clip (see retarget_clip)')
    ap.add_argument('--arm-mode', choices=('refine', 'stock'), default='refine',
                    help='refine: second pass re-solves the arms with robot-anchored wrist targets '
                         '(phase-2 fix, tools/x1_arm_fix.py); stock: phase-1 behaviour')
    ap.add_argument('--drop-head-s', type=float, default=0.2,
                    help='drop this much of each clip after resampling: the IK start pose is a different local '
                         'minimum than the rest of the clip (measured: a 30 rad/s joint-velocity spike in frames 0-1)')
    ap.add_argument('--reject-over-limit', type=float, default=0.35,
                    help='drop a clip whose worst joint exceeds its limit by more than this (rad)')
    ap.add_argument('--reject-root-z', type=float, default=0.45, help='drop a clip whose mean root height is below this')
    ap.add_argument('--reject-tilt', type=float, default=0.35, help='drop a clip whose max 1-R_zz exceeds this')
    ap.add_argument('--reject-qd', type=float, default=25.0, help='drop a clip whose max |joint velocity| exceeds this')
    args = ap.parse_args()
    use_ik_config(args.ik_config)

    files = sorted(glob.glob(str(args.input / '**' / '*.npz'), recursive=True))
    files = [Path(f) for f in files if 'smplx' not in f and 'SMPLX' not in Path(f).name.upper()]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f'no clips under {args.input}')
    print(f'{len(files)} clips, fps_out {args.fps_out}')

    model = X1Model()
    state: dict = {}
    qpos_all, qvel_all, names, report, kept = [], [], [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            qpos, src_fps = retarget_clip(f, args.smplx, retargeter, state, args.ik_passes, args.arm_mode)
        except Exception as e:  # noqa: BLE001
            print(f'[{i + 1}/{len(files)}] SKIP {f.name}: {type(e).__name__}: {e}')
            report.append({'clip': str(f), 'error': f'{type(e).__name__}: {e}'})
            continue
        if len(qpos) < 5:
            report.append({'clip': str(f), 'error': f'too short ({len(qpos)} frames)'})
            continue
        # resample to 50 Hz (same code path as scripts/prepare_motions.py), keep the quat sign continuous
        pos = torch.tensor(qpos[:, :3])
        quat = torch.tensor(qpos[:, 3:7])
        quat = quat / quat.norm(dim=-1, keepdim=True)
        sign = torch.ones(len(quat), dtype=quat.dtype)
        dots = (quat[1:] * quat[:-1]).sum(-1)
        sign[1:] = torch.cumprod(torch.where(dots < 0, -1.0, 1.0).to(quat.dtype), 0)
        quat = quat * sign[:, None]
        dof = torch.tensor(qpos[:, 7:])
        pos, quat, dof = resample(pos, quat, dof, src_fps, args.fps_out)
        qp = torch.cat([pos, quat, dof], dim=-1).float().numpy()
        head = int(round(args.drop_head_s * args.fps_out))
        if len(qp) > head + 20:
            qp = qp[head:]
        # ground the clip on the resampled frames: the sole plane is 0.0428 m below the ankle_roll joint
        sole = model.sole_z(qp.astype(np.float64))
        shift = sole.min(axis=1) if args.ground == 'per-frame' else np.full(len(qp), sole.min())
        qp[:, 2] -= shift
        sole = sole - shift[:, None]
        lin, ang, dof_vel = velocities(torch.tensor(qp[:, :3], dtype=torch.float64),
                                       torch.tensor(qp[:, 3:7], dtype=torch.float64),
                                       torch.tensor(qp[:, 7:], dtype=torch.float64), 1.0 / args.fps_out)
        qv = torch.cat([lin, ang, dof_vel], dim=-1).float().numpy()
        m = metrics(qp, qv, model, sole, args.fps_out)
        m.update({'clip': f.stem.replace('_stageii', ''), 'path': str(f), 'src_fps': src_fps})
        report.append(m)
        ok = (m['joint_over_limit_max_rad'] <= args.reject_over_limit and m['sole_min'] > -0.02
              and m['root_z'][1] >= args.reject_root_z and m['tilt_max'] <= args.reject_tilt
              and m['qd_max'] <= args.reject_qd)
        print(f'[{i + 1}/{len(files)}] {m["clip"]:14s} {m["frames"]:5d}f {m["seconds"]:6.1f}s  '
              f'root_z {m["root_z"][1]:.3f}  sole {m["sole_min"]:+.4f}  '
              f'limit+{m["joint_over_limit_max_rad"]:.3f} ({m["joint_over_limit_worst"]})  '
              f'cadence {m["cadence_hz"]}  {"OK" if ok else "REJECT"}', flush=True)
        if not ok:
            continue
        qpos_all.append(qp)
        qvel_all.append(qv)
        names.append(m['clip'])
        kept.append(len(qp))

    if not qpos_all:
        raise SystemExit('no clip passed; see the report')
    lengths = np.array(kept, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, qpos=np.concatenate(qpos_all), qvel=np.concatenate(qvel_all), starts=starts,
             lengths=lengths, names=np.array(names), fps=np.array(args.fps_out))
    total = int(lengths.sum())
    meta = {
        'out': str(args.out),
        'out_sha256': sha256(args.out),
        'out_bytes': args.out.stat().st_size,
        'clips_kept': len(names),
        'frames': total,
        'hours': round(total / args.fps_out / 3600, 4),
        'fps': args.fps_out,
        'joint_order': list(x1.X1_JOINT_ORDER),
        'asset_sha256': sha256(x1.X1_XML),
        'pipeline': ('GMR smplx->x1 IK (motions/x1_ik_config.json) at the source fps, '
                     'prepare_motions.resample to 50 Hz, grounding on the resampled frames, '
                     'prepare_motions.velocities (np.gradient / quat) recomputed after grounding'),
        'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'foot_geoms': list(x1.X1_FOOT_GEOMS),
        'sole_plane_below_base_m': round(float(x1.X1_INIT_HEIGHT), 4),
        'command': ' '.join(sys.argv),
        'python': sys.executable,
        'elapsed_s': round(time.time() - t0, 1),
        'clips': report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(meta, indent=1))
    print(f'\nwrote {args.out}: {len(names)} clips, {total} frames, {meta["hours"]} h, '
          f'{meta["out_bytes"] / 1e6:.2f} MB; report {args.report}; {meta["elapsed_s"]} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
