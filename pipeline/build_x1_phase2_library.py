"""Build the phase-2 X1 motion library (contract tasks 3+4): AMASS SMPL-H -> category
classification + steady-window selection -> stageii -> GMR retarget (arm fix) ->
grounding -> gates -> quota selection -> train/held-out npz.

Storage red line: raw SMPL-H npz under data/amass/CMU_raw and stageii npz under
data/amass/stageii are intermediates; only the final library npz + meta (+ held-out
npz) are kept. Delete the intermediates afterwards (see --clean).

  PYTHONPATH=. <gmr-venv-python> scripts/build_x1_phase2_library.py --report motions/x1_phase2.meta.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GMR_LOCAL = Path.home() / 'code' / 'X1_29_AMP' / '_gmr_local'
GMR_DIR = GMR_LOCAL / 'GMR'
sys.path.insert(0, str(GMR_DIR))
sys.path.insert(0, str(Path.home() / 'code' / 'X1_29_AMP' / 'data'))  # cmu_select_walk.convert

import torch  # noqa: E402
from cmu_select_walk import convert as convert_stageii  # noqa: E402

from scripts.build_x1_motions import (  # noqa: E402
    FPS_OUT,
    X1Model,
    retarget_clip,
    retargeter,
    use_ik_config,
)
from scripts.prepare_motions import resample, velocities  # noqa: E402
from tools.x1_retarget_gates import GateModel, human_torso_twist, score_clip  # noqa: E402

RAW_EXTRA = Path.home() / 'code' / 'X1_29_AMP' / 'data' / 'amass' / 'CMU_raw'  # phase-1 walk raws (read-only)
RAW_LOCAL = ROOT / 'data' / 'amass' / 'CMU_raw'
STAGE = ROOT / 'data' / 'amass' / 'stageii'
CACHE = ROOT / 'data' / 'phase2_cache'  # overridden by --cache-dir (shard mode)

# subject family -> coarse activity tag for quotas and held-out design
FAMILY = {
    '07': 'walk', '08': 'walk', '16': 'walk', '35': 'walk', '37': 'walk', '38': 'walk',
    '49': 'walk', '82': 'walk', '91': 'walk', '103': 'walk',
    '02': 'walk', '06': 'walk', '17': 'walk', '13': 'walk',
    '09': 'run', '10': 'run',
    '12': 'varied', '15': 'varied', '39': 'varied', '76': 'walk', '77': 'varied',
    '83': 'varied', '84': 'varied', '85': 'varied', '86': 'varied', '87': 'varied',
    '88': 'varied', '89': 'varied', '93': 'varied', '94': 'varied', '102': 'varied',
    '104': 'varied', '105': 'varied', '106': 'varied', '107': 'varied', '108': 'varied',
    '111': 'varied', '113': 'varied', '114': 'varied', '115': 'varied', '118': 'varied',
    '120': 'walk', '122': 'varied', '123': 'varied', '124': 'varied', '125': 'varied',
    '126': 'varied', '127': 'varied', '128': 'varied', '132': 'varied', '133': 'varied',
    '139': 'varied', '140': 'varied', '141': 'varied', '142': 'varied', '144': 'varied',
    '05': 'dance', '60': 'dance', '131': 'dance', '135': 'dance', '136': 'dance',
    '137': 'dance', '143': 'dance',
    '90': 'varied', '138': 'varied',
}
QUOTA = {'walk': 200, 'run': 90, 'dance': 50, 'varied': 40, 'turn': 60}  # run/jog mostly fails the wrist gates (physical boundary, see report)
HELDOUT_SUBJECTS = {'91', '137'}  # whole-family hold-out: 91 = walk, 137 = dance
HELDOUT_EXTRA = {'09_03', '09_06', '09_09', '137_05', '137_10', '137_20', '136_03', '136_12', '143_02', '143_09', '90_05', '90_14'}


def yaw_from_aa(aa: np.ndarray) -> np.ndarray:
    ang = np.linalg.norm(aa, axis=1)
    axis = np.where(ang[:, None] > 1e-8, aa / np.maximum(ang[:, None], 1e-8), 0.0)
    axis = np.where(ang[:, None] > 1e-8, axis, np.array([0.0, 0.0, 1.0]))
    k = axis
    K = np.stack([np.zeros_like(k[:, 0]), -k[:, 2], k[:, 1],
                  k[:, 2], np.zeros_like(k[:, 0]), -k[:, 0],
                  -k[:, 1], k[:, 0], np.zeros_like(k[:, 0])], axis=1).reshape(-1, 3, 3)
    I = np.eye(3)[None]
    A = np.zeros_like(K)
    A[:, 0, 1], A[:, 0, 2], A[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    A[:, 1, 0], A[:, 2, 0], A[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    sin, cos = np.sin(ang), 1 - np.cos(ang)
    K2 = np.einsum('nij,njk->nik', K, K)
    R = I + sin[:, None, None] * K + cos[:, None, None] * K2
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.unwrap(yaw)


def window_features(poses: np.ndarray, trans: np.ndarray, fps: float, i: int, j: int) -> dict:
    """Kinematic features of frames [i, j) from SMPL-H root channels only."""
    xx, yy = trans[i:j, 0], trans[i:j, 1]
    d = np.hypot(np.diff(xx), np.diff(yy))
    travel, net = d.sum(), math.hypot(xx[-1] - xx[0], yy[0])
    if travel < 0.5 and j - i > fps:  # in-place motion (dance)
        v = 0.0
    else:
        v = float(np.median(d) * fps) if len(d) else 0.0
    z = trans[i:j, 2]
    yaw = yaw_from_aa(poses[i:j, :3])
    return {'v': v, 'straight': net / max(travel, 1e-6), 'yaw_net': float(yaw[-1] - yaw[0]),
            'z_rom': float(z.max() - z.min()), 'bob': float(np.std(z)), 'dur': (j - i) / fps}


def pick_window(poses, trans, fps, tag: str, want_s: float = 8.0):
    """Best steady window for the family tag. Returns (i, j, features) or None."""
    n = len(poses)
    w = int(min(want_s * fps, n))
    if w < int(2.0 * fps):
        return None
    best = None
    for i in range(0, n - w + 1, max(1, int(0.5 * fps))):
        j = i + w
        f = window_features(poses, trans, fps, i, j)
        if tag == 'walk':
            score = -abs(f['v'] - 1.2) - 2 * (1 - min(f['straight'], 1)) - abs(min(f['yaw_net'], 0))
        elif tag == 'run':
            score = -abs(f['v'] - 2.6) - (1 - min(f['straight'], 1))
        elif tag == 'dance':
            score = f['bob'] * 10 - 0.1 * f['v']  # rhythmic vertical motion, low travel
        else:  # varied: prefer some travel and any content
            score = -abs(f['v'] - 1.0)
        if best is None or score > best[0]:
            best = (score, i, j, f)
    return best


def refine_tag(feat: dict, arm_rom: float) -> str:
    """Fine tag from measured features (arm_rom = std of |wrist-shoulder|, human)."""
    if feat['v'] > 2.2:
        return 'run'
    if feat['v'] > 1.6:
        return 'jog'
    if abs(feat['yaw_net']) > 1.0 and feat['v'] < 1.6:
        return 'turn'
    if arm_rom > 0.05 and feat['v'] < 0.8:
        return 'dance'
    return 'walk'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-train', type=Path, default=ROOT / 'motions' / 'x1_phase2_train.npz')
    ap.add_argument('--out-heldout', type=Path, default=ROOT / 'motions' / 'x1_phase2_heldout.npz')
    ap.add_argument('--report', type=Path, default=ROOT / 'motions' / 'x1_phase2.meta.json')
    ap.add_argument('--limit', type=int, default=0, help='debug: only first N candidates')
    ap.add_argument('--cache-dir', type=Path, default=ROOT / 'data' / 'phase2_cache')
    ap.add_argument('--no-assemble', action='store_true', help='score only (shard mode); no library output')
    ap.add_argument('--only-new', action='store_true', help='skip candidates whose cache npz already exists')
    ap.add_argument('--shard', default='', help="'i/n': process every n-th candidate starting at i (parallel shards)")
    ap.add_argument('--clean', action='store_true',
                    help='after the libraries are written and verified non-empty, delete the raw SMPL-H '
                         'files (data/amass/CMU_raw), the stageii intermediates and the per-clip cache '
                         '(storage red line: only train/held-out npz + meta are kept)')
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()
    cache_dir: Path = args.cache_dir
    use_ik_config(ROOT / 'motions' / 'x1_ik_config.json')

    STAGE.mkdir(parents=True, exist_ok=True)
    raws = []
    for d in (RAW_LOCAL, RAW_EXTRA):
        for f in sorted(glob.glob(str(d / '*.npz'))):
            name = Path(f).stem.replace('_poses', '')
            sub = name.split('_')[0]
            if sub in FAMILY:
                raws.append((name, sub, Path(f)))
    raws = sorted(set(raws))
    if args.shard:
        i, n = (int(v) for v in args.shard.split('/'))
        raws = raws[i::n]
    if args.limit:
        raws = raws[: args.limit]
    print(f'{len(raws)} raw candidates', flush=True)

    model = X1Model()
    gate_model = GateModel()
    state: dict = {}
    t0 = time.time()
    rows = []
    for idx, (name, sub, path) in enumerate(raws):
        if args.only_new and (cache_dir / f'{name}.npz').exists():
            continue
        try:
            d = np.load(path, allow_pickle=True)
            poses = np.asarray(d['poses'], dtype=np.float32)
            trans = np.asarray(d['trans'], dtype=np.float32)
            fps = float(np.asarray(d['mocap_framerate']).item())
        except Exception as e:  # noqa: BLE001
            rows.append({'clip': name, 'error': f'raw load: {e}'})
            continue
        if poses.ndim != 2 or poses.shape[1] < 66 or len(poses) < 60:
            rows.append({'clip': name, 'error': 'raw too short/malformed'})
            continue
        tag = FAMILY[sub]
        win = pick_window(poses, trans, fps, tag)
        if win is None:
            rows.append({'clip': name, 'error': 'no window'})
            continue
        _, i, j, feat = win
        stage = STAGE / f'{name}_stageii.npz'
        try:
            convert_stageii(path, i, j, stage)
        except Exception as e:  # noqa: BLE001
            rows.append({'clip': name, 'error': f'stageii: {e}'})
            continue
        try:
            qpos, src_fps = retarget_clip(stage, GMR_DIR / 'assets' / 'body_models',
                                          retargeter, state, passes=2, arm_mode='refine')
        except Exception as e:  # noqa: BLE001
            rows.append({'clip': name, 'error': f'retarget: {type(e).__name__}: {e}'})
            stage.unlink(missing_ok=True)
            continue
        # per-frame wrist error vs the anchored target (pre-grounding, in-retarget frame)
        r = state['r']
        import mujoco

        bid = {f'{s}_wrist_roll_link': mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY,
                                                         f'{s}_wrist_roll_link') for s in ('left', 'right')}
        from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast, load_smplx_file

        smplx_data, body_model, smplx_out, h = load_smplx_file(str(stage), str(GMR_DIR / 'assets' / 'body_models'))
        frames, _ = get_smplx_data_offline_fast(smplx_data, body_model, smplx_out, tgt_fps=src_fps)
        dd = mujoco.MjData(r.model)
        sh_bid = r.shoulder_bid
        from tools.x1_arm_fix import D_MAX, D_MIN, RATIO

        werr = []
        arm_rom_h = []
        for k in range(min(len(qpos), len(frames))):
            fr = frames[k]
            dd.qpos[:] = qpos[k]
            mujoco.mj_kinematics(r.model, dd)
            e = []
            for s in ('left', 'right'):
                # anchored intent target: live shoulder FK + human arm direction at the
                # arm-length-preserving distance (the same law tools/x1_arm_fix.py solves for)
                wr_h = np.asarray(fr[f'{s}_wrist'][0], dtype=float)
                sh_h = np.asarray(fr[f'{s}_shoulder'][0], dtype=float)
                v = wr_h - sh_h
                n = np.linalg.norm(v)
                tgt = dd.xpos[sh_bid[s]] + v * (np.clip(RATIO * n, D_MIN, D_MAX) / n) if n > 1e-6 else dd.xpos[sh_bid[s]]
                e.append(np.linalg.norm(dd.xpos[bid[f'{s}_wrist_roll_link']] - tgt))
                arm_rom_h.append(n)
            werr.append(e)
        werr = np.array(werr).max(axis=1)
        arm_rom = float(np.std(arm_rom_h))

        # ground + resample to 50 Hz (phase-1 path)
        pos = torch.tensor(qpos[:, :3])
        quat = torch.tensor(qpos[:, 3:7])
        quat = quat / quat.norm(dim=-1, keepdim=True)
        sign = torch.ones(len(quat), dtype=quat.dtype)
        dots = (quat[1:] * quat[:-1]).sum(-1)
        sign[1:] = torch.cumprod(torch.where(dots < 0, -1.0, 1.0).to(quat.dtype), 0)
        quat = quat * sign[:, None]
        dof = torch.tensor(qpos[:, 7:])
        pos, quat, dof = resample(pos, quat, dof, src_fps, FPS_OUT)
        qp = torch.cat([pos, quat, dof], dim=-1).float().numpy()
        head = int(round(0.2 * FPS_OUT))
        if len(qp) > head + 20:
            qp = qp[head:]
            frames_50 = frames[int(round(0.2 * src_fps)):]
        else:
            frames_50 = frames[:len(qp)]
        sole = model.sole_z(qp.astype(np.float64))
        qp[:, 2] -= sole.min(axis=1)
        twist_h = human_torso_twist(frames_50[:len(qp)])
        qv = torch.cat(velocities(torch.tensor(qp[:, :3], dtype=torch.float64),
                                  torch.tensor(qp[:, 3:7], dtype=torch.float64),
                                  torch.tensor(qp[:, 7:], dtype=torch.float64),
                                  1.0 / FPS_OUT), dim=-1).float().numpy()
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_dir / f'{name}.npz', qpos=qp, qvel=qv)
        ftag = refine_tag(feat, arm_rom)
        sc = score_clip(qp, FPS_OUT, gate_model, wrist_err=werr, chest_yaw_human=twist_h,
                        tag=ftag, human_net_yaw=feat['yaw_net'])
        heldout = sub in HELDOUT_SUBJECTS or name in HELDOUT_EXTRA
        row = {'clip': name, 'subject': sub, 'family': tag, 'tag': ftag, 'heldout': heldout,
               'feat': {k: round(v, 3) for k, v in feat.items()}, 'arm_rom_h': round(arm_rom, 4),
               'wrist_err_p50': round(float(np.median(werr)), 4), 'score': sc}
        rows.append(row)
        stage.unlink(missing_ok=True)
        bad = [g for g, ok in sc['gates'].items() if not ok]
        print(f'[{idx + 1}/{len(raws)}] {name:8s} {ftag:5s} {"HO" if heldout else "  "} '
              f'v={feat["v"]:.2f} yaw={feat["yaw_net"]:+.1f} armROM={arm_rom:.3f} '
              f'werr50={row["wrist_err_p50"]:.3f} pass={sc["pass"]} '
              f'{bad} {[(g, sc["metrics"][g]) for g in bad]}', flush=True)

    if args.no_assemble:
        out_rows = ROOT / 'data' / f'phase2_rows_{args.cache_dir.name}.json'
        out_rows.write_text(json.dumps({'clips': rows}, indent=1, default=float))
        print(f'\nshard done: {sum(1 for r in rows if "error" not in r and r["score"]["pass"])} pass of '
              f'{len(rows)} scored; rows -> {out_rows}')
        return 0

    # merge rows from previous shard runs if present
    for extra in sorted((ROOT / 'data').glob('phase2_rows_*.json')):
        try:
            rows.extend(json.loads(extra.read_text())['clips'])
            print(f'merged rows from {extra.name}')
        except Exception as e:  # noqa: BLE001
            print(f'skip {extra.name}: {e}')

    # ---- assembly ----
    cache = cache_dir
    def take(rows_in, heldout_want: bool):
        sel, counts = [], {}
        for row in rows_in:
            if 'error' in row or not row['score']['pass'] or row['heldout'] != heldout_want:
                continue
            key = 'run' if row['tag'] in ('run', 'jog') else row['tag']
            if counts.get(key, 0) >= QUOTA.get(key, 0):
                continue
            counts[key] = counts.get(key, 0) + 1
            sel.append(row)
        return sel, counts

    train_sel, _ = take(rows, False)
    ho_sel, _ = take(rows, True)

    cache_dirs = [cache_dir] + sorted((ROOT / 'data').glob('phase2_cache_s*'))

    def write_library(sel, out: Path):
        qpos_all, qvel_all, names, kept = [], [], [], []
        for row in sel:
            c = next((d / f"{row['clip']}.npz" for d in cache_dirs if (d / f"{row['clip']}.npz").exists()), None)
            if c is None:
                continue
            d = np.load(c)
            qpos_all.append(d['qpos'])
            qvel_all.append(d['qvel'])
            names.append(f"{row['clip']}_{row['tag']}")
            kept.append(len(d['qpos']))
        if not qpos_all:
            return 0
        lengths = np.array(kept, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, qpos=np.concatenate(qpos_all), qvel=np.concatenate(qvel_all),
                 starts=starts, lengths=lengths, names=np.array(names), fps=np.array(FPS_OUT))
        return len(names)

    n_train = write_library(train_sel, args.out_train)
    n_ho = write_library(ho_sel, args.out_heldout)
    from allmimic.robots import x1_29dof as x1mod

    meta = {
        'out_train': str(args.out_train), 'n_train': n_train,
        'out_heldout': str(args.out_heldout), 'n_heldout': n_ho,
        'fps': FPS_OUT, 'joint_order': list(x1mod.X1_JOINT_ORDER),
        'pipeline': ('SMPL-H window (per-family criteria) -> stageii -> GMR retarget + arm refine '
                     '(tools/x1_arm_fix.py) -> ground per-frame -> 50 Hz resample -> tools/x1_retarget_gates.py'),
        'quota': QUOTA, 'heldout_subjects': sorted(HELDOUT_SUBJECTS), 'heldout_extra': sorted(HELDOUT_EXTRA),
        'gate_summary': {},
        'elapsed_s': round(time.time() - t0, 1),
        'clips': rows,
    }
    ok_rows = [r for r in rows if 'error' not in r]
    gnames = list(ok_rows[0]['score']['gates']) if ok_rows else []
    meta['gate_summary'] = {g: {
        'pass_rate': round(sum(1 for r in ok_rows if r['score']['gates'][g]) / max(len(ok_rows), 1), 3),
        'limit': g_limit(g)} for g in gnames}
    meta['tag_counts_all'] = {}
    for r in ok_rows:
        meta['tag_counts_all'][r['tag']] = meta['tag_counts_all'].get(r['tag'], 0) + 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(meta, indent=1, default=float))
    print(f"\ntrain {n_train} clips, heldout {n_ho} clips; report {args.report}; {meta['elapsed_s']} s")
    if args.clean and n_train > 0:
        import shutil

        for d in (ROOT / 'data' / 'amass' / 'CMU_raw', ROOT / 'data' / 'amass' / 'stageii', cache_dir,
                  *(sorted((ROOT / 'data').glob('phase2_cache_s*')))):
            if d.exists():
                shutil.rmtree(d)
                print(f'cleaned {d}')
    return 0


def g_limit(g: str):
    from tools.x1_retarget_gates import GATES

    return GATES[g][0]


if __name__ == '__main__':
    raise SystemExit(main())
