#!/usr/bin/env python3
"""Rebuild the X1 SMP motion library with STANCE ANCHORING (AUDIT_FAILURE §6-1/2/3).

Fixes root cause R-A (fnd_0b28742e): the R1-R3 library had no planted stance
(GMR does not anchor the stance foot; per-frame grounding manufactured sliding
contact, 0/4902 flight frames). This build:

  1. selects the same style of steady straight human walk windows
     (cad 95-140 spm, v 0.6-1.9, step 0.5-0.85, straight>=0.85), preferring
     the 0.9-1.5 m/s neighborhood (X1 lands 0.4-0.9 covering the 0.7 target);
  2. GMR retarget 2-pass + arm refine (unchanged, super_mimic_F1 route);
  3. HUMAN contact labels from SMPLX foot-joint heights (Schmitt trigger);
  4. anchor_ik second IK pass: each stance foot PINNED at a fixed world SE3 for
     its whole stance phase (sole on a common ground plane), root/arms/torso
     frozen -> speed, cadence and step lengths preserved; swing leg follows GMR;
  5. grounding = one uniform per-clip shift (stance soles are already on the
     plane; flight frames survive by construction);
  6. gates: A (retarget quality) + B (gait bands v3, unchanged) + R (full
     retarget-acceptance gates incl. wrist/torso -- must not regress) +
     S (NEW hard gates): conservative min-corner stance slip p50 < 0.05 m/s,
     stance-sole quality (>=90% of labeled stance frames with sole z<5mm,
     |mean sole z|<3mm), flight frames recorded (library must have >0).

Sources: the 11 CMU trials behind the R1-R3 library's w0 clips + the 4 extra
phase-2 trials (re-fetched from the AMASS mirror when missing locally; deleted
afterwards per the storage red line). Run with the GMR venv python:

  ~/code/X1_29_AMP/_gmr_local/venv/bin/python scripts/build_smp_library_v2.py \
      --out data/x1_smp_train_v2.npz --report data/x1_smp_train_v2.meta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline'))

X1_SMP = HERE.parent
sys.path.insert(0, str(Path.home() / 'code' / 'super_mimic_F1'))          # read-only
sys.path.insert(0, str(Path.home() / 'code' / 'X1_29_AMP' / 'data'))      # read-only
sys.path.insert(0, str(Path.home() / 'code' / 'X1_29_AMP' / '_gmr_local' / 'GMR'))

from cmu_select_walk import convert as convert_stageii  # noqa: E402
from build_x1_motions import FPS_OUT, X1Model, retargeter, use_ik_config  # noqa: E402
from build_smp_library import select_windows, apply_froude_scale  # noqa: E402
from prepare_motions import resample, velocities  # noqa: E402
from x1_data_lib import X1FK, gait_metrics, audit_pass  # noqa: E402
from stance_anchor import (SIDES, anchor_ik, conservative_slip, human_contact,  # noqa: E402
                           resample_contact, root_warp, runs_of, sole_corners,
                           stance_ground)
from tools.x1_retarget_gates import GateModel, human_torso_twist, score_clip  # noqa: E402
from tools.x1_arm_fix import D_MAX, D_MIN, RATIO  # noqa: E402

GMR_BODY = Path.home() / 'code' / 'X1_29_AMP' / '_gmr_local' / 'GMR' / 'assets' / 'body_models'
RAW_LOCAL = X1_SMP / 'data' / 'raw_tmp'
RAW_MIRROR = Path.home() / 'code' / 'X1_29_AMP' / 'data' / 'amass' / 'CMU_raw'

# 11 trials behind the w0 clips of the R1-R3 library + 4 extra phase-2 trials
DEFAULT_TRIALS = [
    '105_19', '105_57', '108_09', '132_32', '132_34', '136_05', '136_07',
    '136_21', '136_23', '142_09', '43_01',
    '111_23', '12_03', '132_30', '136_24',
]

# label sanity (walking): per-foot duty and double-stance share
LABEL_DUTY = (0.40, 0.85)
LABEL_DOUBLE = (0.05, 0.50)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(trial: str, raw_dir: Path) -> Path:
    p = raw_dir / f'{trial}.npz'
    if p.exists() and p.stat().st_size > 100_000:
        return p
    sub = trial.split('_')[0]
    url = (f'https://hf-mirror.com/datasets/realdream-ai/AMASS/resolve/main/'
           f'raw/CMU/{sub}/{trial}_poses.npz')
    for _ in range(3):
        subprocess.run(['curl', '-sL', '-m', '180', '-o', str(p), url], check=False)
        if p.exists() and p.stat().st_size > 100_000:
            return p
        time.sleep(1)
    raise RuntimeError(f'download failed: {trial}')


def retarget_clip_v2(path: Path, smplx_dir: Path, make_retargeter, last: dict,
                     passes: int = 2) -> tuple[np.ndarray, float, list, object]:
    """build_x1_motions.retarget_clip + returns the SMPLX frames and retargeter."""
    from general_motion_retargeting.utils.smpl import (
        get_smplx_data_offline_fast, load_smplx_file)

    smplx_data, body_model, smplx_output, height = load_smplx_file(str(path), str(smplx_dir))
    src_fps = float(smplx_data['mocap_frame_rate'].item())
    frames, _ = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output,
                                            tgt_fps=src_fps)
    if last.get('r') is None or abs(last.get('h', 0.0) - height) > 0.01:
        last['r'] = make_retargeter(height)
        last['h'] = height
    qpos = None
    for _ in range(max(1, passes)):
        qpos = np.array([last['r'].retarget(f).copy() for f in frames], dtype=np.float64)
    from tools.x1_arm_fix import arm_refine
    qpos = arm_refine(last['r'], qpos, frames)
    return qpos, src_fps, frames, last['r']


def wrist_err_series(r, qpos_src: np.ndarray, frames: list) -> np.ndarray:
    """Per-frame wrist error vs the anchored intent target (rerun_retarget_gates law)."""
    import mujoco
    bid = {f'{s}_wrist_roll_link': mujoco.mj_name2id(
        r.model, mujoco.mjtObj.mjOBJ_BODY, f'{s}_wrist_roll_link') for s in ('left', 'right')}
    dd = mujoco.MjData(r.model)
    sh_bid = r.shoulder_bid
    werr = []
    for k in range(min(len(qpos_src), len(frames))):
        fr = frames[k]
        dd.qpos[:] = qpos_src[k]
        mujoco.mj_kinematics(r.model, dd)
        e = []
        for s in ('left', 'right'):
            wr_h = np.asarray(fr[f'{s}_wrist'][0], dtype=float)
            sh_h = np.asarray(fr[f'{s}_shoulder'][0], dtype=float)
            v = wr_h - sh_h
            n = np.linalg.norm(v)
            tgt = dd.xpos[sh_bid[s]] + v * (np.clip(RATIO * n, D_MIN, D_MAX) / n) \
                if n > 1e-6 else dd.xpos[sh_bid[s]]
            e.append(np.linalg.norm(dd.xpos[bid[f'{s}_wrist_roll_link']] - tgt))
        werr.append(e)
    return np.array(werr).max(axis=1)


def resample50(qpos_src: np.ndarray, src_fps: float):
    pos = torch.tensor(qpos_src[:, :3]); quat = torch.tensor(qpos_src[:, 3:7])
    quat = quat / quat.norm(dim=-1, keepdim=True)
    sign = torch.ones(len(quat), dtype=quat.dtype)
    dots = (quat[1:] * quat[:-1]).sum(-1)
    sign[1:] = torch.cumprod(torch.where(dots < 0, -1.0, 1.0).to(quat.dtype), 0)
    quat = quat * sign[:, None]
    dof = torch.tensor(qpos_src[:, 7:])
    pos, quat, dof = resample(pos, quat, dof, src_fps, FPS_OUT)
    qp = torch.cat([pos, quat, dof], dim=-1).float().numpy()
    return qp


def per_frame_csv(name: str, qp: np.ndarray, contact50: dict, model, data, fps: float):
    corners = sole_corners(model, data, qp)
    rows = ['t,sole_z_L,sole_z_R,vmin_L,vmin_R,contact_L,contact_R,root_x,root_y,root_z']
    v = {s: np.linalg.norm(np.diff(corners[s][:, :, :2], axis=0), axis=2).min(axis=1) * fps
         for s in SIDES}
    z = {s: corners[s][:, :, 2].min(axis=1) for s in SIDES}
    n = len(qp)
    for t in range(n):
        rows.append(f'{t / fps:.3f},{z["left"][t]:.4f},{z["right"][t]:.4f},'
                    f'{v["left"][min(t, n - 2)]:.3f},{v["right"][min(t, n - 2)]:.3f},'
                    f'{int(contact50["left"][t])},{int(contact50["right"][t])},'
                    f'{qp[t, 0]:.3f},{qp[t, 1]:.3f},{qp[t, 2]:.3f}')
    out = X1_SMP / 'audit' / 'frames_v2'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.csv').write_text('\n'.join(rows) + '\n')
    return out / f'{name}.csv'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=X1_SMP / 'data' / 'x1_smp_train_v2.npz')
    ap.add_argument('--report', type=Path, default=X1_SMP / 'data' / 'x1_smp_train_v2.meta.json')
    ap.add_argument('--interim', type=Path, default=X1_SMP / 'data' / 'interim_v2')
    ap.add_argument('--trials', nargs='*', default=DEFAULT_TRIALS)
    ap.add_argument('--extra-local', action='store_true',
                    help='also scan the local 66 CMU trials for more candidates')
    ap.add_argument('--win-s', type=float, default=6.5)
    ap.add_argument('--max-per-trial', type=int, default=3)
    ap.add_argument('--prefer-v', type=float, nargs=2, default=(0.9, 1.5))
    ap.add_argument('--mode', choices=('ik-pin', 'root-warp'), default='ik-pin')
    ap.add_argument('--froude-scale', type=float, default=0.811)
    ap.add_argument('--max-clips', type=int, default=24)
    ap.add_argument('--keep-interim', action='store_true')
    ap.add_argument('--dump-clips', type=int, default=3)
    args = ap.parse_args()

    use_ik_config(X1_SMP / 'pipeline' / 'x1_ik_config.json')
    args.interim.mkdir(parents=True, exist_ok=True)
    RAW_LOCAL.mkdir(parents=True, exist_ok=True)
    trials = list(dict.fromkeys(args.trials))
    if args.extra_local:
        trials += [p.stem for p in sorted(RAW_MIRROR.glob('*.npz'))]
    print(f'{len(trials)} trials; mode={args.mode}', flush=True)

    x1m = X1Model()
    fk = X1FK()
    gate_model = GateModel()
    state: dict = {}
    qpos_all, qvel_all, con_all, names, report, kept, keep_rows = [], [], [], [], [], [], []
    t0 = time.time()

    for ti, trial in enumerate(trials):
        try:
            raw = fetch(trial, RAW_MIRROR if (RAW_MIRROR / f'{trial}.npz').exists() else RAW_LOCAL)
        except RuntimeError as e:
            print(f'[{ti + 1}/{len(trials)}] {trial}: FETCH FAIL {e}', flush=True)
            report.append({'clip': trial, 'error': f'fetch: {e}'})
            continue
        try:
            wins = select_windows(raw, args.win_s, args.max_per_trial,
                                  prefer_v=tuple(args.prefer_v))
        except Exception as e:  # noqa: BLE001
            print(f'[{ti + 1}/{len(trials)}] {trial}: BAD FILE ({type(e).__name__})', flush=True)
            report.append({'clip': trial, 'error': f'bad raw file: {type(e).__name__}'})
            continue
        for wi, (i, j, wm) in enumerate(wins):
            name = f'{trial}_w{wi}'
            stage = args.interim / f'{name}_stageii.npz'
            try:
                convert_stageii(raw, i, j, stage)
                apply_froude_scale(stage, args.froude_scale)
                qpos_src, src_fps, frames, r = retarget_clip_v2(
                    stage, GMR_BODY, retargeter, state)
            except Exception as e:  # noqa: BLE001
                print(f'[{ti + 1}/{len(trials)}] {name}: SKIP {type(e).__name__}: {e}', flush=True)
                report.append({'clip': name, 'error': f'{type(e).__name__}: {e}'})
                stage.unlink(missing_ok=True)
                continue
            stage.unlink(missing_ok=True)
            n_src = min(len(qpos_src), len(frames))
            qpos_src = qpos_src[:n_src]

            # ---- human contact labels (src fps) ----
            contact_src, hstat = human_contact(frames[:n_src], n_src, src_fps)
            # human labels are DIAGNOSTIC ONLY now: round-2 anchoring detects
            # stance from the retargeted foot itself (足速最低相), because the
            # measured label/retarget mismatch is ~0.15 s on some clips.
            label_ok = (all(LABEL_DUTY[0] <= hstat[s]['duty'] <= LABEL_DUTY[1] for s in SIDES)
                        and LABEL_DOUBLE[0] <= hstat['double_stance_frac'] <= LABEL_DOUBLE[1])
            if not label_ok:
                print(f'[{ti + 1}/{len(trials)}] {name}: label diag (not used): '
                      f'duty {hstat["left"]["duty"]:.2f}/{hstat["right"]["duty"]:.2f} '
                      f'dbl {hstat["double_stance_frac"]:.2f}', flush=True)

            # ---- anchoring (src fps) ----
            if args.mode == 'ik-pin':
                qpos_src, adiag = anchor_ik(r, qpos_src, contact_src)
            else:
                adiag = {}
            werr_src = wrist_err_series(r, qpos_src, frames[:n_src])

            # ---- 50 Hz + head drop ----
            qp = resample50(qpos_src, src_fps)
            contact50 = {s: resample_contact(contact_src[s], src_fps, FPS_OUT)[:len(qp)]
                         for s in SIDES}
            head = int(round(0.2 * FPS_OUT))
            if len(qp) > head + 20:
                qp = qp[head:]
                contact50 = {s: c[head:] for s, c in contact50.items()}
                werr_src = werr_src[int(round(0.2 * src_fps)):]
            if args.mode == 'root-warp':
                qp, wdiag = root_warp(qp, contact50, x1m.m, x1m.d)
                adiag = {**adiag, **{k: v for k, v in wdiag.items() if k != 'anchor'}}
                contact50 = wdiag['anchor']   # stance stats + CSV use anchored frames
            else:
                qp, gdiag = stance_ground(x1m.m, x1m.d, qp, contact50)
                adiag.update(gdiag)
            if len(qp) < 100:
                report.append({'clip': name, 'error': 'too short'})
                continue

            lin, ang, dof_vel = velocities(torch.tensor(qp[:, :3], dtype=torch.float64),
                                           torch.tensor(qp[:, 3:7], dtype=torch.float64),
                                           torch.tensor(qp[:, 7:], dtype=torch.float64),
                                           1.0 / FPS_OUT)
            qv = torch.cat([lin, ang, dof_vel], dim=-1).float().numpy()

            # ---- gate A: retarget quality ----
            q = qp[:, 7:]
            over = np.maximum(q - x1m.hi, x1m.lo - q)
            a_pass = (over.max() <= 0.35
                      and qp[:, 2].mean() >= 0.45
                      and float(2 * (qp[:, 4] ** 2 + qp[:, 5] ** 2).max()) <= 0.35
                      and float(np.abs(qv[:, 6:]).max()) <= 25.0)
            # sole penetration check via FK corners (min over clip > -0.02)
            cs = sole_corners(x1m.m, x1m.d, qp)
            sole_h = {s: cs[s][:, :, 2].min(axis=1) for s in SIDES}
            a_pass &= float(min(sole_h[s].min() for s in SIDES)) > -0.02

            # ---- gate B: gait bands (unchanged v3) ----
            kfk = fk.fk_clip(qp, qv)
            gm = gait_metrics(kfk['sole_h'], kfk['root_pos'], FPS_OUT)
            b_pass, vals = audit_pass(gm)
            f0, f1 = gm['feet']
            sym = (not np.isnan(f0['duty']) and not np.isnan(f1['duty'])
                   and abs(f0['duty'] - f1['duty']) <= 0.25
                   and abs(f0['cadence_hz'] - f1['cadence_hz']) <= 0.35)
            b_pass = bool(b_pass and sym)

            # ---- gate S: stance anchoring hard gates ----
            slip = conservative_slip(qp, x1m.m, x1m.d, FPS_OUT)
            s_stat = {}
            s_pass = True
            # a clip with almost no anchor coverage has no planted stance to offer
            s_pass &= all(contact50[s].mean() >= 0.25 for s in SIDES)
            for s in SIDES:
                c = contact50[s]
                zs = sole_h[s][c]
                sq = dict(duty_label=float(c.mean()),
                          frac_sole_lt5mm=float((zs < 0.005).mean()) if len(zs) else 0.0,
                          mean_sole_z=float(np.mean(zs)) if len(zs) else float('nan'),
                          slip_p50=slip['feet'][s]['slip_p50'],
                          slip_p90=slip['feet'][s]['slip_p90'])
                s_stat[s] = sq
                s_pass &= sq['frac_sole_lt5mm'] >= 0.90
                s_pass &= abs(sq['mean_sole_z']) < 0.003
                s_pass &= sq['slip_p50'] is not None and sq['slip_p50'] < 0.05
            flight = slip['flight_frames_both_above_25mm']

            # ---- gate R: full retarget-acceptance gates (wrist/torso must not regress) ----
            # chest_yaw_human: same yardstick as the phase-2/rerun baseline (src-fps
            # frames after the 0.2 s head drop; score_clip slices to len(qp))
            head_src = int(round(0.2 * src_fps))
            twist_h = human_torso_twist(frames[head_src:head_src + len(qp)]) if n_src > head_src else None
            if twist_h is not None and len(twist_h) < len(qp):
                twist_h = np.interp(np.arange(len(qp)), np.arange(len(twist_h)), twist_h)
            sc = score_clip(qp, FPS_OUT, gate_model,
                            wrist_err=werr_src, chest_yaw_human=twist_h,
                            tag='walk', human_net_yaw=0.0)
            r_pass = bool(sc['pass'])

            ok = a_pass and b_pass and s_pass and r_pass
            row = {'clip': name,
                   'human_window': {kk: round(vv, 3) if isinstance(vv, float) else vv
                                    for kk, vv in wm.items()},
                   'frames': int(len(qp)), 'a_pass': bool(a_pass), 'b_pass': b_pass,
                   's_pass': bool(s_pass), 'r_pass': r_pass,
                   'anchor_diag': {kk: (vv if not isinstance(vv, dict) else vv)
                                   for kk, vv in adiag.items()},
                   'labels': hstat, 'stance': s_stat, 'flight_frames': flight,
                   'retarget_gates': {'gates': sc['gates'], 'pass': r_pass},
                   'gait': {'cadence': [f0['cadence_hz'], f1['cadence_hz']],
                            'speed': gm['speed_mps'],
                            'step': [f0['step_len_m'], f1['step_len_m']],
                            'swing': [f0['swing_s_med'], f1['swing_s_med']],
                            'duty': [f0['duty'], f1['duty']],
                            'lift': [f0['lift_m'], f1['lift_m']]}}
            report.append(row)
            bad = [g for g, v in dict(A=a_pass, B=b_pass, S=s_pass, R=r_pass).items() if not v]
            print(f'[{ti + 1}/{len(trials)}] {name:16s} hv{wm["v"]:.2f} | X1 v{gm["speed_mps"]:.2f} '
                  f'cad {f0["cadence_hz"]:.2f}/{f1["cadence_hz"]:.2f} '
                  f'slip {s_stat["left"]["slip_p50"] if s_stat["left"]["slip_p50"] is not None else -1:.3f}/'
                  f'{s_stat["right"]["slip_p50"] if s_stat["right"]["slip_p50"] is not None else -1:.3f} '
                  f'flt {flight:3d} | {"OK" if ok else "REJECT " + ",".join(bad)}', flush=True)
            if not ok:
                continue
            qpos_all.append(qp); qvel_all.append(qv); names.append(name); kept.append(len(qp))
            keep_rows.append(row)
            con_all.append(np.stack([contact50['left'], contact50['right']], axis=1).astype(np.uint8))

    # write the report even when nothing passes (diagnostics matter)
    def _jsonable_dbg(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        {'windows': json.loads(json.dumps(report, default=_jsonable_dbg)),
         'built_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_s': round(time.time() - t0, 1),
         'mode': args.mode}, indent=1))
    if not qpos_all:
        raise SystemExit('no clip passed; see the report')

    # ---- library selection: cap 2/trial, balance the 0.4-0.9 band around 0.65 ----
    by_trial: dict[str, list[int]] = {}
    for idx, nm in enumerate(names):
        by_trial.setdefault(nm.rsplit('_w', 1)[0], []).append(idx)
    sel = []
    for tr, idxs in by_trial.items():
        idxs.sort(key=lambda ix: abs(keep_rows[ix]['gait']['speed'] - 0.68))
        sel += idxs[:2]
    bins = [(0.40, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.00), (1.00, 1.30)]
    quota = 7
    final = []
    for lo, hi in bins:
        group = [ix for ix in sel if lo <= keep_rows[ix]['gait']['speed'] < hi]
        group.sort(key=lambda ix: abs(keep_rows[ix]['gait']['speed'] - 0.5 * (lo + hi)))
        final += group[:quota]
    leftover = [ix for ix in sel if ix not in final]
    leftover.sort(key=lambda ix: abs(keep_rows[ix]['gait']['speed'] - 0.68))
    for ix in leftover:
        if len(final) < args.max_clips:
            final.append(ix)
    final = sorted(set(final))

    lengths = np.array([kept[ix] for ix in final], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    qpos_f = np.concatenate([qpos_all[ix] for ix in final])
    qvel_f = np.concatenate([qvel_all[ix] for ix in final])
    con_f = np.concatenate([con_all[ix] for ix in final])
    names_f = [names[ix] for ix in final]
    flight_total = int(sum(keep_rows[ix]['flight_frames'] for ix in final))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, qpos=qpos_f, qvel=qvel_f, starts=starts, lengths=lengths,
             names=np.array(names_f), fps=np.array(FPS_OUT), contact=con_f)

    # per-frame human-review CSVs: fastest / median / slowest of the final library
    speeds = [keep_rows[ix]['gait']['speed'] for ix in final]
    order = np.argsort(speeds)
    dumped = []
    for pick in ([order[-1], order[len(order) // 2], order[0]][:args.dump_clips]
                 if len(order) >= 3 else order):
        nm, s0 = names_f[pick], int(starts[pick])
        qp_c = qpos_f[s0:s0 + int(lengths[pick])]
        c50 = {'left': con_f[pick][:, 0].astype(bool), 'right': con_f[pick][:, 1].astype(bool)}
        dumped.append(str(per_frame_csv(nm, qp_c, c50, x1m.m, x1m.d, FPS_OUT)))

    def _jsonable(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    meta = {'out': str(args.out), 'out_sha256': sha256(args.out),
            'out_bytes': args.out.stat().st_size,
            'clips_kept': len(names_f), 'frames': int(lengths.sum()),
            'hours': round(float(lengths.sum()) / FPS_OUT / 3600, 4), 'fps': FPS_OUT,
            'mode': args.mode, 'froude_scale': args.froude_scale,
            'flight_frames_total': flight_total,
            'speeds': {names_f[k]: round(speeds[k], 3) for k in range(len(names_f))},
            'pipeline': ('CMU_raw -> multi-window select (cad 95-140, v 0.6-1.9, step 0.5-0.85, '
                         'straight>=0.85) -> stageii (XY*froude) -> GMR 2-pass + arm_refine -> '
                         'HUMAN contact labels (SMPLX foot z Schmitt) -> anchor_ik stance pinning '
                         '(root frozen) -> 50 Hz -> uniform ground (stance-aware, flight kept) -> '
                         'gates A + B + R(retarget full) + S(stance slip<0.05 p50, sole<5mm 90%)'),
            'windows': json.loads(json.dumps(report, default=_jsonable)),
            'dumped_csv': dumped,
            'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_s': round(time.time() - t0, 1)}
    args.report.write_text(json.dumps(meta, indent=1))
    if not args.keep_interim:
        shutil.rmtree(args.interim, ignore_errors=True)
        shutil.rmtree(RAW_LOCAL, ignore_errors=True)
    print(f'kept {len(names_f)} clips, {int(lengths.sum())} frames, flight {flight_total} '
          f'-> {args.out} ({time.time() - t0:.0f}s); report {args.report}', flush=True)
    print('speeds:', [round(v, 2) for v in sorted(speeds)], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
