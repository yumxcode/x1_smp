#!/usr/bin/env python3
"""Build the X1 SMP motion library: AMASS CMU -> focused 1 m/s windows -> GMR
retarget (arm-refined) -> grounding -> 50 Hz -> preregistered acceptance.

Run with the GMR venv python (general_motion_retargeting importable) and
PYTHONPATH containing super_mimic_F1 (read-only: allmimic robots, tools.x1_arm_fix,
scripts.prepare_motions) and X1_29_AMP/data (cmu_select_walk metrics):

  ~/code/X1_29_AMP/_gmr_local/venv/bin/python scripts/build_smp_library.py \
      --raw ~/code/X1_29_AMP/data/amass/CMU_raw \
      --smplx ~/code/X1_29_AMP/data/amass \
      --out data/x1_smp_train.npz

Pipeline (docs/SMP_PLAN.md §4):
  1. window selection: all qualifying steady 8 s straight windows per trial
     (human-scale gate: cadence 95-140 spm, 0.7-1.9 m/s, step 0.50-0.85 m,
      straight >= 0.85), preferring the 1 m/s neighborhood, non-overlapping,
     max --max-per-trial per trial;
  2. stageii conversion (cmu_select_walk.convert) to data/interim (deleted at exit);
  3. retarget_clip: GMR 2 IK passes + arm refine (tools/x1_arm_fix) -- the
     super_mimic_F1 phase-2 route, unchanged;
  4. resample 50 Hz, per-frame grounding, velocities;
  5. acceptance A (build_x1_motions reject gates): joint over-limit <= 0.35 rad,
     sole_min > -0.02, root_z mean >= 0.45, tilt <= 0.35, qd_max <= 25;
  6. acceptance B (preregistered gait audit, scripts/x1_data_lib): both feet
     resolved, cadence/foot 0.55-1.2 Hz, speed 0.4-1.3 m/s, step 0.35-0.65 m,
     swing >= 0.24 s, duty <= 0.80, lift >= 25 mm, feet symmetric
     (|duty_L-duty_R| <= 0.25, |cad_L-cad_R| <= 0.35);
  7. storage red line: interim stageii files are deleted; only the final npz +
     meta json are kept.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline'))

X1_SMP = HERE.parent
sys.path.insert(0, str(Path.home() / 'code' / 'super_mimic_F1'))          # allmimic, tools.x1_arm_fix (read-only)
sys.path.insert(0, str(Path.home() / 'code' / 'X1_29_AMP' / 'data'))      # cmu_select_walk (read-only)
sys.path.insert(0, str(Path.home() / 'code' / 'X1_29_AMP' / '_gmr_local' / 'GMR'))

from cmu_select_walk import (TGT, convert as convert_stageii, velocity_heading,  # noqa: E402
                             window_metrics)
from build_x1_motions import (FPS_OUT, X1Model, retarget_clip, retargeter, use_ik_config)  # noqa: E402
from prepare_motions import resample, velocities  # noqa: E402
from x1_data_lib import X1FK, gait_metrics, audit_pass  # noqa: E402


def apply_froude_scale(stageii: Path, scale: float) -> None:
    """Scale the horizontal trajectory of a stageii clip by `scale` (Froude
    similarity for a robot whose leg length is `scale^2` of the human's).

    Root cause (measured 2026-11-03): the ik config scales pelvis height by
    0.6578 but leaves the horizontal trajectory at human scale, so the X1
    walks at the human's 0.85-1.45 m/s with 0.66x leg length -- above its
    geometric feasibility band, and GMR's foot IK responds by dragging one
    foot (cadence unresolved, duty 0.9+; every walk-family clip failed the
    gait audit). Scaling XY by sqrt(0.6578)=0.811 keeps cadence, shortens
    step length and speed by 0.811, and lands the X1 speeds (human
    0.95-1.7 -> X1 0.77-1.38) inside the preregistered 0.4-1.3 band.
    """
    if abs(scale - 1.0) < 1e-6:
        return
    d = dict(np.load(stageii, allow_pickle=True))
    t = np.asarray(d['trans'], dtype=np.float32).copy()
    t[:, :2] *= scale
    d['trans'] = t
    np.savez(stageii, **d)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def select_windows(npz_path: Path, want_dur: float = 8.0, max_per_trial: int = 2,
                   prefer_v: tuple[float, float] = (0.9, 1.5)) -> list[tuple[int, int, dict]]:
    """All qualifying non-overlapping steady windows, best-first."""
    d = np.load(npz_path, allow_pickle=True)
    trans = np.asarray(d['trans'], dtype=np.float32)
    fps = float(np.asarray(d['mocap_framerate']).item())
    h = velocity_heading(trans, fps)
    hy = -np.sin(h) * trans[:, 0] + np.cos(h) * trans[:, 1]
    wx, wy, z = trans[:, 0], trans[:, 1], trans[:, 2]
    N = len(wx)
    w = int(min(want_dur * fps, N))
    cands = []
    for i in range(0, N - w + 1, max(1, int(0.5 * fps))):
        j = i + w
        m = window_metrics(wx, wy, hy, z, fps, i, j)
        if m is None:
            continue
        if not (TGT['cad_lo'] <= m['cad'] <= TGT['cad_hi']
                and TGT['v_lo'] <= m['v'] <= TGT['v_hi']
                and TGT['s_lo'] <= m['step'] <= TGT['s_hi']
                and m['straight'] >= 0.85):
            continue
        # prefer the 1 m/s neighborhood (human scale 0.9-1.5 m/s)
        center = 0.5 * (prefer_v[0] + prefer_v[1])
        score = abs(m['v'] - center) + (1 - m['straight'])
        cands.append((score, i, j, m))
    cands.sort(key=lambda c: c[0])
    chosen, occupied = [], []
    for score, i, j, m in cands:
        if any(not (j <= a or i >= b) for a, b in occupied):
            continue
        chosen.append((i, j, m))
        occupied.append((i, j))
        if len(chosen) >= max_per_trial:
            break
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--smplx', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=X1_SMP / 'data' / 'x1_smp_train.npz')
    ap.add_argument('--report', type=Path, default=X1_SMP / 'data' / 'x1_smp_train.meta.json')
    ap.add_argument('--interim', type=Path, default=X1_SMP / 'data' / 'interim')
    ap.add_argument('--max-per-trial', type=int, default=2)
    ap.add_argument('--win-s', type=float, default=6.5)
    ap.add_argument('--limit-trials', type=int, default=0)
    ap.add_argument('--keep-interim', action='store_true')
    ap.add_argument('--ik-config', type=Path, default=X1_SMP / 'pipeline' / 'x1_ik_config.json')
    ap.add_argument('--froude-scale', type=float, default=0.811,
                    help='scale human XY trajectory by this before retargeting (Froude similarity; '
                         'sqrt(0.6578 pelvis scale); 1.0 disables)')
    args = ap.parse_args()

    use_ik_config(args.ik_config)
    args.interim.mkdir(parents=True, exist_ok=True)
    trials = sorted(p for p in args.raw.glob('[0-9][0-9]*.npz'))
    if args.limit_trials:
        trials = trials[: args.limit_trials]
    print(f'{len(trials)} raw trials under {args.raw}')

    model = X1Model()
    fk = X1FK()
    state: dict = {}
    qpos_all, qvel_all, names, report, kept = [], [], [], [], []
    t0 = time.time()

    for ti, trial in enumerate(trials):
        stem = trial.name.replace('.npz', '')
        try:
            wins = select_windows(trial, args.win_s, args.max_per_trial)
        except Exception as e:  # noqa: BLE001 -- corrupt/incomplete downloads from the mirror
            print(f'[{ti+1}/{len(trials)}] {trial.name}: BAD FILE ({type(e).__name__}), skip', flush=True)
            report.append({'clip': trial.name, 'error': f'bad raw file: {type(e).__name__}'})
            continue
        if not wins:
            continue
        for wi, (i, j, wm) in enumerate(wins):
            name = f'{stem}_w{wi}'
            interim = args.interim / f'{name}_stageii.npz'
            try:
                src_fps, _ = convert_stageii(trial, i, j, interim)
                apply_froude_scale(interim, args.froude_scale)
                qpos, _ = retarget_clip(interim, args.smplx, retargeter, state, passes=2, arm_mode='refine')
            except Exception as e:  # noqa: BLE001
                print(f'[{ti+1}/{len(trials)}] {name}: SKIP {type(e).__name__}: {e}', flush=True)
                report.append({'clip': name, 'error': f'{type(e).__name__}: {e}'})
                interim.unlink(missing_ok=True)
                continue
            interim.unlink(missing_ok=True)
            if len(qpos) < 5:
                report.append({'clip': name, 'error': 'too short'})
                continue
            pos = torch.tensor(qpos[:, :3]); quat = torch.tensor(qpos[:, 3:7])
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
            sole = model.sole_z(qp.astype(np.float64))
            shift = sole.min(axis=1)
            qp[:, 2] -= shift
            sole = sole - shift[:, None]
            lin, ang, dof_vel = velocities(torch.tensor(qp[:, :3], dtype=torch.float64),
                                            torch.tensor(qp[:, 3:7], dtype=torch.float64),
                                            torch.tensor(qp[:, 7:], dtype=torch.float64), 1.0 / FPS_OUT)
            qv = torch.cat([lin, ang, dof_vel], dim=-1).float().numpy()

            # ---- acceptance A: retarget quality gates (build_x1_motions thresholds)
            q = qp[:, 7:]
            over = np.maximum(q - model.hi, model.lo - q)
            a_pass = (over.max() <= 0.35 and sole.min() > -0.02
                      and qp[:, 2].mean() >= 0.45
                      and float(2 * (qp[:, 4] ** 2 + qp[:, 5] ** 2).max()) <= 0.35
                      and float(np.abs(qv[:, 6:]).max()) <= 25.0)
            # ---- acceptance B: preregistered gait audit
            k = fk.fk_clip(qp, qv)
            gm = gait_metrics(k['sole_h'], k['root_pos'], FPS_OUT)
            b_pass, _ = audit_pass(gm)
            f0, f1 = gm['feet']
            sym = (not np.isnan(f0['duty']) and not np.isnan(f1['duty'])
                   and abs(f0['duty'] - f1['duty']) <= 0.25
                   and abs(f0['cadence_hz'] - f1['cadence_hz']) <= 0.35)
            b_pass = bool(b_pass and sym)
            ok = a_pass and b_pass
            report.append({'clip': name, 'human_window': {kk: round(vv, 3) if isinstance(vv, float) else vv
                                                          for kk, vv in wm.items()},
                           'frames': int(len(qp)),
                           'a_pass': bool(a_pass), 'b_pass': b_pass,
                           'gait': {'cadence': [f0['cadence_hz'], f1['cadence_hz']],
                                    'speed': gm['speed_mps'],
                                    'step': [f0['step_len_m'], f1['step_len_m']],
                                    'swing': [f0['swing_s_med'], f1['swing_s_med']],
                                    'duty': [f0['duty'], f1['duty']],
                                    'lift': [f0['lift_m'], f1['lift_m']]}})
            print(f'[{ti+1}/{len(trials)}] {name:16s} human v{wm["v"]:.2f} cad{wm["cad"]:.0f} | '
                  f'X1 v{gm["speed_mps"]:.2f} cad {f0["cadence_hz"]:.2f}/{f1["cadence_hz"]:.2f} '
                  f'step {f0["step_len_m"]:.2f} A{"+" if a_pass else "-"}B{"+" if b_pass else "-"} '
                  f'{"OK" if ok else "REJECT"}', flush=True)
            if not ok:
                continue
            qpos_all.append(qp); qvel_all.append(qv); names.append(name); kept.append(len(qp))

    if not qpos_all:
        raise SystemExit('no clip passed; see the report')

    lengths = np.array(kept, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, qpos=np.concatenate(qpos_all), qvel=np.concatenate(qvel_all),
             starts=starts, lengths=lengths, names=np.array(names), fps=np.array(FPS_OUT))
    def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
    meta = {'out': str(args.out), 'out_sha256': sha256(args.out), 'out_bytes': args.out.stat().st_size,
            'clips_kept': len(names), 'frames': int(lengths.sum()), 'hours': round(float(lengths.sum()) / FPS_OUT / 3600, 4),
            'fps': FPS_OUT, 'joint_order': list(fk.model and __import__('x1_data_lib').X1_JOINT_ORDER),
            'froude_scale': args.froude_scale,
            'pipeline': 'CMU_raw -> multi-window select (cad 95-140, v 0.7-1.9, step 0.5-0.85, straight>=0.85, prefer 0.9-1.5) '
                        '-> stageii (XY*froude_scale) -> GMR 2-pass + arm_refine -> per-frame ground -> 50Hz -> gates A (retarget) + B (gait audit)',
            'windows': json.loads(json.dumps(report, default=_jsonable)), 'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_s': round(time.time() - t0, 1)}
    args.report.write_text(json.dumps(meta, indent=1))
    if not args.keep_interim:
        shutil.rmtree(args.interim, ignore_errors=True)
    print(f'kept {len(names)} clips, {int(lengths.sum())} frames -> {args.out} '
          f'({time.time()-t0:.0f}s); report {args.report}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
