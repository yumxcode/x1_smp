#!/usr/bin/env python3
"""Post-hoc full retarget-acceptance gates for the 11 newly-retargeted clips.

Contract task-2 audit item: build_smp_library applied only gate A (quality
reject thresholds incl. over-limit <= 0.35 rad) + gate B (gait audit); the
FULL gate set from tools/x1_retarget_gates.py (wrist err p50/p90 vs anchored
intent target, arm-at-limit frac, arm qd, lumbar margin, chest twist, tilt,
hard over-limit == 0, sole penetration, root band, self-collision) was never
run on these clips. This script re-derives each clip end-to-end (same
deterministic window selection + Froude scale + GMR 2-pass + arm refine) and
scores the full gate set, writing per-clip results to
data/x1_smp_train.meta.json['retarget_gates'].

Run with the GMR venv python. Raw trials are re-downloaded from the AMASS
mirror (they were deleted per the storage red line) and deleted again after.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline'))
sys.path.insert(0, str(Path.home() / 'code' / 'super_mimic_F1'))
sys.path.insert(0, str(Path.home() / 'code' / 'X1_29_AMP' / 'data'))
sys.path.insert(0, str(Path.home() / 'code' / 'X1_29_AMP' / '_gmr_local' / 'GMR'))

from build_x1_motions import retarget_clip, retargeter, use_ik_config, X1Model, FPS_OUT  # noqa: E402
from build_smp_library import select_windows, apply_froude_scale  # noqa: E402
from cmu_select_walk import convert as convert_stageii  # noqa: E402
from prepare_motions import resample, velocities  # noqa: E402
from tools.x1_retarget_gates import GateModel, human_torso_twist, score_clip  # noqa: E402
from tools.x1_arm_fix import D_MAX, D_MIN, RATIO  # noqa: E402

ROOT = HERE.parent
GMR_BODY = Path.home() / 'code' / 'X1_29_AMP' / '_gmr_local' / 'GMR' / 'assets' / 'body_models'
RAW = ROOT / 'data' / 'raw_tmp'
RAW.mkdir(parents=True, exist_ok=True)

import subprocess


def fetch(trial: str) -> Path:
    p = RAW / f'{trial}.npz'
    if p.exists() and p.stat().st_size > 100_000:
        return p
    url = (f'https://hf-mirror.com/datasets/realdream-ai/AMASS/resolve/main/raw/CMU/'
           f'{trial[:2] if not trial[1].isdigit() else trial[:2]}/{trial}/{trial}_poses.npz')
    # subject prefix handling: names like '02_02' -> subject '02'
    for _ in range(3):
        subprocess.run(['curl', '-sL', '-m', '120', '-o', str(p), url], check=False)
        if p.exists() and p.stat().st_size > 100_000:
            return p
        time.sleep(1)
    raise RuntimeError(f'download failed: {trial}')


def subject_of(trial: str) -> str:
    # '02_02' -> '02'; '105_19' -> '105'; '22_12' -> '22'... subject is the part before '_'
    return trial.split('_')[0]


def fetch2(trial: str) -> Path:
    p = RAW / f'{trial}.npz'
    if p.exists() and p.stat().st_size > 100_000:
        return p
    sub = subject_of(trial)
    url = (f'https://hf-mirror.com/datasets/realdream-ai/AMASS/resolve/main/raw/CMU/'
           f'{sub}/{trial}_poses.npz')
    for _ in range(3):
        subprocess.run(['curl', '-sL', '-m', '120', '-o', str(p), url], check=False)
        if p.exists() and p.stat().st_size > 100_000:
            return p
        time.sleep(1)
    raise RuntimeError(f'download failed: {trial}')


def main() -> int:
    meta = json.load(open(ROOT / 'data' / 'x1_smp_train.meta.json'))
    lib = np.load(ROOT / 'data' / 'x1_smp_train.npz', allow_pickle=True)
    names = [str(n) for n in lib['names']]
    new_clips = [n for n in names if not n.startswith('p2_')]
    print(f'{len(new_clips)} newly-retargeted clips to gate: {new_clips}')

    use_ik_config(ROOT / 'pipeline' / 'x1_ik_config.json')
    gate_model = GateModel()
    model = X1Model()
    state: dict = {}
    interim = ROOT / 'data' / 'interim_gates'
    interim.mkdir(parents=True, exist_ok=True)
    results = {}
    froude = float(meta.get('froude_scale', 0.811))

    for name in new_clips:
        trial, wtag = name.rsplit('_w', 1)
        wi = int(wtag)
        raw = fetch2(trial)
        wins = select_windows(raw, 6.5, 2)
        if wi >= len(wins):
            print(f'{name}: window {wi} not found ({len(wins)} windows) -- SKIP')
            continue
        i, j, wm = wins[wi]
        stage = interim / f'{name}_stageii.npz'
        convert_stageii(raw, i, j, stage)
        apply_froude_scale(stage, froude)
        qpos, src_fps = retarget_clip(stage, GMR_BODY, retargeter, state, passes=2, arm_mode='refine')
        # wrist error vs anchored intent target (same law as the arm fix; phase-2 recipe)
        import mujoco
        r = state['r']
        bid = {f'{s}_wrist_roll_link': mujoco.mj_name2id(
            r.model, mujoco.mjtObj.mjOBJ_BODY, f'{s}_wrist_roll_link') for s in ('left', 'right')}
        from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast, load_smplx_file
        smplx_data, body_model, smplx_out, h = load_smplx_file(str(stage), str(GMR_BODY))
        frames, _ = get_smplx_data_offline_fast(smplx_data, body_model, smplx_out, tgt_fps=src_fps)
        dd = mujoco.MjData(r.model)
        sh_bid = r.shoulder_bid
        werr = []
        for k in range(min(len(qpos), len(frames))):
            fr = frames[k]
            dd.qpos[:] = qpos[k]
            mujoco.mj_kinematics(r.model, dd)
            e = []
            for s in ('left', 'right'):
                wr_h = np.asarray(fr[f'{s}_wrist'][0], dtype=float)
                sh_h = np.asarray(fr[f'{s}_shoulder'][0], dtype=float)
                v = wr_h - sh_h
                n = np.linalg.norm(v)
                tgt = dd.xpos[sh_bid[s]] + v * (np.clip(RATIO * n, D_MIN, D_MAX) / n) if n > 1e-6 else dd.xpos[sh_bid[s]]
                e.append(np.linalg.norm(dd.xpos[bid[f'{s}_wrist_roll_link']] - tgt))
            werr.append(e)
        werr = np.array(werr).max(axis=1)
        # 50 Hz resample + per-frame ground (identical to the library build)
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
            frames_50 = frames[int(round(0.2 * src_fps)):]
        else:
            frames_50 = frames[:len(qp)]
        sole = model.sole_z(qp.astype(np.float64))
        qp[:, 2] -= sole.min(axis=1)
        twist_h = human_torso_twist(frames_50[:len(qp)])
        sc = score_clip(qp, FPS_OUT, gate_model, wrist_err=werr, chest_yaw_human=twist_h,
                        tag='walk', human_net_yaw=0.0)
        results[name] = {'metrics': sc['metrics'], 'gates': sc['gates'], 'pass': sc['pass']}
        bad = [g for g, ok in sc['gates'].items() if not ok]
        print(f'{name:16s} pass={sc["pass"]} bad={bad} '
              f'{[(g, round(sc["metrics"][g], 4) if isinstance(sc["metrics"][g], float) else sc["metrics"][g]) for g in bad]}',
              flush=True)
        stage.unlink(missing_ok=True)

    meta['retarget_gates'] = results
    n_pass = sum(1 for v in results.values() if v['pass'])
    meta['retarget_gates_summary'] = dict(n=len(results), n_pass=n_pass)
    json.dump(meta, open(ROOT / 'data' / 'x1_smp_train.meta.json', 'w'), indent=1, default=str)
    print(f'gated {len(results)}, pass {n_pass}; meta updated')
    # storage red line: remove re-downloaded raws
    import shutil
    shutil.rmtree(RAW, ignore_errors=True)
    shutil.rmtree(interim, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
