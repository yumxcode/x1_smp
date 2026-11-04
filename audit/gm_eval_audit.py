#!/usr/bin/env python3
"""AUDIT re-eval (verification only, no training): rerun the R3 policy rollout
at cmd 0.7 with FULL per-frame foot telemetry and BOTH gait detectors, to
adjudicate whether the shipped eval's "no gait cycle (all-null)" verdict was a
detector false negative.

Measures, per frame (MuJoCo FK on the logged qpos, same model as data audits):
  - sole z (min of 4 corner spheres) per foot, absolute, floor at 0
  - pivot point (lowest corner) horizontal speed per foot
  - flight frames (both soles > 25 mm)
Detectors:
  - REL detector (scripts/x1_gait.touchdown_events, the shipped one; contact =
    within 2 cm of the OTHER foot)  -> cadence/swing/duty as shipped
  - ABS detector (audit-style Schmitt trigger on absolute height, AIR 25 mm /
    GROUND 8 mm) -> swing/stance phases, cadence, stance slip
Everything is printed as one JSON line into the task log AND written to
/workspace/output/audit_gait_<tag>.json (+ npz with the full series).

gm-run X1_SMP/audit/gm_eval_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'audit'))


def install_stack() -> bool:
    import subprocess
    for idx in ('', 'https://pypi.org/simple', 'https://mirrors.aliyun.com/pypi/simple/'):
        flag = ['--index-url', idx] if idx else []
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', 'mjlab==1.6.0',
                            'warp-lang==1.16.0', *flag], capture_output=True, text=True)
        if r.returncode == 0:
            print(f'pip OK idx={idx or "default"}', flush=True)
            return True
    return False


def rollout_qpos(ckpt: str, speed: float, secs: float) -> np.ndarray:
    import smp.rl.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from dataclasses import asdict

    env_cfg = load_env_cfg('Smp-Forward-X1', play=True)
    env_cfg.scene.num_envs = 4
    env_cfg.commands['steering'].resampling_time_range = (secs * 2, secs * 2)
    env_cfg.commands['steering'].tar_speed_min = speed
    env_cfg.commands['steering'].tar_speed_max = speed
    env_cfg.events['init_smp_state'].params['ckpt_path'] = str(
        ROOT / 'data' / 'prior' / 'x1_walk_prior_r3.pt')
    env_cfg.events['init_smp_state'].params['compile_model'] = False
    agent_cfg = load_rl_cfg('Smp-Forward-X1')
    env = ManagerBasedRlEnv(cfg=env_cfg, device='cuda', render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device='cuda')
    runner.load(ckpt, load_cfg={'actor': True}, strict=True, map_location='cuda')
    policy = runner.get_inference_policy(device='cuda')
    obs, _ = env.reset()
    frames = []
    with torch.inference_mode():
        for _ in range(int(secs * 50)):
            obs, _, _, _ = env.step(policy(obs))
            rd = env.unwrapped.scene['robot'].data
            q = torch.cat([rd.root_link_pos_w[0], rd.root_link_quat_w[0],
                           rd.joint_pos[0]]).detach().cpu().numpy()
            frames.append(q)
    env.close()
    return np.stack(frames)


def analyse(qpos: np.ndarray) -> dict:
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets' / 'x1_29dof' / 'x1_29dof.xml'))
    data = mujoco.MjData(model)
    feet = {}
    for fb in ('left_ankle_roll_link', 'right_ankle_roll_link'):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fb)
        feet[fb] = [g for g in range(model.ngeom)
                    if model.geom_bodyid[g] == bid
                    and 'collision' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')]
    T = len(qpos)
    C = {}
    for k, fb in (('L', 'left_ankle_roll_link'), ('R', 'right_ankle_roll_link')):
        corners = np.empty((T, 4, 3))
        for t in range(T):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            corners[t] = [data.geom_xpos[g] for g in feet[fb]]
        C[k] = corners
    out = {'frames': T, 'feet': {}}
    zmin = {k: C[k][:, :, 2].min(axis=1) for k in C}
    rel = zmin['L'] - zmin['R']
    # REL (shipped) detector
    from x1_gait import touchdown_events
    sole = np.stack([zmin['L'], zmin['R']], axis=1)
    ev = touchdown_events(sole)
    out['rel_detector'] = {
        'touchdowns': [int(len(e)) for e in ev],
        'airborne_frac_gt2cm_rel': [float((rel > 0.02).mean()), float((rel < -0.02).mean())]}
    # ABS detector (audit semantics)
    AIR, GROUND = 0.025, 0.008
    for k in ('L', 'R'):
        z = zmin[k]
        air = z > AIR
        phases, state, start = [], 'stance' if z[0] < GROUND else ('swing' if z[0] > AIR else 'band'), 0
        for i in range(1, T):
            if state == 'stance' and z[i] > AIR:
                phases.append(('stance', start, i)); state, start = 'swing', i
            elif state == 'swing' and z[i] < GROUND:
                phases.append(('swing', start, i)); state, start = 'stance', i
            elif state == 'band':
                if z[i] > AIR: state, start = 'swing', i
                elif z[i] < GROUND: state, start = 'stance', i
        phases.append((state, start, T))
        swings = [(i1 - i0) / 50 for kk, i0, i1 in phases if kk == 'swing' and (i1 - i0) >= 5]
        tds = [i0 for kk, i0, i1 in phases if kk == 'stance' and i0 > 0]
        v = np.linalg.norm(np.diff(C[k][:, :, :2], axis=0), axis=2).min(axis=1) / 0.02
        contact = z < 0.003
        m = contact[:-1] & contact[1:]
        out['feet'][k] = {
            'sole_z_p50': float(np.median(z)), 'sole_z_p90': float(np.percentile(z, 90)),
            'sole_z_max': float(z.max()),
            'airborne_abs_frac': float(air.mean()),
            'abs_swings_n': len(swings),
            'abs_swing_s': [round(s, 2) for s in swings[:20]],
            'abs_cadence_hz': float(1.0 / (np.median(np.diff(tds)) / 50)) if len(tds) >= 3 else None,
            'contact_frames': int(contact.sum()),
            'contact_slip_min_corner_p50_mps': float(np.median(v[m])) if m.any() else None,
            'contact_slip_min_corner_p90_mps': float(np.percentile(v[m], 90)) if m.any() else None,
            'slide_frac_gt015': float((v[m] > 0.15).mean()) if m.any() else None,
        }
    out['flight_frames_both_gt25mm'] = int(((zmin['L'] > AIR) & (zmin['R'] > AIR)).sum())
    both_low = (zmin['L'] < 0.012) & (zmin['R'] < 0.012)
    out['both_feet_low_frac'] = float(both_low.mean())
    vxy = np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1) / 0.02
    out['root_speed_p50'] = float(np.median(vxy))
    out['xy_travel_m'] = float(np.hypot(*(qpos[-1, :2] - qpos[0, :2])))
    out['root_z_range'] = [float(qpos[:, 2].min()), float(qpos[:, 2].max())]
    return out


def main() -> int:
    if not install_stack():
        return 2
    ckpt = str(ROOT / 'data' / 'policy' / 'model_r3_3900.pt')
    res = {}
    for speed in (0.7,):
        q = rollout_qpos(ckpt, speed, 20.0)
        r = analyse(q)
        res[f'v{speed}'] = r
        print(f'[audit-eval] v{speed}: {json.dumps(r)}', flush=True)
        try:
            np.save('/workspace/output/audit_rollout_qpos_v0p7.npy', q)
            Path('/workspace/output').mkdir(parents=True, exist_ok=True)
            with open('/workspace/output/audit_gait_v0p7.json', 'w') as f:
                json.dump(r, f, indent=1)
        except Exception as e:  # noqa: BLE001
            print(f'[audit-eval] artifact write skipped: {e}', flush=True)
    with open(ROOT / 'eval' / 'audit_gait_recheck.json', 'w') as f:
        json.dump(res, f, indent=1)
    print('[audit-eval] done -> eval/audit_gait_recheck.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
