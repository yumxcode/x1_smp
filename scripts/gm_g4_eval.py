#!/usr/bin/env python3
"""G4 acceptance rollouts (amendment rev3): 0.45 / 0.5 m/s commands, 20 s,
with the numerical no-slide hard gate + absolute-height detector + per-frame
foot telemetry, on the G3 policy + v3 prior.

Gates per speed (contract G4, rev3 target mapping; see SMP_PLAN Am.3):
  G-SLIDE  stance slip (min-corner, strict z<3mm) p50 < 0.15 m/s, both feet
  G-STEP   per-foot sole z>5mm fraction >= 0.35  [demo-calibrated stepping
           evidence: AMP walk refs 0.40-0.45; the LITERAL contract wording
           "p50 离地>5mm" is structurally unsatisfiable for any walking gait
           (median frame is stance; AMP demos measure p50 = 1.8-1.9 mm) --
           both readings are reported, flagged for user confirmation]
  G-R1     ABS-detector cadence in [0.55, 1.2] Hz per foot
  G-VEL    measured speed >= 0.93 * cmd (0.45 -> 0.42, 0.5 -> 0.46)
  G-SURV   no fall for 20 s (no base_too_low/self_collision termination)
Also logs: rel detector, swing/duty/lift, root z, xy travel, per-frame CSV.

  gm-run X1_SMP/scripts/gm_g4_eval.py
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
sys.path.insert(0, str(ROOT / 'pipeline'))  # stance_anchor lives here (r3 crash)

# G4 ckpt under test: default = G3' variant (audit round 1); override with
# G4_CKPT env var (e.g. to re-evaluate the G3 baseline model_g3_6000.pt).
import os as _os
CKPT = _os.environ.get(
    'G4_CKPT', str(ROOT / 'data' / 'policy' / 'model_g3p_6000.pt'))
print(f'[g4] ckpt={CKPT}', flush=True)
PRIOR = str(ROOT / 'data' / 'prior' / 'x1_walk_prior_v3.pt')
SPEEDS = (0.45, 0.50)
SECS = 20.0


def install_stack() -> bool:
    """G3's exact install spec (proven working at 08:58) + a runtime fix for
    the Isaac prebundled typing_extensions that shadows site-packages and
    lacks TypeForm (tyro>=0.9 import crash, TASK_20261105_387). The pin
    tyro==0.8.14 (r2) conflicted with mjlab's resolver - dropped."""
    import subprocess
    for idx in ('', 'https://pypi.org/simple', 'https://mirrors.aliyun.com/pypi/simple/'):
        flag = ['--index-url', idx] if idx else []
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', 'mjlab==1.6.0',
                            'warp-lang==1.16.0', *flag], capture_output=True, text=True)
        print('pip:', ((r.stdout or '').strip().splitlines() or ['<no output>'])[-1][:160], flush=True)
        if r.returncode == 0:
            print(f'pip OK idx={idx or "default"}', flush=True)
            break
    else:
        return False
    # patch the stale prebundled typing_extensions (needs TypeForm for tyro)
    import typing_extensions
    if not hasattr(typing_extensions, 'TypeForm'):
        import glob, pathlib, shutil, site
        sp_dirs = [p for p in site.getsitepackages() if p.endswith('site-packages')]
        src = pathlib.Path(sp_dirs[0]) / 'typing_extensions.py' if sp_dirs else None
        if src and src.exists():
            for c in glob.glob('/workspace/isaaclab/_isaac_sim/extscache/*/pip_prebundle/typing_extensions.py'):
                pathlib.Path(c).rename(c + '.bak_g4')
                shutil.copy(src, c)
                print('patched prebundle typing_extensions:', c, flush=True)
            import importlib
            importlib.reload(typing_extensions)
            print('TypeForm available after patch:',
                  hasattr(typing_extensions, 'TypeForm'), flush=True)
    return True


def rollout_qpos(ckpt: str, prior: str, speed: float, secs: float) -> tuple[np.ndarray, bool]:
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
    env_cfg.events['init_smp_state'].params['ckpt_path'] = prior
    env_cfg.events['init_smp_state'].params['compile_model'] = False
    agent_cfg = load_rl_cfg('Smp-Forward-X1')
    env = ManagerBasedRlEnv(cfg=env_cfg, device='cuda', render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device='cuda')
    runner.load(ckpt, load_cfg={'actor': True}, strict=True, map_location='cuda')
    policy = runner.get_inference_policy(device='cuda')
    obs, _ = env.reset()
    frames, fell = [], False
    with torch.inference_mode():
        for _ in range(int(secs * 50)):
            obs, _, term, _ = env.step(policy(obs))
            if float(term[0]) > 0 and not env.unwrapped.episode_length_buf[0] >= \
                    env_cfg.episode_length_s * 50 - 1:
                fell = True  # non-timeout termination on env 0
            rd = env.unwrapped.scene['robot'].data
            q = torch.cat([rd.root_link_pos_w[0], rd.root_link_quat_w[0],
                           rd.joint_pos[0]]).detach().cpu().numpy()
            frames.append(q)
    env.close()
    return np.stack(frames), fell


def analyse(qpos: np.ndarray, prior: np.ndarray | None = None) -> dict:
    import mujoco
    from stance_anchor import conservative_slip, SIDES, _segment
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
    zmin = {k: C[k][:, :, 2].min(axis=1) for k in C}
    out = {'frames': T, 'feet': {}}
    # ABS detector phases
    AIR, GROUND = 0.025, 0.008
    slip = {'feet': {}}
    for k in ('L', 'R'):
        z = zmin[k]
        v = np.linalg.norm(np.diff(C[k][:, :, :2], axis=0), axis=2).min(axis=1) / 0.02
        mask = (z[:-1] < 0.003) & (z[1:] < 0.003)
        interior = np.zeros(len(z), dtype=bool)
        for kind, i0, i1 in _segment(z):
            if kind == 'stance':
                lo, hi = i0 + 3, max(i1 - 3, i0 + 3)
                interior[lo:hi] = True
        m = mask & interior[:-1]
        vs = v[m]
        swings, tds = [], []
        for kind, i0, i1 in _segment(z):
            if kind == 'swing' and (i1 - i0) >= 5:
                swings.append(round((i1 - i0) / 50, 2))
            if kind == 'stance' and i0 > 0:
                tds.append(i0)
        cad = float(1.0 / (np.median(np.diff(tds)) / 50)) if len(tds) >= 3 else None
        out['feet'][k] = {
            'sole_z_p50_mm': float(np.median(z) * 1000),
            'sole_z_p90_mm': float(np.percentile(z, 90) * 1000),
            'z_gt5mm_frac': float((z > 0.005).mean()),
            'airborne_gt25mm_frac': float((z > 0.025).mean()),
            'abs_swings_n': len(swings), 'abs_swing_s': swings[:20],
            'abs_cadence_hz': cad,
            'contact_slip_p50': float(np.median(vs)) if len(vs) else None,
            'contact_slip_p90': float(np.percentile(vs, 90)) if len(vs) else None,
            'slide_frac_gt015': float((vs > 0.15).mean()) if len(vs) else None,
        }
    out['flight_frames_both_gt25mm'] = int(((zmin['L'] > AIR) & (zmin['R'] > AIR)).sum())
    vxy = np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1) / 0.02
    lo, hi = int(0.1 * T), max(int(0.9 * T), int(0.1 * T) + 1)
    out['root_speed_p50'] = float(np.median(vxy[lo:hi]))
    out['xy_travel_m'] = float(np.hypot(*(qpos[-1, :2] - qpos[0, :2])))
    out['root_z_range'] = [float(qpos[:, 2].min()), float(qpos[:, 2].max())]
    return out


def main() -> int:
    if not install_stack():
        return 2
    res, gates_all = {}, True
    for speed in SPEEDS:
        q, fell = rollout_qpos(CKPT, PRIOR, speed, SECS)
        r = analyse(q)
        r['fell'] = bool(fell)
        g = {
            'slide_p50<0.15': all((f['contact_slip_p50'] is not None and f['contact_slip_p50'] < 0.15)
                                  for f in r['feet'].values()),
            'step_z>5mm_frac>=0.35': all(f['z_gt5mm_frac'] >= 0.35 for f in r['feet'].values()),
            'r1_cadence_in_band': all(f['abs_cadence_hz'] is not None and 0.55 <= f['abs_cadence_hz'] <= 1.2
                                      for f in r['feet'].values()),
            'speed>=0.93cmd': r['root_speed_p50'] >= 0.93 * speed,
            'no_fall_20s': not fell,
        }
        ok = all(g.values())
        gates_all &= ok
        res[f'v{speed}'] = {'metrics': r, 'gates': g, 'pass': ok}
        print(f'[g4] v{speed}: pass={ok} gates={g}', flush=True)
        print(f'[g4] v{speed} detail: {json.dumps(r)}', flush=True)
        try:
            out_dir = Path('/workspace/output')
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f'g4_rollout_qpos_v{speed}.npy', q)
            with open(out_dir / f'g4_metrics_v{speed}.json', 'w') as f:
                json.dump({'metrics': r, 'gates': g}, f, indent=1)
        except Exception as e:  # noqa: BLE001
            print(f'[g4] artifact write skipped: {e}', flush=True)
    with open(ROOT / 'eval' / 'g4_results.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(f'[g4] ALL PASS={gates_all} -> eval/g4_results.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
