"""Fixed-speed rollout + gait metrics + video for a trained Smp-Forward-X1 policy.

Loads a checkpoint via the mjlab runner (same path as mjlab.scripts.play),
pins the steering command to a fixed +x speed, rolls out --secs seconds at
50 Hz, and measures rhythm_gates-compatible gait metrics from the logged
qpos via MuJoCo FK (scripts/x1_data_lib). Renders an mp4 side video.

Used by scripts/gm_smp_eval.py on the platform.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import smp.rl.tasks  # noqa: F401,E402  registers Smp-Forward-X1
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

TASK_ID = 'Smp-Forward-X1'


def rollout(ckpt: str, speed: float, secs: float, tag: str, num_envs: int = 4,
            prior_ckpt: str | None = None, video: bool = True) -> dict:
    if prior_ckpt is None:
        prior_ckpt = str(ROOT / 'data' / 'prior' / 'x1_walk_prior.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = num_envs
    # pin the command: fixed +x target speed, no resampling drift
    env_cfg.commands['steering'].resampling_time_range = (secs * 2, secs * 2)
    if hasattr(env_cfg.commands['steering'], 'tar_speed_min'):
        env_cfg.commands['steering'].tar_speed_min = speed
        env_cfg.commands['steering'].tar_speed_max = speed
    if prior_ckpt:
        env_cfg.events['init_smp_state'].params['ckpt_path'] = prior_ckpt
    env_cfg.events['init_smp_state'].params['compile_model'] = False

    agent_cfg = load_rl_cfg(TASK_ID)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode='rgb_array' if video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={'actor': True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    frames_q = []  # per-step qpos of env 0
    frames_img = []
    steps = int(secs * 50)
    with torch.inference_mode():
        for t in range(steps):
            act = policy(obs)
            obs, _, _, _ = env.step(act)
            raw = env.unwrapped
            rd = raw.scene['robot'].data
            qpos7 = torch.cat([rd.root_link_pos_w[0], rd.root_link_quat_w[0]]).detach().cpu().numpy()
            jq = rd.joint_pos[0].detach().cpu().numpy()
            frames_q.append(np.concatenate([qpos7, jq]))
            if video and t % 2 == 0:  # 25 fps video
                img = raw.render() if hasattr(raw, 'render') else None
                if img is not None:
                    frames_img.append(np.asarray(img))
    qpos = np.stack(frames_q)

    # gait metrics via FK (env 0 log)
    import mujoco
    from x1_data_lib import X1FK, gait_metrics, X1_XML
    fk = X1FK(X1_XML)
    T = len(qpos)
    qvel = np.zeros((T, 35), dtype=np.float32)
    qpos32 = qpos.astype(np.float64)
    # finite-diff velocities for gait metrics (root world lin vel + joint vel)
    dt = 0.02
    dpos = np.diff(qpos32[:, :3], axis=0) / dt
    qvel[:-1, :3] = dpos
    qvel[:-1, 6:] = np.diff(qpos32[:, 7:], axis=0) / dt
    # root angular vel approximated zero (not used by gait metrics)
    k = fk.fk_clip(qpos32.astype(np.float32), qvel)
    gm = gait_metrics(k['sole_h'], k['root_pos'], 50.0)
    f0, f1 = gm['feet']

    out_dir = ROOT / 'eval'
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / f'rollout_{tag}.npz', qpos=qpos)
    result = dict(
        survival_s=T / 50.0, speed_cmd=speed, speed_measured=gm['speed_mps'],
        cadence_hz=[f0['cadence_hz'], f1['cadence_hz']],
        swing_s=[f0['swing_s_med'], f1['swing_s_med']],
        duty=[f0['duty'], f1['duty']],
        lift_m=[f0['lift_m'], f1['lift_m']],
        step_len_m=[f0['step_len_m'], f1['step_len_m']],
    )
    if video and frames_img:
        try:
            import imageio.v2 as iio
            iio.mimwrite(str(out_dir / f'video_{tag}.mp4'), frames_img, fps=25,
                         quality=7, macro_block_size=None)
            result['video'] = str(out_dir / f'video_{tag}.mp4')
        except Exception as e:  # noqa: BLE001
            result['video_error'] = f'{type(e).__name__}: {e}'
    env.close()
    return {kk: (vv if not isinstance(vv, (list, tuple)) else [float(x) if x == x else None for x in vv])
            for kk, vv in result.items()}


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--secs', type=float, default=20.0)
    args = ap.parse_args()
    print(json.dumps(rollout(args.ckpt, args.speed, args.secs, f'v{args.speed:.1f}'.replace('.', 'p')), indent=1))
