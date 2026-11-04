#!/usr/bin/env python3
"""AUDIT: does the trained prior (59-dim R1 and 63-dim R3) actually reward
stepping over sliding, in feature space?

Tests (all offline, zero platform cost, raw-MSE SDS reward as in
scripts/attribution_prior.py -- relative comparisons only):

  T1 foot-ee freeze       : freeze the 6 foot-ee dims to the window temporal
                            mean (the R2-style ablation, re-run on BOTH priors)
  T2 joint+foot freeze    : additionally freeze all 29 joint-angle channels
  T3 slide signature      : construct the policy's observed slide in feature
                            space on top of real windows: joints frozen, foot
                            ee frozen, ankle_z (both feet) set to the grounded
                            constant (window's own min ankle z), ankle_z_dz=0.
                            Root motion / velocities stay at data values.
                            This is "robot translates, legs frozen, feet on
                            the floor" -- exactly the evaluated R1/R3 gait.
  T4 speed groups         : mean intact reward by clip speed band (is the
                            prior simply a slow-motion prior?)

Verdict convention follows the preregistered H2a thresholds: ratio >= 0.80
"prior cannot distinguish", < 0.50 "distinguishes".

Run: ~/code/X1_29_AMP/_gmr_local/venv/bin/python audit/prior_slide_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from smp.pretrain.model import DiffusionDenoiser  # noqa: E402
from smp.pretrain.scheduler import DDPMScheduler  # noqa: E402

K = (8, 15, 22)
WS = 6.0
N_MC = 4
SEED = 0
J0 = 3 + 6                 # joints start
EE_FOOT = list(range(3 + 6 + 29, 3 + 6 + 29 + 6))
ANKLE_Z = [59, 60]         # 63-dim contact channels
ANKLE_DZ = [61, 62]


def load(ckpt):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = DiffusionDenoiser(feature_dim=int(ck['cfg']['feature_dim']),
                              window_size=int(ck['cfg']['window_size']),
                              d_model=int(ck['cfg']['d_model']), nhead=int(ck['cfg']['nhead']),
                              num_layers=int(ck['cfg']['num_layers']))
    model.load_state_dict(ck['model']); model.eval()
    sched = DDPMScheduler(num_timesteps=int(ck['cfg']['num_timesteps']))
    q_low = torch.from_numpy(np.asarray(ck['q_low'], dtype=np.float32))
    q_high = torch.from_numpy(np.asarray(ck['q_high'], dtype=np.float32))
    return model, sched, q_low, q_high, int(ck['cfg']['feature_dim'])


def sds_reward(model, sched, x0, k=K, ws=WS):
    rs = []
    g = torch.Generator().manual_seed(SEED)
    for _ in range(N_MC):
        errs = torch.zeros(len(x0))
        for t_int in k:
            t = torch.full((len(x0),), t_int, dtype=torch.long)
            noise = torch.randn(x0.shape, generator=g)
            x_t = sched.add_noise(x0, noise, t)
            eps = model(x_t, t)
            errs += ((eps - noise) ** 2).mean(dim=(-1, -2))
        rs.append(torch.exp(-errs / len(k) * ws))
    return torch.stack(rs).mean(0).numpy()


def main():
    out = {}
    # clip speeds from the independent audit (fallback: meta)
    speeds = {}
    try:
        ind = json.load(open(ROOT / 'audit' / 'gait_stats_independent.json'))
        speeds = {n: v['speed_mps'] for n, v in ind.items()}
    except FileNotFoundError:
        pass
    win_files = sorted((ROOT / 'data' / 'windows').glob('*.npz'))
    for tag, ckpt in (('r1_59d', ROOT / 'data/prior/x1_walk_prior.pt'),
                      ('r3_63d', ROOT / 'data/prior/x1_walk_prior_r3.pt')):
        model, sched, q_low, q_high, F = load(ckpt)
        res = {'ckpt': str(ckpt), 'feature_dim': F}
        intact_all, t1_all, t2_all, t3_all = [], [], [], []
        per_clip = {}
        for f in win_files:
            wins = torch.from_numpy(np.load(f)['windows'].astype(np.float32))[:400]
            if F == 59:
                wins = wins[:, :, :59]  # strip round-3 contact channels
            x0 = 2.0 * (wins - q_low) / (q_high - q_low + 1e-8) - 1.0
            x1 = x0.clone(); x1[:, :, EE_FOOT] = x1[:, :, EE_FOOT].mean(dim=1, keepdim=True)
            x2 = x1.clone(); x2[:, :, J0:J0 + 29] = x2[:, :, J0:J0 + 29].mean(dim=1, keepdim=True)
            x3 = x2.clone()
            if F >= 63:
                zc = wins[:, :, ANKLE_Z].amin(dim=(1, 2))  # (N,) grounded constant, raw units
                zl = 2.0 * (zc[:, None] - q_low[ANKLE_Z]) / (q_high[ANKLE_Z] - q_low[ANKLE_Z] + 1e-8) - 1.0
                x3[:, :, ANKLE_Z] = zl[:, None, :].expand(-1, wins.shape[1], 2).clone()
                # dz channels = 0 raw motion
                z0n = 2.0 * (0.0 - q_low[ANKLE_DZ]) / (q_high[ANKLE_DZ] - q_low[ANKLE_DZ] + 1e-8) - 1.0
                x3[:, :, ANKLE_DZ] = torch.as_tensor(z0n, dtype=torch.float32)
            with torch.no_grad():
                r0 = sds_reward(model, sched, x0)
                r1 = sds_reward(model, sched, x1)
                r2 = sds_reward(model, sched, x2)
                r3 = sds_reward(model, sched, x3) if F >= 63 else None
            per_clip[f.stem] = dict(n=len(r0), intact=float(r0.mean()),
                                    t1_freeze_foot=float(r1.mean()),
                                    t2_freeze_joint_foot=float(r2.mean()),
                                    **({'t3_slide': float(r3.mean())} if r3 is not None else {}),
                                    speed=speeds.get(f.stem))
            intact_all.append(r0); t1_all.append(r1); t2_all.append(r2)
            if r3 is not None:
                t3_all.append(r3)
        r0 = float(np.concatenate(intact_all).mean())
        r1 = float(np.concatenate(t1_all).mean())
        r2 = float(np.concatenate(t2_all).mean())
        entry = dict(intact=r0, t1_freeze_foot=r1, ratio_t1=r1 / r0,
                     t2_freeze_joint_foot=r2, ratio_t2=r2 / r0)
        if t3_all:
            r3v = float(np.concatenate(t3_all).mean())
            entry.update(t3_slide_signature=r3v, ratio_t3=r3v / r0)
        # speed groups (intact reward)
        groups = {'<0.6': [], '0.6-0.7': [], '>=0.7': []}
        for n, v in per_clip.items():
            sp = v.get('speed')
            if sp is None:
                continue
            (groups['<0.6'] if sp < 0.6 else groups['0.6-0.7'] if sp < 0.7 else groups['>=0.7']).append(
                (v['intact'], v.get('t3_slide')))
        entry['speed_groups'] = {g: dict(n=len(xs),
                                         intact=float(np.mean([x[0] for x in xs])) if xs else None,
                                         t3=float(np.mean([x[1] for x in xs if x[1] is not None])) if xs else None)
                                 for g, xs in groups.items()}
        res['per_clip'] = per_clip
        res['summary'] = entry
        out[tag] = res
        print(f"== {tag}: intact {r0:.4f} | T1 foot-freeze ratio {entry['ratio_t1']:.3f} | "
              f"T2 +joint-freeze ratio {entry['ratio_t2']:.3f}"
              + (f" | T3 slide-signature ratio {entry['ratio_t3']:.3f}" if 'ratio_t3' in entry else ''), flush=True)
        for g, st in entry['speed_groups'].items():
            if st['n']:
                print(f"   speed {g}: n={st['n']} intact {st['intact']:.4f}"
                      + (f" slide {st['t3']:.4f}" if st.get('t3') else ''), flush=True)
    Path(ROOT / 'audit' / 'prior_slide_test.json').write_text(json.dumps(out, indent=1))
    print('-> audit/prior_slide_test.json')


if __name__ == '__main__':
    main()
