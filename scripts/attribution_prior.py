#!/usr/bin/env python3
"""Offline attribution: can the trained prior (59-dim features) tell stepping
from sliding? Preregistered H2a test (exp_78e7c4ac).

Evidence A (group contrast): per-clip mean SDS reward r = exp(-w_s/|K| * sum_{t in K}
MSE_t) at K={8,15,22} (raw MSE, no adaptive norm -- relative comparison only),
computed over each clip's windows. Clips grouped by gait-audit median swing:
  STEPPING  both-feet mean swing >= 0.30 s
  DRAGGING  both-feet mean swing <  0.22 s  (the slide-adjacent end of the library)

Evidence B (foot-feature ablation): on STEPPING clips, replace the two sole
world-contact channels the features DO NOT carry is impossible offline, so we
test sensitivity the other way: corrupt the two FOOT ee blocks of the feature
(6 of the 15 ee dims, zeroed to the window's temporal mean -> a static foot
relative to root, i.e. the sliding-est corruption expressible in feature space)
and measure the reward drop. If the prior barely reacts, foot semantics are
absent -> feature change is mandatory (H2a holds).

Preregistered thresholds:
  H2a HOLDS  if A: dragging mean reward >= 0.80 * stepping mean reward
              AND B: ablated stepping reward >= 0.80 * intact stepping reward
  H2a FAILS  if dragging mean reward < 0.50 * stepping mean reward
             OR ablated reward < 0.50 * intact (prior is foot-sensitive)

Run: gmr-venv python scripts/attribution_prior.py
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
SEED = 0


def sds_reward(model, sched, x0: torch.Tensor, k=K, ws=WS, n_mc=4) -> np.ndarray:
    """x0: (N, W, F) normalized. Mean over n_mc noise draws of
    exp(-ws/|K| * sum_t MSE_t)."""
    rs = []
    g = torch.Generator().manual_seed(SEED)
    for _ in range(n_mc):
        errs = torch.zeros(len(x0))
        for t_int in k:
            t = torch.full((len(x0),), t_int, dtype=torch.long)
            noise = torch.randn(x0.shape, generator=g)
            x_t = sched.add_noise(x0, noise, t)
            eps = model(x_t, t)
            errs += ((eps - noise) ** 2).mean(dim=(-1, -2))
        rs.append(torch.exp(-errs / len(k) * ws))
    return torch.stack(rs).mean(0).numpy()


def main() -> int:
    ck = torch.load(ROOT / 'data/prior/x1_walk_prior.pt', map_location='cpu', weights_only=False)
    model = DiffusionDenoiser(feature_dim=int(ck['cfg']['feature_dim']),
                              window_size=int(ck['cfg']['window_size']),
                              d_model=int(ck['cfg']['d_model']), nhead=int(ck['cfg']['nhead']),
                              num_layers=int(ck['cfg']['num_layers']))
    model.load_state_dict(ck['model']); model.eval()
    sched = DDPMScheduler(num_timesteps=int(ck['cfg']['num_timesteps']))
    q_low = torch.from_numpy(np.asarray(ck['q_low'], dtype=np.float32))
    q_high = torch.from_numpy(np.asarray(ck['q_high'], dtype=np.float32))

    meta = json.load(open(ROOT / 'data/x1_smp_train.meta.json'))
    swing = {}
    for w in meta['windows']:
        if 'gait' in w:
            sw = [s for s in w['gait']['swing'] if s is not None]
            if len(sw) == 2:
                swing[w['clip']] = float(np.mean(sw))

    stepping, dragging = [], []
    per_clip = {}
    for npz in sorted((ROOT / 'data/windows').glob('*.npz')):
        name = npz.stem
        wins = torch.from_numpy(np.load(npz)['windows'].astype(np.float32))
        x0 = 2.0 * (wins - q_low) / (q_high - q_low + 1e-8) - 1.0
        with torch.no_grad():
            r = sds_reward(model, sched, x0)
        per_clip[name] = dict(mean_reward=float(r.mean()), p10=float(np.percentile(r, 10)),
                              n=len(r))
        sw = swing.get(name)
        if sw is not None:
            (stepping if sw >= 0.30 else (dragging if sw < 0.22 else None) or []) # placeholder
    # regroup cleanly (split-assign above was awkward)
    stepping = [per_clip[n]['mean_reward'] for n, s in swing.items() if s >= 0.30 and n in per_clip]
    dragging = [per_clip[n]['mean_reward'] for n, s in swing.items() if s < 0.22 and n in per_clip]
    mid = [n for n, s in swing.items() if 0.22 <= s < 0.30 and n in per_clip]

    # Evidence B: foot-ee ablation on stepping clips (first 400 windows each)
    ee_foot_dims = list(range(3 + 6 + 29, 3 + 6 + 29 + 6))  # ee block: feet first (order: Lfoot,Rfoot,torso,Lw,Rw)
    intact_rs, ablated_rs = [], []
    with torch.no_grad():
        for n, s in swing.items():
            if s < 0.30 or n not in per_clip:
                continue
            wins = torch.from_numpy(np.load(ROOT / 'data/windows' / f'{n}.npz')['windows'].astype(np.float32))
            x0 = 2.0 * (wins - q_low) / (q_high - q_low + 1e-8) - 1.0
            x_ab = x0.clone()
            x_ab[:, :, ee_foot_dims] = x_ab[:, :, ee_foot_dims].mean(dim=1, keepdim=True)
            intact_rs.append(sds_reward(model, sched, x0[:400]))
            ablated_rs.append(sds_reward(model, sched, x_ab[:400]))
    intact = float(np.concatenate(intact_rs).mean())
    ablated = float(np.concatenate(ablated_rs).mean())

    step_m = float(np.mean(stepping)) if stepping else float('nan')
    drag_m = float(np.mean(dragging)) if dragging else float('nan')
    ratio_a = drag_m / step_m if step_m == step_m and step_m > 0 else float('nan')
    ratio_b = ablated / intact if intact > 0 else float('nan')

    verdict = ('H2a HOLDS (feature change mandatory)' if (ratio_a >= 0.80 and ratio_b >= 0.80)
               else 'H2a FAILS (prior discriminates)' if (ratio_a < 0.50 or ratio_b < 0.50)
               else 'INCONCLUSIVE (between thresholds)')
    out = dict(per_clip=per_clip,
               stepping=dict(clips=len(stepping), mean_reward=step_m),
               dragging=dict(clips=len(dragging), mean_reward=drag_m),
               mid_group_clips=mid,
               ratio_drag_over_step=ratio_a,
               ablation=dict(intact=intact, ablated=ablated, ratio=ratio_b),
               verdict=verdict, thresholds=dict(holds='>=0.80 both', fails='<0.50 either'))
    (ROOT / 'eval').mkdir(exist_ok=True)
    (ROOT / 'eval' / 'attribution_prior.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_clip'}, indent=1))
    print('per-clip:', {n: round(v['mean_reward'], 4) for n, v in per_clip.items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
