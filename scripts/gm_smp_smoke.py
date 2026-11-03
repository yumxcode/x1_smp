#!/usr/bin/env python3
"""P0 platform smoke: verify the mjlab-1.6.0 + X1-SMP stack on a 4090D.

Four checks, all before any paid training:
  1. install mjlab==1.6.0 (+ deps) into the platform image and print resolved
     versions (torch stays the image's CUDA build);
  2. import smp (registers Smp-Forward-X1) and build the X1 EntityCfg, compile
     a Scene, and assert the compiled joint order == X1_JOINT_ORDER;
  3. batched Simulation rollout: 2048 envs, 100 physics steps at 1 kHz (0.5 s)
     with zero joint-position targets, report peak VRAM + max |qd| (contact
     stability under the 200 Hz solref fix);
  4. TinyMDM smoke: 3M-param denoiser forward+backward on random windows
     (batch 512, feature 59, window 10) on CUDA, report step time.

Exit 0 = stack good. Any failure exits nonzero with the traceback in the log.
"""

from __future__ import annotations

import subprocess
import sys
import time


def sh(cmd: str) -> None:
    print(f'+ {cmd}', flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> int:
    print('== [1/4] install stack ==', flush=True)
    print(f'python {sys.version}', flush=True)
    import subprocess as sp
    for idx in ('', 'https://pypi.org/simple', 'https://mirrors.aliyun.com/pypi/simple/'):
        flag = ['--index-url', idx] if idx else []
        r = sp.run([sys.executable, '-m', 'pip', 'install', 'mjlab==1.6.0', *flag],
                   capture_output=True, text=True)
        ok = r.returncode == 0
        print(f'pip mjlab==1.6.0 idx={idx or "default"}: {"OK" if ok else r.stderr.strip().splitlines()[-1][:200]}', flush=True)
        if ok:
            break
    else:
        print('FAIL: no pip index has mjlab==1.6.0'); return 2
    import torch
    print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} '
          f'dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?"}')
    if not torch.cuda.is_available():
        print('FAIL: no CUDA'); return 2
    import mujoco, mujoco_warp  # noqa
    print(f'mujoco {mujoco.__version__}')

    print('== [2/4] X1 entity ==', flush=True)
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    from smp.x1_asset import get_x1_robot_cfg, X1_JOINT_ORDER, x1_action_scale
    cfg = get_x1_robot_cfg()
    print(f'X1 EntityCfg ok; action scale joints={len(x1_action_scale())}')

    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    scene_cfg = SceneCfg(terrain=TerrainEntityCfg(terrain_type='plane'),
                         entities={'robot': cfg}, num_envs=8, extent=2.0)
    device = 'cuda'
    scene = Scene(scene_cfg, device=device)
    model = scene.compile()
    from smp.x1_asset import joint_names_match
    robot = scene['robot']
    joint_names_match(list(robot.joint_names))
    print(f'Scene compiled: nq={model.nq} nv={model.nv}; joint order OK (29)')

    print('== [3/4] batched rollout ==', flush=True)
    from mjlab.sim import SimulationCfg
    from mjlab.sim.sim import Simulation as Sim
    sim_cfg = SimulationCfg()
    scene_cfg.num_envs = 2048
    scene2 = Scene(scene_cfg, device=device)
    model2 = scene2.compile()
    t0 = time.time()
    sim = Sim(num_envs=2048, cfg=sim_cfg, model=model2, device=device)
    scene2.initialize(sim.mj_model, sim.model, sim.data)
    scene2.reset()
    for _ in range(100):
        sim.step()
        scene2.update(sim.mj_model.opt.timestep)
    torch.cuda.synchronize()
    qd = scene2['robot'].data.joint_vel
    print(f'2048-env rollout 100 steps in {time.time()-t0:.1f}s; '
          f'max|qd|={qd.abs().max().item():.2f} (expect O(1-10), not 1e6); '
          f'peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GiB')
    if qd.abs().max().item() > 1e3:
        print('FAIL: contact blew up (solref issue)'); return 3

    print('== [4/4] TinyMDM smoke ==', flush=True)
    from smp.pretrain.model import DiffusionDenoiser
    m = DiffusionDenoiser(feature_dim=59, window_size=10, d_model=128,
                          nhead=4, num_layers=2).to(device)
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.randn(512, 10, 59, device=device)
    t = torch.randint(0, 50, (512,), device=device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    t0 = time.time()
    for _ in range(20):
        eps = m(x, t)
        loss = (eps - torch.randn_like(eps)).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
    print(f'TinyMDM {n_params/1e6:.2f}M params; 20 train steps @ batch512: '
          f'{dt*1000:.0f} ms/step -> {200_000*dt/3600:.1f} h per 200k steps')
    print('P0 SMOKE PASS', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
