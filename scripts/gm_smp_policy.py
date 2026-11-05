#!/usr/bin/env python3
"""gm-run entry for P3/P4: policy training on Smp-Forward-X1 (PPO + SMP prior).

Installs the mjlab 1.6.0 stack (multi-index retry), then launches mjlab's
train script with the X1 task registered. The prior checkpoint is fetched
from data/prior/x1_walk_prior.pt (checked into the repo by the P2 mirror
step) or overridden via SMP_PRIOR_CKPT.

  gm-run X1_SMP/scripts/gm_smp_policy.py                     # default 4096 envs
  python scripts/gm_smp_policy.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK_ID = os.environ.get('SMP_TASK', 'Smp-Forward-X1')


def install_stack() -> bool:
    import subprocess as sp
    for idx in ('', 'https://pypi.org/simple', 'https://mirrors.aliyun.com/pypi/simple/'):
        flag = ['--index-url', idx] if idx else []
        r = sp.run([sys.executable, '-m', 'pip', 'install', 'mjlab==1.6.0', 'warp-lang==1.16.0', *flag],
                   capture_output=True, text=True)
        ok = r.returncode == 0
        print(f'pip mjlab==1.6.0 idx={idx or "default"}: {"OK" if ok else r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else r.stdout.strip().splitlines()[-1][:160]}', flush=True)
        if ok:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--num-envs', type=int, default=int(os.environ.get('SMP_NUM_ENVS', '2048')))
    ap.add_argument('--max-iterations', type=int, default=int(os.environ.get('SMP_ITERS', '10000')))
    ap.add_argument('--resume', default='')
    args, extra = ap.parse_known_args()

    prior = Path(os.environ.get('SMP_PRIOR_CKPT', ROOT / 'data' / 'prior' / 'x1_walk_prior_v3.pt'))
    print(f'[policy] task={TASK_ID} envs={args.num_envs} iters={args.max_iterations} prior={prior}', flush=True)
    if not prior.exists():
        print(f'FAIL: prior checkpoint missing: {prior}', flush=True)
        return 2
    if args.dry_run:
        return 0

    print('[policy] installing stack', flush=True)
    if not install_stack():
        print('FAIL: mjlab install failed on all indexes', flush=True)
        return 2

    # verify the X1 entity compiles before spending training time
    sys.path.insert(0, str(ROOT / 'src'))
    import smp.rl.tasks  # noqa: F401  registers Smp-Forward-X1
    from smp.x1_asset import get_x1_robot_cfg
    get_x1_robot_cfg()
    print('[policy] X1 EntityCfg builds OK', flush=True)

    # runtime effectiveness probe (unconditional): reward composition + band
    from smp.rl.tasks.steering.forward_env_cfg import x1_forward_smp_env_cfg
    _c = x1_forward_smp_env_cfg(play=True)
    print('[policy] task_terms:', [(f.__name__, w) for f, w, _ in
          _c.rewards['task_smp_additive'].params['task_terms']],
          '| band', _c.commands['steering'].tar_speed_min, '-',
          _c.commands['steering'].tar_speed_max, flush=True)

    os.environ.setdefault('WANDB_MODE', 'disabled')

    # resource watchdog: log GPU/RAM every 60 s so silent instance kills
    # (two deaths at iter ~405/471 with no traceback) leave evidence behind
    import threading
    def _watchdog():
        import subprocess as sp, time as _t
        while True:
            _t.sleep(60)
            try:
                gpu = sp.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                              '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
                ram = sp.run(['free', '-m'], capture_output=True, text=True).stdout.splitlines()[1]
                print(f'[watchdog] GPU {gpu} | RAM {ram}', flush=True)
            except Exception:
                pass
    threading.Thread(target=_watchdog, daemon=True).start()
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src') + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    cmd = [sys.executable, str(ROOT / 'scripts' / 'train_x1.py'), TASK_ID,
           f'--env.scene.num-envs={args.num_envs}',
           f'--agent.max-iterations={args.max_iterations}',
           f'--env.events.init_smp_state.params.ckpt_path={prior}'] + extra
    if args.resume:
        cmd += ['--resume', args.resume]
    print('[policy] exec:', ' '.join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print(f'[policy] train exited {proc.returncode} after {(time.time()-t0)/60:.1f} min', flush=True)

    # mirror newest checkpoints to output/ (outside the git work tree)
    out_dir = Path('/workspace/output') if Path('/workspace').exists() else ROOT / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cand in (ROOT / 'logs', ROOT / 'runs'):
        if cand.exists():
            for ck in sorted(cand.rglob('model_*.pt'), key=lambda p: p.stat().st_mtime)[-4:]:
                shutil.copy2(ck, out_dir / f'policy_{ck.name}')
                n += 1
    print(f'[policy] mirrored {n} checkpoints to {out_dir}', flush=True)
    return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())
