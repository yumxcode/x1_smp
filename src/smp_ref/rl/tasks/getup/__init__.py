"""SMP getup task — registers ``Smp-Getup-G1`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.getup.getup_env_cfg import g1_getup_smp_env_cfg

_getup_rl = unitree_g1_smp_ppo_runner_cfg()
_getup_rl.experiment_name = "smp_getup_g1"
_getup_rl.run_name = "smp_getup_g1"

register_mjlab_task(
  task_id="Smp-Getup-G1",
  env_cfg=g1_getup_smp_env_cfg(play=False),
  play_env_cfg=g1_getup_smp_env_cfg(play=True),
  rl_cfg=_getup_rl,
)

__all__ = ["g1_getup_smp_env_cfg"]
