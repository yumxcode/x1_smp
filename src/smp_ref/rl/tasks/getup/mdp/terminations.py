"""Termination terms for the getup task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["smp_too_low", "stood_up"]


def stood_up(
  env: ManagerBasedRlEnv,
  head_height: float = 1.2,
  max_speed: float = 0.5,
  hold_steps: int = 10,
) -> torch.Tensor:
  """Truncate once STABLY standing (success): head ≥ ``head_height`` and base
  speed < ``max_speed`` for ``hold_steps`` consecutive steps (counter zeroed by
  ``reset_stand_counter``).  Wire ``time_out=True`` so it's a TRUNCATION — the
  value bootstraps from the standing state, else standing looks worthless."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  z = robot.data.site_pos_w[:, head_idx, 2]
  speed = torch.linalg.norm(robot.data.root_link_lin_vel_w, dim=-1)
  is_standing = (z >= head_height) & (speed < max_speed)
  cnt = getattr(env, "_getup_stand_count", None)
  if cnt is None:
    cnt = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  cnt = torch.where(is_standing, cnt + 1, torch.zeros_like(cnt))
  env._getup_stand_count = cnt  # type: ignore[attr-defined]
  return cnt >= hold_steps


def smp_too_low(
  env: ManagerBasedRlEnv,
  threshold: float = 0.02,
  ws: float = 6.0,
  grace_steps: int = 15,
) -> torch.Tensor:
  """Terminate when the SMP score collapses (off-manifold): end if
  ``exp(-ws·env._smp_raw_err) < threshold`` past ``grace_steps``.  Uses the RAW MSE
  (stable absolute scale), so ``ws`` must match the reward's.  Kills the "violent
  get-up" shortcut — leaving the manifold drives the score to 0."""
  raw_err = getattr(env, "_smp_raw_err", None)
  if raw_err is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  raw_smp = torch.exp(-ws * raw_err)
  past_grace = env.episode_length_buf >= grace_steps
  return (raw_smp < threshold) & past_grace
