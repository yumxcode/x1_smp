"""Steering reward components: linear-velocity tracking + face alignment.

SMP-gated via the generic ``smp.rl.rewards.smp_product``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from smp.rl.tasks.steering.mdp.commands import SteeringCommand


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def steering_target_velocity(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``exp(-vel_err_scale * ‖tar_speed·tar_dir - root_vel_xy‖²)``, zeroed when
  root velocity projects negatively onto the target dir (no reward for walking
  the wrong way)."""
  asset = env.scene[asset_cfg.name]
  cmd: "SteeringCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  root_vel_xy = asset.data.root_link_lin_vel_w[:, :2]
  tar_vel = cmd.tar_speed.unsqueeze(-1) * cmd.tar_dir_w
  vel_err = ((tar_vel - root_vel_xy) ** 2).sum(dim=-1)

  proj_speed = (cmd.tar_dir_w * root_vel_xy).sum(dim=-1)
  reward = torch.exp(-vel_err_scale * vel_err)
  reward = torch.where(proj_speed < 0, torch.zeros_like(reward), reward)
  return reward


def steering_face_direction(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``max(face_dir · char_face_dir, 0)`` — both unit world-xy vectors."""
  asset = env.scene[asset_cfg.name]
  cmd: "SteeringCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  heading_w = asset.data.heading_w
  char_face_w = torch.stack([torch.cos(heading_w), torch.sin(heading_w)], dim=-1)
  face_dot = (cmd.face_dir_w * char_face_w).sum(dim=-1)
  return face_dot.clamp_min(0.0)


def contact_guard(
  env: "ManagerBasedRlEnv",
  z0: float = 0.060,
  z_width: float = 0.012,
  v_half: float = 0.20,
) -> torch.Tensor:
  """Explicit contact-consistency guard (AUDIT_FAILURE sect. 6-5): per foot,
  ``sigmoid((z0 - ankle_z)/z_width) * exp(-ln2 * (|v_ankle_xy|/v_half)^2)`` --
  ~1 for a planted, non-sliding foot; ~0 for a near-ground foot translating
  fast (the sliding failure mode); ~0 for a swung foot (no free reward).

  Unit calibration on the v3 physics-solved library (audit 2026-11-05):
  ankle-link XY speed on strict-contact frames (sole z<3mm) p50=0.006,
  p90=0.078, p99=0.34; v_half=0.20 keeps data p50 -> 0.999 while the R3
  sliding policy (both feet ~1mm, v~0.6) -> 0.002. Reward accounting at
  0.5/0.5 additive + guard weight 0.5: proper walk ~1.11 vs slide ~0.66 vs
  stand ~0.65 -- walking wins decisively, standing gains nothing.
  """
  import math

  robot = env.scene["robot"]
  idx = env._smp_ee_indexes[:2]  # type: ignore[attr-defined]  (ankles)
  foot_z = robot.data.body_link_pos_w[:, idx, 2] - env.scene.env_origins[:, 2:3]
  contact = torch.sigmoid((z0 - foot_z) / z_width)
  v = robot.data.body_link_lin_vel_w[:, idx, :2].norm(dim=-1)
  per_foot = contact * torch.exp(-math.log(2.0) * (v / v_half) ** 2)
  return per_foot.mean(dim=-1)
