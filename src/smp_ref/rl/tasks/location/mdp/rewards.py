"""Location reward component: position tracking.

SMP-gated via the generic ``smp.rl.rewards.smp_product``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from smp.rl.tasks.location.mdp.commands import LocationCommand


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def location_position(
  env: "ManagerBasedRlEnv",
  command_name: str,
  pos_err_scale: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``exp(-pos_err_scale * ‖tar_pos_xy - root_pos_xy‖²)``."""
  asset = env.scene[asset_cfg.name]
  cmd: "LocationCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  pos_diff = cmd.tar_pos_w - asset.data.root_link_pos_w[:, :2]
  pos_err = (pos_diff * pos_diff).sum(dim=-1).sqrt()
  return torch.exp(-pos_err_scale * pos_err)
