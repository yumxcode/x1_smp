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


def swing_bonus(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  z_swing: float = 0.073,
  z_width: float = 0.006,
  v_full: float = 0.30,
) -> torch.Tensor:
  """True-stepping positive term (G3' variant, targets the G4 micro-shuffle
  failure mode fnd_9a894dca): per foot,
  ``sigmoid((ankle_z - z_swing)/z_width) * clamp(|v_ankle_xy|/v_full, 0, 1)``
  -- pays only for a foot that is CLEARLY off the ground (ankle_z > ~73mm ~
  sole >30mm, i.e. above the ABS-detector swing threshold) AND translating
  forward-speed-sized. Physics-solved v3 demo: swing frames carry ankle XY
  speed p50 1.18 m/s, so real swings saturate the clamp; pivot-rolling
  (G4 policy: soles <4mm, ~0 swing frames/20s) earns ~0.

  Unit calibration on the v3 library (audit/gait_reward_calibration_v2.json,
  2026-11-05): demo mean 0.205 (18.7% of foot-frames above z_swing, ankle
  swing-speed p50 1.18 m/s -> clamp saturated); the G4 pivot-shuffle policy
  (~0 swing phases / 20 s) earns ~0. Wired at weight 0.5.
  """
  robot = env.scene["robot"]
  cmd: "SteeringCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  idx = env._smp_ee_indexes[:2]  # type: ignore[attr-defined]  (ankles)
  foot_z = robot.data.body_link_pos_w[:, idx, 2] - env.scene.env_origins[:, 2:3]
  off_gnd = torch.sigmoid((foot_z - z_swing) / z_width)
  v_vec = robot.data.body_link_lin_vel_w[:, idx, :2]
  # G3'' (exp_5f6ae123): FORWARD PROJECTION instead of |v| -- the G3' per-frame
  # isotropic speed term paid more for short QUICK steps (cadence 2.63-2.94 Hz
  # vs R1 band [0.55,1.2], eval/g4_results_g3p.json) and paid sideways swings
  # (training error_face ~0.9 rad). v.tar_dir integrates to forward swing
  # DISPLACEMENT: frequency-neutral (pay per meter of forward swing travel),
  # and lateral swings earn 0.
  v_fwd = (v_vec * cmd.tar_dir_w.unsqueeze(1)).sum(dim=-1).clamp_min(0.0)
  per_foot = off_gnd * (v_fwd / v_full).clamp(0.0, 1.0)
  return per_foot.mean(dim=-1)


def midband_penalty(
  env: "ManagerBasedRlEnv",
  z_lo: float = 0.046,
  z_hi: float = 0.073,
  z_width: float = 0.004,
  v_free: float = 0.05,
  v_full: float = 0.15,
) -> torch.Tensor:
  """Micro-shuffle penalty (G3' variant): per foot, a NEGATIVE term active
  only in the dead band BETWEEN the guard's contact zone and the swing zone
  (ankle_z in (z_lo, z_hi) ~ sole (3, 30mm)) while translating
  ``-band(z) * clamp((|v|-v_free)/(v_full-v_free), 0, 1)``. The G4 policy
  bought translation with 3-8mm hops exactly inside this band, invisible to
  both the slip metric (sole z<3mm) and the swing detector (z>25mm) while
  still collecting partial guard credit. Physics-solved v3 demo only passes
  through the band briefly at lift-off/touchdown (mean value 0.036), so the
  penalty discriminates shuffle from walk.

  CALIBRATION FALSIFIED pre-training (audit/gait_reward_calibration_v2.json):
  the v3 demo itself occupies the band 29.3% of foot-frames (mean 0.260)
  -- MORE than the G4 micro-shuffle policy (~8-12%, soles <4mm on 90% of
  frames: its translation is sub-band pivot rolling, not in-band hops).
  This penalty would punish real walking harder than the failure mode, so it
  is NOT wired into forward_env_cfg. Kept for the record.
  """
  robot = env.scene["robot"]
  idx = env._smp_ee_indexes[:2]  # type: ignore[attr-defined]  (ankles)
  foot_z = robot.data.body_link_pos_w[:, idx, 2] - env.scene.env_origins[:, 2:3]
  band = (torch.sigmoid((foot_z - z_lo) / z_width)
          * (1.0 - torch.sigmoid((foot_z - z_hi) / z_width)))
  v = robot.data.body_link_lin_vel_w[:, idx, :2].norm(dim=-1)
  moving = ((v - v_free) / (v_full - v_free)).clamp(0.0, 1.0)
  per_foot = band * moving
  return -per_foot.mean(dim=-1)


def phase_gait(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  swing_width: float = 0.40,
  swing_center_left: float = 0.84,
  edge_soft: float = 0.04,
) -> torch.Tensor:
  """Clock-conditioned gait reward (G3-C1, amendment rev4): pays for feet that
  follow the commanded gait phase -- each foot should be AIRBORNE inside its
  phase swing window (raised-cosine-soft, width `swing_width` of the cycle,
  duty = 1 - swing_width ~ 0.60, the v3 AMP demo regime) and PLANTED outside
  it; left/right windows are antiphase. Requires cfg gait_clock=True on the
  steering command (sin/cos/freq observed by the policy).

  Unit calibration on the v3 physics-solved library against its own gait
  clock: see audit/gait_clock_calibration.json. Rationale: three reward-only
  rounds never pulled cadence into R1 [0.55,1.2] Hz (fnd_63ef26fd) -- here
  cadence is COMMANDED (sampled in-band) and the phase reward schedules the
  alternation, so the clock is both observable and rewarded.
  """
  import math

  robot = env.scene["robot"]
  cmd: "SteeringCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  idx = env._smp_ee_indexes[:2]  # type: ignore[attr-defined]  (ankles)
  foot_z = robot.data.body_link_pos_w[:, idx, 2] - env.scene.env_origins[:, 2:3]
  air = torch.sigmoid((foot_z - 0.073) / 0.006)          # clearly off ground
  plant = torch.sigmoid((0.060 - foot_z) / 0.012)         # guard's contact zone
  v_vec = robot.data.body_link_lin_vel_w[:, idx, :2]
  v_fwd = (v_vec * cmd.tar_dir_w.unsqueeze(1)).sum(dim=-1).clamp_min(0.0)
  travel = (v_fwd / 0.30).clamp(0.0, 1.0)                 # forward swing travel

  p = cmd.gait_phase / (2.0 * math.pi)                    # cycle pos [0,1)
  swings = []
  for k, center in ((0, swing_center_left), (1, swing_center_left - 0.5)):
    d = torch.remainder(p - center + 0.5, 1.0) - 0.5      # signed dist to center
    swings.append(torch.sigmoid((swing_width / 2 - torch.abs(d)) / edge_soft))
  s_L, s_R = swings
  # G3-C2 payment form (exp_cc15ed32): swing pays only with REAL forward travel
  # (lift + step, closes the lift-without-stepping loophole); stance pays only
  # while the OTHER foot is inside its swing window (true support timing); the
  # double-stance windows pay 0. A permanently planted foot loses all income
  # outside the partner's swing windows -> single-foot gait (fnd_b234164e,
  # G3-C1: L never lifted) is no longer near-optimal.
  r_L = s_L * air[:, 0] * travel[:, 0] + (1.0 - s_L) * plant[:, 0] * s_R
  r_R = s_R * air[:, 1] * travel[:, 1] + (1.0 - s_R) * plant[:, 1] * s_L
  return 0.5 * (r_L + r_R)
