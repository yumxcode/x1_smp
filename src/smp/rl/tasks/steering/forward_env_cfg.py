"""G1 forward task — fixed +x heading, variable target speed.

A specialization of the steering task with the world-frame target and face
directions pinned to ``+x``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg, x1_smp_env_cfg
from smp.rl.rewards import task_smp_additive, task_smp_product
from smp.rl.tasks.steering import mdp


def g1_forward_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 forward env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Commands ------------------------------------------------------------
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rand_tar_dir=False,
    rand_face_dir=False,
    tar_speed_min=0.5,
    tar_speed_max=5.0,
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "steering"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Rewards -------------------------------------------------------------
  # task = velocity tracking, gated by SMP.
  # round 3 (exp_c8a50775): paper's ADDITIVE reward + 0.4-0.9 command band
  # (median 0.65 ~= prior manifold center 0.62 -> speed-discount tension gone)
  cfg.rewards["task_smp_additive"] = RewardTermCfg(
    func=task_smp_additive,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.steering_target_velocity,
          1.0,
          {"command_name": "steering", "vel_err_scale": 0.5},
        ),
      ),
      "w_task": 0.5,
      "w_smp": 0.5,
    },
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_loco.pt"
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg


def x1_forward_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """X1 forward env: fixed +x heading, low-cadence speed band 0.4-1.4 m/s.

  The band covers the walk05 (0.5) and walk10 (1.0) evaluation commands with
  margin on both sides; the prior library itself spans 0.4-1.3 m/s
  (data/x1_smp_train.npz, gait-audited).
  """
  cfg = x1_smp_env_cfg(play=play)
  # X1 port: drop the periodic GSI-pool refresh. Both 4096-env and 2048-env
  # instances died silently (no traceback, system-level kill) at iter ~405/471,
  # i.e. right after the 4th 2400-step refresh; the static 4096-window pool
  # suffices (upstream play cfg disables refresh as well).
  cfg.events.pop("gsi_refresh", None)
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(6.0, 12.0),
    rand_tar_dir=False,
    rand_face_dir=False,
    # G3 (amendment rev3): target 0.45 m/s; band centered on the v3 physics
    # library (0.455-0.482 m/s, cad 0.78 Hz) with margin both sides.
    tar_speed_min=0.35,
    tar_speed_max=0.60,
    # G3-C1 (amendment rev4): gait clock -- cadence commanded in-band
    # [0.70, 1.00] Hz (R1 band [0.55, 1.2]), phase observed (sin/cos/freq),
    # phase_gait reward schedules the stance/swing alternation.
    gait_clock=True,
    gait_freq_min=0.70,
    gait_freq_max=1.00,
    debug_vis=False,
  )
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "steering"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs
  # round 3 (exp_c8a50775): paper's ADDITIVE reward + 0.4-0.9 command band
  # (median 0.65 ~= prior manifold center 0.62 -> speed-discount tension gone)
  cfg.rewards["task_smp_additive"] = RewardTermCfg(
    func=task_smp_additive,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.steering_target_velocity,
          1.0,
          # G3' variant: 0.5 -> 3.0. At 0.5 the shuffle speed discount (0.27 vs
          # cmd 0.45, err^2=0.032) cost only exp(-0.016)=1.6% of the task term
          # -- no pressure to track the command. At 3.0 the same discount costs
          # 9%, and demos (92% tracking) stay at 99.5%+.
          {"command_name": "steering", "vel_err_scale": 3.0},
        ),
        # G3 (sect. 6-5): explicit contact guard (unit-calibrated, see
        # mdp.rewards.contact_guard docstring) on top of the additive form.
        (
          mdp.contact_guard,
          0.5,
          {},
        ),
        # G3' variant (audit round 1), calibrated on the v3 library
        # (audit/gait_reward_calibration_v2.json). The G4 failure mode is
        # sub-band pivot rolling (fnd_9a894dca: soles <4mm on 90% frames, ~0
        # swing phases / 20 s). swing_bonus pays only above the ABS swing
        # threshold with forward-sized speed: demo mean 0.205 vs pivot-gait
        # ~0; weight 0.5 restores the intended per-step magnitude. The
        # midband penalty was DESIGNED but calibration falsified it (demo
        # occupies the band 29% -> would punish walking more than the
        # shuffle) -- see mdp.midband_penalty docstring; NOT wired.
        (
          mdp.swing_bonus,
          0.3,
          {"command_name": "steering"},
        ),
        # G3-C1: the clock-conditioned alternation reward (see
        # mdp.rewards.phase_gait) -- cadence is now COMMANDED and scheduled;
        # swing_bonus drops to 0.3 (timing handover to phase_gait).
        (
          mdp.phase_gait,
          0.5,
          {},
        ),
        # G3-C1: heading term (fnd_8ec2bc0a -- swing rewards decouple foot
        # motion from body heading; error_face drifted 0.9-1.3 rad).
        (
          mdp.steering_face_direction,
          0.3,
          {"command_name": "steering"},
        ),
      ),
      "w_task": 0.5,
      "w_smp": 0.5,
    },
  )
  import os
  cfg.events["init_smp_state"].params["ckpt_path"] = os.environ.get(
    "SMP_PRIOR_CKPT", "data/prior/x1_walk_prior_v3.pt")
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": 0.28, "asset_cfg": SceneEntityCfg("robot")},
  )
  return cfg
