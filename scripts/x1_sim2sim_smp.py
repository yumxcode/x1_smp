"""X1 SMP sim2sim: replay an SMP policy checkpoint in CPU MuJoCo (G5 deliverable).

Replicates the training env's control loop with plain MuJoCo (no mjlab/CUDA):
  - model: assets/x1_29dof/x1_29dof.xml, stock actuators deleted, per-joint PD
    position servos from smp.x1_asset.X1_JOINT_PARAMS (kp/kd/armature/frictionloss/
    forcerange=±effort) -- the same constants mjlab's BuiltinPositionActuatorCfg
    compiles on the platform;
  - foot geoms: condim 3, priority 1, friction 0.6, solref (0.02, 1.0) (asset cfg;
    training DR-sampled [0.3, 1.2], we take the mid value);
  - physics: timestep 0.002, decimation 10 -> 50 Hz policy (SimulationCfg default);
  - obs (101-dim, training layout, no corruption): [body_lin_vel(3, imu site),
    body_ang_vel(3, imu gyro), projected_gravity(3), joint_pos_rel(29),
    joint_vel_rel(29), last_action(29), command(5: tar_dir_b(2), tar_speed,
    face_dir_b(2))];
  - actor: ckpt['actor_state_dict'], mlp [512,256,128] ELU, deterministic mean,
    obs normalized by the ckpt's obs_normalizer;
  - action: ctrl = a * (0.25*effort/kp) + default(0) (JointPositionActionCfg);
  - reset: from a v3-library walking frame (the GSI distribution the platform
    G4 rollouts reset from), root yaw kept, world aligned to +x command.

Output per rollout: qpos npz + rendered mp4 (frames -> PNGs -> encoded with the
repo .venv's imageio-ffmpeg; the GMR venv has no ffmpeg) + metrics via the same
analyse() as scripts/gm_g4_eval.py.

  ~/code/X1_29_AMP/_gmr_local/venv/bin/python scripts/x1_sim2sim_smp.py \
      --ckpt data/policy/model_g3p_6000.pt --speeds 0.45 0.5 --out sim2sim/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'pipeline'))

import mujoco  # noqa: E402

from smp.x1_asset import (  # noqa: E402
    MESH_DIR, X1_INIT_HEIGHT, X1_JOINT_ORDER, X1_JOINT_PARAMS, X1_XML,
)
from x1_data_lib import X1_JOINT_ORDER as LIB_JOINTS  # noqa: E402  (sanity)


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(X1_XML))
    spec.assets = {f'{spec.meshdir}/{p.name}': p.read_bytes()
                   for p in sorted(MESH_DIR.iterdir()) if p.is_file()}
    for act in list(spec.actuators):
        spec.delete(act)
    for name in X1_JOINT_ORDER:
        p = X1_JOINT_PARAMS[name]
        j = spec.joint(name)
        j.armature = p['armature']        # mjlab applies these at the joint
        j.frictionloss = p['frictionloss']
        spec.add_actuator(
            name=f'servo_{name}',
            target=name,
            trntype=int(mujoco.mjtTrn.mjTRN_JOINT),
            gainprm=[p['kp']] + [0.0] * 9,
            biasprm=[0.0, -p['kp'], -p['kd']] + [0.0] * 7,
            forcerange=[-p['effort'], p['effort']],
            ctrlrange=[-4.0, 4.0],
            ctrllimited=True,
        )
    # FLOOR: mjlab's TerrainEntityCfg adds the plane at scene level; the robot
    # XML has none (audit round 2 caught the first replay free-falling 1954 m).
    plane = spec.worldbody.add_geom()
    plane.name = 'floor'
    plane.type = mujoco.mjtGeom.mjGEOM_PLANE
    plane.size = [0.0, 0.0, 0.1]
    plane.pos = [0.0, 0.0, 0.0]
    plane.contype = 1
    plane.conaffinity = 1
    plane.condim = 3
    plane.friction = [1.0, 0.005, 0.0001]
    m = spec.compile()
    # foot contact geom params (x1_collision_cfgs; friction 0.6 = mid of the
    # training DR range [0.3, 1.2])
    foot_geoms = {f'{s}_foot_{k}_collision' for s in ('left', 'right') for k in (1, 2, 3, 4)}
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if n in foot_geoms:
            m.geom_contype[g] = 1   # CollisionCfg sets contype/conaffinity=1 on
            m.geom_conaffinity[g] = 1  # all '.*_collision' geoms; the stock XML
            m.geom_condim[g] = 3    # leaves them 0 (no contact) -- the second
            m.geom_priority[g] = 1  # replay fell THROUGH the floor without this
            m.geom_friction[g, 0] = 0.6
            m.geom_solref[g, 0] = 0.02
        elif n.endswith('_collision'):
            m.geom_contype[g] = 1
            m.geom_conaffinity[g] = 1
            m.geom_condim[g] = 1
    m.opt.timestep = 0.002
    return m


class Actor:
    def __init__(self, ckpt_path: str):
        import torch
        ck = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        a = ck['actor_state_dict']
        self.mean = a['obs_normalizer._mean'].numpy().ravel()
        self.var = a['obs_normalizer._var'].numpy().ravel()
        self.layers = []
        for i in (0, 2, 4, 6):
            w = a[f'mlp.{i}.weight'].numpy()
            b = a[f'mlp.{i}.bias'].numpy()
            self.layers.append((w, b))
        self.hidden = self.layers[:-1]
        self.out = self.layers[-1]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        for w, b in self.hidden:
            x = w @ x + b
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 20.0)))  # ELU
        w, b = self.out
        return w @ x + b


def quat_to_mat(q):  # wxyz
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def sensor_slices(m: mujoco.MjModel):
    adr, dim = {}, {}
    for name in ('body-linear-vel', 'body-angular-velocity'):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        assert sid >= 0, name
        adr[name] = m.sensor_adr[sid]
        dim[name] = m.sensor_dim[sid]
    return adr, dim


def rollout(actor: Actor, speed: float, secs: float, init_frame: np.ndarray,
            render: bool, out_prefix: Path):
    m = build_model()
    d = mujoco.MjData(m)
    adr, _ = sensor_slices(m)
    scale = np.array([0.25 * X1_JOINT_PARAMS[n]['effort'] / X1_JOINT_PARAMS[n]['kp']
                      for n in X1_JOINT_ORDER])
    n_steps = int(secs * 50)
    dec = 10

    # reset from the v3-library walking frame (GSI distribution)
    d.qpos[:] = init_frame[:len(d.qpos)]
    d.qvel[:] = init_frame[len(d.qpos):len(d.qpos) + len(d.qvel)]
    mujoco.mj_forward(m, d)

    qpos_log = np.zeros((n_steps, len(d.qpos)))
    fell = False
    last_a = np.zeros(29)
    frames_dir = out_prefix.parent / f'{out_prefix.name}_frames'
    rend = cam = None
    if render:
        frames_dir.mkdir(exist_ok=True)
        rend = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.distance = 3.2
        cam.azimuth = -90
        cam.elevation = -25

    for t in range(n_steps):
        R = quat_to_mat(d.qpos[3:7])
        grav_b = R.T @ np.array([0.0, 0.0, -1.0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(-yaw), np.sin(-yaw)
        tar_b = np.array([c, s])          # world +x in heading frame
        face_b = np.array([c, s])
        obs = np.concatenate([
            d.sensordata[adr['body-linear-vel']:adr['body-linear-vel'] + 3],
            d.sensordata[adr['body-angular-velocity']:adr['body-angular-velocity'] + 3],
            grav_b,
            d.qpos[7:7 + 29],
            d.qvel[6:6 + 29],
            last_a,
            [tar_b[0], tar_b[1], speed, face_b[0], face_b[1]],
        ])
        a = actor(obs)
        d.ctrl[:] = a * scale
        for _ in range(dec):
            mujoco.mj_step(m, d)
        last_a = a
        qpos_log[t] = d.qpos
        if not np.isfinite(d.qpos).all() or d.qpos[2] < 0.28:
            # training termination: base_too_low minimum_height 0.28
            fell = True
            qpos_log = qpos_log[:t + 1]
            break
        if rend is not None and t % 4 == 0:
            cam.lookat[:] = d.qpos[:3]
            rend.update_scene(d, cam)
            import imageio.v2 as iio
            iio.imwrite(frames_dir / f'f{t:05d}.png', rend.render())
    if rend is not None:
        rend.close()
    return qpos_log, fell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='data/policy/model_g3p_6000.pt')
    ap.add_argument('--speeds', nargs='+', type=float, default=[0.45, 0.5])
    ap.add_argument('--secs', type=float, default=20.0)
    ap.add_argument('--out', default='sim2sim')
    ap.add_argument('--render', type=int, default=1)
    ap.add_argument('--clip', default=0, type=int, help='v3 library clip index for reset')
    ap.add_argument('--frame', default=0, type=int, help='frame within the clip for reset')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    lib = np.load(ROOT / 'data' / 'x1_smp_train_v3.npz', allow_pickle=True)
    start = int(lib['starts'][args.clip]) + args.frame
    # init frame = [qpos(36)] + [qvel(35)]; the rollout() maps them onto the model
    init = np.concatenate([lib['qpos'][start], lib['qvel'][start]])

    from gm_g4_eval import analyse
    actor = Actor(args.ckpt)
    report = {}
    for speed in args.speeds:
        tag = f'v{speed:.2f}'.replace('.', '')
        qpos, fell = rollout(actor, speed, args.secs, init, bool(args.render), out / tag)
        np.savez_compressed(out / f'{tag}_qpos.npz', qpos=qpos)
        r = analyse(qpos)
        r['speed_cmd'] = speed
        r['fell'] = fell
        report[tag] = r
        print(f'[s2s] {tag}: fell={fell} speed_p50={r["root_speed_p50"]:.3f} '
              f'heading={r["heading_offset_deg_p50"]:.0f}deg '
              f'cad={r["feet"]["L"]["abs_cadence_hz"]}/{r["feet"]["R"]["abs_cadence_hz"]} '
              f'z5mm={r["feet"]["L"]["z_gt5mm_frac"]:.2f}/{r["feet"]["R"]["z_gt5mm_frac"]:.2f}',
              flush=True)
    json.dump(report, open(out / 'sim2sim_metrics.json', 'w'), indent=1)
    print('->', out / 'sim2sim_metrics.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
