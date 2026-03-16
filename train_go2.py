#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Symmetry-Guided RL Training for Unitree Go2 Quadrupedal Gaits
Paper: "Towards Dynamic Quadrupedal Gaits" (arXiv:2510.10455v1)

Usage:
    conda activate py38_env
    python train_go2.py

This trains a PPO policy on the Go2 robot using unitree_mujoco's MJCF model
with symmetry-guided rewards (temporal, morphological, time-reversal).
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Add current dir to path so we can import gait_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gait_utils


# ═══════════════════════════════════════════════════════════════════════════════
# Go2 Constants (from go2.xml)
# ═══════════════════════════════════════════════════════════════════════════════
FOOT_GEOM_NAMES = ["FL", "FR", "RL", "RR"]
LEG_SLICES = {
    "FL": slice(3, 6),
    "FR": slice(0, 3),
    "RL": slice(9, 12),
    "RR": slice(6, 9),
}
HIP_INDICES = [0, 3, 6, 9]
HOME_QPOS = np.array([0, 0.9, -1.8, 0, 0.9, -1.8,
                       0, 0.9, -1.8, 0, 0.9, -1.8])


# ═══════════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════════
class Go2SymmetryEnv(gym.Env):
    """Unitree Go2 with symmetry-guided rewards for diverse gait training."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, xml_path=None):
        super(Go2SymmetryEnv, self).__init__()

        if xml_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(
                base, "unitree_mujoco", "unitree_robots", "go2", "scene.xml"
            )
        self.xml_path = xml_path
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Timing (paper: 50 Hz control, 200 Hz sim)
        self.sim_dt = 0.005
        self.ctrl_dt = 0.02
        self.mj_model.opt.timestep = self.sim_dt
        self.n_substeps = int(round(self.ctrl_dt / self.sim_dt))
        self.max_steps = 1000  # 20s episodes

        # PD controller (paper: Kp=30, Kd=0.65)
        self.kp = 30.0
        self.kd = 0.65
        self.action_scale = 0.5

        # IDs
        self._base_id = self.mj_model.body("base_link").id
        self._floor_id = self.mj_model.geom("floor").id
        self._foot_geom_ids = [
            self.mj_model.geom(n).id for n in FOOT_GEOM_NAMES
        ]
        self._foot_body_ids = [
            self.mj_model.body(n + "_foot").id for n in FOOT_GEOM_NAMES
        ]

        # Spaces: obs=52, act=12
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(52,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(12,), dtype=np.float32
        )

        # Reward weights (paper Table II + regularization)
        self.rw = {
            "track_lin": 2.0, "track_ang": 1.0, "base_h": -2.0,
            "torque_d": -0.001, "hip": -0.5, "clearance": -1.0,
            "stance_v": -2.0, "swing_grf": -2.0, "morph": -1.0,
            "orient": -5.0, "vz": -0.5, "wxy": -0.05,
            "act_rate": -0.01, "term": -1.0, "energy": -0.001,
        }
        self.sigma = 0.25
        self.render_mode = render_mode
        self._renderer = None
        self._rng = np.random.default_rng()
        self._reset_state()

    def _reset_state(self):
        self._step_n = 0
        self._phase = 0.0
        self._last_act = np.zeros(12)
        self._last_torque = np.zeros(12)
        self._offsets = np.zeros(4)
        self._cmd = np.zeros(3)
        self._T = 1.0
        self._beta = 0.6
        self._gait = "trot"

    def reset(self, seed=None, options=None):
        super(Go2SymmetryEnv, self).reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        self.mj_data.qpos[0:2] += self._rng.uniform(-0.1, 0.1, 2)
        self.mj_data.qvel[0:6] = self._rng.uniform(-0.3, 0.3, 6)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self._gait, self._offsets = gait_utils.sample_gait(self._rng)
        self._cmd = gait_utils.sample_velocity_command(self._rng)
        self._T = gait_utils.compute_stride_period(
            self._cmd[0], self._rng.uniform(-0.05, 0.05)
        )
        self._beta = gait_utils.compute_duty_factor(
            self._cmd[0], self._rng.uniform(-0.02, 0.02)
        )
        self._reset_state()

        return self._obs(), {"gait": self._gait, "cmd": self._cmd.copy()}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # PD control
        tgt = HOME_QPOS + action * self.action_scale
        jpos = self.mj_data.qpos[7:19]
        jvel = self.mj_data.qvel[6:18]
        torque = self.kp * (tgt - jpos) - self.kd * jvel
        cr = self.mj_model.actuator_ctrlrange
        torque = np.clip(torque, cr[:, 0], cr[:, 1])
        self.mj_data.ctrl[:] = torque

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.mj_model, self.mj_data)

        self._phase = (self._phase + self.ctrl_dt / self._T) % 1.0
        self._step_n += 1

        # Leg phases with time-reversal
        lp = gait_utils.compute_leg_phases(self._phase, self._offsets)
        lp = gait_utils.apply_time_reversal(lp, self._cmd[0], self._offsets)

        # Contacts
        contacts = self._contacts()

        # Rewards
        r = self._rewards(action, lp, contacts, torque)
        total = sum(self.rw.get(k, 0) * v for k, v in r.items())
        total = float(np.clip(total * self.ctrl_dt, 0.0, 100.0))

        # Termination
        up = self._up_vec()
        terminated = bool(up[2] < 0.0)
        truncated = self._step_n >= self.max_steps

        # Resample velocity at t=10s
        if self._step_n == int(10.0 / self.ctrl_dt):
            self._cmd = gait_utils.sample_velocity_command(self._rng)
            self._T = gait_utils.compute_stride_period(
                self._cmd[0], self._rng.uniform(-0.05, 0.05)
            )
            self._beta = gait_utils.compute_duty_factor(
                self._cmd[0], self._rng.uniform(-0.02, 0.02)
            )

        # Resample gait at t=20s
        if self._step_n == int(20.0 / self.ctrl_dt):
            self._gait, self._offsets = gait_utils.sample_gait(self._rng)

        self._last_act = action.copy()
        self._last_torque = torque.copy()

        return self._obs(), total, terminated, truncated, {
            "gait": self._gait, "rewards": r, "contacts": contacts
        }

    # ── Observation (52-dim) ──────────────────────────────────────────────────

    def _obs(self):
        d = self.mj_data
        q = d.qpos[3:7]
        grav = self._rot_inv(q, np.array([0., 0., -1.]))
        jpos = d.qpos[7:19]
        jvel = d.qvel[6:18]
        lp = gait_utils.compute_leg_phases(self._phase, self._offsets)
        lp = gait_utils.apply_time_reversal(lp, self._cmd[0], self._offsets)
        clock = gait_utils.compute_clock_inputs(lp)

        return np.concatenate([
            grav, jpos - HOME_QPOS, jvel, self._last_act,
            self._cmd, clock, self._offsets,
            [self._beta, 1.0 - self._beta],
        ]).astype(np.float32)

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _rewards(self, act, lp, contacts, torque):
        d = self.mj_data
        q = d.qpos[3:7]
        lv = self._rot_inv(q, d.qvel[0:3])
        av = self._rot_inv(q, d.qvel[3:6])
        beta = self._beta
        r = {}

        # Command tracking
        r["track_lin"] = float(np.exp(
            -np.sum((self._cmd[:2] - lv[:2])**2) / self.sigma
        ))
        r["track_ang"] = float(np.exp(
            -(self._cmd[2] - av[2])**2 / self.sigma
        ))

        # Base height
        h = d.xpos[self._base_id, 2]
        r["base_h"] = max(0, 0.20 - h)**2 + max(0, h - 0.35)**2

        # Smoothness
        r["torque_d"] = float(np.sum((torque - self._last_torque)**2))
        ha = np.abs(act[HIP_INDICES])
        w = np.exp(ha * 5); w = w / (w.sum() + 1e-8)
        r["hip"] = float(np.sum(w * ha))

        # Foot clearance + temporal symmetry
        fc = sv = sg = 0.0
        for i in range(4):
            fz = d.xpos[self._foot_body_ids[i], 2]
            sw_i = gait_utils.swing_indicator(lp[i], beta)
            st_i = gait_utils.stance_indicator(lp[i], beta)
            fc += sw_i * gait_utils.foot_clearance_cost(fz, lp[i], beta, 0.05)
            fv = np.linalg.norm(d.cvel[self._foot_body_ids[i], 3:6])
            sv += st_i * fv**2
            sg += sw_i * float(contacts[i])
        r["clearance"] = fc
        r["stance_v"] = sv
        r["swing_grf"] = sg

        # Morphological symmetry
        jp = d.qpos[7:19]
        legs = [jp[LEG_SLICES[n]] for n in FOOT_GEOM_NAMES]
        mc = 0.0
        for i in range(4):
            for j in range(i+1, 4):
                if abs(self._offsets[i] - self._offsets[j]) <= 0.01:
                    mc += np.sum((legs[i] - legs[j])**2)
        r["morph"] = mc

        # Regularization
        up = self._up_vec()
        r["orient"] = float(np.sum(up[:2]**2))
        r["vz"] = float(d.qvel[2]**2)
        r["wxy"] = float(np.sum(d.qvel[3:5]**2))
        r["act_rate"] = float(np.sum((act - self._last_act)**2))
        r["term"] = float(up[2] < 0.0)
        r["energy"] = float(np.sum(np.abs(d.qvel[6:18]) * np.abs(torque)))

        return r

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _contacts(self):
        c = np.zeros(4, dtype=bool)
        for i in range(self.mj_data.ncon):
            ct = self.mj_data.contact[i]
            for fi, fid in enumerate(self._foot_geom_ids):
                if (ct.geom1 == fid and ct.geom2 == self._floor_id) or \
                   (ct.geom2 == fid and ct.geom1 == self._floor_id):
                    c[fi] = True
        return c

    def _up_vec(self):
        return self._rot(self.mj_data.qpos[3:7], np.array([0., 0., 1.]))

    @staticmethod
    def _rot(q, v):
        w, x, y, z = q
        u = np.array([x, y, z])
        t = 2.0 * np.cross(u, v)
        return v + w * t + np.cross(u, t)

    @staticmethod
    def _rot_inv(q, v):
        w, x, y, z = q
        return Go2SymmetryEnv._rot([w, -x, -y, -z], v)

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.mj_model, 640, 480)
        self._renderer.update_scene(self.mj_data)
        return self._renderer.render()

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None


# ═══════════════════════════════════════════════════════════════════════════════
# Training Callback
# ═══════════════════════════════════════════════════════════════════════════════
class TrainLogger(BaseCallback):
    def __init__(self, log_interval=10000, verbose=1):
        super(TrainLogger, self).__init__(verbose)
        self.log_interval = log_interval
        self.rewards = []
        self.timesteps_log = []
        self.t0 = None

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self):
        if self.num_timesteps % self.log_interval == 0:
            elapsed = time.time() - self.t0
            sps = self.num_timesteps / max(elapsed, 1)

            # Get mean reward from SB3's rollout buffer (last 100 episodes)
            mean_r = 0.0
            if len(self.model.ep_info_buffer) > 0:
                mean_r = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            
            self.rewards.append(mean_r)
            self.timesteps_log.append(self.num_timesteps)

            if self.verbose:
                if mean_r == 0.0:
                    print(
                        f"Step {self.num_timesteps:>8,} | "
                        f"Reward: (waiting for ep finish) | "
                        f"Time: {elapsed:6.1f}s | "
                        f"SPS: {sps:,.0f}"
                    )
                else:
                    print(
                        f"Step {self.num_timesteps:>8,} | "
                        f"Reward: {mean_r:8.2f} | "
                        f"Time: {elapsed:6.1f}s | "
                        f"SPS: {sps:,.0f}"
                    )
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def make_env(xml_path):
    def _init():
        return Go2SymmetryEnv(xml_path=xml_path)
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train symmetry-guided gait policy for Go2"
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps (default: 500000)"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--save-path", type=str, default="go2_policy",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--eval-only", type=str, default=None,
        help="Path to saved model to evaluate (skip training)"
    )
    args = parser.parse_args()

    xml = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "unitree_mujoco", "unitree_robots", "go2", "scene.xml"
    )

    if args.eval_only:
        # ── Evaluation mode ──
        print(f"Loading model from {args.eval_only}...")
        env = Go2SymmetryEnv(xml_path=xml)
        model = PPO.load(args.eval_only)
        evaluate(env, model)
        env.close()
        return

    # ── Training ──
    print("=" * 60)
    print("Symmetry-Guided RL for Unitree Go2 Quadrupedal Gaits")
    print("Paper: arXiv:2510.10455v1")
    print("=" * 60)
    print(f"Training timesteps: {args.timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print()

    # Create vectorized environment
    env_fns = [make_env(xml) for _ in range(args.n_envs)]
    if args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # PPO config (paper: MLP [512,256,128], lr=1e-3, gamma=0.99, clip=0.2)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-3,
        n_steps=512,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=1.0,
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
            activation_fn=__import__('torch').nn.ELU,
        ),
        verbose=0,
    )

    print("Starting training...")
    callback = TrainLogger(log_interval=10000)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}.zip")

    vec_env.close()

    # ── Plot training curve ──
    if callback.rewards:
        plt.figure(figsize=(10, 5))
        plt.plot(callback.timesteps_log, callback.rewards, 'b-o', ms=4)
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Reward")
        plt.title("PPO Training - Go2 Symmetry-Guided Gaits")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_curve.png", dpi=150)
        print("Training curve saved to training_curve.png")

    # ── Quick evaluation ──
    print("\nRunning evaluation...")
    eval_env = Go2SymmetryEnv(xml_path=xml)
    evaluate(eval_env, model)
    eval_env.close()


def evaluate(env, model, n_episodes=5):
    """Run evaluation episodes and print metrics."""
    all_rewards = []
    all_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        total_r = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            steps += 1
            done = terminated or truncated

        all_rewards.append(total_r)
        all_lengths.append(steps)
        print(
            f"  Episode {ep+1}: reward={total_r:.2f}, "
            f"steps={steps}, gait={info['gait']}"
        )

    print(f"\nMean reward: {np.mean(all_rewards):.2f} +/- "
          f"{np.std(all_rewards):.2f}")
    print(f"Mean length: {np.mean(all_lengths):.0f} steps")


if __name__ == "__main__":
    main()
