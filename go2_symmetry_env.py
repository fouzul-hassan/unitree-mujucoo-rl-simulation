# Symmetry-Guided RL Environment for Unitree Go2
# Based on: "Towards Dynamic Quadrupedal Gaits" (arXiv:2510.10455v1)
# Uses unitree_mujoco's Go2 MJCF model directly with Gymnasium.
"""
Standalone Gymnasium environment for symmetry-guided quadrupedal gait
training on the Unitree Go2 robot using raw MuJoCo.

Compatible with Stable-Baselines3 for PPO training.
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

import gait_utils


# Go2 joint layout (matching go2.xml actuator order):
# 0: FR_hip,  1: FR_thigh,  2: FR_calf
# 3: FL_hip,  4: FL_thigh,  5: FL_calf
# 6: RR_hip,  7: RR_thigh,  8: RR_calf
# 9: RL_hip, 10: RL_thigh, 11: RL_calf

# Foot geom names in go2.xml
FOOT_GEOM_NAMES = ["FL", "FR", "RL", "RR"]
# Leg joint slices (3 joints per leg)
LEG_SLICES = {
    "FL": slice(3, 6),   # actuators 3,4,5
    "FR": slice(0, 3),   # actuators 0,1,2
    "RL": slice(9, 12),  # actuators 9,10,11
    "RR": slice(6, 9),   # actuators 6,7,8
}
# Hip joint indices (for hip action penalty)
HIP_INDICES = [0, 3, 6, 9]

# Home pose from go2.xml keyframe (joint positions only, 12 values)
# qpos order: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#             RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
HOME_QPOS = np.array([0, 0.9, -1.8, 0, 0.9, -1.8,
                       0, 0.9, -1.8, 0, 0.9, -1.8])


class Go2SymmetryEnv(gym.Env):
    """Unitree Go2 environment with symmetry-guided rewards.

    Observation space (52-dim):
        - Gravity orientation (3)
        - Joint positions relative to home (12)
        - Joint velocities (12)
        - Last action (12)
        - Velocity command [vx, vy, omega_yaw] (3)
        - Clock inputs sin(2*pi*phi_i) for 4 legs (4)
        - Phase offsets [theta_FL, theta_FR, theta_RL, theta_RR] (4)
        - Stance/swing ratio [beta, 1-beta] (2)

    Action space (12-dim):
        - Delta joint position targets, scaled and added to home pose.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        xml_path=None,
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=1500,
        kp=30.0,
        kd=0.65,
        action_scale=0.5,
        # Reward weights (from paper Table II + regularization)
        reward_weights=None,
    ):
        super().__init__()

        # ── Load MuJoCo model ──
        if xml_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(
                base, "unitree_mujoco", "unitree_robots", "go2", "scene.xml"
            )
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Timing
        self.mj_model.opt.timestep = sim_dt
        self.ctrl_dt = ctrl_dt
        self.sim_dt = sim_dt
        self.n_substeps = max(1, int(round(ctrl_dt / sim_dt)))
        self.episode_length = episode_length
        self.max_steps = episode_length

        # PD controller
        self.kp = kp
        self.kd = kd
        self.action_scale = action_scale

        # ── Identify bodies/geoms ──
        self._base_body_id = self.mj_model.body("base_link").id
        self._floor_geom_id = self.mj_model.geom("floor").id
        self._foot_geom_ids = np.array([
            self.mj_model.geom(name).id for name in FOOT_GEOM_NAMES
        ])
        self._foot_body_ids = np.array([
            self.mj_model.body(name + "_foot").id for name in FOOT_GEOM_NAMES
        ])

        # Joint limits
        self._jnt_range = self.mj_model.jnt_range[1:]  # skip freejoint
        self._home_qpos = HOME_QPOS.copy()

        # ── Reward weights ──
        self.reward_weights = reward_weights or {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.0,
            "base_height": -2.0,
            "torque_change": -0.001,
            "hip_action": -0.5,
            "foot_clearance": -1.0,
            "stance_vel": -2.0,
            "swing_grf": -2.0,
            "morphological": -1.0,
            "orientation": -5.0,
            "lin_vel_z": -0.5,
            "ang_vel_xy": -0.05,
            "action_rate": -0.01,
            "termination": -1.0,
            "energy": -0.001,
        }
        self.tracking_sigma = 0.25
        self.base_height_range = (0.20, 0.35)

        # ── Spaces ──
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # ── State tracking ──
        self._step_count = 0
        self._last_action = np.zeros(12)
        self._last_torque = np.zeros(12)
        self._global_phase = 0.0
        self._phase_offsets = np.zeros(4)
        self._gait_name = "trot"
        self._command = np.zeros(3)
        self._stride_period = 1.0
        self._duty_factor = 0.6
        self._rng = np.random.default_rng()

        # Rendering
        self.render_mode = render_mode
        self._renderer = None

    def _get_obs(self):
        """Build the 52-dim observation vector."""
        d = self.mj_data

        # Gravity in body frame (from IMU quaternion)
        quat = d.sensordata[
            self.mj_model.sensor("imu_quat").adr[0]:
            self.mj_model.sensor("imu_quat").adr[0] + 4
        ]
        # Rotate world gravity [0,0,-1] into body frame
        gravity = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))

        # Joint positions and velocities (12 each)
        joint_pos = d.qpos[7:19]
        joint_vel = d.qvel[6:18]

        # Compute leg phases and clock
        leg_phases = gait_utils.compute_leg_phases(
            self._global_phase, self._phase_offsets
        )
        leg_phases = gait_utils.apply_time_reversal(
            leg_phases, self._command[0], self._phase_offsets
        )
        clock = gait_utils.compute_clock_inputs(leg_phases)

        obs = np.concatenate([
            gravity,                                # 3
            joint_pos - self._home_qpos,            # 12
            joint_vel,                              # 12
            self._last_action,                      # 12
            self._command,                          # 3
            clock,                                  # 4
            self._phase_offsets,                    # 4
            np.array([self._duty_factor,
                      1.0 - self._duty_factor]),    # 2
        ]).astype(np.float32)
        # Total: 3+12+12+12+3+4+4+2 = 52

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset to home keyframe
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        # Add small randomization
        self.mj_data.qpos[0:2] += self._rng.uniform(-0.1, 0.1, 2)
        self.mj_data.qvel[0:6] = self._rng.uniform(-0.3, 0.3, 6)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Sample gait and command
        self._gait_name, self._phase_offsets = gait_utils.sample_gait(
            self._rng
        )
        self._command = gait_utils.sample_velocity_command(self._rng)
        noise_T = self._rng.uniform(-0.05, 0.05)
        noise_b = self._rng.uniform(-0.02, 0.02)
        self._stride_period = gait_utils.compute_stride_period(
            self._command[0], noise_T
        )
        self._duty_factor = gait_utils.compute_duty_factor(
            self._command[0], noise_b
        )

        # Reset state
        self._step_count = 0
        self._global_phase = 0.0
        self._last_action = np.zeros(12)
        self._last_torque = np.zeros(12)

        obs = self._get_obs()
        info = {
            "gait": self._gait_name,
            "command": self._command.copy(),
        }
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # PD control: target = home + action * scale
        target_pos = self._home_qpos + action * self.action_scale
        joint_pos = self.mj_data.qpos[7:19]
        joint_vel = self.mj_data.qvel[6:18]
        torque = self.kp * (target_pos - joint_pos) - self.kd * joint_vel
        # Clip torques to actuator limits
        ctrl_range = self.mj_model.actuator_ctrlrange
        torque = np.clip(torque, ctrl_range[:, 0], ctrl_range[:, 1])
        self.mj_data.ctrl[:] = torque

        # Step simulation
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.mj_model, self.mj_data)

        # Update phase
        self._global_phase = (
            self._global_phase + self.ctrl_dt / self._stride_period
        ) % 1.0
        self._step_count += 1

        # Compute leg phases
        leg_phases = gait_utils.compute_leg_phases(
            self._global_phase, self._phase_offsets
        )
        leg_phases = gait_utils.apply_time_reversal(
            leg_phases, self._command[0], self._phase_offsets
        )

        # Detect foot contacts
        contacts = self._get_foot_contacts()

        # Compute rewards
        rewards = self._compute_rewards(
            action, leg_phases, contacts, torque
        )
        total_reward = sum(
            self.reward_weights.get(k, 0.0) * v for k, v in rewards.items()
        )
        total_reward = float(np.clip(total_reward * self.ctrl_dt, 0.0, 100.0))

        # Termination
        up_vec = self._get_up_vector()
        terminated = bool(up_vec[2] < 0.0)  # Flipped over
        truncated = self._step_count >= self.max_steps

        # Resample velocity at t=10s
        vel_resample_step = int(10.0 / self.ctrl_dt)
        if self._step_count == vel_resample_step:
            self._command = gait_utils.sample_velocity_command(self._rng)
            self._stride_period = gait_utils.compute_stride_period(
                self._command[0], self._rng.uniform(-0.05, 0.05)
            )
            self._duty_factor = gait_utils.compute_duty_factor(
                self._command[0], self._rng.uniform(-0.02, 0.02)
            )

        # Resample gait at t=20s
        gait_resample_step = int(20.0 / self.ctrl_dt)
        if self._step_count == gait_resample_step:
            self._gait_name, self._phase_offsets = gait_utils.sample_gait(
                self._rng
            )

        # Update history
        self._last_action = action.copy()
        self._last_torque = torque.copy()

        obs = self._get_obs()
        info = {
            "gait": self._gait_name,
            "command": self._command.copy(),
            "rewards": rewards,
            "contacts": contacts.copy(),
        }

        return obs, total_reward, terminated, truncated, info

    # ─── Reward Functions ─────────────────────────────────────────────────────

    def _compute_rewards(self, action, leg_phases, contacts, torque):
        d = self.mj_data
        beta = self._duty_factor

        # Base velocities in body frame
        base_quat = d.qpos[3:7]
        base_linvel_world = d.qvel[0:3]
        base_angvel_world = d.qvel[3:6]
        local_linvel = self._quat_rotate_inv(base_quat, base_linvel_world)
        local_angvel = self._quat_rotate_inv(base_quat, base_angvel_world)

        rewards = {}

        # ── Command tracking ──
        lin_err = np.sum((self._command[:2] - local_linvel[:2]) ** 2)
        rewards["tracking_lin_vel"] = float(
            np.exp(-lin_err / self.tracking_sigma)
        )

        ang_err = (self._command[2] - local_angvel[2]) ** 2
        rewards["tracking_ang_vel"] = float(
            np.exp(-ang_err / self.tracking_sigma)
        )

        # Base height
        base_h = d.xpos[self._base_body_id, 2]
        h_min, h_max = self.base_height_range
        below = max(0.0, h_min - base_h)
        above = max(0.0, base_h - h_max)
        rewards["base_height"] = float(below ** 2 + above ** 2)

        # ── Smoothness ──
        rewards["torque_change"] = float(
            np.sum((torque - self._last_torque) ** 2)
        )

        hip_actions = np.abs(action[HIP_INDICES])
        weights = np.exp(hip_actions * 5.0)
        weights = weights / (weights.sum() + 1e-8)
        rewards["hip_action"] = float(np.sum(weights * hip_actions))

        # Foot clearance
        fc_cost = 0.0
        for i, name in enumerate(FOOT_GEOM_NAMES):
            foot_z = d.xpos[self._foot_body_ids[i], 2]
            sw = gait_utils.swing_indicator(leg_phases[i], beta)
            fc_cost += sw * gait_utils.foot_clearance_cost(
                foot_z, leg_phases[i], beta, 0.05
            )
        rewards["foot_clearance"] = fc_cost

        # ── Temporal symmetry ──
        stance_cost = 0.0
        swing_cost = 0.0
        for i in range(4):
            st = gait_utils.stance_indicator(leg_phases[i], beta)
            sw = gait_utils.swing_indicator(leg_phases[i], beta)
            # Foot velocity (approximate from body positions)
            foot_vel = np.linalg.norm(d.cvel[self._foot_body_ids[i], 3:6])
            stance_cost += st * foot_vel ** 2
            swing_cost += sw * float(contacts[i])
        rewards["stance_vel"] = stance_cost
        rewards["swing_grf"] = swing_cost

        # ── Morphological symmetry ──
        eps = 0.01
        joint_pos = d.qpos[7:19]
        leg_joints = [
            joint_pos[LEG_SLICES["FL"]],
            joint_pos[LEG_SLICES["FR"]],
            joint_pos[LEG_SLICES["RL"]],
            joint_pos[LEG_SLICES["RR"]],
        ]
        morph_cost = 0.0
        offsets = self._phase_offsets
        for i in range(4):
            for j in range(i + 1, 4):
                if abs(offsets[i] - offsets[j]) <= eps:
                    morph_cost += np.sum((leg_joints[i] - leg_joints[j]) ** 2)
        rewards["morphological"] = morph_cost

        # ── Regularization ──
        up_vec = self._get_up_vector()
        rewards["orientation"] = float(np.sum(up_vec[:2] ** 2))
        rewards["lin_vel_z"] = float(base_linvel_world[2] ** 2)
        rewards["ang_vel_xy"] = float(np.sum(base_angvel_world[:2] ** 2))
        rewards["action_rate"] = float(
            np.sum((action - self._last_action) ** 2)
        )
        rewards["termination"] = float(up_vec[2] < 0.0)
        rewards["energy"] = float(
            np.sum(np.abs(d.qvel[6:18]) * np.abs(torque))
        )

        return rewards

    # ─── Contact Detection ────────────────────────────────────────────────────

    def _get_foot_contacts(self):
        """Check which feet are in contact with the floor."""
        contacts = np.zeros(4, dtype=bool)
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = c.geom1, c.geom2
            for fi, foot_id in enumerate(self._foot_geom_ids):
                if (g1 == foot_id and g2 == self._floor_geom_id) or \
                   (g2 == foot_id and g1 == self._floor_geom_id):
                    contacts[fi] = True
        return contacts

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_up_vector(self):
        """Get the body z-axis in world frame (up vector)."""
        quat = self.mj_data.qpos[3:7]
        return self._quat_rotate(quat, np.array([0.0, 0.0, 1.0]))

    @staticmethod
    def _quat_rotate(q, v):
        """Rotate vector v by quaternion q (w,x,y,z format)."""
        w, x, y, z = q
        t = 2.0 * np.cross(np.array([x, y, z]), v)
        return v + w * t + np.cross(np.array([x, y, z]), t)

    @staticmethod
    def _quat_rotate_inv(q, v):
        """Rotate vector v by inverse of quaternion q."""
        w, x, y, z = q
        q_inv = np.array([w, -x, -y, -z])
        return Go2SymmetryEnv._quat_rotate(q_inv, v)

    # ─── Rendering ────────────────────────────────────────────────────────────

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.mj_model, width=640, height=480
            )
        self._renderer.update_scene(self.mj_data, camera="track")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
