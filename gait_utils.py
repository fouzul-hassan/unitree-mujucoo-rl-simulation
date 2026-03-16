# Symmetry-Guided RL Gait Utilities
# Based on: "Towards Dynamic Quadrupedal Gaits" (arXiv:2510.10455v1)
"""Gait parameterization and symmetry utilities for quadrupedal locomotion.

Pure NumPy implementation — no JAX dependency. Compatible with Python 3.8+.
"""

import numpy as np


# ─── Gait Library (Table I from the paper) ───────────────────────────────────
# Phase offsets: [θ_FL, θ_FR, θ_RL, θ_RR]
# Maps to Go2 leg order: FL, FR, RL, RR (matching actuator order in go2.xml)
GAIT_LIBRARY = {
    "trot":       np.array([0.5, 0.0, 0.0, 0.5]),   # Diagonal pairs
    "bound":      np.array([0.0, 0.0, 0.5, 0.5]),   # Front/rear pairs
    "half_bound": np.array([0.0, 0.0, 0.5, 0.3]),   # Asymmetric
    "gallop":     np.array([0.1, 0.0, 0.6, 0.5]),   # Sequential
}

GAIT_NAMES = list(GAIT_LIBRARY.keys())

# Fitted constants from Eq. 1 (Section III-E)
A_T, B_T, C_T = 2.55, 0.20, 0.975
A_BETA, B_BETA, C_BETA = 0.5588, 0.20, 0.681
GRAVITY = 9.81
LEG_LENGTH = 0.213  # Go2 calf length in meters


def compute_stride_period(vx_cmd, noise=0.0):
    """Compute stride period T from forward command velocity.

    T = a_T * exp(-b_T * |vx*|) + c_T + noise
    where vx* = vx_cmd / sqrt(g*l).

    Args:
        vx_cmd: Forward command velocity (m/s).
        noise: Optional perturbation (default 0).

    Returns:
        Stride period T in seconds.
    """
    vx_norm = abs(vx_cmd) / np.sqrt(GRAVITY * LEG_LENGTH)
    T = A_T * np.exp(-B_T * vx_norm) + C_T + noise
    return float(np.clip(T, 0.3, 3.5))


def compute_duty_factor(vx_cmd, noise=0.0):
    """Compute duty factor beta from forward command velocity.

    beta = a_beta * exp(-b_beta * |vx*|) + c_beta + noise

    Args:
        vx_cmd: Forward command velocity (m/s).
        noise: Optional perturbation (default 0).

    Returns:
        Duty factor beta in (0.1, 0.9).
    """
    vx_norm = abs(vx_cmd) / np.sqrt(GRAVITY * LEG_LENGTH)
    beta = A_BETA * np.exp(-B_BETA * vx_norm) + C_BETA + noise
    return float(np.clip(beta, 0.1, 0.9))


def compute_leg_phases(global_phase, phase_offsets):
    """Compute individual leg phases from global phase and offsets.

    phi_i = (global_phase + theta_i) mod 1

    Args:
        global_phase: Global normalized phase in [0, 1).
        phase_offsets: Phase offsets for each leg [theta_FL, theta_FR, theta_RL, theta_RR].

    Returns:
        Leg phases array of shape (4,) in [0, 1).
    """
    return (global_phase + np.array(phase_offsets)) % 1.0


def compute_clock_inputs(leg_phases):
    """Compute clock inputs sin(2*pi*phi_i) for observation.

    Args:
        leg_phases: Individual leg phases (4,).

    Returns:
        Clock inputs (4,).
    """
    return np.sin(2.0 * np.pi * np.array(leg_phases))


def apply_time_reversal(leg_phases, vx_cmd, phase_offsets):
    """Apply time-reversal symmetry when vx < 0 (Eq. 9).

    When vx_cmd < 0: phi_i = (1 - phi_i + theta_i) mod 1.

    Args:
        leg_phases: Current leg phases (4,).
        vx_cmd: Forward command velocity.
        phase_offsets: Gait phase offsets (4,).

    Returns:
        Modified leg phases (4,).
    """
    if vx_cmd < 0:
        return (1.0 - np.array(leg_phases) + np.array(phase_offsets)) % 1.0
    return np.array(leg_phases)


def von_mises_indicator(phase, center, kappa=8.0):
    """Smooth phase indicator using Von Mises distribution.

    Args:
        phase: Leg phase in [0, 1).
        center: Center of distribution on [0, 1).
        kappa: Concentration parameter.

    Returns:
        Smooth indicator in (0, 1).
    """
    cos_diff = np.cos(2.0 * np.pi * (phase - center))
    return float(np.exp(kappa * (cos_diff - 1.0)))


def swing_indicator(phase, beta):
    """Smoothed swing phase indicator. Swing is [0, 1-beta)."""
    swing_center = (1.0 - beta) / 2.0
    return von_mises_indicator(phase, swing_center, kappa=6.0)


def stance_indicator(phase, beta):
    """Smoothed stance phase indicator. Stance is [1-beta, 1)."""
    stance_center = 1.0 - beta / 2.0
    return von_mises_indicator(phase, stance_center, kappa=6.0)


def foot_clearance_cost(foot_z, phase, beta, h_cl_min=0.05):
    """Compute foot clearance shortfall (Eq. 7).

    c_foot = max(0, h_cl_min * s_i - z_i)
    where s_i = clip(phi_i / (1-beta), 0, 1).

    Args:
        foot_z: Vertical position of foot.
        phase: Leg phase.
        beta: Duty factor.
        h_cl_min: Desired minimal clearance.

    Returns:
        Clearance cost.
    """
    s = np.clip(phase / max(1.0 - beta, 1e-6), 0.0, 1.0)
    target = h_cl_min * s
    return float(max(0.0, target - foot_z))


def sample_gait(rng=None):
    """Sample a random gait from the library with perturbation +/- 0.02.

    Args:
        rng: Optional numpy RandomState or Generator.

    Returns:
        Tuple of (gait_name, phase_offsets).
    """
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, len(GAIT_NAMES))
    name = GAIT_NAMES[idx]
    offsets = GAIT_LIBRARY[name].copy()
    offsets += rng.uniform(-0.02, 0.02, size=4)
    offsets = offsets % 1.0
    return name, offsets


def sample_velocity_command(rng=None):
    """Sample a velocity command: vx in [-2, 2], vy=0, omega_yaw=0.

    Args:
        rng: Optional numpy RandomState or Generator.

    Returns:
        Command array [vx, vy, omega_yaw].
    """
    if rng is None:
        rng = np.random.default_rng()
    vx = rng.uniform(-2.0, 2.0)
    return np.array([vx, 0.0, 0.0])
