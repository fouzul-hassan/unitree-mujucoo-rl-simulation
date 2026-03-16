#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize the trained Go2 Policy using MuJoCo's interactive 3D viewer.
"""

import os
import time
import argparse
import mujoco
import mujoco.viewer
import numpy as np

from stable_baselines3 import PPO
from train_go2 import Go2SymmetryEnv

def main():
    parser = argparse.ArgumentParser(description="Visualize the trained Go2 policy")
    parser.add_argument("--model-path", type=str, default="go2_policy.zip", help="Path to the saved model")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        print("Please train the model first by running: python train_go2.py")
        return

    xml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "unitree_mujoco", "unitree_robots", "go2", "scene.xml"
    )

    print("Loading environment...")
    env = Go2SymmetryEnv(xml_path=xml_path)
    
    print(f"Loading model from {args.model_path}...")
    model = PPO.load(args.model_path)

    obs, info = env.reset()
    print(f"Initial gait commanded: {info['gait']}")
    print("\nStarting simulation viewer. Press ESC to exit.")

    # Launch the interactive MuJoCo viewer
    with mujoco.viewer.launch_passive(env.mj_model, env.mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # The policy predicts the best action based on the observation
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Sync the viewer with the updated physics state
            viewer.sync()

            if terminated or truncated:
                print("Episode ended, resetting robot...")
                obs, info = env.reset()
                print(f"New gait commanded: {info['gait']}")

            # Try to run in real-time
            time_until_next_step = env.ctrl_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
