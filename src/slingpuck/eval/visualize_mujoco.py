import time
from src.slingpuck.config import load_config
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from src.slingpuck.envs.goalkeeper_env import GoalkeeperEnv

def visualize(model_path, params_version="latest"):
    config = load_config(params_version)
        
    env = GoalkeeperEnv(config)
    policy = PPO.load(model_path)
    
    mj_model = mujoco.MjModel.from_xml_path("assets/mujoco/slingpuck.xml")
    mj_data = mujoco.MjData(mj_model)
    
    puck_x_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "puck_x")
    puck_y_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "puck_y")
    
    # --- MENAGERIE UPDATE ---
    # Change "Joint1" to match whatever the base pan joint is named in the menagerie XML
    pan_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "Joint1")
    
    # We also need the IDs of the other 5 joints so we can lock them in a "blocking" pose
    # Update these string names based on the menagerie XML
    locked_joint_names = ["Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
    locked_joint_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in locked_joint_names]
    # Adjust these static angles so the paddle sits nicely in front of the gate
    locked_joint_angles = [0.5, -0.5, 0.0, 1.57, 0.0] 

    print("Launching MuJoCo viewer. Close the viewer window to stop.")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            obs, _ = env.reset()
            done = False
            
            while not done and viewer.is_running():
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Sync Puck
                mj_data.qpos[puck_x_id] = env.puck_x
                # Subtract 0.20 because we shifted the board body forward in the XML
                mj_data.qpos[puck_y_id] = env.puck_y - 0.20 
                
                # Sync active Pan joint
                mj_data.qpos[pan_id] = -float(env.servo.pos) 
                
                # Lock the rest of the arm
                for j_id, j_angle in zip(locked_joint_ids, locked_joint_angles):
                    mj_data.qpos[j_id] = j_angle
                
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()
                
                time.sleep(env.control_dt)
                
            time.sleep(0.5)

if __name__ == "__main__":
    model_file = "logs/ppo_goalkeeper_latest/final_model.zip" 
    
    import os
    if os.path.exists(model_file):
        visualize(model_file)
    else:
        print(f"Model not found at {model_file}. Run training first.")