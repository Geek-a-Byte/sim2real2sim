import os
import yaml
import subprocess
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from src.slingpuck.envs.goalkeeper_env import GoalkeeperEnv

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

if __name__ == "__main__":
    # 1. Load config
    with open("configs/physics_params.dev.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    run_id = f"ppo_goalkeeper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"logs/{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config and git hash for reproducibility
    config['git_hash'] = get_git_hash()
    with open(f"{log_dir}/run_config.yaml", "w") as f:
        yaml.dump(config, f)

    # 2. Vectorize environments
    env = make_vec_env(lambda: GoalkeeperEnv(config), n_envs=8, seed=42)

    # 3. Initialize PPO
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./tensorboard_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64
    )

    # 4. Train
    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path=log_dir, name_prefix="gk_model")
    
    print(f"Starting training: {run_id}")
    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback, tb_log_name=run_id)
    
    model.save(f"{log_dir}/final_model")
    print("Training complete.")