import argparse
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from slingpuck.config import load_config, save_run_config
from slingpuck.envs.goalkeeper_env import GoalkeeperEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-version", default="latest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--strict", action="store_true", help="Refuse uncalibrated placeholder params")
    args = parser.parse_args()

    config = load_config(args.params_version, strict=args.strict)
    set_random_seed(args.seed)

    run_id = f"ppo_goalkeeper_{config['meta']['params_version']}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = f"logs/{run_id}"
    save_run_config(config, log_dir, seed=args.seed)

    env = make_vec_env(lambda: GoalkeeperEnv(config), n_envs=args.n_envs, seed=args.seed)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path=log_dir, name_prefix="gk_model")
    print(f"Starting training: {run_id}")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, tb_log_name=run_id)
    model.save(f"{log_dir}/final_model")
    print("Training complete.")


if __name__ == "__main__":
    main()
