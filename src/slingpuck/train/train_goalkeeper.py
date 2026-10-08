"""Train the Phase 1 goalkeeper with PPO.

Example:
    python -m slingpuck.train.train_goalkeeper --seeds 0 1 2
Each seed writes logs/goalkeeper/<run_id>/ with run_config.yaml (physics params
version, train config, git hash, seed), checkpoints and final_model.zip, and
TensorBoard logs under tensorboard_logs/goalkeeper/.
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from slingpuck.config import REPO_ROOT, load_config, save_run_config
from slingpuck.policies.asymmetric import AsymmetricActorCriticPolicy
from slingpuck.train.common import OutcomeLoggerCallback, goalkeeper_env_fn, load_train_config, new_run_dir


def train_one(config: dict, train_cfg: dict, seed: int) -> str:
    set_random_seed(seed)
    asymmetric = train_cfg["policy"] == "asymmetric"
    run_id, run_dir = new_run_dir("goalkeeper", config["meta"]["params_version"], seed)
    save_run_config({**config, "train": train_cfg}, run_dir, seed=seed)

    vec_cls = SubprocVecEnv if train_cfg["vec_env"] == "subproc" else DummyVecEnv
    env = make_vec_env(goalkeeper_env_fn(config, asymmetric), n_envs=train_cfg["n_envs"], seed=seed,
                       vec_env_cls=vec_cls)
    model = PPO(
        AsymmetricActorCriticPolicy if asymmetric else "MlpPolicy",
        env,
        policy_kwargs={"net_arch": list(train_cfg["net_arch"])},
        tensorboard_log=str(REPO_ROOT / "tensorboard_logs" / "goalkeeper"),
        seed=seed,
        verbose=0,
        **train_cfg["ppo"],
    )
    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(train_cfg["checkpoint_freq"] // train_cfg["n_envs"], 1),
                           save_path=str(run_dir), name_prefix="gk_model"),
        OutcomeLoggerCallback("goalkeeper"),
    ])
    print(f"Training {run_id}")
    model.learn(total_timesteps=train_cfg["total_timesteps"], callback=callbacks, tb_log_name=run_id)
    model.save(run_dir / "final_model")
    env.close()
    print(f"Saved {run_dir / 'final_model.zip'}")
    return str(run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default=str(REPO_ROOT / "configs" / "train_goalkeeper.yaml"))
    parser.add_argument("--params-version", default="latest")
    parser.add_argument("--seeds", type=int, nargs="*", help="Override the seeds in the train config")
    parser.add_argument("--timesteps", type=int, help="Override total_timesteps")
    parser.add_argument("--strict", action="store_true", help="Refuse uncalibrated placeholder params")
    args = parser.parse_args()

    config = load_config(args.params_version, strict=args.strict)
    train_cfg = load_train_config(args.train_config)
    if args.timesteps:
        train_cfg["total_timesteps"] = args.timesteps
    for seed in args.seeds if args.seeds else train_cfg["seeds"]:
        train_one(config, train_cfg, seed)


if __name__ == "__main__":
    main()
