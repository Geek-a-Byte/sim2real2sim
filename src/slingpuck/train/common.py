"""Shared training helpers: env factories, run directories, outcome logging, PPO runs."""
import argparse
from datetime import datetime
from functools import partial
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import BaseCallback

from src.slingpuck.config import REPO_ROOT
from src.slingpuck.envs.goalkeeper_env import GoalkeeperEnv
from src.slingpuck.envs.sling_env import SlingEnv
from src.slingpuck.envs.wrappers import PrivilegedObsWrapper


def make_goalkeeper_env(config: dict, asymmetric: bool, backend: str = "2d"):
    """backend: "2d" (Fast2DPuckSim + ServoModel) or "mujoco" (MuJoCo board, puck and SO-101)."""
    if backend == "mujoco":
        from src.slingpuck.envs.mujoco_goalkeeper_env import MujocoGoalkeeperEnv
        env = MujocoGoalkeeperEnv(config)
    elif backend == "2d":
        env = GoalkeeperEnv(config)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return PrivilegedObsWrapper(env) if asymmetric else env


def goalkeeper_env_fn(config: dict, asymmetric: bool, backend: str = "2d"):
    """Picklable env factory for SubprocVecEnv."""
    return partial(make_goalkeeper_env, config, asymmetric, backend)


def make_sling_env(config: dict, asymmetric: bool):
    env = SlingEnv(config)
    return PrivilegedObsWrapper(env) if asymmetric else env


def sling_env_fn(config: dict, asymmetric: bool):
    """Picklable env factory for SubprocVecEnv."""
    return partial(make_sling_env, config, asymmetric)


def load_trained_run(run_dir, device: str = "cpu"):
    """Load final_model.zip from a training run. Returns (model, asymmetric, run_cfg).

    The model file stores the policy class by its module path. custom_objects
    replaces it with the class imported here, so a run loads whether the package
    was imported as `slingpuck` or `src.slingpuck` when it was trained.
    """
    from stable_baselines3 import PPO

    from src.slingpuck.policies.asymmetric import AsymmetricActorCriticPolicy

    run_dir = Path(run_dir)
    run_cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text())
    asymmetric = run_cfg["train"]["policy"] == "asymmetric"
    custom = {"policy_class": AsymmetricActorCriticPolicy} if asymmetric else None
    model = PPO.load(run_dir / "final_model.zip", device=device, custom_objects=custom)
    return model, asymmetric, run_cfg


def load_train_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def new_run_dir(task: str, params_version: str, seed: int) -> tuple[str, Path]:
    run_id = f"{task}_{params_version}_s{seed}_{datetime.now():%Y%m%d_%H%M%S}"
    return run_id, REPO_ROOT / "logs" / task / run_id


class OutcomeLoggerCallback(BaseCallback):
    """Log the fraction of episode outcomes (info["outcome"]) per rollout to TensorBoard."""

    def __init__(self, prefix: str, outcomes=("save", "goal", "none")):
        super().__init__()
        self.prefix = prefix
        self.outcomes = outcomes
        self._counts = dict.fromkeys(outcomes, 0)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and info.get("outcome") in self._counts:
                self._counts[info["outcome"]] += 1
        return True

    def _on_rollout_end(self) -> None:
        total = sum(self._counts.values())
        if total:
            for k, n in self._counts.items():
                self.logger.record(f"{self.prefix}/{k}_rate", n / total)
            self.logger.record(f"{self.prefix}/episodes", total)
        self._counts = dict.fromkeys(self.outcomes, 0)


def train_ppo(task: str, env_fn_factory, outcomes, config: dict, train_cfg: dict, seed: int) -> Path:
    """One reproducible PPO run. env_fn_factory(config, asymmetric) returns a picklable env factory.

    Writes logs/<task>/<run_id>/ with run_config.yaml (merged config, train config,
    params version, git hash, seed), checkpoints and final_model.zip, and TensorBoard
    logs under tensorboard_logs/<task>/.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from src.slingpuck.config import save_run_config
    from src.slingpuck.policies.asymmetric import AsymmetricActorCriticPolicy

    set_random_seed(seed)
    asymmetric = train_cfg["policy"] == "asymmetric"
    run_id, run_dir = new_run_dir(task, config["meta"]["params_version"], seed)
    save_run_config({**config, "train": train_cfg}, run_dir, seed=seed)
    vec_cls = SubprocVecEnv if train_cfg["vec_env"] == "subproc" else DummyVecEnv
    env = make_vec_env(env_fn_factory(config, asymmetric), n_envs=train_cfg["n_envs"], seed=seed,
                       vec_env_cls=vec_cls)
    model = PPO(
        AsymmetricActorCriticPolicy if asymmetric else "MlpPolicy",
        env,
        policy_kwargs={"net_arch": list(train_cfg["net_arch"])},
        tensorboard_log=str(REPO_ROOT / "tensorboard_logs" / task),
        seed=seed,
        verbose=0,
        **train_cfg["ppo"],
    )
    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(train_cfg["checkpoint_freq"] // train_cfg["n_envs"], 1),
                           save_path=str(run_dir), name_prefix="model"),
        OutcomeLoggerCallback(task, outcomes),
    ])
    print(f"Training {run_id}")
    model.learn(total_timesteps=train_cfg["total_timesteps"], callback=callbacks, tb_log_name=run_id)
    model.save(run_dir / "final_model")
    env.close()
    print(f"Saved {run_dir / 'final_model.zip'}")
    return run_dir


def train_cli(task: str, env_fn_factory, outcomes, default_train_config: str):
    """Command-line entry point shared by the train_* scripts."""
    from src.slingpuck.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default=str(REPO_ROOT / "configs" / default_train_config))
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
        train_ppo(task, env_fn_factory, outcomes, config, train_cfg, seed)
