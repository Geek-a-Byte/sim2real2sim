"""Shared training helpers: env factories, run directories, outcome logging."""
from datetime import datetime
from functools import partial
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import BaseCallback

from slingpuck.config import REPO_ROOT
from slingpuck.envs.goalkeeper_env import GoalkeeperEnv
from slingpuck.envs.wrappers import PrivilegedObsWrapper


def make_goalkeeper_env(config: dict, asymmetric: bool):
    env = GoalkeeperEnv(config)
    return PrivilegedObsWrapper(env) if asymmetric else env


def goalkeeper_env_fn(config: dict, asymmetric: bool):
    """Picklable env factory for SubprocVecEnv."""
    return partial(make_goalkeeper_env, config, asymmetric)


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
