"""Train the Phase 1 goalkeeper with PPO.

Example:
    python -m src.slingpuck.train.train_goalkeeper --seeds 0 1 2
See train.common.train_ppo for what each run writes.
"""
from src.slingpuck.train.common import goalkeeper_env_fn, train_cli

if __name__ == "__main__":
    train_cli("goalkeeper", goalkeeper_env_fn, ("save", "goal", "none"), "train_goalkeeper.yaml")
