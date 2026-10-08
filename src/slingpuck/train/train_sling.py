"""Train the Phase 2 targeted shot with PPO.

Example:
    python -m src.slingpuck.train.train_sling --seeds 0 1 2
See train.common.train_ppo for what each run writes.
"""
from src.slingpuck.train.common import sling_env_fn, train_cli

if __name__ == "__main__":
    train_cli("sling", sling_env_fn, ("success", "miss"), "train_sling.yaml")
