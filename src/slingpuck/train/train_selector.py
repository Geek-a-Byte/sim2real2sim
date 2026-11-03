"""Train the PPO selector (layer 3) on the event-level match.

    python -m src.slingpuck.train.train_selector                       # configs/train_selector.yaml
    python -m src.slingpuck.train.train_selector --seeds 0 --timesteps 200000
    python -m src.slingpuck.eval.match_eval --policies greedy_rule greedy_sling --runs logs/selector/<run>

The match uses the shot model (shot.selection in env.yaml), so fit it first: python -m src.slingpuck.shot.model
"""
from src.slingpuck.train.common import match_env_fn, train_cli

if __name__ == "__main__":
    train_cli("selector", match_env_fn, ("win", "loss", "time"), "train_selector.yaml")
