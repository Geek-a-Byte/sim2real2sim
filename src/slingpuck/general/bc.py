"""Behaviour-cloning warm start: record scripted-expert demos, then train the PPO actor to copy them.

    python -m src.slingpuck.general.bc --objects puck --episodes 1500        # demos + BC -> logs/general/bc_<...>/
    python -m src.slingpuck.general.train_teacher --objects puck --init logs/general/bc_<...>/bc_model.zip

Demos: normal starts only (the expert does the whole task). Half of the episodes run with DART noise on the
executed action, and the recorded label is always the clean expert action, so the policy also learns how to
recover from small errors. Only episodes that reach at least one goal are kept.

BC: the actor (paired-point encoder + MLP + action head) is trained with an MSE loss between its mean action
and the expert action. The critic is not trained here; train_teacher --critic-warmup fits it with the actor
frozen before PPO changes the actor.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import torch as th

from src.slingpuck.config import REPO_ROOT, save_run_config
from src.slingpuck.general.common import load_general_config
from src.slingpuck.general.expert import run_expert
from src.slingpuck.general.goal_env import GoalEnv
from src.slingpuck.general.reachability import load_or_build_map

ACTOR_PARTS = ("pi_features_extractor", "mlp_extractor.policy_net", "action_net")


def collect(cfg, rmap, objects, episodes: int, seed: int, noise: float):
    env = GoalEnv(cfg, reach_map=rmap, objects=objects, grasped_start_prob=0.0, pregrasp_start_prob=0.0)
    keep = {"points": [], "proprio": [], "privileged": [], "action": []}
    kept, goals = 0, []
    for k in range(episodes):
        stats, data = run_expert(env, 1, seed=seed + k, noise=noise if k % 2 else 0.0, record=True)
        goals.append(stats[0]["goals"])
        if stats[0]["goals"] >= 1:
            kept += 1
            for key in keep:
                keep[key].append(data[key])
    out = {k: np.concatenate(v) for k, v in keep.items()}
    return out, {"episodes": episodes, "kept": kept, "goals_per_episode": float(np.mean(goals)),
                 "samples": int(len(out["action"]))}


def actor_parameters(policy):
    params = []
    for part in ACTOR_PARTS:
        mod = policy
        for name in part.split("."):
            mod = getattr(mod, name)
        params += list(mod.parameters())
    return params


def train_bc(model, data, epochs: int = 30, batch: int = 1024, lr: float = 1e-3, seed: int = 0, log=print):
    """Train the actor's mean action on (obs, expert action) pairs. Returns the per-epoch train losses."""
    policy = model.policy
    policy.set_training_mode(True)
    opt = th.optim.Adam(actor_parameters(policy), lr=lr)
    rng = np.random.default_rng(seed)
    obs = {k: th.as_tensor(data[k], dtype=th.float32) for k in ("points", "proprio", "privileged")}
    act = th.as_tensor(data["action"], dtype=th.float32)
    n = len(act)
    losses = []
    for epoch in range(epochs):
        idx = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            dist = policy.get_distribution({k: v[b] for k, v in obs.items()})
            loss = ((dist.distribution.mean - act[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        losses.append(total / n)
        if epoch % 5 == 0 or epoch == epochs - 1:
            log(f"  BC epoch {epoch:3d}: MSE {losses[-1]:.4f}")
    policy.set_training_mode(False)
    return losses


def main():
    from stable_baselines3 import PPO

    from src.slingpuck.general.policy import PairedPointPolicy

    p = argparse.ArgumentParser()
    p.add_argument("--objects", nargs="*", default=["puck"])
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--noise", type=float, default=0.3, help="DART noise std on half of the episodes")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-std-init", type=float, default=-1.0)
    args = p.parse_args()

    cfg = load_general_config()
    rmap = load_or_build_map(cfg)
    run_dir = REPO_ROOT / "logs" / "general" / f"bc_{'_'.join(args.objects)}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Recording {args.episodes} expert episodes ...")
    data, meta = collect(cfg, rmap, args.objects, args.episodes, 10_000 + args.seed * 100_000, args.noise)
    np.savez_compressed(run_dir / "demos.npz", **data)
    print(f"  kept {meta['kept']}/{meta['episodes']} episodes, {meta['samples']} samples, "
          f"{meta['goals_per_episode']:.1f} goals per episode")

    env = GoalEnv(cfg, reach_map=rmap, objects=args.objects)
    model = PPO(PairedPointPolicy, env, seed=args.seed, verbose=0,
                policy_kwargs={"net_arch": [256, 256], "log_std_init": args.log_std_init})
    losses = train_bc(model, data, epochs=args.epochs, seed=args.seed)
    model.save(run_dir / "bc_model")
    save_run_config({**cfg, "bc": {**vars(args), **meta, "final_mse": losses[-1]}}, run_dir, seed=args.seed)
    print(f"Saved {run_dir / 'bc_model.zip'}")


if __name__ == "__main__":
    main()
