"""Frozen block primitive for Phase 3: a measured save-probability table.

Phase 3 runs millions of decisions, so it does not re-simulate the goalkeeper.
Instead it uses this table, measured once on the Phase 1 env (GoalkeeperEnv,
domain-randomized physics) with a frozen block policy:

  p_gate(speed)          paddle settled at the gate when the shot launches
  p_edge(speed, delay)   paddle at the edge of the pan range (just back from the
                         band), free `delay` seconds after the launch

The table is cached in cache/ under a hash of everything that changes it.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.slingpuck.config import REPO_ROOT

CACHE_DIR = REPO_ROOT / "cache"
_HASH_KEYS = ("goalkeeper", "servo", "camera", "board", "puck", "robot", "sim", "tracker", "randomization")


@dataclass(frozen=True)
class BlockTable:
    speeds: np.ndarray
    delays: np.ndarray
    p_gate: np.ndarray       # (n_speeds,)
    p_edge: np.ndarray       # (n_speeds, n_delays)

    def p_save(self, speed: float, at_gate: bool, delay: float = 0.0) -> float:
        s = float(np.clip(speed, self.speeds[0], self.speeds[-1]))
        if at_gate:
            return float(np.interp(s, self.speeds, self.p_gate))
        d = float(np.clip(delay, self.delays[0], self.delays[-1]))
        if delay > self.delays[-1]:
            return 0.0 if self._edge_at(s, self.delays[-1]) < 0.05 else self._edge_at(s, d)
        return self._edge_at(s, d)

    def _edge_at(self, s, d):
        per_speed = np.array([np.interp(d, self.delays, row) for row in self.p_edge])
        return float(np.interp(s, self.speeds, per_speed))


def _cache_key(cfg, policy, speeds, delays, episodes, seed):
    blob = json.dumps({k: cfg.get(k) for k in _HASH_KEYS} | {"params": cfg["meta"]["params_version"],
                       "policy": str(policy), "speeds": list(map(float, speeds)),
                       "delays": list(map(float, delays)), "episodes": episodes, "seed": seed},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _policy(cfg, policy):
    if policy == "center":
        from src.slingpuck.policies.scripted import CenterBlocker
        return CenterBlocker(), False
    from src.slingpuck.train.common import load_trained_run
    model, asymmetric, _ = load_trained_run(policy)
    return model, asymmetric


def build_block_table(cfg: dict, policy="center", speeds=None, delays=None, episodes: int = 40,
                      seed: int = 50_000, verbose: bool = False) -> BlockTable:
    from src.slingpuck.train.common import make_goalkeeper_env
    bt = cfg["match"]["block_table"]
    speeds = np.linspace(*bt["speeds_m_s"][:2], int(bt["speeds_m_s"][2])) if speeds is None else np.asarray(speeds)
    delays = np.linspace(*bt["delays_s"][:2], int(bt["delays_s"][2])) if delays is None else np.asarray(delays)
    pol, asymmetric = _policy(cfg, policy)
    env = make_goalkeeper_env(cfg, asymmetric)
    pan_max = env.unwrapped.geom.pan_max

    def rate(speed, start_pan, delay):
        saves = n = 0
        for k in range(episodes):
            obs, info = env.reset(seed=seed + k, options={
                "speed": speed, "start_pan": start_pan * (1 if k % 2 else -1), "release_delay": delay})
            if not info["threat"]:
                continue
            while True:
                action, _ = pol.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            n += 1
            saves += info["outcome"] == "save"
        return saves / max(n, 1)

    p_gate = np.array([rate(s, 0.0, 0.0) for s in speeds])
    p_edge = np.zeros((len(speeds), len(delays)))
    for i, s in enumerate(speeds):
        for j, d in enumerate(delays):
            p_edge[i, j] = rate(s, pan_max, d)
        if verbose:
            print(f"  block table speed {s:.2f} m/s: edge {np.round(p_edge[i], 2)}")
    return BlockTable(speeds, delays, p_gate, p_edge)


def load_or_build_block_table(cfg: dict, verbose: bool = True) -> BlockTable:
    bt = cfg["match"]["block_table"]
    speeds = np.linspace(*bt["speeds_m_s"][:2], int(bt["speeds_m_s"][2]))
    delays = np.linspace(*bt["delays_s"][:2], int(bt["delays_s"][2]))
    key = _cache_key(cfg, bt["policy"], speeds, delays, bt["episodes_per_cell"], 50_000)
    path = CACHE_DIR / f"block_table_{key}.npz"
    if path.exists():
        d = np.load(path)
        return BlockTable(d["speeds"], d["delays"], d["p_gate"], d["p_edge"])
    if verbose:
        print(f"Building the block table (Phase 1 env, policy {bt['policy']}); cached at {path}")
    table = build_block_table(cfg, bt["policy"], speeds, delays, bt["episodes_per_cell"], verbose=verbose)
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(path, speeds=table.speeds, delays=table.delays, p_gate=table.p_gate, p_edge=table.p_edge)
    return table
