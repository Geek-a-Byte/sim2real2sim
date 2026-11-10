"""Shot model (layer 2): P(through | launch x, aim, pull), learned from simulated shots.

An ensemble of small MLPs, each trained on a bootstrap sample. The mean is the success estimate, the
spread across the ensemble is its uncertainty (for the UCB exploration bonus). Inference is plain NumPy
and is precomputed on the candidate grid, so the match env stays fast.

The model is tied to the physics it was fitted on: the file stores a fingerprint of the board, puck,
band and release-noise values, and loading it under other values raises an error. Fit it again after
a calibration (S4) or a new board:

    python -m src.slingpuck.shot.model                  # cache/shot_model.npz from 4000 simulated shots
    python -m src.slingpuck.shot.model --shots 8000 --out cache/shot_model.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.slingpuck.config import REPO_ROOT

FINGERPRINT_KEYS = ("board", "puck", "band")
NOISE_KEYS = ("release_x_std_m", "release_angle_std_rad", "release_speed_std_frac", "release_speed_factor",
              "release_x_bias_m", "release_angle_bias_rad")


def fingerprint(cfg: dict) -> str:
    parts = {k: cfg[k] for k in FINGERPRINT_KEYS}
    parts["noise"] = {k: cfg["shot"][k] for k in NOISE_KEYS}
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def shot_bounds(cfg: dict) -> np.ndarray:
    s = cfg["shot"]
    return np.array([s["launch_x_range_m"], s["aim_range_rad"], s["pull_range_m"]], float)   # (3, 2)


def sample_shots(cfg: dict, n: int, seed: int = 0):
    """n shots with uniform random parameters in the configured ranges, with release noise."""
    from src.slingpuck.physics.shot_sim import ShotSim

    sim, rng = ShotSim(cfg), np.random.default_rng(seed)
    b = shot_bounds(cfg)
    X = rng.uniform(b[:, 0], b[:, 1], (n, 3))
    outcomes = [sim.shoot(*x, rng=rng).outcome for x in X]
    return X, np.array([o == "through" for o in outcomes], float), outcomes


class ShotModel:
    def __init__(self, layers: list, lo, hi, fp: str, meta: dict | None = None):
        self.layers = layers            # Per member: [(W, b), ...]
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.fingerprint, self.meta = fp, meta or {}

    def _norm(self, X):
        return 2 * (np.asarray(X, float) - self.lo) / (self.hi - self.lo) - 1

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Mean and ensemble std of P(through) for shots X (n, 3)."""
        Z = self._norm(np.atleast_2d(X))
        ps = []
        for member in self.layers:
            h = Z
            for k, (W, b) in enumerate(member):
                h = h @ W + b
                if k < len(member) - 1:
                    h = np.maximum(h, 0.0)
            ps.append(1 / (1 + np.exp(-h[:, 0])))
        ps = np.array(ps)
        return ps.mean(0), ps.std(0)

    @classmethod
    def fit(cls, X, y, cfg: dict, members: int = 5, hidden: int = 32, epochs: int = 400, seed: int = 0):
        import torch as th

        lo, hi = shot_bounds(cfg).T
        th.manual_seed(seed)
        rng = np.random.default_rng(seed)
        Z = th.as_tensor(2 * (X - lo) / (hi - lo) - 1, dtype=th.float32)
        Y = th.as_tensor(y, dtype=th.float32)
        layers = []
        for _ in range(members):
            idx = th.as_tensor(rng.integers(len(X), size=len(X)))
            net = th.nn.Sequential(th.nn.Linear(3, hidden), th.nn.ReLU(), th.nn.Linear(hidden, hidden), th.nn.ReLU(),
                                   th.nn.Linear(hidden, 1))
            opt = th.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
            for _ in range(epochs):
                opt.zero_grad()
                loss = th.nn.functional.binary_cross_entropy_with_logits(net(Z[idx])[:, 0], Y[idx])
                loss.backward()
                opt.step()
            lin = [m for m in net if isinstance(m, th.nn.Linear)]
            layers.append([(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in lin])
        meta = {"shots": int(len(X)), "through_rate": float(np.mean(y)),
                "params_version": cfg.get("meta", {}).get("params_version", "?")}
        return cls(layers, lo, hi, fingerprint(cfg), meta)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"m{i}_l{k}_{n}": a for i, member in enumerate(self.layers) for k, wb in enumerate(member)
                  for n, a in zip("Wb", wb)}
        np.savez_compressed(path, lo=self.lo, hi=self.hi, fingerprint=self.fingerprint,
                            meta=json.dumps(self.meta), n_members=len(self.layers), n_layers=len(self.layers[0]),
                            **arrays)

    @classmethod
    def load(cls, path, cfg: dict | None = None) -> "ShotModel":
        z = np.load(path)
        layers = [[(z[f"m{i}_l{k}_W"], z[f"m{i}_l{k}_b"]) for k in range(int(z["n_layers"]))]
                  for i in range(int(z["n_members"]))]
        model = cls(layers, z["lo"], z["hi"], str(z["fingerprint"]), json.loads(str(z["meta"])))
        if cfg is not None and model.fingerprint != fingerprint(cfg):
            raise ValueError(f"{path} was fitted on other board, puck, band or noise values. "
                             "Fit it again: python -m src.slingpuck.shot.model")
        return model


def load_for(cfg: dict) -> ShotModel:
    path = REPO_ROOT / cfg["shot"]["model_file"]
    if not path.exists():
        raise FileNotFoundError(f"No shot model at {path}. Fit one: python -m src.slingpuck.shot.model")
    return ShotModel.load(path, cfg)


def main():
    from src.slingpuck.config import load_config

    p = argparse.ArgumentParser()
    p.add_argument("--params-version", default="latest")
    p.add_argument("--shots", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    args = p.parse_args()
    cfg = load_config(args.params_version)
    X, y, outcomes = sample_shots(cfg, args.shots, args.seed)
    n_fit = int(0.8 * len(X))
    model = ShotModel.fit(X[:n_fit], y[:n_fit], cfg, seed=args.seed)
    p_val, _ = model.predict(X[n_fit:])
    acc = float(np.mean((p_val > 0.5) == (y[n_fit:] > 0.5)))
    brier = float(np.mean((p_val - y[n_fit:]) ** 2))
    model = ShotModel.fit(X, y, cfg, seed=args.seed)       # Final model on all shots
    out = Path(args.out) if args.out else REPO_ROOT / cfg["shot"]["model_file"]
    model.save(out)
    counts = {o: outcomes.count(o) for o in sorted(set(outcomes))}
    print(f"{len(X)} shots {counts}. Held-out accuracy {acc:.1%}, Brier {brier:.3f}. Saved {out}")


if __name__ == "__main__":
    main()
