"""Synthetic logs from the simulator with known "true" parameters.

Used to test the sysid pipeline before real logs exist: generate logs with a
truth config, fit them, and check that the fit recovers the truth.

    python -m src.slingpuck.sysid.synthetic --out data/real_logs/synthetic
writes band_bench.csv, servo_step.csv, shots.csv and truth.yaml.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.slingpuck.config import load_config, set_path
from src.slingpuck.envs.sling_env import launch_and_fly, realize_pull
from src.slingpuck.physics.band_model import BandModel
from src.slingpuck.physics.servo_model import ServoModel
from src.slingpuck.sensing.camera_model import CameraModel

# "True" physics for the synthetic test: deliberately different from v0.
DEFAULT_TRUTH = {
    "band.stiffness_k_n_m": 230.0,
    "band.stiffness_exponent": 1.2,
    "band.natural_length_m": 0.152,
    "band.hysteresis_loss_factor": 0.22,
    "band.energy_transfer": 0.62,
    "arm.deflection_compliance_m_per_n": 0.0014,
    "puck.friction_kinetic": 0.24,
    "board.restitution": 0.78,
    "servo.command_latency_s": 0.042,
    "servo.lag_tau_s": 0.065,
    "servo.max_rate_deg_s": 250.0,
    "servo.deadband_deg": 0.4,
}

GAUGE_NOISE_N = 0.05
ENCODER_NOISE_RAD = 0.002
ENCODER_COUNTS = 4096  # STS3215: 12-bit encoder


def truth_config(base: dict, truth: dict) -> dict:
    cfg = copy.deepcopy(base)
    for path, value in truth.items():
        set_path(cfg, path, value)
    return cfg


def band_bench_log(cfg, rng, cycles=3, points=25) -> pd.DataFrame:
    band = BandModel.from_config(cfg)
    d_max = cfg["band"]["max_pull_m"]
    rows = []
    for c in range(cycles):
        top = d_max * (0.7 + 0.3 * c / max(cycles - 1, 1))  # Different peak per cycle
        s_max = band.elongation((0.0, band.band_y - top))
        for phase, ds in (("load", np.linspace(0, top, points)), ("unload", np.linspace(top, 0, points))):
            for d in ds:
                pos = (0.0, band.band_y - d)
                s = band.elongation(pos)
                t = band.tension_load(s) if phase == "load" else band.tension_unload(s, s_max)
                force = float(t) * band.direction_sum(pos)[1]
                rows.append((c, phase, d, force + rng.normal(0, GAUGE_NOISE_N)))
    return pd.DataFrame(rows, columns=["cycle", "phase", "pull_m", "force_n"])


def servo_step_log(cfg, rng, sample_dt=0.005, duration=0.6) -> pd.DataFrame:
    steps = [(0.0, 0.35), (0.35, -0.35), (-0.35, 0.0), (0.0, 0.1), (0.1, 0.05), (0.05, -0.2)]
    rows = []
    dt = 0.001
    q = 2 * np.pi / ENCODER_COUNTS
    for i, (start, target) in enumerate(steps):
        servo = ServoModel(cfg["servo"], dt, init_pos=start)
        n = int(round(duration / dt))
        every = int(round(sample_dt / dt))
        for k in range(n):
            cmd = start if k * dt < 0.05 else target
            pos = float(servo.step(cmd))
            if k % every == 0:
                meas = np.round((pos + rng.normal(0, ENCODER_NOISE_RAD)) / q) * q
                rows.append((i, k * dt, cmd, meas))
    return pd.DataFrame(rows, columns=["test_id", "t_s", "cmd_rad", "meas_rad"])


def _frames(cfg, rng, positions_fn, t0, t1):
    """Camera frames (capture times, noisy positions) of a trajectory between t0 and t1."""
    cam = CameraModel({**cfg["camera"], "latency_s": 0.0}, rng=rng)
    cam.reset(t0=t0)
    out = []
    for t in np.arange(t0, t1, 0.001):
        for f in cam.observe(t, positions_fn(t)):
            out.append((f.t_capture, *f.xy))
    return out


def shots_log(cfg, rng, n_gate=40, n_divider=25) -> pd.DataFrame:
    """Shots near the gate (most succeed), plus off-center shots that bounce off the divider."""
    band = BandModel.from_config(cfg)
    rows = []
    plans = [("gate", rng.uniform(-0.002, 0.002), rng.uniform(0.012, 0.03)) for _ in range(n_gate)]
    plans += [("divider", rng.choice([-1, 1]) * rng.uniform(0.03, 0.06), rng.uniform(0.02, 0.03))
              for _ in range(n_divider)]
    for shot_id, (_, slide, pull_cmd) in enumerate(plans):
        x0 = rng.uniform(-0.06, 0.06)
        angle = rng.uniform(-0.1, 0.1)
        rest = np.array([x0, band.band_y])
        slide_act = slide + rng.normal(0, cfg["sling"]["place_noise_m"])
        pull = realize_pull(cfg, band, slide_act, angle, pull_cmd, rng)
        flight = launch_and_fly(cfg, band, pull.pos, rng, record=True)
        traj = flight.trajectory
        meta = (slide, angle, pull_cmd, int(flight.success))
        for t, x, y in _frames(cfg, rng, lambda t: rest, 0.0, 0.3):
            rows.append((shot_id, "rest", t - 0.6, x, y, *meta))
        for t, x, y in _frames(cfg, rng, lambda t: pull.pos, 0.0, 0.3):
            rows.append((shot_id, "pulled", t - 0.3, x, y, *meta))
        if len(traj):
            t_end = min(traj[-1, 0], 1.0)

            def pos_at(t, traj=traj):
                return np.array([np.interp(t, traj[:, 0], traj[:, 1]), np.interp(t, traj[:, 0], traj[:, 2])])

            for t, x, y in _frames(cfg, rng, pos_at, traj[0, 0], t_end):
                rows.append((shot_id, "flight", t, x, y, *meta))
    return pd.DataFrame(rows, columns=["shot_id", "phase", "t_s", "puck_x_m", "puck_y_m", "cmd_slide_m",
                                       "cmd_angle_rad", "cmd_pull_m", "success"])


def generate(out_dir, base_version="v0", truth=None, seed=0, n_gate=40, n_divider=25) -> dict:
    truth = dict(DEFAULT_TRUTH if truth is None else truth)
    cfg = truth_config(load_config(base_version), truth)
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    band_bench_log(cfg, rng).to_csv(out / "band_bench.csv", index=False)
    servo_step_log(cfg, rng).to_csv(out / "servo_step.csv", index=False)
    shots_log(cfg, rng, n_gate, n_divider).to_csv(out / "shots.csv", index=False)
    (out / "truth.yaml").write_text(yaml.safe_dump(
        {"synthetic": True, "base_version": base_version, "seed": seed, "truth": truth}, sort_keys=False))
    return truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/real_logs/synthetic")
    parser.add_argument("--base-version", default="v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-gate", type=int, default=40)
    parser.add_argument("--n-divider", type=int, default=25)
    args = parser.parse_args()
    generate(args.out, args.base_version, seed=args.seed, n_gate=args.n_gate, n_divider=args.n_divider)
    print(f"Wrote synthetic logs to {args.out}")


if __name__ == "__main__":
    main()
