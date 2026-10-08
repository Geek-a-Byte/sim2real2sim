# slingpuck

Sim2real2sim RL pipeline for an SO-101 arm playing tabletop sling puck
(moopok Fast Sling Puck, size M).

## Setup

```bash
pip install -e ".[dev]"
pytest -q
```

## Configs

`slingpuck.config.load_config(params_version)` merges three files from `configs/`:

| File | Content |
|---|---|
| `servo.yaml` | Servo model (rate limit, lag, deadband, command latency) |
| `physics_params_v<N>.yaml` | Measured / fitted physics. sysid writes a new version for each fit |
| `env.yaml` | Sim timing, tracker tuning, opponent, scoring, domain randomization ranges |

- Every uncalibrated value has a `TODO` comment and is listed under `placeholders:`.
  `load_config(..., strict=True)` refuses to load while any placeholder remains.
- `save_run_config(cfg, run_dir, seed)` writes the merged config, params version,
  git hash, dirty flag and seed next to each training run.
- `sample_randomized(cfg, rng)` returns a domain-randomized copy of the config.

## Board frame

Origin at the board center. x is across the width, y is along the length.
The agent's half is y < 0, the opponent's half is y > 0, and the center divider
with the gate is at y = 0.

## Milestones

| Milestone | Status |
|---|---|
| M1: physics, servo, camera, tracker, unit tests | Done, waiting for review |
| M2: Phase 1 env, PPO baseline, save rate vs speed | Old version present; must be rebuilt on `Fast2DPuckSim` |
| M3: Phase 2 env, band and deflection models, sysid | Old version present; not reviewed |
| M4: opponent, Phase 3 env, rule-based baseline | Old version present; known bugs (see below) |
| M5: selector training, ablations, ONNX, latency | Old version present; not reviewed |

Known bugs that later milestones must fix:
- Opponent shots move away from the gate (`opponent/scripted_opponent.py`).
- `MatchEnv` does not count goals during sling recovery.
- The robot stub maps action 1.0 to 0.5 rad; the sim maps it to ~0.19 rad.
