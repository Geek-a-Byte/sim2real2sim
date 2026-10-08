# slingpuck

Sim2real2sim reinforcement-learning pipeline for an **SO-101 arm** (6-DOF,
LeRobot-compatible, Feetech STS3215 servos) that plays tabletop **sling puck**
on a **moopok Fast Sling Puck board, size M**. Deployment inference must run on
a Jetson Orin Nano.

The project has three phases:

| Phase | Task | Control variable |
|---|---|---|
| 1. Goalkeeper | A paddle on the arm blocks pucks that come through the center gate. | Pan joint target |
| 2. Targeted shot | The arm hooks the puck with a passive slip-release end effector, pulls back, and releases it through the gate. | Strike angle and pull-back distance |
| 3. Adversarial | A high-level policy chooses **block**, **sling** or **hold** against an opponent. Research question: *when is it safe to commit to a sling?* | Discrete selector over frozen primitives |

---

## Contents

1. [Status](#status)
2. [Setup](#setup)
3. [Quick start](#quick-start)
4. [Repository layout](#repository-layout)
5. [Board frame and geometry](#board-frame-and-geometry)
6. [Configs and parameter versioning](#configs-and-parameter-versioning)
7. [Physics and sensing models](#physics-and-sensing-models)
8. [Phase 1: goalkeeper](#phase-1-goalkeeper)
9. [Tests](#tests)
10. [Reproducibility](#reproducibility)
11. [Known problems and open items](#known-problems-and-open-items)
12. [Values to measure on the real hardware](#values-to-measure-on-the-real-hardware)

---

## Status

| Milestone | Content | Status |
|---|---|---|
| M1 | Physics, servo, camera, tracker, config versioning, unit tests | Done |
| M2 | Phase 1 env, asymmetric PPO, save-rate eval, viewer | Code done. 3-seed training in progress. Waits for review |
| M3 | Phase 2 env, band and deflection models, sysid with synthetic logs | Old code only. Not reviewed |
| M4 | Opponent model, Phase 3 hierarchical env, rule-based baseline | Old code only. Has known bugs |
| M5 | Selector training, ablations, ONNX export, latency benchmark | Old code only. Not reviewed |

"Old code" is code from before the M1 rework. It runs, but it was not
rebuilt on the new physics and sensing models. Each later milestone replaces it.

Git history:

| Commit | Content |
|---|---|
| `10d29f9` | Baseline snapshot of the original code |
| `c20b8ea` | M1 |
| `7810cbe` | User-measured puck mass and gate width |
| `36e0faa` | M2: goalkeeper env, asymmetric PPO, eval |
| `69ce734` | M2: recovery task, grid eval |
| `74d72f9` | Top-down goalkeeper viewer |

---

## Setup

Requirements: Python 3.10 or later. The project was tested with Python 3.14
(miniconda). Everything runs on CPU.

```bash
pip install -e ".[dev]"
pytest -q            # 72 tests, about 6 s
```

On the development machine, use the miniconda interpreter
(`~/miniconda3/bin/python3`). The Homebrew Python 3.10 does not have pytest or
PyBullet installed.

Main dependencies: Gymnasium, NumPy, SciPy, PyTorch, Stable-Baselines3,
TensorBoard, PyYAML, Matplotlib, Pillow (GIF output). PyBullet and MuJoCo are
installed but are not used by the current Phase 1 pipeline.

---

## Quick start

```bash
# Train the goalkeeper (seeds and hyperparameters from configs/train_goalkeeper.yaml)
python -m slingpuck.train.train_goalkeeper
python -m slingpuck.train.train_goalkeeper --seeds 0 --timesteps 200000   # shorter run

# Watch training
tensorboard --logdir tensorboard_logs/goalkeeper

# Evaluate: save rate vs puck speed, with 95% CIs, against scripted baselines
python -m slingpuck.eval.save_rate_vs_speed --runs logs/goalkeeper/goalkeeper_v0_s*_*

# Watch episodes top-down in slow motion
python -m slingpuck.eval.visualize_goalkeeper --run logs/goalkeeper/<run_dir>
python -m slingpuck.eval.visualize_goalkeeper --policy center --speed 2.6 --start-offset 0.9 --save gk.gif
```

---

## Repository layout

```
configs/
  physics_params_v0.yaml   Measured/fitted physics, version 0 (uncalibrated). sysid writes v1, v2, ...
  servo.yaml               Servo model parameters
  env.yaml                 Sim timing, tracker tuning, goalkeeper task, opponent, scoring, randomization
  train_goalkeeper.yaml    PPO hyperparameters and seeds for Phase 1
assets/
  urdf/so101.urdf          SO-101 URDF (not used yet)
  mujoco/                  SO-101 MuJoCo model (RobotStudio) and an old board scene
data/real_logs/schema.md   CSV schema for real teleoperation logs (sysid input)
src/slingpuck/
  config.py                Config loader, params versioning, placeholders, strict mode, randomization, run records
  kinematics.py            GoalkeeperGeometry: the one action -> pan -> joint mapping (sim and robot)
  physics/
    backend.py             PuckPhysicsBackend interface, PaddleState, StepEvents, GateCrossing
    puck_dynamics.py       Fast2DPuckSim: friction, walls, divider + gate, puck-puck, kinematic paddle
    servo_model.py         Latency -> deadband -> first-order lag -> rate limit
    band_model.py          Old linear band model (M3 replaces it)
    arm_deflection.py      Old compliance model (M3 replaces it)
  sensing/
    camera_model.py        Frame rate, latency, Gaussian noise, dropouts; seeded
    kalman_tracker.py      Constant-velocity Kalman filter with latency compensation
  envs/
    goalkeeper_env.py      Phase 1 env (recovery task)
    wrappers.py            PrivilegedObsWrapper (Dict obs for the asymmetric critic)
    sling_env.py           Old Phase 2 env (M3 replaces it)
    match_env.py           Old Phase 3 env (M4 replaces it; has known bugs)
  policies/
    asymmetric.py          AsymmetricActorCriticPolicy for SB3 PPO
    scripted.py            CenterBlocker, HoldStart baselines
  train/
    common.py              Env factories, run directories, outcome logging callback
    train_goalkeeper.py    Phase 1 PPO training
    train_selector.py      Old Phase 3 training (M5 replaces it)
  eval/
    save_rate_vs_speed.py  Phase 1 eval with CIs, CSV tables and a plot
    stats.py               Wilson and Student-t confidence intervals
    visualize_goalkeeper.py  Top-down slow-motion viewer (window or GIF)
    visualize_mujoco.py    Old 3D viewer (does not run, see known problems)
    vulnerability_window.py  Old Phase 3 baseline (M4 replaces it)
  sysid/fit_physics.py     Old sysid (M3 replaces it)
  deploy/                  Old ONNX export, latency benchmark, LeRobot stub (M5)
tests/                     pytest suite
logs/, tensorboard_logs/, results/   Training runs and eval outputs (not in git)
```

---

## Board frame and geometry

**Frame:** the origin is at the board center. x is across the board width, and y
is along the board length. The **agent's half is y < 0**, the **opponent's half
is y > 0**, and the center divider with the gate is at y = 0.

**Reference board:** moopok Fast Sling Puck, size M
([Amazon B086KWXPTY](https://www.amazon.com/moopok-Fast-Sling-Puck-Game/dp/B086KWXPTY/?th=1)).
Each value in `configs/physics_params_v0.yaml` has a source tag:

| Tag | Meaning |
|---|---|
| `[spec]` | From the size M listing text |
| `[image]` | From the listing image. The image shows size S, and we assume that size M is the same |
| `[user]` | Measured or estimated by the user |
| `[derived]` | Computed from other tagged values |
| `TODO` | Unknown. Must be measured or fitted |

| Quantity | Value | Source |
|---|---|---|
| Outer length × width × height | 0.3658 × 0.2159 × 0.0249 m (14.4 × 8.5 × 0.98 in) | spec |
| Frame (wall) thickness | 0.01524 m (0.6 in) | image |
| Inner play area | 0.3353 × 0.1854 m | derived |
| Gate width | 0.03175 m (1.25 in) | user: "a little over 1.19 in". Still TODO |
| Divider thickness | 0.010 m | TODO |
| Puck diameter × thickness | 0.0302 × 0.0102 m (1.19 × 0.4 in) | image |
| Puck mass | 0.027 kg | user. See the note below |
| Band rest length | 0.1854 m (assumed to span the inner width) | derived |
| Band distance from the end wall | 0.05 m | TODO |

**Note on the puck mass:** 27 g in a 1.19 × 0.4 in puck gives a density of
3.7 g/cm³. Wood is 0.5 to 0.75 g/cm³. 27 g / 6 = 4.5 g, which is the expected
mass of one wood puck. Weigh one puck alone to confirm. The mass does not affect
Phase 1. It does affect the launch speed in Phase 2.

**Note on the gate:** the gate is only about 1.5 mm wider than the puck. A
straight shot passes only when its path is within about ±3 mm of the gate center
(the gate corners deflect slightly off-center pucks through). This fact drives
the Phase 1 task design (see below).

---

## Configs and parameter versioning

### Merge order

`slingpuck.config.load_config(params_version)` merges three files. A later file
overrides an earlier one, key by key:

1. `servo.yaml`
2. `physics_params_v<N>.yaml` (its `servo:` section overrides `servo.yaml` after sysid)
3. `env.yaml`

```python
from slingpuck.config import load_config, sample_randomized, save_run_config
cfg = load_config("latest")          # or "v0", 0, "v3"
cfg["meta"]                          # params_version, params_file, calibrated, placeholders, git_hash, git_dirty
```

### Versioned physics parameters

- sysid (M3) writes a new file `physics_params_v<N+1>.yaml` for each fit.
- `"latest"` loads the highest version. The loader checks that the `version:`
  field in the file matches the file name.
- Each training run records the params version that it used.

### Placeholders and strict mode

- Every value that is not measured has a `TODO` comment **and** is listed under
  `placeholders:` in its file. There are 22 in `physics_params_v0.yaml` and 4 in
  `servo.yaml`.
- `load_config(..., strict=True)` raises `ConfigError` while any placeholder or
  null value remains. Use strict mode for every result that you report against
  real data. Training accepts `--strict`.

### Domain randomization

`env.yaml` → `randomization.params` maps a dotted config path to a rule:

```yaml
puck.friction_kinetic: {scale: [0.7, 1.3]}           # value * U(0.7, 1.3)
board.gate_width_m:    {range: [0.0307, 0.0330]}     # U(lo, hi)
board.restitution:     {scale: [0.85, 1.1], clip: [0.0, 1.0]}
```

Randomized values: puck friction and restitution, wall and paddle restitution,
gate width, band stiffness and hysteresis, arm compliance, camera latency, noise
and dropout rate, and servo latency, lag and maximum rate. `sample_randomized(cfg,
rng)` returns a changed copy and records the sampled values in
`cfg["meta"]["randomized"]`. The goalkeeper env samples new values at every reset.
The ranges are placeholders until sysid gives real uncertainty.

---

## Physics and sensing models

### Servo (`physics/servo_model.py`)

Each step applies, in this order:

1. **Command latency:** exactly `round(latency / dt)` steps.
2. **Deadband:** the servo ignores errors smaller than the deadband.
3. **First-order lag:** exact discretization of `tau * dx/dt = target - x`.
4. **Rate limit:** default 300 deg/s (about 50 RPM).

The model works for one joint or an array of joints. The values are in
`servo.yaml`, and all of them are placeholders.

### Camera (`sensing/camera_model.py`)

- 60 Hz frames, configurable latency, Gaussian position noise, random dropouts.
- Each frame has its capture time and its arrival time (capture + latency).
- `observe(t, true_xy)` returns only the frames that arrive at this step, so a
  frame is never delivered twice.
- It uses a seeded `numpy.random.Generator`, so episodes repeat from a seed.

### Kalman tracker (`sensing/kalman_tracker.py`)

- Constant-velocity model with state `[x, y, vx, vy]` and a white-noise
  acceleration process model.
- It starts from the first frame. The measurement noise comes from the camera config.
- **Latency compensation:** `estimate(t_now)` predicts from the last frame to the
  current time. In tests, the error is less than 25% of the raw-frame error.
- Tuning value `accel_noise_std_m_s2 = 2.0`. This is about μg, the friction
  deceleration. Steady state at 60 Hz with 3 mm noise: 0.056 m/s velocity error
  and 5 mm position error after 50 ms of latency.
- **Limits:** the filter does not model wall bounces or a sudden launch from rest
  (see known problems).

### Fast 2D puck sim (`physics/puck_dynamics.py`)

`Fast2DPuckSim` implements the `PuckPhysicsBackend` interface, so a PyBullet (or
later Isaac) backend can replace it without env changes.

- Coulomb sliding friction: constant deceleration μg. The puck stops and does not reverse.
- Side and end walls, and the divider with the gate gap. A contact scales the
  normal velocity by −restitution.
- Puck-puck contacts with an equal-mass impulse.
- **Kinematic paddle:** an oriented box with linear and angular velocity. The
  contact uses the paddle surface velocity, so a moving paddle can push the puck.
- `step(dt)` returns `StepEvents`: gate crossings (with direction) and paddle contacts.
- Without a moving paddle, the kinetic energy never increases. A test checks this.
- Speed: the per-puck code uses Python floats, because NumPy call overhead is
  larger than the arithmetic for 1 or 2 values. It is 5 times faster than the
  NumPy version and gives the same events (max state difference 7e-13 over 200
  random trials).

### Goalkeeper kinematics (`kinematics.py`)

`GoalkeeperGeometry` is the **only** mapping from a policy action to a pan angle
to the SO-101 `shoulder_pan` joint command. The env and
`deploy/lerobot_interface.py` both use it, so the sim and the robot cannot
disagree. (The original code mapped action 1.0 to 0.19 rad in the sim and to
0.5 rad on the robot.)

| Quantity | Value |
|---|---|
| Arm base (pan axis) | (0, −0.2177) m. Placeholder: 5 cm behind the agent end wall |
| Arm radius to the paddle center | 0.2047 m |
| Pan range | ±20° (keeps the paddle inside the board) |
| Paddle | 30 × 6 mm. Placeholder |
| Paddle center at pan = 0 | (0, −0.013) m |
| Paddle face standoff from the divider | 5 mm |

---

## Phase 1: goalkeeper

### Why the task is a "recovery" task

The gate is only about 1.5 mm wider than the puck, so a paddle at the gate center
blocks every shot. When the paddle has time to get there, "go to the center" is
the best possible policy, and it does not need to see the puck. With the original
setup, PPO and the center policy both saved 100% of shots at every speed.

So Phase 1 models the case where the arm is busy (for example, after a sling)
when the opponent shoots:

- The paddle is **locked at a random start offset** until the puck launches,
  plus an optional `release_delay_s`.
- `reset()` runs the locked period internally. The first agent step is at the
  release time, and the camera and tracker are already warm.
- The result is a **time-to-cover** measurement: the save rate as a function of
  puck speed and paddle start offset. Phase 3 uses it for the vulnerability window.

### Episode

1. Domain-randomized physics are sampled.
2. The puck rests at the opponent band line at a random x for 0.1 to 0.5 s.
3. The puck launches toward a random aim point within ±6 mm of the gate center,
   with a speed from `speed_range_m_s` (0.8 to 3.0 m/s, a placeholder).
4. With `threats_only: true`, shots are resampled until the shot would score
   when no paddle is present. A shot counts as a threat when it enters the agent
   half before it turns back or stops.
5. The paddle is free at launch + release delay. The policy then acts at 30 Hz.
   The physics runs at 1 kHz.

### Observation, action, reward

**Action:** one value in [−1, 1] → pan target in [−20°, +20°].

**Policy observation** (8 values, from deployable sensors only):

| Name | Content |
|---|---|
| `trk_x`, `trk_y` | Kalman position estimate / half board length |
| `trk_vx`, `trk_vy` | Kalman velocity estimate / 3 m/s |
| `trk_valid` | 1 after the first camera frame, else 0 (and the estimate values are 0) |
| `pan`, `pan_vel` | Pan encoder / pan range, pan velocity / max servo rate |
| `prev_action` | Previous action |

**Privileged observation** (16 values, critic only, in `info["privileged"]`):
true puck position and velocity, launched flag, time to launch, true pan and pan
velocity, touched flag, and the randomized friction, paddle and wall restitution,
camera latency, servo latency, gate width and servo rate (relative to nominal).

**Outcomes and reward:**

| Outcome | Condition | Reward |
|---|---|---|
| save | The puck touched the paddle and then moved back toward the opponent | +1 |
| goal | The puck got past the paddle line, or stopped in the agent half | −1 |
| none | The puck never threatened (only when `threats_only` is false) | 0 |

A smoothness penalty of `0.01 × |a_t − a_{t−1}|` is added at each step.
Episodes are short: about 3 agent steps after release.

### Asymmetric actor-critic

`policies/asymmetric.py` defines `AsymmetricActorCriticPolicy` for SB3 PPO, used
with `envs/wrappers.PrivilegedObsWrapper`:

- The **actor** reads only `obs["policy"]` (8 values). Only the actor is
  deployed, so the robot never needs privileged inputs.
- The **critic** reads `obs["policy"]` and `obs["privileged"]` (24 values).
- A test checks that a change to the privileged input does not change the actor
  output, and does change the critic value.

Set `policy: mlp` in `configs/train_goalkeeper.yaml` to train a normal PPO
policy, where the actor and the critic both see only the tracked observation.

### Training

```bash
python -m slingpuck.train.train_goalkeeper [--seeds 0 1 2] [--timesteps N] [--params-version v0] [--strict]
```

- Hyperparameters: `configs/train_goalkeeper.yaml` (PPO, 8 subprocess envs,
  `n_steps` 512, batch 256, γ = 0.98, net [64, 64], 1M steps, seeds 0, 1, 2).
- Each seed writes `logs/goalkeeper/goalkeeper_<params>_s<seed>_<time>/` with
  `run_config.yaml` (full merged config, train config, params version, git hash,
  dirty flag, seed), checkpoints every 250k steps, and `final_model.zip`.
- TensorBoard: `tensorboard_logs/goalkeeper/`. Besides the standard SB3 values,
  the script logs `goalkeeper/save_rate`, `goal_rate` and `none_rate` for each rollout.
- Run time: about 40 min for each 1M-step seed on the development laptop. The
  episodes are short, so `reset()` (pre-launch time and threat sampling) is most
  of the cost.

### Evaluation

```bash
python -m slingpuck.eval.save_rate_vs_speed --runs <run_dir> [<run_dir> ...] \
    [--episodes-per-bin 300] [--n-speed-bins 6] [--release-delay 0.0] [--nominal]
```

- Speed bins are stratified: each bin gets the same number of shots, with the
  speed uniform within the bin. Only threatening shots count.
- **Paired:** all policies get the same shots (same eval seeds).
- Policies: every PPO run given, plus the scripted baselines
  `center` (always command pan = 0) and `hold` (never move).
- **Confidence intervals (95%):** PPO uses a Student-t interval across training
  seeds. The deterministic scripted baselines use a Wilson interval over episodes.
- Outputs in `results/goalkeeper/<time>/`:
  - `save_rate_vs_speed.csv`
  - `save_rate_vs_start_offset.csv`
  - `save_rate_grid.csv` (speed × start offset, the time-to-cover table for Phase 3)
  - `save_rate.png`
  - `eval_meta.yaml` (params version, randomization, seeds, runs, git hash)

### Viewer

```bash
python -m slingpuck.eval.visualize_goalkeeper --run <run_dir>            # live window
python -m slingpuck.eval.visualize_goalkeeper --policy center --save gk.gif
```

A top-down slow-motion view (default 10 times slower). It shows the true puck
(filled), the last camera frame (×), the Kalman estimate (blue ring and velocity
arrow), and the paddle (faded while locked). The header shows the time from
launch, the shot speed, and the outcome.

| Option | Effect |
|---|---|
| `--episodes N` | Number of episodes (default 3) |
| `--speed` | Shot speed, m/s |
| `--start-offset` | Paddle start, −1 to 1 of the pan range |
| `--release-delay` | Seconds from launch until the paddle is free |
| `--slowmo` | Playback speed factor (default 0.1) |
| `--frame-ms` | Sim time between frames (default 4 ms) |
| `--nominal` | No domain randomization |
| `--save file.gif` | Write a GIF instead of opening a window |

### Results so far (seed 0 only, preliminary)

Eval with 150 shots per speed bin, domain-randomized physics, release delay 0,
start offset uniform over the pan range. 95% Wilson intervals. The final result
needs all 3 seeds.

| Speed (m/s) | PPO seed 0 | Center | Never move |
|---|---|---|---|
| 0.80–1.17 | 100.0% | 100.0% | ~45% |
| 1.17–1.53 | 100.0% | 99.3% | ~46% |
| 1.53–1.90 | 96.7% [92.4, 98.6] | 90.0% [84.2, 93.8] | ~44% |
| 1.90–2.27 | 86.7% [80.3, 91.2] | 76.7% [69.3, 82.7] | ~34% |
| 2.27–2.63 | 64.7% [56.7, 71.9] | 50.7% [42.7, 58.6] | ~29% |
| 2.63–3.00 | 57.3% [49.3, 65.0] | 53.3% [45.4, 61.1] | ~38% |

The "never move" values are from a separate 100-shot run.

**Why PPO does better than "go to center":** PPO commands the far side of the pan
range (action −1 from a start of +0.9), not the center. With the first-order
servo lag, a far target keeps the paddle at full speed for longer. A scripted
"overshoot" policy gives the same save rate as PPO at 1.9 to 2.6 m/s (79.0% vs
79.0%, center 72.3%). The size of this gain depends on the servo lag time
constant, which is still a placeholder. Measure the real servo step response
before you rely on it.

---

## Tests

```bash
pytest -q
```

| File | What it checks |
|---|---|
| `test_config.py` | Version resolution, `latest`, version mismatch, strict mode, servo overrides, randomization ranges and seeding, run records |
| `test_servo.py` | Exact latency in steps, latency on any signal, rate limit, 63% at τ, deadband, vector joints |
| `test_camera.py` | Frame rate, **latency shifts the signal correctly**, each frame once, dropout rate, noise std, seeding |
| `test_tracker.py` | Start state, convergence, latency compensation, dropouts, covariance symmetric and positive |
| `test_puck_dynamics.py` | Friction stop, wall restitution, **energy never increases through bounces**, energy kept without losses, divider blocks, gate crossing, puck-puck swap, paddle restitution, moving and rotating paddle |
| `test_kinematics.py` | Action mapping and inverse, paddle geometry, range check, robot stub uses the shared mapping |
| `test_goalkeeper_env.py` | **Gymnasium `check_env`** (Box and Dict), seeded repeat, threat sampling, save and goal outcomes, smoothness penalty, policy sees only tracked data, randomization, locked paddle, release delay, viewer |
| `test_asymmetric_policy.py` | Actor ignores privileged input, critic uses it, save and load |
| `test_stats.py` | Wilson and t intervals against known values |
| `test_band.py` | Old band hysteresis (M3 replaces it with a closed-loop test) |

---

## Reproducibility

- Every env, camera and opponent uses the Gymnasium `np_random` generator, so an
  episode repeats exactly from its seed (tested).
- Training seeds PyTorch, NumPy and the envs (`set_random_seed`, PPO `seed`,
  `make_vec_env(seed=...)`).
- Each run saves the full merged config, the physics params version, the git
  hash and dirty flag, and the seed. Commit your changes before a run that you
  will report, so that `git_dirty` is false.
- The eval saves its own `eval_meta.yaml` with the same information.

---

## Known problems and open items

### Phase 1 (M2)

- **Tracker lag after launch:** about 80 ms after launch, the Kalman estimate is
  about 10 cm behind the puck. With 50 ms latency, only 1 or 2 frames from after
  the launch have arrived, and the constant-velocity filter reacts slowly to a
  sudden launch from rest. The goalkeeper is not affected (the best policy does
  not need the puck position), but Phases 2 and 3 will be. Possible fix: launch
  (maneuver) detection with a higher process noise.
- **Training speed:** `reset()` is most of the cost, because episodes are only
  about 3 steps. A shorter pre-launch time, or a fast-forward of the locked
  period, would make training faster.
- The pan encoder in the observation has no noise or quantization yet (TODO).
- A scripted overshoot baseline is not yet part of the eval.

### Later milestones (old code, not yet rebuilt)

- **M3:** `band_model.py` is linear, and its hysteresis is not a closed loop.
  `sling_env.py` is a one-step bandit with a straight-line gate check. The
  sysid fits only 3 parameters and has no synthetic-log generator or example CSV.
- **M4:** `opponent/scripted_opponent.py` fires shots **away** from the gate (sign
  error), and it uses the global NumPy random state. `match_env.py` does not
  count goals during sling recovery, and a sling resolves in one step. The opponent
  gives a noisy angle, not a hand position and velocity.
- **M5:** `deploy/lerobot_interface.py` builds the old 6-value observation; it
  must build the 8-value `GoalkeeperEnv` observation. `train_selector.py` loads
  model paths that do not exist.
- **3D viewer:** `eval/visualize_mujoco.py` does not run. The scene `include`
  path is wrong, the joint names are wrong (the SO-101 model uses `shoulder_pan`,
  `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`), the board
  is the old size, and the script reads env attributes that no longer exist. A fix
  needs inverse kinematics to put the paddle in front of the gate.
- **PyBullet backend:** not implemented yet. The `PuckPhysicsBackend` interface
  is ready for it.

---

## Values to measure on the real hardware

These values are placeholders now. Each one is listed under `placeholders:` in
its config file. Strict mode does not run until you replace them.

| Value | Config path | Why it matters |
|---|---|---|
| Gate width | `board.gate_width_m` | Sets the Phase 2 target size and the Phase 1 threat zone |
| Divider thickness | `board.divider_thickness_m` | Paddle position and gate geometry |
| Puck mass (one puck alone) | `puck.mass_kg` | Phase 2 launch speed |
| Puck diameter and thickness on size M | `puck.radius_m`, `puck.thickness_m` | All contacts |
| Band position and maximum stretch | `band.band_offset_from_end_wall_m`, `band.max_stretch_m` | Shot start point, Phase 2 range |
| Band force vs stretch (loading and unloading) | `band.*` | Phase 2 band model (M3) |
| Arm base position | `robot.base_xy_m` | Paddle arc |
| Pan joint zero offset | `robot.pan_zero_offset_rad` | Sim-to-robot angle mapping |
| Paddle (end effector) width and thickness | `robot.paddle_width_m`, `robot.paddle_thickness_m` | Blocking geometry |
| Servo speed, step response, deadband, latency | `servo.*` | Reaction time; the size of the PPO overshoot gain |
| Camera latency, noise, dropout rate | `camera.*` | Tracker accuracy |
| Real shot speeds | `goalkeeper.speed_range_m_s` | Phase 1 speed range |
| Game scoring rules | `scoring` | Phase 3 reward |
