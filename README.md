# slingpuck

An **SO-101 arm** (6-DOF, LeRobot-compatible, Feetech STS3215 servos) plays
tabletop **sling puck** against a human, on a **moopok Fast Sling Puck board,
size M**. Deployment inference must run on a Jetson Orin Nano.

**Research question:** *when is it safe to commit to a sling?* A sling takes the
arm away from the gate, so the gate is open until the arm comes back. A
high-level policy chooses **block**, **sling** or **hold** against the opponent.

---

## Contents

1. [Status](#status)
2. [Setup](#setup)
3. [Quick start](#quick-start)
4. [Repository layout](#repository-layout)
5. [Board frame and geometry](#board-frame-and-geometry)
6. [Configs and parameter versioning](#configs-and-parameter-versioning)
7. [The match](#the-match)
8. [MuJoCo match](#mujoco-match)
9. [Real arm](#real-arm)
10. [Overhead camera: where the pucks are](#overhead-camera-where-the-pucks-are)
11. [Where RL is used](#where-rl-is-used)
12. [Tests](#tests)
13. [Reproducibility](#reproducibility)
14. [Known problems and open items](#known-problems-and-open-items)
15. [Values to measure on the real hardware](#values-to-measure-on-the-real-hardware)
16. [Future work](#future-work)

---

## Status

Status on 2026-11-10. The plan and the milestone notes are in [docs/PLAN.md](docs/PLAN.md).

| Milestone | Content | Status |
|---|---|---|
| M4 | Opponent model, event-level match env, primitives, rule-based baselines, match eval, MuJoCo match | Done. Waits for review |
| S1 | Shot model: P(through \| launch x, aim, pull) from simulated shots, and the shot time T(puck) | Done. Fitted again on 2026-11-09 with the current config (held-out accuracy 95.4%) |
| S2 | Greedy game rule, per-puck positions, greedy shot choice, PPO selector, new metrics | Done. The PPO selector learns "always sling" with the placeholder times |
| S3 | Scripted arm skills in the MuJoCo match: pick (with push-out from a row), seat, draw, release, block, hold | Done in sim. In review. Not committed |
| Real arm | Joint map, board fit (touch test), surface height (vertical stop), record and replay of sim shots | Board fit done (7.7 mm RMS). Replay done only 30 mm above the board, at 0.25× speed. **No shot on the board yet.** Not committed |
| Camera | Overhead camera: puck positions, colours and speeds in the board frame, used by the arm script | Code and tests done. Tested only on sample media, not on the real camera. Not committed |
| S4 | Physics calibration from 40–60 real shots | Not started |
| S5 | Opponent profiles (4 team members), human games | Not started |
| S6 | Deploy: live closed loop, opponent state from the camera, ONNX on the Jetson | Started (camera puck positions only) |

This branch (`slingpuck`) is the sling puck task: step 7 ("outcome layer for
dynamic tasks") of the Sim2Real2Sim synthesis. The general any-task method is
on the branch `generalized`.

The earlier goalkeeper (Phase 1), targeted shot (Phase 2), system
identification and digital twin code is not on this branch. It is on `main`
(up to commit `0221bb5`).

---

## Setup

Requirements: Python 3.10 or later. The project was tested with Python 3.14
(miniconda). Everything runs on CPU.

```bash
pip install -e ".[dev]"
pip install -e ".[vision]"   # OpenCV, for the overhead camera (src/slingpuck/sensing/)
pytest -q            # 133 tests, about 20 s
```

**Which Python:** on the lab Mac, `~/.zshrc` aliases `python` to Homebrew Python 3.10, which has no pytest.
Use `python3` (miniconda, Python 3.14) for this project, and the `lerobot` conda env for the scripts that talk to
the real arm (see [Real arm](#real-arm)).

**Run all commands from the project root folder**, in the form
`python -m src.slingpuck.<module>`. The code imports the package as
`src.slingpuck`, so it works without `pip install -e .`.

Main dependencies: Gymnasium, NumPy, SciPy, PyTorch, Stable-Baselines3,
PyYAML, Matplotlib, Pillow (GIF output), MuJoCo 3.x.

**MuJoCo live window on macOS:** the MuJoCo viewer window must run under
`mjpython` (installed with MuJoCo), not `python`. GIF output works with any Python.

---

## Quick start

```bash
# Fit the shot model once (and again after any change to the board, puck, band or release noise)
python -m src.slingpuck.shot.model                           # cache/shot_model.npz, about 15 s

# Scripted selectors against the scripted opponent (event-level match)
python -m src.slingpuck.eval.match_eval
python -m src.slingpuck.eval.match_eval --policies greedy_sling greedy_rule --selection greedy random fixed
python -m src.slingpuck.train.train_selector --seeds 0      # PPO selector (configs/train_selector.yaml)
python -m src.slingpuck.eval.match_eval --policies greedy_sling reload_rule tell_rule --tell 0 0.5 1
python -m src.slingpuck.eval.match_eval --no-hand            # ablation: the policy does not see the hand

# MuJoCo match: the SO-101 (black pucks) against a scripted human (white pucks), face to face
mjpython -m src.slingpuck.eval.view_mujoco_match                                     # live 3D window
mjpython -m src.slingpuck.eval.view_mujoco_match --camera opponent                   # the human's view
mjpython -m src.slingpuck.eval.view_mujoco_match --games 0 --speed 0.5               # until the window closes, half speed
python   -m src.slingpuck.eval.view_mujoco_match --save match.gif --max-time 20       # GIF

# S3: scripted arm skills and full-physics matches (no training)
python -m src.slingpuck.eval.skill_eval                       # reach, shots, block, hold, row clear -> results/skills/
python -m src.slingpuck.eval.timing_sweep                     # when does blocking start to matter?
python -m src.slingpuck.eval.mujoco_match_eval --speeds 1 2   # SO-101 skills against the scripted human

# The full scene only (table, board, robot, 10 pucks)
python -m src.slingpuck.so101_slingpuck               # interactive viewer (Ctrl + right drag flicks a puck)
python -m src.slingpuck.so101_slingpuck --no-viewer   # build and check only
python -m src.slingpuck.so101_slingpuck --save        # write assets/mujoco/robotstudio_so101/so101_slingpuck.xml
```

---

## Repository layout

```
configs/
  physics_params_v0.yaml   Measured physics, version 0 (uncalibrated). Calibrated values go in v1, v2, ...
  env.yaml                 Sim timing, opponent, match timings and primitives, scoring, arm skills
  train_selector.yaml      PPO selector: network, PPO settings, seeds, steps
  real_arm_joint_map.yaml  Sim <-> real joint zeros and signs (deploy/calibrate_joints.py)
  real_arm_board.yaml      Real board relative to the real arm (deploy/touch_test.py, vertical_stop.py)
  overhead_camera.yaml     Camera corners and detector settings (sensing/overhead_camera.py calibrate; not yet made)
assets/
  mujoco/robotstudio_so101/  SO-101 MuJoCo model (RobotStudio MJCF, STS3215 actuators)
src/slingpuck/
  config.py                Config loader, params versioning, placeholders, strict mode, run records
  so101_slingpuck.py       The full scene (table, board, bands, robot, 10 pucks): viewer, --save, --no-viewer
  physics/
    band_model.py          Pre-stretched band: V geometry, nonlinear tension, closed hysteresis loop, release
    shot_sim.py            One robot shot: band release, release noise, puck flight in a board-only MuJoCo scene
    mujoco_match.py        MuJoCo match: SO-101 (black pucks) against a scripted human (white pucks), all pucks in the physics
  envs/
    match_env.py           Event-level match (block / sling / hold over frozen primitives)
    wrappers.py            PrivilegedObsWrapper (Dict obs for the asymmetric critic)
  opponent/
    scripted_opponent.py   Human-like opponent: reload, place, pull, hold, release; hand trajectory; tell
  primitives/
    block.py               Block primitive: save probability from the arm's return time and the puck's arrival time
    sling.py               Sling primitive: fixed model, and ShotSelector (puck and shot with the best P(through) / T)
  shot/
    model.py               Shot model: ensemble MLP for P(through | launch x, aim, pull); fit with python -m
  policies/
    asymmetric.py          AsymmetricActorCriticPolicy for SB3 PPO (for the M5 selector)
    match_scripted.py      AlwaysBlock, GreedySling, ReloadRule, TellRule, OracleReload, GreedyRule
  train/
    common.py              Match env factory, run loader, shared PPO runner and CLI, outcome logging
    train_selector.py      PPO selector over block / sling / wait
  eval/
    match_eval.py          Win/loss, puck difference, tell sweep, hand ablation
    stats.py               Wilson and Student-t confidence intervals
    view_mujoco_match.py   3D MuJoCo match viewer (cameras: overview, opponent)
    skill_eval.py          S3: reach map, shot model against the scripted skill, block, hold, row clear
    timing_sweep.py        Selectors over a range of robot timings
    mujoco_match_eval.py   Full-physics matches: scripted arm skills against the scripted human
  deploy/
    joint_map.py           Sim <-> real joint angles (zero and sign per joint), gripper map
    calibrate_joints.py    Measure the joint map (lerobot env)
    fix_ranges.py          Record the range of motion again for some joints
    mirror.py              The sim arm follows the real arm live (check the joint map)
    board_frame.py         Board <-> robot frame, the touch marks, the board fit
    touch_test.py          Touch the marks with the jaw tips: board position, yaw, joint zeros
    vertical_stop.py       Move the jaws straight down to contact: the board surface height
    record_trajectory.py   Record one sim shot as joint angles (--camera: the puck where the camera sees it)
    replay_trajectory.py   Play a recorded shot on the real arm, with limit, speed and stall checks
    export_onnx.py, latency_benchmark.py   ONNX export and latency of a selector (M5)
  sensing/
    board_view.py          Camera image <-> board frame: homography from the 4 inner corners, rectified view, parallax
    puck_detector.py       Hough circles on the rectified view, black/white check, board (x, y) of each puck
    puck_tracker.py        Multi-puck constant-velocity Kalman tracks (position, velocity, latency compensation)
    overhead_camera.py     CLI: calibrate, live, snapshot; locate_pucks() for the arm scripts
assets/vision/             Sample board photo and video (from python-opencv-messaround) for the camera demo
tests/                     pytest suite
docs/TESTS.md              What each test checks, and why
docs/figures/              Figures used in this README
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
| `TODO` | Unknown. Must be measured |

| Quantity | Value | Source |
|---|---|---|
| Outer length × width × height | 0.3658 × 0.2159 × 0.0249 m (14.4 × 8.5 × 0.98 in) | spec |
| Frame (wall) thickness | 0.01524 m (0.6 in) | image |
| Inner play area | 0.330 × 0.203 m | user: measured 2026-11-05 (was 0.3353 × 0.1854 derived) |
| Gate width | 0.034 m | user: measured 3.4 cm |
| Divider thickness | 0.007 m | user: measured 2026-11-05 |
| Puck diameter × thickness | 0.0254 × 0.0102 m (1 × 0.4 in) | user (diameter), image (thickness) |
| Puck mass | 0.027 kg | user. See the note below |
| Band anchor span | 0.203 m (the band goes into the side walls: the inner width) | user: 2026-11-05 |
| Band distance from the end wall | 0.05 m | user: measured 2026-11-05 |

**Note on the puck mass:** 27 g in a 1 × 0.4 in puck gives a density of
5.2 g/cm³. Wood is 0.5 to 0.75 g/cm³. 27 g / 6 = 4.5 g, which is the expected
mass of one wood puck. Weigh one puck alone to confirm. The mass changes the
launch speed in the MuJoCo match.

**Note on the gate:** a straight shot goes through the gate when its launch x is
within ±(gate width / 2 − puck radius) = ±4.3 mm of the center
(`opponent.scripted_opponent.threat_half_width`). The opponent aims its threats
inside this band and its misses at least 6 mm off center.

---

## Configs and parameter versioning

### Merge order

`load_config(params_version)` merges two files. The later file overrides the
earlier one, key by key:

1. `physics_params_v<N>.yaml`
2. `env.yaml`

```python
from src.slingpuck.config import load_config, save_run_config
cfg = load_config("latest")          # or "v0", 0, "v3"
cfg["meta"]                          # params_version, params_file, calibrated, placeholders, git_hash, git_dirty
```

### Versioned physics parameters

- Put calibrated values in a new file `physics_params_v<N+1>.yaml`.
- `"latest"` loads the highest version. The loader checks that the `version:`
  field in the file matches the file name.
- Each training run records the params version that it used.

### Placeholders and strict mode

- Every value that is not measured has a `TODO` comment. In
  `physics_params_v0.yaml`, each one is also listed under `placeholders:` (15 now).
- `load_config(..., strict=True)` raises `ConfigError` while any placeholder or
  null value remains. Use strict mode for every result that you report against
  real data. Training accepts `--strict`.
- The match timings, the primitive values and the scoring rules in `env.yaml`
  are also placeholders (`TODO`), but strict mode does not check them.

---

## The match

### Design: an event-level match over frozen primitives

The selector needs millions of high-level decisions, so `envs/match_env.py` does
not simulate the puck and the arm at 1 kHz. It runs at 10 ms steps and uses:

- **Opponent shots:** straight flights with Coulomb friction (analytic arrival
  time at the gate). A pre-stretched band launches almost straight
  (`tests/test_band.py`), so the launch x decides whether a shot is a threat.
- **Block primitive** (`primitives/block.py`): the robot guards the gate with its
  closed gripper. A threat is saved if the gripper covers the gate before the
  puck gets there:

  ```
  time_to_cover = (time the puck reaches the gate) - (time the arm got back to the gate)
  p_save = block_save_prob * Phi((time_to_cover - cover_time_s) / cover_time_std_s)
  ```

  Phi is the standard normal CDF. An arm that is away from the gate saves
  nothing. An arm that stayed at the gate saves with `block_save_prob`.
- **Sling primitive** (`primitives/sling.py`, `shot.selection` in `env.yaml`):
  - `greedy` (default): the shot model (`shot/model.py`) gives P(through) for each
    candidate shot (launch x, aim, pull), and the time of a shot depends on where
    the puck lies: `T = pick + (|arm → puck| + |puck → seat|) / arm_speed + seat_release + pull / draw_speed`.
    The robot takes the puck and shot with the highest (P + UCB bonus) / T. Pucks
    jammed against a side wall go last.
  - `random`: a random puck with the most likely shot (the "random puck order" baseline).
  - `fixed`: the old model, fixed timings and `sling_success_prob`.
- **Shot model** (`shot/model.py`): 5 small MLPs, each on a bootstrap sample of
  4000 simulated shots (`physics/shot_sim.py`: band release, release noise, puck flight
  in MuJoCo). Held-out accuracy 95.4% (fit of 2026-11-09; 97.9% before the S3 friction change). The file stores a fingerprint of the board,
  puck, band and noise values, and refuses to load under other values.

All primitive values are placeholders in `env.yaml` (`match:`) until they are
measured on the robot. Speed: about 70,000 decisions per second (one process).

### Opponent (`opponent/scripted_opponent.py`)

- Cycle: **reload** (hand fetches a puck) → **place** (to the band) → **pull** (pull ~
  shot speed) → **hold** (random pause) → **release**, about 1.2 s per shot.
- The hand covers the loaded puck from the overhead camera. The camera sees the
  hand (position, velocity, 5 mm noise).
- **`tell_strength`:** during place, pull and hold, the hand x is
  `tell · x_launch + (1 − tell) · x_decoy`, with the decoy from the same
  distribution. 0 = the hand says nothing about the threat. 1 = the hand shows
  the launch x exactly (tested: correlation ~0 at 0, exact at 1).
- 50% of shots are threats (placeholder). The generator is seeded.

### Actions, arm states, observation, reward

| Action | Effect |
|---|---|
| 0 block | From the band: return to the gate (0.6 s). At the gate: stay; the block primitive defends |
| 1 sling | From the gate: move to the band (0.6 s), then fetch and sling the chosen puck (time from the shot choice). From the band: fetch and sling. Ignored when busy or out of pucks |
| 2 wait | Keep the arm where it is (at the band: ready for another sling, gate open). Called `hold` before; renamed because the adaptation plan uses HOLD for "press the puck on the band" |

Arm states: gate → to_band → slinging → at_band → to_gate → gate. The gate is
open from the sling command until the arm is back (the explicit recovery time).
After the return, the gripper needs `cover_time_s` to cover the gate.

- **Observation (22 values):** tracked incoming opponent puck (only after the
  camera latency), opponent hand (x, y, vx, vy), arm state (one-hot, time left,
  gate covered), puck counts, own puck in flight, the robot's best shot
  (P(through) and its time T), time left.
  `match.observe_hand: false` sets the hand values to zero (the M5 ablation).
- **Privileged (critic only):** opponent phase, time to release, next shot
  (threat, x, speed), incoming threat and its time to the gate.
- **Reward (placeholder, `scoring`, TODO: real rules):** +1 per puck sent, −1 per
  puck received. The first side with no pucks wins (+5 / −5). 5 pucks per side,
  60 s limit.

### Baselines (`policies/match_scripted.py`) and results

```bash
python -m src.slingpuck.eval.match_eval [--policies ...] [--matches 400] [--tell 0 0.5 1] [--no-hand]
```

| Policy | Rule |
|---|---|
| `always_block` | Never slings |
| `greedy_sling` | Slings whenever possible, never returns |
| `reload_rule` | **The plan's baseline:** sling only when the opponent reloads (hand away from its band line); otherwise go back to the gate |
| `tell_rule` | Like `reload_rule`, but also slings while the opponent aims if the hand x shows a likely miss |
| `oracle_reload` | `reload_rule` with the opponent's true phase (privileged) |
| `greedy_rule` | **The adaptation plan's layer 3:** sling when the best shot's P(through) is more than the threats that blocking would save in the same time T_shot (race of rates) |

The hand-based reload detector is reliable where it matters: it marks the
opponent as aiming in more than 98% of pull/hold frames (tested).

**S2 results: shot selection** (400 matches per row, tell 0.8, placeholder
times; `results/match/s2_selection/`). Net = pucks sent − received per minute:

| Policy | Greedy shot choice: win [95% CI], net/min | Random puck order | Old fixed sling |
|---|---|---|---|
| `greedy_sling` | 70% [66, 75], +7.3 | 58% [54, 63], +5.7 | 30% [25, 34], +1.1 |
| `greedy_rule` | 59% [54, 64], +6.6 | 52% [47, 56], +4.6 | 0% (never slings) |
| `tell_rule` | 34% [29, 39], +1.1 | 24% [20, 29], −0.9 | 11% [8, 14], −4.6 |
| `reload_rule` | 8% [6, 11], −4.8 | 6% [4, 9], −5.6 | 4% [3, 6], −7.4 |
| `always_block` | 0%, 0 | 0%, 0 | 0%, 0 |

- **The greedy shot choice (layer 2) helps every policy that slings.** Near pucks
  and clean shots first give about 33 shots per minute.
- **The greedy rule (layer 3) is below "always sling" here.** Blocking pays only
  when the arm is at the gate in time, and with these times it saves 1–2% of the
  threats. With the old fixed sling (0.95 in 2.5 s from the gate), the rule
  calculates that blocking saves more than a shot clears, so it never slings: the
  race-of-rates condition says that this robot cannot win by slinging.
- **The PPO selector learns to always sling.** One seed, 1 million steps
  (`results/match/s2_ppo/`): 70% [66, 75] wins and +7.3 net/min, the same matches
  as `greedy_sling`. With these placeholder times, slinging is the best strategy,
  and the learned policy finds it without help. The comparison is worth repeating
  with the measured times, where blocking can pay. This run is from 2026-11-03, before the S3 changes
  (jaw contacts, friction against the velocity, shot model refit). It was not repeated after them.

The older results below use the old fixed sling (`shot.selection: fixed`).

![Match baselines](docs/figures/match_baselines.png)

| Policy (400 matches, tell 0.8, fixed sling) | Win [95% CI] | Loss | Sent − received [95% CI] | Conceded while away |
|---|---|---|---|---|
| `always_block` | 0% [0, 1] | 0% | +0.0 | 0 / match |
| `greedy_sling` | 34% [29, 38] | 18% | +1.1 [+0.8, +1.5] | 20 / match |
| `tell_rule` | 12% [9, 16] | 44% | −1.7 [−2.1, −1.4] | 19 / match |
| `reload_rule` | 4% [3, 6] | 64% | −3.3 [−3.6, −3.0] | 17 / match |
| `oracle_reload` | 4% [2, 6] | 61% | −3.3 [−3.6, −3.0] | 17 / match |

![Tell sweep](docs/figures/match_tell_sweep.png)

### Findings (with placeholder timings and rules)

1. **The plan's baseline does badly.** A sling away from the gate takes about 3.1 s
   (out, fetch, sling, back). The opponent's reload is about 0.6 s. Thus "sling
   only while the opponent reloads" always leaves the gate open for the next
   2–3 shots. Even with the true phase (oracle), it loses most matches.
2. **The tell helps, but little.** `tell_rule` beats `reload_rule`, but its gain
   does not grow clearly with the tell. To know that the *current* shot is a
   miss protects only one of the ~2.5 shots that come during a sling.
3. **Under the "first empty side wins" rule, the match is a throughput race.**
   With the robot timings scaled ×0.5 (sling-and-return cycle 1.6 s), every
   strategy that slings wins 100%. `always_block` never slings, so it still ends
   every match on time (0% wins). At the default 3.1 s, greedy slinging beats every
   defensive rule. Strategy is important only in a narrow band, where the
   robot's sling rate is close to the opponent's threat rate.

Thus the answer to the research question depends on values that are not
measured yet: **the real scoring rules**, the robot's fetch, sling, move and
cover times, and the human's cycle time and accuracy. M5 (selector training,
ablations, vulnerability window) must run after those values are known, or
over a sweep of them.

---

## MuJoCo match

`physics/mujoco_match.py` plays a full match in MuJoCo physics. The scene comes
from `so101_slingpuck.build`: the table, the board, both bands, 5 black robot
pucks and 5 white human pucks, and a simple mocap hand. All 10 pucks slide with
Coulomb friction and bounce off the walls, the divider and each other.

- **Scene frame:** the origin is at the robot base, x forward, y left, z up.
  The robot and the board stand on the same table. The board lies lengthwise and
  centered, with the robot behind the agent end wall. The human sits at the far
  end (camera `opponent`). One light from straight above.
- **Robot (scripted skills, S3):** the closed gripper stands in the gate. When the
  game rule (`skills.game_rule`: `reload` or `greedy_sling`) says sling, the arm
  takes the puck and shot that the shot model chooses. It rises, travels, comes
  down, and grips the puck with jaw friction. If a neighbour leaves no room for
  the open jaws, it first pushes the puck out of the row with the jaw tips. It
  seats the puck on the band, pulls back (aim and pull from the shot), releases,
  and goes back to the gate. A fetch that has not grasped yet stops when the
  human starts to pull.
- **S3 results in sim** (all times are placeholders): at 2× arm speed, 96 of 100
  shots go through with a 2.37 s cycle (1×: 95 of 100, 3.49 s). In 30 full-physics
  matches per point, greedy sling wins 6/30 at 1× and 18/30 at 2×; the reload rule
  wins 4/30 and 14/30. 11–16 matches per point end on time (cause not checked).
  Details: [docs/PLAN.md](docs/PLAN.md), notes on S3.
- **Human (scripted, timings and aim from `env.yaml` `opponent`):** reload, place
  on its band, pull back, hold, release.
- **Band:** a held puck behind its band line gets the band force (`BandModel`).
  A stiff connect constraint passes the force to the arm or the hand. On release,
  the unloading band pushes the puck to its band line, and then the puck keeps
  `energy_transfer` of its energy.
- **Contacts:** the gripper collision boxes touch the puck colliders. Puck
  collisions use a sphere of the puck radius at mid-height (MuJoCo's
  cylinder-box contact gives a wrong normal at the gate corners). Contact
  damping comes from `board.restitution`.

The MuJoCo match does not feed the event-level match. It is a check that the
scripted plan is physically possible with the SO-101 and the real board sizes.

---

## Real arm

The real SO-101 plays shots that the sim records: the sim plans the shot (IK, skills), and the real arm replays
the joint angles open loop. Two Python envs: the **project env** (MuJoCo) records, and the **`lerobot` env**
(LeRobot, Python 3.12) talks to the motors. In the `lerobot` env, use `python3`, not `python`. Port and id below
are the lab values.

| Step | Command | Writes |
|---|---|---|
| 1. LeRobot calibration | `lerobot-calibrate` (LeRobot) | LeRobot calibration file |
| 2. Joint map: sim angles <-> real angles | `python3 -m src.slingpuck.deploy.calibrate_joints --port /dev/tty.usbmodem5A7A0563651 --id group5` | `configs/real_arm_joint_map.yaml` |
| 2a. Check the joint map (sim arm follows the real arm) | `mjpython -m src.slingpuck.deploy.mirror --port ... --id group5` | — |
| 3. Board position: touch 6 marks with the jaw tips | `python3 -m src.slingpuck.deploy.touch_test --port ... --id group5` | `configs/real_arm_board.yaml`, new joint zeros |
| 3a. Board surface height: jaws straight down to contact | `python3 -m src.slingpuck.deploy.vertical_stop --port ...` | Surface height in the board file |
| 4. Record a sim shot (project env) | `python3 -m src.slingpuck.deploy.record_trajectory [--hover-mm 30] [--camera 0]` | `results/real/traj_<time>/` |
| 5. Check, then play it on the arm | `python3 -m src.slingpuck.deploy.replay_trajectory results/real/traj_<time> --port ... [--execute --speed 0.25]` | `replay_<time>.npz` (commanded and measured angles) |

Without `--execute`, the replay only checks the trajectory against the joint limits and a speed limit. With it,
you hold the arm near the first frame, the torque comes on there, and a stall guard stops the arm if a joint stays
behind its command.

**Order of the first replays** (from `replay_trajectory.py`): hover 30 mm at 0.25× speed, then 0.5×, then on
the board at 0.25× with `--until grasp` and a puck on the mark, then the full shot.

**Results so far (2026-11-05):**

| Item | Value |
|---|---|
| Board fit (6 marks, pan, lift and elbow zeros fitted) | RMS 7.7 mm, board yaw −0.03° |
| Arm pan axis in the board frame | (0.0025, −0.2264) m (`robot.base_xy_m` in `configs/real_arm_board.yaml`) |
| Board surface height | 19.3 mm in the robot frame, after the vertical stop moved it up 7.5 mm (median of 4 contacts) |
| Replays | 30 mm hover trajectory (puck on `mid_center`, sim outcome "through"), 0.25× speed, 3 runs of 13 s |
| Shots on the board | None yet |

---

## Overhead camera: where the pucks are

Before this module, the arm script (`deploy/record_trajectory.py`) knew a puck position only from a touch mark
or from `--puck X Y`. The package `src/slingpuck/sensing/` gives the arm the positions that the overhead camera
sees. It comes from the group prototype
[python-opencv-messaround](https://github.com/CS6341-Robotics-Group5/python-opencv-messaround) (median blur and
the Hough circle transform), with these changes:

| Prototype | Here |
|---|---|
| Circles in image pixels | Board (x, y) in meters, the same frame as the sim and `configs/real_arm_board.yaml` |
| `minRadius` / `maxRadius` tuned by hand per image | The search runs on a rectified top-down view, where the puck radius in pixels comes from `puck.radius_m` |
| Every circle is a puck | Colour check (black or white against the board median) and a fill check: printed graphics are rejected |
| One frame | Multi-puck Kalman tracks: velocity of a moving puck, prediction over the camera latency |

**Steps** (OpenCV: `pip install -e ".[vision]"`):

```bash
# 1. Once per camera position: click the 4 inner corners of the play area (the window says the order:
#    robot end left, robot end right, opponent end right, opponent end left). Writes configs/overhead_camera.yaml
python -m src.slingpuck.sensing.overhead_camera calibrate --source 0
#    Optional parallax correction: --camera-height-m 0.6 --camera-xy 0 0   (lens above the surface, board xy under it)

# 2. Check: the board view with the board (x, y) in mm on each puck, and arrows on moving pucks
python -m src.slingpuck.sensing.overhead_camera live --source 0 --print

# 3. The arm: record a sim shot of a puck where the camera sees it, then replay it (deploy/replay_trajectory.py)
python -m src.slingpuck.deploy.record_trajectory --camera 0                 # greedy choice of the puck
python -m src.slingpuck.deploy.record_trajectory --camera 0 --pick nearest  # or --pick 2 (index in the printed list)
python -m src.slingpuck.deploy.record_trajectory --camera 0 --auto-shot     # also the greedy shot (needs the shot model)

# Demo without a camera (sample media; the photo's board is not the measured board, so a few mm are off)
python -m src.slingpuck.sensing.overhead_camera calibrate --source assets/vision/sample_board.mp4 \
    --corners 503 228 503 858 1422 858 1422 228 --calib /tmp/cam_demo.yaml
python -m src.slingpuck.sensing.overhead_camera live --source assets/vision/sample_board.mp4 --calib /tmp/cam_demo.yaml
```

With `--camera`, every puck on the robot half goes into the sim at its seen position, so the sim script sees the
neighbours (and pushes a puck out of a row first, as on the real board). The trajectory's `meta.yaml` records the
camera positions and the choice. `--pick greedy` uses the shot model (`ShotSelector`, as in the match); if the
model does not load (a fingerprint mismatch after a config change), the script uses the nearest puck and says so.

**Measured on the sample media** (1920 × 1080 video, 2.9 px/mm): 20 of 20 pucks in 149 of 150 frames, no false
puck, 8 ms per frame (rectify, detect, track) on a laptop CPU. The first frame has no confirmed tracks (a track
needs 2 detections).

**Limits and open items:**

- The detector expects dark and white pucks on a light board, seen from above. Check `live` under the real light.
  The settings are in `configs/overhead_camera.yaml` under `detect:` (see `DEFAULTS` in `puck_detector.py`).
- No lens distortion correction. A wide-angle webcam can bend the board edges. If the corners of the rectified view
  fit but the middle does not, add a camera calibration (`cv2.calibrateCamera`) before the homography.
- A hand above a puck hides it. The snapshot needs each puck in 60% of the frames, and still.
- The live tracks are not yet an input of a closed-loop controller on the arm. The real arm still plays
  recorded trajectories (record, then replay). A live loop is part of S6.
- The camera-to-board fit and the arm-to-board fit (`touch_test.py`) are independent. A puck that the camera sees
  at a mark must also be at that mark for the arm. Check this once: put a puck on `mid_center` and compare.

---

## Where RL is used

**Short answer:** on this branch, reinforcement learning (PPO) is used in **one place only**: the high-level
**selector** (layer 3) that chooses BLOCK, SLING or WAIT during a match. The arm skills, the shot choice, the
camera and the real-arm control do not use RL.

| Part of the system | Method | RL? | Code |
|---|---|---|---|
| **Game decision: BLOCK, SLING or WAIT** | **PPO** with an asymmetric actor-critic. The scripted rules (`greedy_rule`, `reload_rule`, ...) are the baselines | **Yes** | `train/train_selector.py`, `train/common.py`, `policies/asymmetric.py`, `envs/wrappers.py`, `envs/match_env.py`, `configs/train_selector.yaml` |
| Shot choice: which puck, which shot | Greedy: highest (P(through) + UCB bonus) / T | No | `primitives/sling.py` |
| Shot model P(through \| launch x, aim, pull) | Supervised learning: an ensemble of 5 small MLPs on 4000 labelled simulated shots | No (supervised) | `shot/model.py` |
| Arm skills: pick, seat, draw, release, block, hold | Scripted moves with IK in MuJoCo | No | `physics/mujoco_match.py` |
| Opponent | Scripted (reload, place, pull, hold, release) | No | `opponent/scripted_opponent.py` |
| Puck positions | OpenCV (Hough circles) and Kalman filters | No | `sensing/` |
| Real arm | Open-loop replay of recorded sim joint angles | No | `deploy/` |

### The RL problem

| Item | Value |
|---|---|
| Environment | `MatchEnv` (Gymnasium): the event-level match. Each step is one decision of the selector. About 70,000 steps per second |
| Observation (actor) | 22 values that the real robot can sense: the tracked incoming puck (after the camera latency), the opponent's hand, the arm state, puck counts, the best shot's P(through) and time, time left |
| Privileged observation (critic only) | Sim truth: the opponent's phase, time to its release, its next shot (threat, x, speed), the incoming threat and its time to the gate |
| Action | Discrete(3): 0 block, 1 sling, 2 wait |
| Reward (placeholder scoring) | +1 per puck sent, −1 per puck received, +5 / −5 for a win / loss. 5 pucks per side, 60 s limit |
| Algorithm | Stable-Baselines3 PPO. The actor reads only the 22 deployable values; the critic also reads the privileged values. Thus the deployed actor needs only what the camera and the arm give |
| Settings (`configs/train_selector.yaml`) | 8 parallel envs, net [128, 128], n_steps 1024, batch 2048, 10 epochs, γ 0.99, λ 0.95, lr 3e-4, entropy 0.01, clip 0.2. Configured: 3 seeds × 3M steps |

**Training done so far:** one seed, 1M steps (2026-11-03, `logs/selector/selector_v0_s0_20261103_032545/`,
results in `results/match/s2_ppo/`). The policy won 70% [66, 75] of 400 matches with +7.3 net pucks per minute:
the same matches as `greedy_sling`. **It learned "always sling".** With the placeholder times, slinging is the
best strategy, so this is the correct answer, but it does not yet show a benefit of RL over the scripted rules.
The run is from before the S3 changes and was not repeated.

### Why RL is used only for the selector

1. **The plan makes RL a challenger, not the default.** The adaptation plan
   ([docs/papers/slingpuck-adaptation-plan.md](docs/papers/slingpuck-adaptation-plan.md)) treats the match as a
   race of rates. With a known P(through) and time T for each action, the greedy one-step rule is a strong
   baseline. The PPO selector replaces the greedy rule only if it wins more matches in the event sim.
2. **RL for low-level control did not transfer before.** On `main`, Phase 1 (goalkeeper) trained PPO for the
   paddle. In 2D physics, PPO saved 85.0% of the 1.90–2.27 m/s shots against 78.0% for "go to center". But PPO
   had only learned to command the far end of the pan range, a scripted "overshoot" did the same (79.0% vs
   79.0%), and the gain did not transfer to the MuJoCo servo model. Thus the skills on this branch are scripted,
   and learned skills come in only if a script fails on the real arm.
3. **One shot is not a sequential decision.** P(through) has direct labels from simulated shots, so supervised
   learning fits it. RL is not necessary there.

**Where RL can still help:** when blocking pays. The timing sweep (`eval/timing_sweep.py`) shows that the
strategy matters only in a narrow band, where the robot's sling rate is close to the opponent's threat rate.
With the measured robot and human times, the selector can face a real trade-off between BLOCK and SLING, and the
privileged critic can learn from the opponent's hidden state.

### RL on the other branches

| Branch | RL use |
|---|---|
| `main` (up to `0221bb5`) | Phase 1 goalkeeper: asymmetric PPO, action = pan target (`logs/goalkeeper/`). Phase 2 targeted shot: PPO, action = strike angle and pull-back distance (`logs/sling/`) |
| `generalized` | One goal-conditioned policy for any manipulation task, with a paired-point PPO teacher and a failure-guided curriculum |

---

## Tests

```bash
pytest -q
```

[docs/TESTS.md](docs/TESTS.md) describes each test. The table below is a summary.

| File | What it checks |
|---|---|
| `test_config.py` | Version resolution, `latest`, version mismatch, strict mode, placeholders, run records |
| `test_stats.py` | Wilson and t intervals against known values |
| `test_band.py` | **Hysteresis loop closes** and loses the set fraction (3 exponents × 3 losses), monotonic tension, pre-tension, force direction, release energy balance, energy transfer, almost-straight launch, tiny pull does not launch |
| `test_match.py` | Opponent: tell 1 shows the launch x exactly, tell 0 hides it, threats aim inside the gate (width from the geometry), phases in order, smooth hand. Match: `check_env`, **pucks conserved**, rewards add up, a covered gate saves every threat, greedy concedes every threat while away, sling timing, seeded repeat, hand ablation, puck hidden before the camera latency, reload detection, block primitive values, a late return concedes, bad gate geometry rejected |
| `test_asymmetric_policy.py` | On the match env: the actor ignores privileged input, the critic uses it, save and load |
| `test_slingpuck_scene.py` | Board in front of the robot, turning axis matches the config, board and robot on the same table, flicked puck slows with Coulomb friction, wall bounce keeps the configured restitution, opponent camera faces the robot |
| `test_match_mujoco.py` | 5 black and 5 white pucks, both players shoot, no blow-up (speeds stay below 6 m/s, all pucks on the board), guard pose stands in the gate |
| `test_overhead_camera.py` | Homography against OpenCV, rectified orientation, every puck found within 2 mm with its colour in two synthetic camera views (no false puck from printed lines, a ring or a "glove"), parallax, calibration file, tracker speed and ids, snapshot rejects a moving puck, puck choice for the arm |
| `test_skills.py` | Scripted arm skills in the MuJoCo match: friction grip, push-out from a row, block, hold, timing sweep |
| `test_joint_map.py` | Sim <-> real joint map, gripper map, calibration script, mirror (no hardware) |
| `test_real_arm.py` | Board fit, joint-zero fit, range check, trajectory record and replay checks, vertical stop (no hardware) |
| `test_shot.py` | Shot simulator outcomes, shot model accuracy and fingerprint, greedy shot choice (near pucks first, jammed last), shot time model, greedy match bookkeeping, greedy rule decisions |

---

## Reproducibility

- The match env and the opponent use the Gymnasium `np_random` generator, so a
  match repeats exactly from its seed (tested).
- Training seeds PyTorch, NumPy and the envs (`set_random_seed`, PPO `seed`,
  `make_vec_env(seed=...)`).
- Each run saves the full merged config, the physics params version, the git
  hash and dirty flag, and the seed. Commit your changes before a run that you
  will report, so that `git_dirty` is false.
- The eval saves its own `eval_meta.yaml` with the same information.

---

## Known problems and open items

- **The game balance depends on placeholder values.** 15 physics values, all arm and human timings, and the
  scoring rules are not measured (see the findings and the table below). Thus no strategy result is final.
- **The shot model is too confident.** It gives about 0.9 on shots where the scripted sim arm gets 66–82 of 100
  through (up to 91). Next: fit it on arm shots (`eval/skill_eval.py`), then on real shots (S4).
- **Full-physics matches time out.** 11–16 of 30 matches per point in `mujoco_match_eval.py`. Cause not checked.
- **Event-level fidelity:** in `MatchEnv`, block and sling outcomes come from primitive models, not physics. One
  opponent puck at a time is resolved at the gate. Puck-puck collisions in the gate are not modeled. The opponent
  never blocks, and its accuracy does not depend on the match state.
- **PPO selector:** one seed only, from before S3, and it learned "always sling" (see
  [Where RL is used](#where-rl-is-used)). `deploy/export_onnx.py` exports an `MlpPolicy` only, not the asymmetric
  policy.
- **Real arm:** no shot on the board yet. The arm plays recorded trajectories open loop. It does not react to the
  camera during a move.
- **Camera:** not tested on the real camera. No lens distortion correction. The camera-to-board and arm-to-board
  fits are independent and not yet compared.
- **Version control:** S3, the real-arm code and the camera code are not committed.

---

## Values to measure on the real hardware

| Value | Config path | Why it is important |
|---|---|---|
| **Game scoring rules** | `scoring` | Reward and win rule; decides whether the match is a throughput race |
| **Robot fetch, sling and move times** | `match.fetch_time_s`, `match.sling_core_s`, `match.move_*` | Length of the vulnerability window |
| **Gate cover time after a return, and its spread; block save rate** | `match.cover_time_s`, `match.cover_time_std_s`, `match.block_save_prob` | Block primitive |
| Robot sling success rate | `match.sling_success_prob` | Sling primitive |
| **Human opponent timing and accuracy** (video of human play) | `opponent.*` | Opponent cycle, threat rate, tell |
| Real shot speeds | `opponent.shot_speed_range_m_s` | Puck arrival time at the gate |
| Camera latency | `camera.latency_s` | When the robot sees an incoming puck |
| Overhead camera height and position (optional) | `configs/overhead_camera.yaml` `camera:` | Parallax: the camera sees the top face of the puck |
| Wall height (divider thickness: measured, 7 mm) | `board.wall_height_m` | MuJoCo board |
| Puck mass (one puck alone), diameter in mm, thickness | `puck.mass_kg`, `puck.radius_m`, `puck.thickness_m` | Launch speed, all contacts, threat width |
| Sliding friction, wall restitution | `puck.friction_kinetic`, `board.restitution` | Puck flight and bounces |
| Natural (unstretched) band length, band stiffness and hysteresis, energy transfer | `band.*` | Launch speed in the MuJoCo match |
| Arm base position: **measured** by the touch test, (0.0025, −0.2264) m in `configs/real_arm_board.yaml`. Copy it into `physics_params_v1.yaml` | `robot.base_xy_m` | Scene layout and reach |

Measured on 2026-11-05 and already in `physics_params_v0.yaml`: inner play length and width, divider thickness,
band span, band distance from the end wall.

---

## Future work

In priority order:

1. **Commit the work** (S3, real arm, camera) in separate commits.
2. **Set up the real camera:** mount it above the board, run `overhead_camera calibrate`, check `live` under the
   room light, and put a puck on `mid_center` to compare the camera position with the arm position.
3. **Finish the real-arm steps:** hover replay at 0.5×, then on the board at 0.25× with `--until grasp`, then the
   full shot, then the same with `--camera`.
4. **Measure the placeholder values:** arm fetch, sling and move times, puck mass, friction, band values, camera
   latency, and the scoring rules. Write `physics_params_v1.yaml`.
5. **S4, physics calibration:** record 40–60 real shots with camera tracks (a 120 fps camera is necessary), fit
   band stiffness, friction and release noise, and fit the shot model again, on real arm shots.
6. **Find the cause of the match timeouts** in `mujoco_match_eval.py`.
7. **Train the PPO selector again (M5)** with the measured times, 3 seeds, and compare it with the greedy rule and
   the baselines on paired seeds. Export the actor of the asymmetric policy to ONNX.
8. **S5, opponents:** record video of each team member and measure their shot timing and accuracy with the
   tracker.
9. **S6, deploy:** a live closed loop (camera, then selector, then arm), the opponent's hand from the camera, and
   the selector's speed on the Jetson Orin Nano.
10. **Lens distortion correction** if the real webcam needs it.
