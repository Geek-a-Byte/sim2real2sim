# Tests

This document describes each test in `tests/`: what it checks, and why the
check is important.

Run all tests from the project root folder:

```bash
pytest -q                          # 49 tests, about 10 s
pytest -q tests/test_match.py      # one file
pytest -q -k hysteresis            # tests whose name contains "hysteresis"
```

The MuJoCo tests (`test_slingpuck_scene.py`, `test_match_mujoco.py`) need the
`mujoco` package. If it is not installed, pytest skips these tests.

## Contents

| File | Tests | Area |
|---|---|---|
| [`conftest.py`](#conftestpy) | (fixture) | Shared config |
| [`test_config.py`](#test_configpy) | 6 | Config loading and param versions |
| [`test_stats.py`](#test_statspy) | 2 | Confidence intervals |
| [`test_band.py`](#test_bandpy) | 16 | Rubber band physics |
| [`test_match.py`](#test_matchpy) | 15 | Opponent, match env, block primitive |
| [`test_asymmetric_policy.py`](#test_asymmetric_policypy) | 1 | Asymmetric actor-critic policy |
| [`test_slingpuck_scene.py`](#test_slingpuck_scenepy) | 6 | MuJoCo scene |
| [`test_match_mujoco.py`](#test_match_mujocopy) | 3 | MuJoCo match |

---

## `conftest.py`

| Fixture | What it gives |
|---|---|
| `cfg` | The merged config `load_config("v0")`. Scope: one per test session. Most tests use it |

The tests use version `v0` on purpose. A new `physics_params_v1.yaml` does not
change the test results.

---

## `test_config.py`

Code under test: `src/slingpuck/config.py`.

The local fixture `cfg_dir` copies `env.yaml` and `physics_params_v0.yaml` to
a temporary folder. Thus the tests can add config files without a change to
`configs/`.

| Test | What it checks | Why |
|---|---|---|
| `test_v0_loads_and_lists_placeholders` | v0 loads. `meta` shows `params_version: v0` and `calibrated: False`. `board.divider_thickness_m` is a placeholder. `board.gate_width_m` is not a placeholder (the user measured it). `env.yaml` is merged in (`pucks_per_side == 5`) | Each run must record which values are real and which are guesses |
| `test_strict_rejects_placeholders` | `load_config("v0", strict=True)` raises `ConfigError`, and the message names a placeholder | Strict mode must stop a run that reports real results with guessed values |
| `test_latest_loads_the_newest_version` | Writes a calibrated `v2` file. `"latest"` loads `v2`, with its new friction value and no placeholders. Strict mode accepts it | Calibrated values go in new version files. `"latest"` must find the newest one |
| `test_version_mismatch_rejected` | A file `physics_params_v5.yaml` that contains `version: 0` raises `ConfigError` ("expected 5") | Prevents a copied file with a wrong version number |
| `test_missing_version_rejected` | `load_config("v999")` raises `ConfigError` | A wrong version name must fail clearly |
| `test_save_run_config_records_version_seed_and_git` | `save_run_config` writes the params version, the seed and the git hash | Each training or eval run must be traceable and repeatable |

---

## `test_stats.py`

Code under test: `src/slingpuck/eval/stats.py`.

| Test | What it checks | Why |
|---|---|---|
| `test_wilson_known_values` | Wilson 95% interval for 50/100 is [0.4038, 0.5962]. For 60/60, the upper limit is 1.0 and the lower limit is 0.9398 | The win rates in the README use this interval. The values come from a reference table |
| `test_t_ci` | Student-t 95% interval for [0.8, 0.9, 1.0]: the mean is 0.9, and the half-width is 4.303 × 0.1 / √3 | The puck difference in the README uses this interval |

---

## `test_band.py`

Code under test: `src/slingpuck/physics/band_model.py`.

The helper `band(...)` makes a `BandModel` with a 0.1854 m anchor span and a
0.15 m natural length (a pre-stretched band), with optional changes to the
exponent, loss, energy transfer and stiffness.

| Test | What it checks | Why |
|---|---|---|
| `test_hysteresis_loop_closes_and_loses_the_set_fraction` (9 cases: exponent 1.0, 1.5, 2.0 × loss 0.0, 0.15, 0.4) | The unloading curve meets the loading curve at both ends (closed loop). Unloading tension is never above loading tension. The loading work equals `work_load` (numerical integral). The lost work fraction equals `hysteresis_loss_factor` | An open loop or a wrong loss gives wrong launch speeds. The test covers linear and stiffening bands |
| `test_tension_monotonic_and_pretension` | Tension never decreases with stretch. The rest stretch is `anchor span − natural length`. The tension at rest is above zero | The band is pre-stretched. The pre-tension gives most of the launch energy |
| `test_force_points_forward_and_is_zero_on_the_band_line` | No force on the band line. A straight pull back gives a force forward (+y) with no sideways part | Checks the V geometry and the force direction |
| `test_release_energy_balance` | With no friction and full energy transfer, the band work and the puck kinetic energy both equal (1 − loss) × loading work. The puck leaves at the band line | Energy conservation check for the release simulation |
| `test_energy_transfer_scales_exit_speed` | Energy transfer 0.5 gives √0.5 times the speed of transfer 1.0 | `energy_transfer` scales energy, not speed |
| `test_angled_pull_launches_almost_straight` | A straight pull launches at 0°. A pull at 30° off center launches less than 3° off straight | `MatchEnv` assumes straight shots. This test supports that assumption |
| `test_config_band` | `BandModel.from_config` puts the band line at the configured distance from the end wall. `energy_transfer = 0` raises `ValueError` | The config and the model must agree. Bad values must fail |
| `test_tiny_pull_does_not_launch` | A 0.2 mm pull gives less force than sliding friction: no launch and zero speed. A 20 mm pull launches | A puck that cannot beat friction must stay behind the band |

---

## `test_match.py`

Code under test: `src/slingpuck/opponent/scripted_opponent.py`,
`src/slingpuck/envs/match_env.py`, `src/slingpuck/primitives/block.py`,
`src/slingpuck/policies/match_scripted.py`.

Helpers:

- `env_for(cfg, **match)`: a `MatchEnv` in which a covered gate saves every
  threat (`block_save_prob = 1`, `cover_time_std_s = 0`). Keyword arguments
  change more `match` values.
- `play(env, policy, seed)`: plays one full match. Returns the last `info` and
  the total reward.
- `cycles(cfg, tell)`: runs 300 opponent shots. For each shot, it records the
  hand x at the middle of the hold, the launch x, and the threat flag.

### Opponent

| Test | What it checks | Why |
|---|---|---|
| `test_tell_one_shows_launch_x_and_zero_hides_it` | At tell 1, the hand x equals the launch x. At tell 0, the correlation is below 0.15 | `tell_strength` is an experiment variable. Its two ends must mean what the README says |
| `test_threat_shots_aim_inside_the_gate` | `threat_half_width` = gate width / 2 − puck radius. Every threat launches inside this band. Every miss launches outside it. The threat fraction is near `threat_prob` | A threat must be able to go through the gate. A miss must not |
| `test_opponent_phases_in_order_and_hand_continuous` | The phases come in the order reload → place → pull → hold. The hand never jumps more than 1 cm between samples | The robot sees the hand. A hand that jumps is not realistic |

### Match env

| Test | What it checks | Why |
|---|---|---|
| `test_check_env` | Gymnasium `check_env` accepts `MatchEnv` and the `PrivilegedObsWrapper` version | Stable-Baselines3 needs a correct Gymnasium env |
| `test_pucks_are_conserved_and_rewards_add_up` | 20 matches with random actions. At every step, the pucks on both sides plus the pucks in flight equal 10. The total reward equals the sum from the stats (sent, received, win or loss) | A puck must never appear or disappear. The reward must agree with the scoring rules |
| `test_settled_blocker_saves_every_threat` | `AlwaysBlock` saves every threat and receives nothing. The match ends on time with total reward 0 | The block primitive and the gate state work together |
| `test_greedy_slinger_concedes_every_threat_while_away` | `GreedySling` saves nothing. Every threat is received while the arm is away | An open gate must not save shots |
| `test_sling_timing` | One sling: the arm goes `to_band`, and the gate becomes open. The release comes at move + fetch + core time (within one decision step). One puck leaves the agent side. `BLOCK` sends the arm `to_gate`, and the gate becomes ready on arrival | The arm state machine must follow the configured times. These times define the vulnerability window |
| `test_seeded_matches_repeat` | Two `ReloadRule` matches with seed 11 give the same stats and reward | Results must be repeatable from the seed |
| `test_hand_ablation_zeros_hand_features` | With `observe_hand: false`, the four hand values are 0 for 30 steps | The M5 ablation must really remove the hand input |
| `test_incoming_puck_hidden_before_camera_latency` | `puck_valid` is 1 only when the camera latency has passed since the launch | The robot must not see a puck sooner than a real camera can |
| `test_reload_detection_from_the_hand` | Over 3000 steps, `ReloadRule.reloading` is true in less than 2% of pull and hold frames, and in more than 70% of reload frames | The baseline must not sling while the opponent aims. The README reports this detector as reliable |
| `test_late_return_concedes_fast_threats` | With a 10 s cover time, an arm at the gate does not save the first threat | A return that is too late must not count as a block |
| `test_bad_gate_geometry_is_rejected` | A gate too narrow for a miss margin raises `ValueError` in `threat_half_width` | Bad board values must fail clearly |

### Block primitive

| Test | What it checks | Why |
|---|---|---|
| `test_block_primitive` | Hard cut (std 0): 0.9 at 0.06 s, 0 at 0.04 s, 0.9 at infinite time. Soft (std 0.02 s): 0.5 at the cover time, 0.977 at 2 std early, 0.023 at 2 std late | Checks the formula `save_prob × Phi((time_to_cover − cover_time) / std)` |

---

## `test_asymmetric_policy.py`

Code under test: `src/slingpuck/policies/asymmetric.py`,
`src/slingpuck/envs/wrappers.py`.

| Test | What it checks | Why |
|---|---|---|
| `test_actor_ignores_privileged_critic_uses_it` | Trains PPO for 64 steps on the wrapped `MatchEnv`. The actor input size equals the policy observation. The critic input size equals policy + privileged. A change to the privileged input does not change the action probabilities, but it changes the value. The saved and loaded model gives the same action | The deployed robot does not have the privileged values. The actor must never use them |

---

## `test_slingpuck_scene.py`

Code under test: `src/slingpuck/so101_slingpuck.py`. Needs `mujoco`.

The fixture `scene` builds and compiles the full scene once, and resets it to
keyframe 0.

| Test | What it checks | Why |
|---|---|---|
| `test_board_lies_in_front_of_the_robot` | The agent end of the board is in front of the robot base axis. The board center is centered left to right | The scene layout must match the real setup |
| `test_turning_axis_matches_config` | The `shoulder_pan` joint is at `robot.base_xy_m` (within 0.1 mm) | The robot position in the scene comes from the config |
| `test_board_and_robot_stand_on_the_same_table` | The top of the table is at z = 0. The old wall `lb_wall_back` is not in the scene | The robot and the board must stand on one surface |
| `test_flicked_puck_slows_with_coulomb_friction` | A puck at 0.5 m/s slows at `friction_kinetic × 9.81` (within 3%) | MuJoCo friction must match the config. `MatchEnv` uses the same friction law |
| `test_wall_bounce_keeps_the_configured_restitution` | A puck at 1.5 m/s into the side wall bounces back at `restitution` times its speed (within 0.03) | The contact settings must give the configured bounce |
| `test_opponent_camera_faces_the_robot` | The `opponent` camera looks toward −x (the robot) | The human's view must show the robot |

---

## `test_match_mujoco.py`

Code under test: `src/slingpuck/physics/mujoco_match.py`. Needs `mujoco`.

The fixture `game` plays 12 s of a MuJoCo match with seed 0. It records the
highest puck speed and whether each player held a puck.

| Test | What it checks | Why |
|---|---|---|
| `test_colours_split_five_and_five` | 5 black pucks start on the robot side. 5 white pucks start on the human side | The start state must match the game rules |
| `test_both_players_shoot_and_physics_stays_sane` | Both players hold and shoot at least one puck. The highest puck speed is below 6 m/s. All pucks stay on the board. The count is 10 | Shows that the scripted plan works. Catches a physics blow-up (a blow-up gives speeds of tens to thousands of m/s) |
| `test_robot_guard_pose_blocks_the_gate` | At the start, the gripper site is at the guard pose (within 3 mm). The guard pose is centered (x = 0) on the agent side (y < 0) | The closed gripper must stand in the gate to block shots |

---

## Add a test

1. Put the test in the file for the code that it tests.
2. Use the `cfg` fixture. Do not load `"latest"`, so that new param versions do
   not change the result.
3. Use a fixed seed for all random values.
4. Add a row for the test to this document.
