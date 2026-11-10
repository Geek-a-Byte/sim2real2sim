# Tests

This document describes each test in `tests/`: what it checks, and why the
check is important.

Run all tests from the project root folder:

```bash
pytest -q                          # 133 tests, about 20 s
pytest -q tests/test_match.py      # one file
pytest -q -k hysteresis            # tests whose name contains "hysteresis"
```

The MuJoCo tests (`test_slingpuck_scene.py`, `test_match_mujoco.py`, `test_shot.py`)
need the `mujoco` package. If it is not installed, pytest skips these tests.

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
| [`test_shot.py`](#test_shotpy) | 17 | Shot simulator, shot model, shot choice, greedy match, greedy rule |
| [`test_skills.py`](#test_skillspy) | 16 | Scripted arm skills, friction grip, block, timing sweep |
| [`test_joint_map.py`](#test_joint_mappy) | 14 | Sim-to-real joint map, gripper map, calibration script, mirror (no hardware) |
| [`test_real_arm.py`](#test_real_armpy) | 23 | Board fit, joint-zero fit, range check, trajectory record and replay checks, vertical stop test (no hardware) |
| [`test_overhead_camera.py`](#test_overhead_camerapy) | 14 | Overhead camera: homography, puck detection, tracker, snapshot, puck choice (needs `opencv-python`) |

---

## `conftest.py`

| Fixture | What it gives |
|---|---|
| `cfg` | The merged config `load_config("v0")` with `shot.selection: fixed` (the old fixed sling), so these tests need no shot model file. Scope: one per test session. Most tests use it |
| `greedy_cfg` | `load_config("v0")` with `shot.selection: greedy` |
| `shot_model` | A small shot model (3 members) fitted on 600 simulated shots. The real model uses 4000 shots and 5 members |

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
| `test_v0_loads_and_lists_placeholders` | v0 loads. `meta` shows `params_version: v0` and `calibrated: False`. `board.wall_height_m` is a placeholder. `board.gate_width_m`, `board.divider_thickness_m` and `band.band_offset_from_end_wall_m` are not placeholders (the user measured them). `env.yaml` is merged in (`pucks_per_side == 5`) | Each run must record which values are real and which are guesses |
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

## `test_shot.py`

Code under test: `src/slingpuck/physics/shot_sim.py`, `src/slingpuck/shot/model.py`,
`src/slingpuck/primitives/sling.py` (`ShotSelector`), `src/slingpuck/envs/match_env.py` (greedy
selection), `src/slingpuck/policies/match_scripted.py` (`GreedyRule`). Needs `mujoco`.

| Test | What it checks | Why |
|---|---|---|
| `test_straight_center_shot_goes_through` | A straight 2 cm pull from the center goes through, faster than 0.5 m/s, and reaches the divider at x ≈ 0 | The basic shot must work |
| `test_off_center_shot_hits_the_divider` | 1 cm off center hits the gate edge; 5 cm off center hits the divider wall | The outcome classes (edge, miss) that the calibration logs |
| `test_small_pull_is_weak` | Pulls of 1 mm and 5 mm do not reach the divider | Weak shots are one of the real failure types |
| `test_angled_pull_still_launches_almost_straight` | A 0.3 rad pull launches and reaches the divider less than 8 mm off center | The pre-stretched band launches almost straight |
| `test_release_noise_is_seeded` | The same seed gives the same noisy shots, and noise changes the shot | Repeatable samples for the shot model |
| `test_shot_model_learns_where_shots_go_through` | The model gives P > 0.7 for a center shot, and P < 0.1 for a 4 cm off-center or a 6 mm pull | The model must learn the physics |
| `test_shot_model_save_load_and_fingerprint` | Save and load give the same predictions. Another gate width changes the fingerprint, and the load then refuses | A shot model fitted on another board must never be used |
| `test_greedy_choice_takes_the_near_puck_and_jammed_pucks_go_last` | Between a near and a far puck, the greedy choice takes the near one. A puck jammed against a wall goes last, but is taken when it is the only one | The adaptation plan: near pucks first, jammed pucks last |
| `test_shot_time_follows_the_time_model` | T = pick + carry / arm speed + seat and release + pull / draw speed | The time model that the choice uses |
| `test_random_selection_varies_the_puck` | Random selection takes at least 4 of 5 pucks in 40 choices | The "random puck order" baseline |
| `test_a_puck_in_a_row_costs_the_push_out_time` | Two touching pucks each cost `free_puck_s` more; a puck with room costs nothing extra | The shot choice must see the time of the push-out |
| `test_release_applies_the_measured_jaw_effect` | The exit speed is scaled by `release_speed_factor`, and the bias moves the shot toward +x | The shot model must learn from what the arm does |
| `test_a_puck_keeps_its_direction_under_friction` | A 0.03 rad shot is 3 mm off center at the divider | Per-axis friction removed small sideways speeds and hid angle errors |
| `test_greedy_match_conserves_pucks_and_removes_the_chosen_puck` | With greedy selection, the puck count never changes, and the released puck is the chosen one | Puck bookkeeping with positions |
| `test_best_shot_observation` | The `shot_p` observation equals the best shot's P. From the gate, T includes the move out; at the band it is shorter | The greedy rule and the PPO selector read these values |
| `test_greedy_shot_choice_beats_random_puck_order` | Over 40 matches, greedy selection gives a better puck difference than random puck order | Layer 2 must help |
| `test_greedy_rule_decisions` | The rule slings with a good shot while the opponent reloads, blocks a threat that arrives in 50 ms when the shot is poor, and blocks with no pucks | Layer 3 decisions in clear cases |

---

## Add a test

1. Put the test in the file for the code that it tests.
2. Use the `cfg` fixture. Do not load `"latest"`, so that new param versions do
   not change the result.
3. Use a fixed seed for all random values.
4. Add a row for the test to this document.

---

## `test_skills.py`

Code under test: `src/slingpuck/physics/mujoco_match.py` (skills, friction grip),
`src/slingpuck/eval/skill_eval.py`, `src/slingpuck/eval/timing_sweep.py`. Needs `mujoco`.

| Test | What it checks | Why |
|---|---|---|
| `test_jaws_collide_with_pucks` | The body-level collision masks let both jaws touch a puck | With MuJoCo's compile-time masks the jaws passed through the pucks |
| `test_scripted_shot_grips_and_goes_through` | A straight 2.5 cm shot grips without a slip and goes through the gate. The times are in order | The whole skill chain works with friction |
| `test_start_row_pucks_touch` | Neighboring pucks start one puck diameter apart | The real game's start row |
| `test_middle_puck_is_pushed_out_of_the_row_then_gripped` | A middle puck is pushed out and gripped. Its neighbours move less than 2 mm | No jaw fits between touching pucks |
| `test_row_shots_keep_the_jaws_above_the_board` | During row shots, the jaw tips stay above the board surface | The sim has no jaw-board contact, so the script must not rely on one |
| `test_fast_arm_does_not_knock_the_puck_before_the_grasp` | At double speed the arm still grips, and the jaws stay above the board | The arm must rise first and wait for arrival before a descent |
| `test_spread_widens_the_hole_in_the_row` | After a middle puck is pushed out, the pucks at its sides are more than 5 mm further apart | A shot needs room to pass through the row |
| `test_greedy_sling_rule_does_not_go_back_to_the_gate` | With `game_rule: greedy_sling`, the plan has no move to the gate | The rule must do what it names |
| `test_low_jaw_friction_loses_the_puck_in_the_pull` | With jaw friction 0.02, the puck slips under the band force | The grip is real physics, not a constraint |
| `test_hold_keeps_the_puck_under_the_band_force` | With the default grip, a puck pulled back 3 cm drifts less than 3 mm in 0.5 s | HOLD must keep the shot |
| `test_grasp_of_an_empty_spot_is_a_missed_grasp` | A wrong camera position gives the outcome `missed_grasp` and no held puck | Failure labels for the real2sim step |
| `test_closed_gripper_blocks_a_shot_and_an_open_gate_does_not` | Shots at both edges of the threat band stop at the closed gripper. With the jaws 6 cm away, a shot passes | The block is physical, and the test can fail |
| `test_calibration_error_moves_the_arm` | A joint offset of 1 degree moves the grasp site more than 1 mm | The calibration test must change the arm's pose |
| `test_constraint_grasp_model_still_works` | `skills.grasp_model: constraint` still holds a puck | The old model stays as a comparison |
| `test_selector_gives_the_shot` | With a `ShotSelector`, the arm's shot comes from the shot model's range | Layer 2 drives layer 1 |
| `test_timing_sweep_scales_only_the_robot_times` | The sweep scales the robot times, not the opponent's, and does not change its input | The sweep must change only what it names |

---

## `test_joint_map.py`

Code under test: `src/slingpuck/deploy/joint_map.py`, `src/slingpuck/deploy/calibrate_joints.py`,
`src/slingpuck/deploy/mirror.py`.
Needs no arm and no LeRobot: a fake arm gives the readings.

| Test | What it checks | Why |
|---|---|---|
| `test_sim_to_real_and_back_is_the_identity` | Sim angles → LeRobot action → sim angles gives the same angles. Sim 0 gives the zero reading | Commands and readings must use the same map |
| `test_real_to_sim_accepts_names_with_and_without_pos` | `real_to_sim` reads `joint` and `joint.pos` keys, and applies the sign | LeRobot's bus and robot give different key names |
| `test_sign_needs_a_large_move` | A move of 4 deg is refused. Up gives +1, down gives −1 | A small move is too close to hand shake and backlash |
| `test_zero_from_a_non_zero_reference_pose` | The zero comes out right from a reference pose that is not sim 0 | The sim zero pose can be hard to reach on the real arm |
| `test_save_load_and_calibration_change_warning` | The map saves and loads. A changed LeRobot calibration file gives a warning | The LeRobot zeros move after each `lerobot-calibrate` |
| `test_arm_joint_order_matches_the_sim` | `joint_map.ARM_JOINTS` equals `mujoco_match.ARM_JOINTS`. Needs `mujoco` | The map is a copy, so that the lerobot env does not need MuJoCo |
| `test_sign_and_zero_steps_recover_the_true_map` | The sign, zero and gripper steps, with a fake arm and a fake person, find the true zeros, signs and gripper readings | The whole procedure works before it runs on the arm |
| `test_a_pose_that_moves_during_the_reading_is_asked_again` | A reading with more than 1 deg spread is asked again | A moving arm gives a wrong zero |
| `test_a_joint_that_falls_back_before_enter_is_asked_again` | A joint that falls back to its start before Enter gives no sign, a hint that names the fall, and a new try | The first run on the arm: `shoulder_lift` fell to its stop when the person let go |
| `test_live_reading_shows_the_move` | While the script waits for Enter, it prints the joint angle and the move from the start pose | The person must see the move before pressing Enter |
| `test_gripper_closed_and_open_map_to_the_sim_jaws` | The closed reading gives the sim closed angle, the open reading gives closed + span. Sim → real → sim is the identity. Out-of-range values clip. No gripper map gives an error | The real and sim jaws must open by the same angle |
| `test_sim_gripper_constants_match_the_model` | The gripper constants equal the `so101.xml` joint range, and the jaws are closed at the lower limit. Needs `mujoco` | The map is a copy, so that it works without MuJoCo |
| `test_mirror_overlay_shows_every_joint` | The mirror's text shows 5 joints and the gripper, sim and real. Needs `mujoco` | The person compares the numbers with the arm |
| `test_live_table_shows_sim_and_real_in_rad` | The live check shows the sim angle and the real angle of each joint in rad, and the sim gripper angle | The person compares the sim row with the viewer sliders, which are in rad |

---

## `test_real_arm.py`

Code under test: `src/slingpuck/deploy/board_frame.py`, `touch_test.py`, `fix_ranges.py`, `record_trajectory.py`,
`replay_trajectory.py`, `vertical_stop.py`. Needs no arm and no LeRobot.

| Test | What it checks | Why |
|---|---|---|
| `test_board_fit_recovers_yaw_offset_and_surface` | Exact touches give back the yaw, the offset and the surface height | The fit must be right before it meets noise |
| `test_board_fit_with_noise_and_too_few_marks` | With 1 mm noise the RMS error is 0.3–2 mm. Two marks are refused | Real touches have noise. Two points cannot fix a yaw well |
| `test_pan_axis_matches_the_sim_board_placement` | `base_xy_m` from the fit, put into `so101_slingpuck.board_to_robot`, gives back the touched points. `BASE_AXIS_X` equals the model value. Needs `mujoco` | The sim must put the board where the real board is |
| `test_board_file_round_trip` | The board file saves and loads `base_xy_m` and the yaw | `record_trajectory.py` reads this file |
| `test_marks_are_on_the_robot_half_and_left_is_negative_x` | Every mark is on the robot's half. "left" marks have board x < 0 | A left/right mistake turns the fit by 180 degrees |
| `test_range_check` | A full range passes. The 487-step pan range is refused. A range at the encoder wrap is refused | The second calibration had a 487-step pan range |
| `test_limits_from_calibration` | Stops in LeRobot degrees from a calibration file (487 steps = ±21.4 deg) | The replay must refuse targets past the motor limits |
| `test_replay_checks_limits_and_speed` | No gripper map gives no gripper command. A ramp's speed is measured exactly. 172 deg/s passes, 200 deg/s fails. A target past a stop fails | The replay must refuse unsafe trajectories before the torque comes on |
| `test_single_frame_jump_is_not_a_speed_problem` | A 3 deg one-frame step gives no speed problem, and is reported as a jump | The sim script starts some moves a few mm from the last end. That is not a sustained speed |
| `test_cut_until_and_sample` | `--until` keeps the frames up to the end of a move. Interpolation is linear and clamped | Partial replays (for example up to the grasp) |
| `test_action_has_the_gripper_only_with_a_map` | The gripper is commanded only with a gripper map | A wrong gripper command can crush a puck or drop it |
| `test_record_setup_uses_the_board_file_and_hover` | The record config takes `base_xy_m` and the surface height from the board file, and adds the hover. Needs `mujoco` | The sim must match the real setup, and hover runs must stay above the board |
| `test_touch_fit_finds_zero_errors_and_the_board` | Touches made with known zero errors (pan −6, lift +2, elbow −1.5, wrist flex +1 deg): the fit gives the pan zero within 0.3 deg, the others within 1 deg, tip errors below 0.5 mm (xy) and 1 mm (z), the board offset and height, and a leave-one-out error below 1 mm. Needs `mujoco` | With the jaws down at every mark, lift, elbow and wrist flex trade off: the fit must put the tips right, and the test must not ask for each true zero |
| `test_pan_change_turns_the_tip_about_the_pan_axis` | A sim pan change of +4 deg equals a turn of the tip by −4 deg about the pan axis. Needs `mujoco` | This is why a board yaw can go into the pan zero |
| `test_runner_limits_steps_from_the_last_command_not_from_the_present` | With a joint that does not move, the commands still advance 4 deg per step from the last command | The first run on the arm: LeRobot's `max_relative_target` held the goal 5 deg from the present position, which also capped the elbow's force, and the elbow stalled under gravity |
| `test_runner_stops_a_stalled_joint` | A joint blocked at 80 deg stops the approach with `Stalled` and names the joint. The log keeps the commands | A blocked or overloaded joint must stop the arm before the power supply or a motor does |
| `test_runner_passes_a_joint_that_follows` | A joint that follows reaches the end of the approach with no stop | The stall guard must not stop a normal move |
| `test_vertical_stop_ik_moves_the_tip_straight_down` | Five 2 mm IK steps move the jaw tip 10 mm straight down (within 0.2 mm). The jaw direction turns less than 1 deg. Wrist roll does not move. Needs `mujoco` | The test must go straight down, so that the contact height is where the jaws are |
| `test_vertical_stop_rules_and_frames` | The descent targets, the auto-stop 10 mm under the expected surface and after the travel limit, robot → board xy as the inverse of board → robot, and the load in % (LeRobot already gives the signed value) | The arm must stop by itself if nobody presses Enter |
| `test_vertical_stop_records_the_contact` | With an arm that follows every command, Enter at the 4th step records a contact at the board height (within 0.3 mm), and the arm goes back to the start. Needs `mujoco` | The whole procedure works before it runs on the arm. With a step period of 0 s, the first version never checked for Enter |
| `test_vertical_stop_correction_uses_the_median` | The median of the contacts (7.2, 13.0, 7.8, 7.0 mm: the first real run) moves the board surface up 7.5 mm. Fewer than 2 contacts give no correction | One outlier (a contact near a wall) must not move the surface |
| `test_replay_start_check` | A hand-held start within 20 deg per joint and 40 mm at the tips passes. The rest pose fails. Tips 5 cm away fail | The joint-space move from the rest pose went through a stretched-out pose that the elbow could not hold |
| `test_runner_log_has_loads_or_nan` | The replay log has one load row (lift, elbow, wrist flex) per command, NaN when the arm cannot report loads | The summary warns about loads near the overload protection |

---

## `test_overhead_camera.py`

Code under test: `src/slingpuck/sensing/` and the camera part of `deploy/record_trajectory.py`. Needs
`opencv-python` (skipped without it). The frames are synthetic: a top-down board with pucks, printed lines, a ring
and a "glove" (a dark disk with light notches at the rim), warped into a camera view with a known homography.

| Test | What it checks | Why |
|---|---|---|
| `test_homography_matches_opencv_and_round_trips` | The numpy homography equals `cv2.getPerspectiveTransform`. Image → board → image is exact. More than 4 pairs give the same map | Every puck position goes through this map |
| `test_rectified_view_puts_the_robot_end_at_the_bottom` | Robot end at the bottom, robot's right at the right, the board length vertical | A left/right or end mistake sends the arm to the wrong puck |
| `test_detects_every_puck_within_2mm_with_its_colour` | In a tilted view and in a view turned 90 deg: every puck is found within 2 mm, with its colour, and the robot-half count is right. No false puck from the lines, the ring or the glove | The arm grasps where the camera says. The jaws have a few mm of room |
| `test_no_pucks_no_detections` | An empty board gives no puck | Printed graphics must not look like pucks |
| `test_parallax_correction` | The corrected point is where the ray from the lens through the top face hits the surface. No camera values: no change | The camera sees the top face, 10 mm above the surface |
| `test_bad_settings_and_corners_rejected` | A misspelled detector setting and corners on one line are refused | A silent typo leaves a default in use |
| `test_calibration_file_round_trip` | The calibration file saves and loads the map, the camera values and the detector settings | `record_trajectory.py --camera` reads this file |
| `test_tracker_follows_a_fast_puck_and_keeps_ids` | A 2.5 m/s shot at 60 fps keeps one track, with its velocity within 0.15 m/s and its colour. A resting puck stays below 0.2 m/s with 1 mm noise. Prediction over the latency | The block decision needs the incoming puck's velocity |
| `test_tracker_drops_lost_pucks_and_needs_two_hits` | A track is confirmed after 2 detections, and ends after `max_missed_s` without one | One noisy frame must not make a puck |
| `test_snapshot_rejects_a_moving_puck` | A sliding puck is not in the snapshot. A resting one is, within 2 mm | The arm must not go to a puck that moves away |
| `test_snapshot_from_an_image_file` | The snapshot finds every puck, the robot-half filter, `--pick nearest` and `--pick <index>` | The path from the camera to the arm script |
| `test_greedy_pick_uses_the_shot_selector` | `--pick greedy` gives the same puck and shot as `ShotSelector`. Needs `mujoco` | The arm takes the puck that the match logic would take |
| `test_greedy_pick_checks_the_model_against_the_unmoved_board` | With the sim board moved up 3 cm (as `setup_config()` does for the real surface height and a hover), `--pick greedy` checks the shot model against the unmoved board, and the greedy choice works. Needs `mujoco` | Before this fix, every `--camera` run failed the fingerprint check and fell back to the nearest puck |
