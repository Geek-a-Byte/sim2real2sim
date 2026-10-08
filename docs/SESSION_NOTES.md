# Session notes: 2026-10-07 to 2026-10-08

Work log of the Claude Code session that reworked the project and built
milestones M1 to M4. Read this first to continue the work. The full technical
report is `docs/slingpuck_report.pdf`; the README has the commands.

## Starting point

- The folder had code for all five milestones from an earlier attempt, one
  trained goalkeeper (`logs/ppo_goalkeeper_latest/`), and no git.
- The review found bugs that made the old results invalid: opponent shots flew
  away from the gate, goals during a sling recovery were not counted, the sim
  and the robot stub used different action-to-angle mappings (0.19 vs 0.5 rad),
  stale camera frames were reused, the camera fell back to ground truth, random
  numbers were not seeded, and a servo latency test failed.
- The original plan (5 milestones, stop after each for review, no invented
  numbers) is the plan in force.

## What was done, in order

| Time (2026-10-08) | Commit | Work |
|---|---|---|
| 00:53 | `10d29f9` | Git created; baseline snapshot of the old code |
| 01:08 | `c20b8ea` | **M1**: config versioning (servo.yaml, physics_params_v0.yaml, env.yaml), placeholders and strict mode, domain randomization, fixed servo/camera/Kalman, Fast2DPuckSim behind a backend interface, tests |
| 01:15 | `7810cbe` | User values: puck mass 27 g, gate "a little over 1.19 in" (set to 1.25 in) |
| 01:35 | `36e0faa` | **M2**: goalkeeper env on Fast2DPuckSim, kinematic paddle, shared kinematics, asymmetric actor-critic, training, save-rate eval |
| 01:55 | `69ce734` | **M2**: recovery task (user decision), grid eval |
| 02:38 | `74d72f9` | 2D slow-motion viewer |
| 02:42 | `d8b2414` | Full README |
| 02:45 | `88ebf85` | User switched imports to `src.slingpuck`; models now load with either path |
| 03:19, 03:25 | `6f285e8`, `c2947e0` | **MuJoCo** backend (user request): real SO-101, IK paddle pose, calibrated contacts, 3D viewer, sim-to-sim eval |
| 07:42, 07:44 | `d354cd8`, `90506f0` | **M3**: band model (pre-tension, closed hysteresis), arm deflection, Phase 2 env with placement (user decision) |
| 08:46, 08:54 | `acb49f4`, `3a1b0d0` | **M3**: sysid (schemas, synthetic logs, fits, run_sysid), hole-rate and parity eval |
| 09:09 | `2c192eb` | **M4**: opponent model, event-level match env, frozen primitives, baselines, match eval |
| 09:19, 09:27 | `2e4f826`, `e014813` | Project report PDF; figures, eval tables, constraints, improvements |

## Decisions made by the user

1. Use the moopok Fast Sling Puck, size M, as the reference board.
2. Puck mass 27 g (questioned: a wood puck of this size is ~4.5 g; 27 g / 6 = 4.5 g).
3. Gate width "a little over 1.19 in" (1.25 in used; still a placeholder).
4. **Phase 1 = recovery task**: the paddle starts away from the gate at the launch.
5. **MuJoCo with full physics** (not only a 3D replay).
6. **Phase 2 aims by placement**: action = (slide x, pull angle, pull distance).
7. Continue M3 and M4 without a review stop; the user reviews M1-M4 later.

## Main findings

- The gate is ~1.5 mm wider than the puck, so a settled paddle blocks every shot.
- PPO beat "go to center" in 2D with a servo-lag trick (3-8 points at 1.5-3 m/s),
  but not in MuJoCo, where the arm is torque-limited (37 ms vs 81 ms to move
  halfway). All policies lose 20-40 points in MuJoCo. The servo model is the
  weakest part of Phase 1.
- A pre-stretched band launches almost straight; the pull angle barely aims.
- Phase 2 success: PPO 94.6%, slide + correction 93.8%, slide to center 92.5%.
- Sysid recovers known values from synthetic logs (within ~1 standard error over
  6 data sets). Parity: v0 transit time -9.5 ms off; the fit +0.7 ms.
- Phase 3 with placeholder timings and "first empty side wins" is a throughput
  race; the plan's reload rule loses; the tell helps only a little.

## Where the outputs are

| Path | Content | In git |
|---|---|---|
| `logs/goalkeeper/goalkeeper_v0_s{0,1,2}_*` | Phase 1 PPO models (3 seeds, 1M steps), run configs | No (archived) |
| `logs/sling/sling_v0_s{0,1,2}_*` | Phase 2 PPO models (3 seeds, 400k steps), run configs | No (archived) |
| `logs/ppo_goalkeeper_latest/` | Model from the old code (invalid env) | No (archived) |
| `tensorboard_logs/` | Training curves of all runs | No (archived) |
| `results/goalkeeper/m2_final_{2d,mujoco}/` | Phase 1 eval tables and plots | No (archived) |
| `results/sling/m3_final/`, `m3_parity_synthetic/` | Phase 2 eval, parity | No (archived) |
| `results/match/m4_baselines/`, `m4_tell_sweep/` | Phase 3 eval | No (archived) |
| `cache/block_table_*.npz` | Block primitive table (rebuilt automatically, ~40 s) | No (archived) |
| `session_archives/session_2026-10-08.tar.gz` | Archive of all the rows above | No (local file) |
| `docs/figures/`, `docs/slingpuck_report.pdf` | Figures and the report | Yes |
| `data/real_logs/synthetic_example/` | Small synthetic sysid logs | Yes |

To restore the archive: `tar -xzf session_archives/session_2026-10-08.tar.gz` in the project root.

## Environment notes

- Use the miniconda Python: `~/miniconda3/bin/python3` (3.14; all packages).
  `python` in the shell is an alias for Homebrew Python 3.10, which has no
  scipy or pytest.
- Run commands from the project root as `python -m src.slingpuck.<module>`.
- The live MuJoCo window needs `~/miniconda3/bin/mjpython` on macOS.
- `pytest -q`: 127 tests, all passing at `e014813`.

## Open questions for the user

1. Real scoring rules of the game (Phase 3 reward and win rule).
2. Is 27 g the mass of one puck?
3. Exact gate width, divider thickness, band position and natural length.
4. M5: wait for real values, or run it now as a sweep over them?
5. Review and approval of M1 to M4.

## How to continue

1. Read `docs/slingpuck_report.pdf` (sections 10 to 13: to-do, constraints, improvements).
2. Do the real measurements in `data/real_logs/schema.md` and run `run_sysid`.
3. Start M5: selector training on `MatchEnv`, ablations (`--no-hand`, `--tell`),
   vulnerability window, ONNX export and the Jetson latency test.
