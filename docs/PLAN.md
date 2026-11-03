# Plan: sling puck as a dynamic task (branch `slingpuck`)

This branch adapts the sling puck project to step 7 of the synthesis
"Which ideas fit your problem: one policy for any task, via Sim2Real2Sim".
The general any-task method is on the branch `generalized`.

## Where sling puck sits in the general method

The synthesis puts sling puck in the **outcome layer for dynamic tasks**: tasks
where the result happens after the gripper lets go. The reachable-goal filter
is the switch:

- The robot cannot reach the goal itself (the puck past the gate).
- The robot can reach the **setup goals**: pick the puck, seat it on the band,
  draw the band back.
- The **last goal is parameterized**: aim (launch x) and pull distance.
- A learned **outcome model** P(success | state, parameters) selects the
  parameters greedily: the best success per second, with a small exploration bonus.
- **Real2sim for this layer = physics calibration:** fit friction, restitution
  and band elasticity to logged real shots, then fit the outcome model again.
- The **task-level strategy** (block or sling against the opponent) is a
  task-specific selector on top. It is outside the general method, and the
  report must say so.

## What changes in the code

| Part now | Change |
|---|---|
| `primitives/sling.py`: fixed timings, fixed `sling_success_prob` | Sling with parameters (aim, pull). Its time and its success come from the outcome model |
| `physics/mujoco_match.py`: scripted fetch, fixed 2.5 cm pull | Setup goals (pick, seat, draw) and the parameterized release. Source of the outcome model's sim samples |
| `physics/band_model.py`, `physics_params_v0.yaml` | Parameters to calibrate from real shots (`physics_params_v1.yaml`) |
| `envs/match_env.py`, `policies/` | The selector stays. It reads the sling time and success from the outcome model, not from fixed values |
| `opponent/scripted_opponent.py`: one opponent | One profile for each of the 4 team members |

## Milestones

Each milestone stops for review.

| Milestone | Content | Needs from the team |
|---|---|---|
| S1. Outcome model in sim | Sweep aim and pull in the MuJoCo release. Record success (puck through the gate) and time. Fit P(success \| state, aim, pull). Select the shot by success per second, with an exploration bonus. Replace the fixed sling primitive | New puck diameter, thickness and mass (one puck). New gate width |
| S2. Setup goals on the arm | Pick, seat and draw as reachable goals for the SO-101 in MuJoCo, then the parameterized release | Arm base position |
| S3. Physics calibration (real2sim) | Log real shots (aim, pull, result, puck track). Fit friction, restitution, band stiffness, hysteresis and energy transfer. Write `physics_params_v1.yaml`. Fit the outcome model again | Logged real shots with an overhead camera (120 fps or more) |
| S4. Selector and opponents | 4 opponent profiles (one per team member), sampled per match. Sweep the robot and opponent times and rates. Train the selector (PPO with the asymmetric critic) | Video of each team member playing; the real scoring rules |
| S5. Deploy | ONNX export of the selector, latency on the Jetson Orin Nano | Jetson access |

## Experiments

1. **Main result:** P(success) per shot, sim against real, before and after the
   physics calibration. This shows real2sim working for a task where scene
   rebuilding alone does not help.
2. **Outcome model:** success per second with the learned shot selection,
   against a fixed aim and pull.
3. **Selector:** win rate against each team member, with confidence intervals.
   Baselines: `always_block`, `greedy_sling`, `reload_rule`, `tell_rule`.
4. **Sweep:** which strategy wins as a function of the robot sling rate and the
   opponent threat rate.

## Real trial budget

The synthesis assumes about 150 real trials. Spend them on shots that the
outcome model is not sure about, and on failures, not on easy shots.

## Open questions

- The real scoring rules (the placeholder is "first side with no pucks wins").
- The new puck and gate sizes. The opponent model accepts a clearance
  (gate width / 2 − puck radius) only between 0 and 6 mm.
- The file `slingpuck-adaptation-plan.md` that the synthesis names is not in
  the repository.
