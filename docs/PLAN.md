# Plan: win sling puck (branch `slingpuck`)

This branch follows [slingpuck-adaptation-plan.md](papers/slingpuck-adaptation-plan.md)
and step 7 of the synthesis
[synthesis-generalize-any-task-sim2real2sim.md](papers/synthesis-generalize-any-task-sim2real2sim.md).
The notes on the 9 papers are in [papers/](papers/). The general any-task method is on the branch
`generalized`.

## The goal: a race of rates

The rules are standard sling puck, the same as the `scoring` placeholder in
`env.yaml`: the first side with no pucks wins, and the opponent's shots send
pucks back. Your side empties when:

```
net clearing rate = (your shots per min × P(through gap)) − (pucks arriving from the opponent per min) > 0
```

Thus, on every turn, take the action with the highest expected pucks cleared
per second. This is a one-step (greedy) decision. There are three ways to
raise the rate:

1. **Accuracy:** a higher P(through) per shot.
2. **Speed:** a shorter pick, seat, draw and release cycle.
3. **Defense:** block the gap when that saves more pucks than a shot clears in
   the same time.

## The four layers

| Layer | Content | Status in this repository |
|---|---|---|
| 1. Skills | PICK(puck i), SEAT, DRAW(aim, pull), BLOCK(x), HOLD (press the puck on the band so that it does not rebound). A time penalty in the reward | Scripted in `physics/mujoco_match.py` (fixed 2.5 cm pull, fixed aim). Later: the goal-conditioned skills from the `generalized` branch |
| 2. Shot model | P(through \| puck position, aim, pull) and cycle time T(puck i). A small MLP or a Gaussian process. Greedy: the puck and shot with the highest P(through) / T, with a UCB exploration bonus. Pucks jammed against walls go last | Not present. `primitives/sling.py` has a fixed success probability and a fixed fetch time |
| 3. Game decision | Greedy rule: SLING value = P(through) / T_shot; BLOCK value = P(incoming and blockable) × P(block succeeds) / T_block; take the larger. The PPO selector may replace the rule only if it wins in the event sim | `MatchEnv`, the rule baselines, and the asymmetric PPO policy exist. The greedy rule does not |
| 4. Real2sim | Skill failures (missed grasp, puck not seated, slip in the draw) → the `generalized` pipeline. Shot failures (overshoot, gap edge, weak shot) → physics calibration: fit band stiffness, puck–board friction and release noise to webcam tracks of 40–60 real shots, fit the shot model again, oversample shots near real misses | `band_model.py` and versioned physics params exist. No fit code |

## Milestones

Each milestone stops for review.

| Milestone | Content | Needs from the team |
|---|---|---|
| S1. Shot model in sim | Sample aim and pull in the MuJoCo release (band model, gate, friction). Fit P(through \| puck position, aim, pull). Model T(puck i) from the distance of the puck to the pocket | New puck diameter, thickness, mass (one puck) and gate width. Until then, v0 values |
| S2. Greedy game rule | Give `MatchEnv` the position of each agent puck. Use the shot model for the sling time and success. Add the greedy SLING-or-BLOCK rule. New metrics: shots per minute and net clearing rate | — |
| S3. Skills on the arm | PICK, SEAT, DRAW(aim, pull), BLOCK(x) and the pressing HOLD in the MuJoCo match, with the shot model's choice. Scripted first | Arm base position |
| S4. Physics calibration | Log 40–60 real shots (aim, pull, webcam track, result). Fit band stiffness, friction and release noise. Write `physics_params_v1.yaml`. Fit the shot model again | Real shots, overhead webcam at 120 fps or more |
| S5. Opponents and games | A scripted opponent with a fixed shot rate and accuracy, then one profile per team member (4 people), then human games | Video of each team member playing |
| S6. Deploy | Opponent state (drawing or not, pucks left) from the webcam. ONNX export, Jetson latency | Jetson access |

## Evaluation

1. **Per shot (the real2sim result):** P(through) in sim against real, before
   and after the calibration.
2. **Per game:** shots per minute, net clearing rate, and win rate against the
   scripted opponent, then against each team member.
3. **Comparison:** the greedy rule against random puck order and against the PPO
   selector. The greedy rule is the baseline.

## Paper story

Sling puck shows the boundary of the general method. A dynamic task needs one
extra part, a learned outcome model over skill parameters, and its real2sim step
is physics calibration, not scene rebuilding. The main live demo stays the
general task (Task 1) on the `generalized` branch.

## Open questions

- **HOLD has two meanings.** In `MatchEnv`, HOLD keeps the arm where it is. In
  the adaptation plan, HOLD presses the puck on the band. S2 must rename one.
- **Where the shot model trains.** The plan says "the event-level sim", but
  `MatchEnv` does not simulate a shot. S1 takes the samples from the MuJoCo
  release and uses the fitted model inside `MatchEnv`.
- **Gate clearance.** The opponent model accepts gate width / 2 − puck radius
  only between 0 and 6 mm. The new puck and gate must fit, or the limit changes.
- **Rules.** The plan assumes standard sling puck. Confirm that the course
  version is the same.
