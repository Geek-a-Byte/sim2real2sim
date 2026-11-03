# Adapting the pipeline to win sling puck

**Rules assumed:** standard sling puck. Two sides, a gap in the middle, and an elastic band on each side. You win when your side is empty first, and the opponent's shots send pucks back to you. Tell me if your course version differs.

## What "winning" means mathematically
It's a **race of rates**. Your side empties when

> **net clearing rate = (your shots per min × P(through gap)) − (pucks arriving from the opponent per min) > 0**

and the bigger that number, the faster you win. So "sling greedily" means: **on every turn, take the action with the highest expected pucks cleared per second.** That is a one-step decision. You don't need long-horizon RL for it.

The three ways to raise the rate:
1. **Accuracy:** a higher P(through) per shot.
2. **Speed:** a shorter pick, seat, draw, and release cycle.
3. **Defense:** block the gap when that saves more pucks than a shot would clear in the same time.

## The adapted pipeline: three layers

### Layer 1: skills (the goal-conditioned policy from the main pipeline, unchanged)
These are all object-pose or gripper-pose goals, so the recommended pipeline applies directly, including the reachable-goal filter:
- **PICK(puck i):** grasp a puck at its tracked position.
- **SEAT:** place the puck in the band pocket.
- **DRAW(aim, pull):** pull the pocket back to a pose given by the aim angle and pull distance, then release.
- **BLOCK(x):** put the gripper or a paddle at gap position x.
- **HOLD:** keep a puck pressed against the band so it doesn't rebound.

Add a **time penalty** to the shared reward, because speed directly raises the clearing rate.

### Layer 2: a shot model that picks the best shot greedily
- Learn **P(through | puck position, aim, pull)** and the **cycle time T(puck i)** (how far the puck is from the pocket).
- Train it in your **event-level sim**, calibrated with real shots (see Layer 4). A small MLP or a Gaussian-process regressor is enough.
- **Greedy choice each turn:** pick the puck and shot that maximize **P(through) / T**. Near pucks and clean angles go first. Skip pucks jammed against walls until they're the last ones left.
- Add a small exploration bonus (e.g. UCB) so the model keeps learning where it's uncertain.

### Layer 3: game decision (SLING vs. BLOCK vs. HOLD)
- Compare expected value per second:
  - **SLING:** P(through) / T_shot
  - **BLOCK:** P(opponent shot is incoming and blockable) × P(block succeeds) / T_block
- Take the argmax, which is a *greedy rule*.
- Your existing **T2 PPO selector** over BLOCK/SLING/HOLD can replace this rule if a learned policy beats it in the event sim. Keep the greedy rule as the baseline, because it's simple, explainable, and fits the "greedy" brief.
- Opponent state (are they drawing, how many pucks do they have left) comes from the webcam.

### Layer 4: real2sim (the failure-guided part, adapted)
There are two kinds of failure, each with its own fix:
| Failure | Example | Fix |
|---|---|---|
| **Skill failure** | missed grasp, puck not seated, slipped during the draw | The standard pipeline: rebuild the scene in MuJoCo, oversample nearby tasks, and retrain the goal-conditioned policy. |
| **Shot failure** | overshoot, hit the gap's edge, weak shot | **Calibrate the sim (system identification).** Fit band stiffness, puck–board friction, and release noise to the logged real trajectories (tracked by the webcam). Then refit the shot model, and oversample shot parameters near real misses. |

Shots are cheap trials (a few seconds each), so about **40–60 calibration shots** from your ~150-trial budget should be enough to fit the 3–4 physics parameters.

## Evaluation
- **Per shot:** P(through) in sim vs. real, before and after calibration. This is the real2sim result.
- **Per game:** shots per minute, net clearing rate, and **win rate** against a scripted opponent (a fixed shot rate and accuracy) and then a human.
- **Comparison:** greedy selector vs. random puck order vs. the T2 PPO selector.

## How it fits the paper story
- **One method:** goal-conditioned skills plus failure-guided Sim2Real2Sim.
- **Sling puck shows the method's boundary.** Dynamic tasks need one extra piece, a learned outcome model over skill parameters, and their real2sim step becomes *physics calibration* instead of *scene rebuilding*. That's a clean, honest extension to report, while Task 1 stays the main live demo.
