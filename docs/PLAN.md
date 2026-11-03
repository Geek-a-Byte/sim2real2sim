# Plan: one policy for any task (branch `generalized`)

This branch follows the synthesis "Which ideas fit your problem: one policy
for any task, via Sim2Real2Sim". The sling puck task is on the branch
`slingpuck`. In the general method, sling puck is step 7 (the outcome layer
for dynamic tasks).

**Problem:** train one policy in sim that does any manipulation task on the
SO-101 when it gets a goal. Deploy it on the real arm, send the real failures
back into sim (real2sim), and train again, so that it generalizes better with
each loop.

**Contribution:** no earlier paper closes the loop with real failures for a
task-agnostic policy.

**Sources:** the synthesis and the notes on the 9 papers are in
[papers/](papers/) (start with
[synthesis-generalize-any-task-sim2real2sim.md](papers/synthesis-generalize-any-task-sim2real2sim.md)).

## Design decisions from the synthesis

| Decision | From |
|---|---|
| A task is a sequence of object goal poses ("any pose to any pose", AP2AP). No reward per task | Dex4D, SimToolReal |
| Goals are object-centric: object points, not language or a video embedding | Octo, BC-Z, Dex4D, Track2Act |
| Paired Point Encoding: each object point is [current xyz, target xyz], then PointNet pooling. 32 points or fewer | Dex4D |
| Chained random goals in one episode (the next goal near the pose reached) | Dex4D |
| Privileged PPO teacher, then a DAgger student with masked, noisy points (one RGB webcam, no depth) | Dex4D |
| Staged dense reward: reach, lift, track the goal, plus smoothness penalties for the servos | SO-101 PPO paper |
| Entity transformer: one token per object plus a "which object" vector | Cho et al. |
| Real failures rebuilt as scene graphs. Curriculum score = feasibility + diversity + closeness to real failures, plus 10% uniform sampling | ATR, plus this project's extension |

Do not take: dexterous-hand details, tactile sensing, huge real datasets,
video generation as the planner, or unsupervised MI pre-training as a core part.

## What this repository already gives

| Part | Use in the general method |
|---|---|
| `assets/mujoco/robotstudio_so101/` | The SO-101 model for sim training and for the reachability check |
| `config.py` (versioned physics params, placeholders, strict mode, run records) | The same config system, with new config files |
| `policies/asymmetric.py` (actor without privileged input, critic with it) | The start of the privileged teacher |
| `eval/stats.py` | Confidence intervals for success rates |
| `deploy/` | ONNX export and latency benchmark |
| The sling puck code (`envs/`, `opponent/`, `primitives/`, `physics/`) | Stays for now. It becomes the step 7 task later, or is replaced by the `slingpuck` branch |

## Milestones

Each milestone stops for review.

| Milestone | Content | Status |
|---|---|---|
| G1. Reachable-goal filter | One check: is an object goal pose achievable by the SO-101? (a) IK finds the grasp pose within the joint limits, with no collision. (b) The approach is one that a 5-DoF arm can give. (c) The gripper opens wide enough for the object. A precomputed reachability map makes the check fast | **Done:** `general/reachability.py` |
| G2. Goal-conditioned sim env | MuJoCo env with objects, AP2AP goals that pass the filter, chained goals, staged reward. Include the sling puck skills as goals (PICK, SEAT, DRAW(aim, pull), BLOCK(x), HOLD), with a time penalty. The `slingpuck` branch uses them in its layer 1 | **Done for AP2AP** (cube, bar, tube, puck): `general/goal_env.py`. The sling puck skill goals (SEAT, DRAW, BLOCK) are not in it yet |
| G3. PPO teacher | Privileged teacher with paired-point goals and scene tokens | **Code done, not trained:** `general/policy.py`, `general/train_teacher.py`. No scene tokens yet. Needs long runs (see the open questions) |
| G4. Webcam student | DAgger distillation with masked, noisy points from one RGB camera. 3-D scale from the known object size | The student observation (masked, noisy points) is in the env. DAgger is not written |
| G5. Task from a human demo | Human video → CoTracker3 point tracks → a rigid transform per step → a goal sequence, through the filter | Not started |
| G6. Deploy and log failures | Zero-shot on the SO-101 with online point tracking. Go to the next goal when the point error is below a threshold. Log each trial: goal step, grasp, transport or place, object pose. Log unreachable goals separately | **Trial log done** (same format for sim and real): `general/failures.py`, `general/evaluate.py`. The real robot interface and the online tracker are not written |
| G7. Real2sim curriculum | Rebuild each failure, make variations, score them (feasibility + diversity + closeness to real failures, reachable tasks only), and train again | **Done:** `general/curriculum.py` (modes `uniform`, `atr`, `failure`). Feasibility is a k-NN success estimate, not a learned value |
| G8. (Optional) Real residual | A small real residual on one demo task for last-centimeter contact errors | Not started |

## How to run (G1, G2, G3, G6, G7)

```bash
python -m src.slingpuck.general.reachability                      # build cache/reach_map.npz (a few minutes, once)
mjpython -m src.slingpuck.general.view                            # live 3D window: scripted IK pick-and-place
mjpython -m src.slingpuck.general.view --model logs/general/<run>/final_model.zip   # a trained policy
python -m src.slingpuck.general.view --save goal.gif --episodes 2   # GIF instead of a window
python -m src.slingpuck.general.train_teacher --curriculum uniform --timesteps 2000000      # (A)
python -m src.slingpuck.general.train_teacher --curriculum atr --timesteps 2000000          # (B)
python -m src.slingpuck.general.evaluate --model logs/general/<run>/final_model.zip --episodes 100
python -m src.slingpuck.general.train_teacher --curriculum failure \
    --failures results/general/real_trials.jsonl --resume logs/general/<run>/final_model.zip  # (C), next round
```

**The real2sim loop:** run the policy on a task (real or sim), write one trial record per trial
(`general/failures.py`, one JSON line each), and give the file to `--failures`. The curriculum rebuilds
each policy failure in sim (start where the object lay, the goal that failed), makes variations near
it, and samples training tasks near the failures more often. Unreachable goals are tagged and do not
count as policy failures.

## Experiments

1. **Main comparison at matched compute:** (A) uniform task sampling,
   (B) a sim-only curriculum (ATR or PLR), (C) the real-failure curriculum.
2. **Generalization:** a held-out task (a new arrangement or push target),
   given only as a human demo.
3. **Ablations:** paired against separate point encoding; with and without
   scene tokens.
4. **Dynamic task:** P(success) per sling puck shot, sim against real, before
   and after physics calibration (from the `slingpuck` branch).

The synthesis assumes about 150 real trials. Spend them on finding failures,
not on demos.

## Open questions

- **Compute.** The synthesis says "you already run 4,096 MuJoCo envs". This
  repository runs CPU MuJoCo with Stable-Baselines3, one env per process. The
  paper notes say that the 4,096 envs belong to the T3 (pushing) task. G3
  needs that code, or a parallel GPU sim (for example MuJoCo MJX) and a GPU.
- **Code for the other tasks.** The paper notes refer to T1 (tube to rack,
  with 52 lift and 22 place real trials), T2 (the sling puck selector) and T3
  (pushing). Only the sling puck code is in this repository.
- **Task set.** Which tasks and objects to start with (the synthesis names
  tube to rack, pushing and puzzles).
- **Camera.** The webcam model, its position, and its frame rate.
- The SO-101 PPO paper note uses only the abstract. Its full text is needed
  to check its results.
