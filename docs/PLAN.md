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

| Milestone | Content |
|---|---|
| G1. Reachable-goal filter | One check: is an object goal pose achievable by the SO-101? (a) IK finds the grasp pose within the joint limits, with no collision. (b) The wrist roll and pitch are inside the 5-DoF range. (c) The gripper opens wide enough for the object. Precompute a reachability map (a voxel grid of feasible grasp orientations) so that the check is fast |
| G2. Goal-conditioned sim env | MuJoCo env with objects, AP2AP goals that pass the filter, chained goals, staged reward. Include the sling puck skills as goals (PICK, SEAT, DRAW(aim, pull), BLOCK(x), HOLD), with a time penalty. The `slingpuck` branch uses them in its layer 1 |
| G3. PPO teacher | Privileged teacher with paired-point goals and scene tokens. Needs a parallel sim (see the open questions) |
| G4. Webcam student | DAgger distillation with masked, noisy points from one RGB camera. 3-D scale from the known object size |
| G5. Task from a human demo | Human video → CoTracker3 point tracks → a rigid transform per step → a goal sequence, through the filter |
| G6. Deploy and log failures | Zero-shot on the SO-101 with online point tracking. Go to the next goal when the point error is below a threshold. Log each trial: goal step, grasp, transport or place, object pose. Log unreachable goals separately |
| G7. Real2sim curriculum | Rebuild each failure as a scene graph, make variations, score them (feasibility + diversity + closeness to real failures, reachable tasks only), and train again |
| G8. (Optional) Real residual | A small real residual on one demo task for last-centimeter contact errors |

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
  repository runs CPU MuJoCo with Stable-Baselines3, one env per process. G3
  needs a parallel GPU sim (for example MuJoCo MJX) and a GPU.
- **Task set.** Which tasks and objects to start with (the synthesis names
  tube to rack, pushing and puzzles).
- **Camera.** The webcam model, its position, and its frame rate.
- The `papers/` folder that the synthesis names is not in the repository.
  The sling puck adaptation plan is in `docs/slingpuck-adaptation-plan.md` on
  the `slingpuck` branch.
