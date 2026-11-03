# Which ideas fit your problem: one policy for any task, via Sim2Real2Sim

**Your problem, stated in one line:** train *one* policy in sim that can do *any* manipulation task on SO-101 when given a goal. Deploy it on the real arm, send real failures back into sim (real2sim), and retrain so it generalizes better with each loop.

This synthesis draws on 9 papers already explained in `papers/`: Dex4D, Track2Act, OmniTacTune, Cho et al. (unsupervised skills), ATR, RT-1, Octo, BC-Z, and the SO-101 PPO paper.

---

## The big picture: what the papers agree on
1. **"Any task" comes from the *goal*, not from training on many tasks.** Dex4D, Track2Act, and SimToolReal all train *one* task-agnostic skill: move the object from its current pose to a target pose. The *task* is just a sequence of object goals. You don't need a reward per task.
2. **The goal should be object-centric: object poses or object points**, not language or a whole video. Octo found goal images beat language by 25%. BC-Z found human-video embeddings nearly useless (4% vs. 32% with language). Dex4D and Track2Act succeed by tracking *object points*.
3. **Which tasks and scenes you train on matters more than how many.** RT-1 says diversity beats quantity, and ATR picks feasible and diverse tasks automatically. This is exactly where *real failures* can do better.
4. **No paper closes the loop with real failures for a task-agnostic policy.** Dex4D, Track2Act, and Octo are zero-shot or imitation. ATR's curriculum is sim-only. BC-Z fixes failures with a *human* teleoperator. OmniTacTune fixes them with real RL, but per task. **That gap is your contribution.**

---

## The ideas to use, by stage of your loop

### Stage 1: Define "any task" (the goal interface)
| Idea | From | Why it fits |
|---|---|---|
| **Task = a sequence of object goals** ("anypose-to-anypose", AP2AP) | Dex4D (and SimToolReal) | One training objective covers tube-to-rack, pushing, and puzzles. It answers the professor's "one problem, one method". |
| **Paired Point Encoding:** each object point is written as [current xyz, target xyz], followed by PointNet pooling | Dex4D | Its biggest ablation (20% to 60% success). It's cheap, and 32 or fewer points works, which suits a webcam. |
| **Human demo video, then object point tracks, then a rigid transform per step, then a goal sequence** | Track2Act (PnP step), Dex4D (SAM2 + CoTracker3) | This is how a new task gets specified at test time with no new training: record a human doing it once. It is gripper-friendly (Track2Act uses a 2-finger gripper). |
| Avoid language-only and video-embedding goals | Octo, BC-Z | Both were weaker than explicit goals. |

**SO-101 note:** you have one RGB webcam and no depth. Use the known object size or model for 3-D scale when converting tracks to poses.

### Stage 2: Train the policy in sim
| Idea | From | Why it fits |
|---|---|---|
| **Chain random goals within an episode** (next goal sampled near the achieved pose) | Dex4D | It trains recovery and long sequences, not just single reaches. |
| **Privileged PPO teacher, then a DAgger student with masked, noisy points** | Dex4D | It bridges sim state to a single-camera view. You already run 4,096 MuJoCo envs. |
| **Staged dense reward:** reach, lift, track the goal, plus smoothness penalties | SO-101 PPO paper | A template already shown on SO-101 at 4,096 envs on one consumer GPU. Smoothness terms protect cheap servos. |
| **Entity transformer:** a token per object plus a "which object" intention vector, no positional embedding, average pooling | Cho et al. | Handles a variable number of objects and distractors (multiple tubes, puzzle pieces). |
| **Restrict goals to SO-101-reachable poses** (step 1 of the recommended pipeline) | (lesson from Dex4D's hand vs. your gripper) | Otherwise the policy trains on goals it can never reach. |
| *(Stretch)* Shared frozen primitives with a cheap goal or task gate | Cho et al. | Makes each retraining round cheaper (retrain the gate only). Optional. |

### Stage 3: Deploy on the real arm (sim2real)
| Idea | From | Why it fits |
|---|---|---|
| **Zero-shot deployment with closed-loop online point tracking.** Advance to the next goal when the point error falls below a threshold | Dex4D | No real fine-tuning is needed for the base policy. |
| **Log every real trial as a structured failure:** which goal step, grasp vs. transport vs. place, object pose | Dex4D and Track2Act failure analyses, BC-Z intervention rate | Gives the real2sim stage something precise to rebuild. Their observed failures (bad grasps, knocked-over objects, lost tracks, the last centimeter) are the ones to expect. |
| *(Optional)* **A small real residual** for contact-level "last-mile" errors that sim can't model | OmniTacTune, Track2Act | Residuals beat raw actions (Track2Act), and warm-start the critic from logged trials (OmniTacTune). Affordable for one demo task only. |

### Stage 4: Real failures back to sim, then retrain (real2sim, the core novelty)
| Idea | From | Why it fits |
|---|---|---|
| **Task vector with a scene graph** (objects, relations, goal, camera) and a procedural generator | ATR | A clean way to rebuild a real failure as a sim task and to vary it. It also gives you *scene-aware goals* (rack slots, walls, gates as graph nodes), which Dex4D says it lacks. |
| **Curriculum score = feasibility + diversity**, plus 10% uniform sampling | ATR | A ready formula. **Your extension adds a third term, closeness to real failures**, so tasks near real-failure embeddings get oversampled. That is a precise, novel statement of your method. |
| **Failure data is worth more than new data** | BC-Z (interventions ≈ 2× demos), RT-1 (diversity > quantity) | Justifies spending your roughly 150 real trials on finding failures, not on collecting demos. |
| **Sim data absorbs well into a real policy** | RT-1 | Supports adding failure-derived sim scenes without hurting other skills. |

---

## How your method differs from each paper (for the related-work section)
| Paper | What it has | What it lacks that you add |
|---|---|---|
| Dex4D | Task-agnostic goal-conditioned sim policy, zero-shot | Scene awareness; learning from real failures |
| Track2Act | Human/web video to object tracks to robot plan | Closed-loop RL policy (its plan is open-loop, its residual is BC); learning from failures |
| OmniTacTune | Real-world residual RL that fixes failures | Task-agnostic policy (per task); sim curriculum |
| ATR | Automatic sim task curriculum | Real-world failure signal; one goal-conditioned policy |
| BC-Z | Learning from failure points (human interventions) | Autonomy (needs a teleoperator); sim |
| Cho et al. | Task-agnostic pre-training and transfer | Perception; real-failure loop |
| RT-1 / Octo | Generalist policies from huge datasets | Feasibility at about 150 real trials; RL; improving from failures |
| SO-101 PPO | Goal-conditioned PPO on SO-101 in sim | Sim-to-real; multiple tasks; real2sim |

---

## Recommended pipeline (from the strongest ideas)
1. **Reachable-goal filter (applies everywhere):** build one check that says whether an object goal pose is achievable by SO-101. It covers three things: (a) IK finds a solution for the grasp pose implied by the goal, within joint limits and with no collisions; (b) the wrist roll and pitch the goal needs are within SO-101's 5-DoF range (most arbitrary 6-D orientations aren't); (c) the gripper can open wide enough for the object. Use it in **all three places goals are created**:
   - **Sim training:** reject or resample random AP2AP goals that fail the check.
   - **Task spec from human demos:** project each demo waypoint to the nearest reachable pose, or flag the demo as infeasible.
   - **Real2sim generator:** discard failure variations that are unreachable, so the curriculum never oversamples impossible tasks. Unreachable goals have zero feasibility anyway, but filtering them up front saves compute and keeps the "failure" signal meaningful.
   Precompute a reachability map of the workspace once, e.g. a voxel grid of feasible grasp orientations, so the check is fast inside 4,096 envs.
2. **Policy:** a goal-conditioned PPO teacher in MuJoCo with paired-point goals, chained random (reachable) goals, scene tokens, and a staged reward. Distil it with DAgger into a webcam student with masked points.
3. **Task spec:** a human demo, then CoTracker3 tracks, then per-step object poses as goals, passed through the reachable-goal filter.
4. **Deploy** zero-shot on SO-101 and log structured failures. Tag any failure whose goal turned out to be unreachable separately, so it doesn't count as a policy failure.
5. **Real2sim:** rebuild each failure as an ATR-style scene graph, score candidate tasks by feasibility + diversity + closeness to real failures (reachable tasks only), and retrain.
6. *(Optional)* a small real residual on the demo task for last-mile contact errors.
7. **Outcome layer, only for dynamic tasks** (sling puck, flicking, sliding, tossing: anything where the result happens *after* the gripper lets go). The reachable-goal filter acts as the switch:
   - **Goal pose reachable** (tube to rack, push, puzzle): use steps 1–6 unchanged. The goal *is* the outcome.
   - **Goal not reachable, but its outcome is reachable through a release** (puck past the gap): the policy executes reachable *setup* goals (pick, seat, draw), and the last goal is **parameterized** (e.g. aim and pull). A learned **outcome model** P(success | state, parameters) picks the parameters greedily (best success per second, with a small exploration bonus).
   - **Real2sim for this layer = physics calibration.** Fit restitution, elasticity, and friction to the logged real outcomes, then refit the outcome model. Calibration is also worth doing for quasi-static tasks (e.g. friction for pushing), so it isn't sling-specific.
   - Task-level *strategy* (e.g. block vs. shoot in a competitive game) sits on top as a task-specific rule or selector. It is outside the general method and is reported as such. See `slingpuck-adaptation-plan.md`.

## Experiments this suggests
- **Main comparison at matched compute:** (A) uniform task sampling, (B) **ATR or PLR sim-only curriculum**, (C) **your real-failure curriculum**. Adding ATR as B makes the result much harder to dismiss.
- **Generalization test:** hold out a task that was *never trained on*, such as a new tube arrangement or a new push target, and specify it only through a human demo.
- **Dynamic-task extension:** P(success) per shot in sim vs. real, before and after physics calibration. This shows real2sim working for a task where scene rebuilding alone wouldn't help.
- **Cheap ablations:** paired vs. separate point encoding (replicating Dex4D on SO-101), and with vs. without scene tokens.

## What not to take
- Dexterous-hand specifics (Dex4D), tactile sensing (OmniTacTune), and huge real datasets (RT-1, BC-Z, Octo).
- Video *generation* as the planner (Dex4D's main failure source). Use real human demos instead.
- Unsupervised MI pre-training (Cho et al.) as a core part. It's finicky, so keep it as a stretch goal at most.
