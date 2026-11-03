# Track2Act (ECCV 2024): explained

**Title:** Track2Act: Predicting Point Tracks from Internet Videos Enables Generalizable Robot Manipulation
**Authors:** Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, Shubham Tulsiani (CMU / Meta). ECCV 2024, LNCS vol. 15134, pp. 306–324
**Links:** [Springer](https://link.springer.com/chapter/10.1007/978-3-031-73116-7_18) · [arXiv 2405.01527](https://arxiv.org/abs/2405.01527) · [project](https://homangab.github.io/track2act/)

The authors are from the same group as Dex4D (Tulsiani). This paper is the earlier gripper-based version of the same "object point tracks as the plan" idea.

## 1. The problem
The goal is to manipulate unseen objects in new scenes out of the box, with no test-time training and without a huge robot dataset. Behavior cloning needs very large in-domain data. Test-time-training approaches need a human demo plus hours of robot practice for every new task.

**Core idea:** learn *how points in the image should move* to reach a goal, from web videos of humans and robots. Points don't depend on the embodiment, so a human video can teach a robot. Then convert the predicted point motion into robot motion.

## 2. Method
**1. Track prediction (learned from web video)**
- A DiT diffusion model takes the **start image, a goal image, and a set of query points**, and denoises their future 2-D positions.
- It was trained on about 400k short clips from Something-Something-v2 and EPIC-Kitchens (humans) and RT-1 and BridgeData (robots). Labels come from CoTracker on a 400-point grid.

**2. Tracks to a rigid object trajectory to a gripper plan (no learning)**
- Keep the moving points and remove outliers with RANSAC. Then, for each step, solve a PnP-style problem for the object's 3-D rigid transform, using the depth at the start frame and the camera intrinsics.
- Grasping is a heuristic: move to the center of the object's points with the starting orientation and close the gripper. After that, apply the rigid transforms to the gripper pose, which gives an **open-loop end-effector trajectory**.

**3. Closed-loop residual policy (small robot dataset)**
- A DiT transformer reads the current point tracks plus the open-loop plan and predicts **corrections** for the next 4 steps, of which it executes 1.
- It was trained by BC on about **400 teleoperated trajectories** (10 tasks, 3 locations), which they say is 3–4 orders of magnitude less than prior generalist BC.

**Hardware:** a Boston Dynamics Spot arm with a 2-finger gripper, one fixed RealSense RGB-D camera, 5 Hz control, 50-step rollouts.

## 3. Results
**Track prediction** (Δ accuracy, higher is better): 0.67–0.77 across the four datasets, against 0.17–0.42 for flow and video-generation baselines.

**Real robot:** 25 tasks in 5 locations, 20 rollouts per level, at increasing difficulty: mild (MG), standard (G), combinatorial (CG), and new object type (TG).

| Method | MG | G | CG | TG |
|---|---|---|---|---|
| Goal-conditioned BC | 60% | 20% | 0% | 0% |
| Hand-object-mask BC (best baseline) | 70% | 40% | 25% | 20% |
| Track2Act open-loop only | 35% | 25% | 30% | 25% |
| Predicts actions (not residuals) | 70% | 45% | 30% | 30% |
| **Track2Act (plan + residual)** | **70%** | **60%** | **55%** | **40%** |

- The plan alone generalizes better than BC on hard levels, but it is brittle. The residual roughly doubles it.
- Predicting **residuals** beats predicting raw actions, because the plan carries the generalization.

**Failure modes:** bad grasp points from the simple heuristic, no recovery from mid-task errors (open-loop), getting stuck on infeasible motions, and trajectories that don't match the goal image.

## 4. Limitations
- Short-horizon, single-object tasks only.
- It needs a goal image of the finished scene, a fixed camera, and depth.
- The grasp heuristic is crude, and success on hard generalization levels is still modest (40–55%).
- There is no RL and no learning from failures. The residual is plain BC from teleop.

## 5. How it relates to your SO-101 project
**Most relevant parts**
- **Gripper-friendly recipe.** Unlike Dex4D's dexterous hand, this is a 2-finger gripper, so it's much closer to SO-101. "Point tracks to a rigid object transform to a gripper pose" is exactly how you would turn a human demo video into **object pose goals** for your goal-conditioned policy.
- **You don't need their track predictor.** You have actual human demos, so run CoTracker3 on them directly to get tracks. The PnP or rigid-fit step is the reusable part. One catch: they use depth for it, and with a single RGB webcam you'll need the known object model (the tube or puck size) for scale. Plan for that.
- **Residual over a plan.** This is the same lesson as OmniTacTune: a base plan carries the generalization, and a small residual fixes execution. In your pitch the base is the sim-trained goal-conditioned PPO policy, not an open-loop plan.

**Where your project goes further**
- Track2Act's open-loop plan can't recover, and its residual is BC from 400 teleop demos. You replace both with a **closed-loop goal-conditioned RL policy trained in MuJoCo**, which is the Dex4D/SimToolReal step forward, plus a **failure-guided curriculum**. Their failure list (bad grasps, infeasible motions) is exactly what a curriculum could target.
- It ignores the scene, the same gap as Dex4D, which supports your **scene-aware goals** contribution.

**How to cite it:** as the origin of "object point tracks from human video as an embodiment-agnostic goal/plan", and as evidence that tracks give better generalization than goal-conditioned BC (CG level: 55% vs 0%). Dex4D is the follow-up that swaps its BC residual for a sim-trained RL policy, and your project adds failure-guided learning on top of that.
