# OmniTacTune (arXiv 2607.03723): explained

**Title:** OmniTacTune: Policy-Agnostic Real-World RL for Tactile Residual Adaptation of Visual Policies
**Authors:** Kelin Yu, Haode Zhang, Harish Ravichandar, Yunhai Han, Ruohan Gao (July 2026)
**Project page:** https://colinyu1.github.io/omnitactune-site/

## 1. The problem
Visual policies (ACT, Diffusion Policy, π0.5, or policies learned from human videos) get the robot *near* the right place, but they fail at the "last mile" of contact-rich work: tight insertions, levering, holding contact while moving. A camera can't see force or how the contact sits. Tactile data could help, but it is scarce and doesn't transfer well across sensors, robots, or tasks. Simulating touch is also hard, so the corrections have to be learned on the real robot.

**Core idea:** vision supplies the task structure, and touch supplies a small learned correction on top of it.

## 2. Method
The executed action is the **base action plus a scaled residual**:

  a_t = a_base + s_t · a_residual

The base visual policy stays **frozen**. Only a small residual actor and a critic are trained, with SAC on the real robot.

**Stage I: warm-start (about 12 minutes)**
- Run the frozen base policy with the residual set to zero, and log transitions (proprioception, object-flow features, tactile frames, base action, reward).
- Train a twin-Q critic on these rollouts, so RL doesn't start from a random critic.
- Adapt a pretrained tactile encoder (e.g., AnyTouch2). Only a small adapter and projection are trained, using the critic loss plus a reconstruction loss, on contact frames only.
- Augment: ControlTac generates extra tactile trajectories with different normal forces, with the same actions and rewards.

**Stage II: online residual RL**
- SAC, asynchronous (actor and learner run in parallel, as in HIL-SERL).
- The residual scale ramps 0.05 → 0.10 → 0.15 every 500 steps, so early corrections are gentle. A larger scale (0.5) did worse.
- Residual inputs: proprioception, object-flow features, the base action and its action chunk, and tactile features that are **gated** (zeroed until the gel markers actually move, i.e., real contact).
- **Policy-agnostic:** the residual only sees the base policy's *actions*, never its internals, so the same residual design attaches to ACT, Diffusion Policy, π0.5, or a flow policy.

**Reward** (shaped, object-centric):
- reaching (before the grasp)
- tactile contact/grasp
- **object-flow tracking**: follow 30 keypoint subgoals predicted from human videos
- a safety penalty

A human gives the terminal success label and does the resets. Too much tactile deformation ends the episode.

**Base policies:** a human-video object-flow policy (keypoints tracked with DINOv2, SAM, and CoTracker3, plus a LoRA-tuned flow generator trained on 50 demos), a teleoperation flow policy, ACT, Diffusion Policy, and π0.5.

**Hardware:** xArm7 with GelSight Mini fingertips, one RealSense D435, 10 Hz control, episodes of at most 300 steps.

## 3. Results
Four real tasks: peg-in-hole, charger insertion (~2 mm clearance), cap opening, and box opening (~1.3 mm lever edge).

| Method | Peg | Charger | Cap | Box | Avg |
|---|---|---|---|---|---|
| Base policy | 40% | 10% | 5% | 5% | 15% |
| PLD (visual-only residual RL) | 60% | 30% | 40% | 20% | 37.5% |
| ViTAL | 50% | 50% | 45% | 30% | 43.8% |
| PLD* (with touch) | 65% | 60% | 50% | 35% | 52.5% |
| **OmniTacTune** | **100%** | **100%** | **90%** | **85%** | **93.8%** |

- Real-robot training took **40 to 80 minutes per task**, including warm-start.
- Five different base policies all improved on peg-in-hole, from 15–50% to 75–100%.
- It beats visuo-tactile *imitation* given more demos. For example, ACT+tactile reached 60%, and ACT+OmniTacTune reached 80%.
- Every tactile encoder tested helped. Encoders without dynamic pretraining (T3, Sparsh) were weaker on the charger task.
- Human-video demos gave smoother motion priors than teleoperation (measured with the SPARC and LDLJ smoothness metrics).
- Ablations: every reward term, the warm-start, the contact gate, and the residual-scale schedule each mattered.

## 4. Limitations
- It needs manual resets and a human success label, so it is not autonomous.
- Repeated contact wears out hardware, especially fragile gel sensors.
- It was shown on one robot (xArm7) with one sensor family. Transfer to other robots is claimed, not tested.
- It only fixes the **last mile**. If the base policy never reaches the contact region, the residual can't help, because the correction is capped at 0.15× the base action.
- The four tasks are short single-stage manipulations, and the scene or goal doesn't change much.
- The shaped reward needs a human-video flow generator for each task.

## 5. How it relates to your SO-101 project
**What transfers directly**
- **Residual RL on a frozen base policy** is a cheap, safe way to do the *real-world* half of a failure-guided loop. Your sim-trained goal-conditioned policy becomes the frozen base, and a small residual learns from real rollouts. This also follows the professor's "one method" feedback: the base stays general, and the residual is task-agnostic because it only sees actions.
- **Warm-starting the critic from base rollouts** is a good use of real trials you already have (e.g., the 52 lift and 22 place trials). They are exactly "base policy, residual = 0" data.
- **The object-flow reward from human videos** fits your digital-twin-from-human-demos direction. Tracking object keypoints with CoTracker gives both a goal signal and a dense reward without hand-designed rewards.
- **The residual-scale ramp and the safety termination** are practical tricks for a fragile, low-cost arm.

**What doesn't transfer**
- **You have no tactile sensor.** The paper's main gain is touch: the visual-only residual (PLD) plateaued at 37.5% versus 93.8%. On SO-101, stand-ins could be motor current or load from the Feetech servos, gripper encoder error (commanded vs. actual), or wrist-camera contact cues. Expect smaller gains than theirs.
- **Budget:** 40–80 minutes of real RL is roughly 100–200 episodes *per task*, which matches your whole ~150-trial semester. You could afford it for one demo task at most.

**How it fits your failure-guided curriculum**
OmniTacTune fixes failures *in the real world*. Fail2Progress-style curricula fix them *in sim* by generating scenes around real failures. These two are complementary, which gives a clean two-level story:
1. Real failures go into MuJoCo, where they shape the scene and goal curriculum for the goal-conditioned base policy (the **coarse** fix).
2. The remaining contact-level failures, the ones sim can't model (you have no force sensing and contact sim is poor), get a small real-world residual (the **last-mile** fix).

Cite it as evidence that real-world residual RL works within a small, fixed time budget, and as motivation for the residual stage.
