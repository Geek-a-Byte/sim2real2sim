# Unsupervised RL for Transferable Manipulation Skill Discovery (RA-L 2022): explained

**Title:** Unsupervised Reinforcement Learning for Transferable Manipulation Skill Discovery
**Authors:** Daesol Cho, Jigang Kim, H. Jin Kim (Seoul National University)
**Venue:** IEEE Robotics and Automation Letters 7(3), July 2022, pp. 7455–7462. DOI 10.1109/LRA.2022.3171915 (IEEE Xplore doc 9770346)

## 1. The problem
RL policies are usually trained for one task with one reward, so they don't carry over to new tasks. *Unsupervised RL* pre-trains an agent **without any task reward** and then adapts it to downstream tasks. Earlier skill-discovery methods (DIAYN, DADS) explore based on the robot's *own* state. In manipulation that mostly produces arm-waving, because what matters is **interacting with objects**.

**Core idea:** pre-train with an intrinsic reward that pays the robot for *interacting with objects in diverse ways*, and store what it learns in a set of reusable **primitive policies**. Then solve new tasks by training only a small **gating network** that mixes those frozen primitives.

## 2. Method
**Phase 1: reward-free pre-training (SAC)**
The intrinsic reward has two mutual-information (MI) terms:
- **I(object pos ; gripper pos)**, estimated with a Jensen–Shannon (JSD) MI estimator. This term is high when the object's position is uncertain overall but predictable from the gripper, i.e. the robot is carrying the object around. JSD replaced the Donsker–Varadhan estimator used by MISC, which was unstable and overestimated the MI, so the robot collapsed to one behavior after grasping.
- **I(next object state ; skill z | object state)**, the DADS objective applied to the *object*. Different skills z move the object in different but predictable ways. Unlike DIAYN, it pays nothing if the object doesn't move, so the robot can't game it by shoving the object off the table.

**Policy:** a Multiplicative Compositional Policy (MCP). There are N = 8 Gaussian **primitives** πᵢ(a|s), multiplied together with weights from a gating network Wᵢ(s, z). Multiplicative mixing lets several primitives be active at once.

**Phase 2: transfer**
- **Goal-conditioned RL:** freeze the primitives, replace z with the goal g, and retrain **only the gating network** with SAC plus HER on a *sparse* success reward.
- **Multiple objects:** a transformer encoder (no positional embedding, pre-LayerNorm, average pooling) reads per-object tokens [robot state, object state, one-hot object ID, intention w]. Here w is a one-hot "which object I care about" vector. Because the encoder pools, it handles any number of objects: they trained with 4 and transferred to 6 and 8.
- **Multitask RL:** the gating network also gets a task one-hot. The SAC temperature α is learned *per task*, so exploration doesn't stop once the easy tasks are solved.

**Setup:** pre-training analysis in OpenAI Gym Fetch. Transfer used a custom MuJoCo UR3 with **end-effector position plus gripper actions (4-D, 20 Hz)**. Real pick-and-place ran on a UR3, zero-shot from sim. Observations are low-dimensional state (positions and velocities), not images.

## 3. Results
Most results are only in plots, so exact numbers aren't given in the text.
- **Exploration quality:** the proposed JSD+DADS reward gave the highest grasp ratio and object-position entropy, against MISC and MISC+DIAYN (DIAYN mostly pushed objects away). The primitives also stayed *diverse*, measured by the entropy of a Fisher-sensitivity score, instead of collapsing to one or two.
- **Primitives encode interaction:** even with a *random* gating network, the learned primitives keep the gripper near the object more often than baselines.
- **Goal-conditioned pick-and-place:** the paper claims **2–5× better sample efficiency** overall. It beats MI baselines, and for 4, 6, and 8 objects with distractors, *only this method succeeded* within 1M steps. Single-object, it beat SAC, MCP-from-scratch, and Composable SAC.
- **Real UR3 (zero-shot):** success was slightly below sim, but the ranking held. Some failures came from UR3 safety stops on jerky motions. A real 4-object sequential arranging task worked, with w chosen by a *scripted* rule.
- **Multitask (goal-randomized Meta-World subset, sparse reward):** pick-and-place with and without an obstacle, push, drawer and door open/close, and button press. It was faster than MT-SAC, multi-head MT-SAC, and PCGrad on every task. The gain was smallest on push, where the gripper stays closed.

## 4. Limitations
- **State-based only.** It assumes known object positions with no vision, so there is no perception gap in the real tests.
- Only object *position* is modeled, not orientation, and tasks are simple pick, place, push, and open/close.
- Real-world evaluation is limited: pick-and-place plus one scripted sequential demo, with no reported trial counts in the text.
- Numbers are mostly in plots. It is a 2022 paper, so the baselines are older (no comparison with goal-conditioned policies trained at scale).
- The high-level choice of *which object* (w) is scripted. Real HRL is left out.
- Future work they name: reset-free or offline RL for more practical real-world training.

## 5. How it relates to your SO-101 project
**Why it's relevant**
- **Very close action space.** UR3 with **end-effector xyz plus gripper at 20 Hz** is basically how SO-101 is usually controlled through IK. Their sim-to-real was zero-shot, which supports doing all RL in MuJoCo.
- **"One policy, many tasks" without one giant network.** Shared, frozen *interaction primitives* plus a cheap task or goal gate are a middle ground between three specialists (what the professor objected to) and one monolithic generalist (expensive). For T1 tube-to-rack and T3 pushing, one primitive set could be shared, with a small gate per task or goal. That gives one method across tasks.
- **Entity transformer with an intention vector.** This is the same design you sketched earlier (entity tokens plus a task token). Its "no positional embedding, average pooling" trick handles different object counts, which helps with puzzle pieces and multiple tubes.
- **Per-task learned SAC temperature** is a cheap fix if you go multi-task. It stops easy tasks from killing exploration on hard ones.

**How it fits with goal-conditioned, failure-guided learning**
- Their downstream stage is goal-conditioned RL with sparse reward and HER, so it is goal-conditioned too. But goals are just object *positions*, with no orientation, point tracks, or scene. Dex4D/SimToolReal-style object-pose or point goals are the stronger goal interface, and this paper's pre-training could **warm-start** that policy. Treat it as an optional "Phase 0", not the core.
- The failure-guided curriculum could act on the *gating* stage only. Real failures go to MuJoCo scenes, then you retrain the gate with the primitives frozen. That makes each retraining round much cheaper, which matters at roughly 150 real trials and limited compute. This would be a novel combination.

**Cautions**
- Unsupervised MI pre-training is finicky (MI estimators, skill collapse) and adds a whole extra training phase. With a semester deadline, it's a stretch goal at most.
- It has no perception. You'd still need your webcam object-tracking pipeline, and you'd have to model the SO-101's weak gripper and limited reach in sim.

**How to cite it:** as related work for *task-agnostic pre-training plus transfer to goal-conditioned and multitask manipulation*, and as the reason to share one skill backbone across tasks instead of training separate policies.
