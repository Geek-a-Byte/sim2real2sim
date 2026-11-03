# Efficient Parallel PPO for Generalized Pick and Place on SO-101 (QPAIN 2026): explained

**Title:** Efficient Parallel PPO-Based Reinforcement Learning for Generalized Pick and Place with Dense Reward Shaping
**Authors:** Md Eashir Arafat, Piyash Ahmed Kheyal, Arijit Das, Md. Firoj Ali. IEEE QPAIN 2026, DOI 10.1109/QPAIN69676.2026.11545619

> **Note:** I could only get the abstract (from your Elicit export). The full text is behind IEEE, which blocks my access, and I couldn't find an open copy. Upload the PDF if you want results and details checked.

## 1. The problem
The goal is to train an **SO-101** arm to do single-object pick-and-place with **randomized object and goal poses** using RL in sim, without programming each task.

## 2. Method (from the abstract)
- **PPO in Isaac Lab (Isaac Sim, PhysX 5)**, with **4,096 parallel envs on one RTX 5060 Ti (16 GB)**.
- **Privileged state observations:** joint positions and velocities, object pose, target pose, and the previous action. No cameras.
- **Staged dense reward:** reaching the end-effector, then lifting the object, then progressively tracking the goal, plus action-rate and joint-velocity penalties.
- The claimed novelty is combining staged dense reward shaping with large-scale parallel PPO.

## 3. Results
The abstract gives no numbers. It claims efficient training and generalization across randomized object and goal poses, with qualitative checks in Isaac Sim.

## 4. Limitations (inferred from the abstract)
- It is **sim-only**. There is no sim-to-real or real SO-101 evaluation mentioned, only "future extensions toward vision and sim-to-real".
- It uses privileged state, with no perception.
- Single object, single task type, and a heavily hand-shaped reward.
- A small regional conference paper, so treat its claims carefully until you've seen the numbers.

## 5. How it relates to your SO-101 project
- **It's the nearest *same-robot* precedent.** It shows that a goal-conditioned (random goal pose) PPO pick-and-place policy for SO-101 trains fine on a single consumer GPU at 4,096 envs. That supports the feasibility of your plan, and it's close to your T1 lift setup and T3's 4,096 MuJoCo envs.
- **Its reward structure is a usable template** for the privileged teacher: reach, then lift, then track the goal, plus smoothness penalties. The smoothness terms matter on SO-101's cheap servos.
- **It leaves open exactly what you're doing:** sim-to-real on the actual arm, perception from a webcam, multiple tasks or scene-aware goals, and learning from real failures. Cite it as "prior SO-101 RL stops at sim; we close the loop with real failures."
- **Differences to note:** Isaac Lab vs. your MuJoCo. It also uses goal *position/pose* only, while Dex4D-style point goals would be richer.
