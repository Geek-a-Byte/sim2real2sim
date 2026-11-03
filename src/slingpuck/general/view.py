"""3D MuJoCo viewer of the goal-conditioned env: the SO-101 moves an object to its goal (green ghost).

With --model, a trained policy acts. Without it, a scripted IK pick-and-place runs (approach, grasp,
carry to the goal, release), to show the scene and the physics before a policy exists.

Live window (on macOS the MuJoCo viewer must run under mjpython):
    mjpython -m src.slingpuck.general.view                                   # scripted, random tasks
    mjpython -m src.slingpuck.general.view --object bar --episodes 0         # until the window closes
    mjpython -m src.slingpuck.general.view --model logs/general/<run>/final_model.zip
GIF (any Python, offscreen rendering):
    python -m src.slingpuck.general.view --save goal.gif --episodes 2
"""
import argparse
import time

import mujoco
import numpy as np

from src.slingpuck.general.common import load_general_config
from src.slingpuck.general.goal_env import GoalEnv
from src.slingpuck.general.reachability import SEED_POSES, approach_dir, load_or_build_map


class ScriptedPickPlace:
    """IK waypoints for one goal: above the object, down, close, lift, over the goal, down, open."""

    def __init__(self, env: GoalEnv):
        self.env, self.kin = env, env.checker.kin
        self.plan, self.q = [], env.d.qpos[env.qadr[:5]].copy()

    def _ik(self, p, jaw):
        p = np.array(p, float)
        for tilt in (0.0, 30.0, 60.0):
            for q0 in (self.q, *SEED_POSES):
                ok, q, *_ = self.kin.ik(p, approach_dir(p, tilt), jaw, q0=q0)
                if ok:
                    self.q = q
                    return q
        return self.q

    def make_plan(self):
        env = self.env
        pos, yaw = env.object_pose()
        goal = np.array(env.goal[:3])
        jaw = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        width = env.shape.width_along(jaw, yaw)
        g_open = self.kin.gripper_for_opening(width + 0.03)
        g_close = self.kin.gripper_for_opening(max(width - 0.01, 0.004))
        z_grasp = max(pos[2], 0.01)
        steps = [((pos[0], pos[1], z_grasp + 0.05), g_open, 15), ((pos[0], pos[1], z_grasp), g_open, 15),
                 ((pos[0], pos[1], z_grasp), g_close, 10), ((pos[0], pos[1], z_grasp + 0.05), g_close, 15),
                 ((goal[0], goal[1], goal[2] + 0.04), g_close, 20), (tuple(goal), g_close, 15),
                 (tuple(goal), g_open, 10)]
        self.plan = [(self._ik(p, jaw), g, n) for p, g, n in steps]

    def step(self):
        """One control step. Returns False when the plan is done."""
        if not self.plan:
            return False
        q, g, n = self.plan[0]
        env = self.env
        env.d.ctrl[:5], env.d.ctrl[5] = q, g
        mujoco.mj_step(env.m, env.d, nstep=env.n_sub)
        self.plan[0] = (q, g, n - 1)
        if n - 1 <= 0:
            self.plan.pop(0)
        return True


def episodes(env, model, n_episodes, seed, max_steps, on_frame):
    """Run episodes and call on_frame(env, text) after every control step."""
    k = 0
    while n_episodes <= 0 or k < n_episodes:
        options = {"task": env.random_task()} if env.task_pool is None else None
        obs, info = env.reset(seed=seed + k, options=options)
        script = None if model else ScriptedPickPlace(env)
        if script:
            script.make_plan()
        text = f"episode {k + 1}: {env.task.obj}, {'policy' if model else 'scripted IK'}"
        for step in range(max_steps):
            if model:
                obs, _, terminated, truncated, info = env.step(model.predict(obs, deterministic=True)[0])
                if terminated or truncated:
                    break
            elif not script.step():
                break
            d = float(np.linalg.norm(env._object_points() - env._goal_points(), axis=1).mean())
            on_frame(env, f"{text}  |  goal distance {100 * d:.1f} cm  |  goals reached {env.goals_reached}")
        d = float(np.linalg.norm(env._object_points() - env._goal_points(), axis=1).mean())
        print(f"episode {k + 1}: {env.task.obj}, final goal distance {100 * d:.1f} cm, "
              f"goals reached {env.goals_reached}")
        k += 1


def live(args, env, model):
    import mujoco.viewer
    dt = env.cfg["robot"]["control_dt_s"] / args.speed
    with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
        viewer.cam.lookat[:], viewer.cam.distance = [0.25, 0.0, 0.05], 0.75
        viewer.cam.azimuth, viewer.cam.elevation = 135, -30

        def on_frame(e, text):
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, None)])
            viewer.sync()
            time.sleep(dt)

        try:
            episodes(env, model, args.episodes, args.seed, args.max_steps, on_frame)
        except KeyboardInterrupt:
            pass


def save_gif(args, env, model):
    from PIL import Image, ImageDraw

    frames = []

    def on_frame(e, text):
        img = Image.fromarray(e.render())
        draw = ImageDraw.Draw(img, "RGBA")
        box = draw.textbbox((10, 8), text)
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(255, 255, 255, 200))
        draw.text((10, 8), text, fill=(20, 20, 20))
        frames.append(img)

    episodes(env, model, max(args.episodes, 1), args.seed, args.max_steps, on_frame)
    ms = int(round(1000 * env.cfg["robot"]["control_dt_s"] / args.speed))
    frames[0].save(args.save, save_all=True, append_images=frames[1:], duration=ms, loop=0)
    print(f"Saved {args.save} ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Trained policy (final_model.zip); default: scripted IK pick-and-place")
    p.add_argument("--object", nargs="*", help="Object names from configs/general.yaml (default: all)")
    p.add_argument("--episodes", type=int, default=3, help="Live: 0 = until the window closes")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.5 = half speed)")
    p.add_argument("--student", action="store_true", help="Masked, noisy points (webcam view) for the policy")
    p.add_argument("--save", help="Write a GIF instead of opening a window")
    args = p.parse_args()

    cfg = load_general_config()
    env = GoalEnv(cfg, student=args.student, reach_map=load_or_build_map(cfg), objects=args.object,
                  render_mode="rgb_array")
    model = None
    if args.model:
        from stable_baselines3 import PPO

        from src.slingpuck.general.policy import PairedPointPolicy
        model = PPO.load(args.model, custom_objects={"policy_class": PairedPointPolicy})
    if args.save:
        save_gif(args, env, model)
    else:
        live(args, env, model)


if __name__ == "__main__":
    main()
