"""3D MuJoCo viewer for the Phase 1 goalkeeper (MuJoCo physics, SO-101 arm).

Runs MujocoGoalkeeperEnv with a trained run or a scripted policy. The blue ring
is the Kalman estimate (what the policy sees); the small blue sphere is the last
camera frame.

Live window (on macOS the MuJoCo viewer must run under mjpython):
    mjpython -m src.slingpuck.eval.view_mujoco --run logs/goalkeeper/<run_dir>
GIF (any Python, offscreen rendering):
    python -m src.slingpuck.eval.view_mujoco --policy center --speed 2.4 --start-offset 0.9 --save gk3d.gif
Cameras: gate (default), side, top. In the live window, double-click a body to
track it, or press [ and ] to switch cameras.
"""
import argparse
import time

import mujoco
import numpy as np

from src.slingpuck.config import load_config
from src.slingpuck.eval.visualize_goalkeeper import OUTCOME_TEXT, load_policy

TRACK_RGBA = np.array([0.16, 0.47, 0.84, 0.55])
CAM_RGBA = np.array([0.16, 0.47, 0.84, 0.9])


def add_markers(scene: mujoco.MjvScene, env, last_cam):
    """Kalman estimate ring and last camera frame, drawn above the puck."""
    gk = env.unwrapped
    r = gk.cfg["puck"]["radius_m"]
    z = gk.cfg["puck"]["thickness_m"] + 0.002
    est = gk.tracker.estimate(gk.t)
    markers = []
    if est is not None:
        markers.append((mujoco.mjtGeom.mjGEOM_CYLINDER, [r, 0.0008, 0], [est[0][0], est[0][1], z], TRACK_RGBA))
    if last_cam is not None:
        markers.append((mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], [last_cam[0], last_cam[1], z], CAM_RGBA))
    for gtype, size, pos, rgba in markers:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype, np.array(size, float), np.array(pos, float),
                            np.eye(3).flatten(), rgba.astype(np.float32))
        scene.ngeom += 1


def episode_options(args, env, k):
    options = {"randomize": not args.nominal, "release_delay": args.release_delay}
    if args.speed is not None:
        options["speed"] = args.speed
    if args.start_offset is not None:
        options["start_pan"] = args.start_offset * env.unwrapped.geom.pan_max
    return options


def run_episode(env, policy, seed, options, on_frame, frame_dt, pre_launch_s):
    """Step one episode; call on_frame(env, last_cam) every frame_dt of sim time near the launch."""
    gk = env.unwrapped
    state = {"next": 0.0, "cam": None}

    def hook(e, frames):
        if frames:
            state["cam"] = frames[-1].xy.copy()
        if e.t + 1e-12 < state["next"] or e.t < e.shot.prelaunch_s - pre_launch_s:
            return
        state["next"] = e.t + frame_dt
        on_frame(e, state["cam"])

    gk.substep_hook = hook
    obs, info = env.reset(seed=seed, options=options)
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    gk.substep_hook = None
    return info, state["cam"]


def live(args, env, policy):
    import mujoco.viewer
    gk = env.unwrapped
    model, data = gk.mj.model, gk.mj.data
    frame_dt = args.frame_ms / 1000
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera(args.camera).id
        clock = {"wall": time.perf_counter(), "sim": None}

        def on_frame(e, last_cam):
            if not viewer.is_running():
                raise KeyboardInterrupt
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                add_markers(viewer.user_scn, env, last_cam)
            viewer.sync()
            if clock["sim"] is not None:
                wait = (e.t - clock["sim"]) / args.slowmo - (time.perf_counter() - clock["wall"])
                if wait > 0:
                    time.sleep(wait)
            clock["sim"], clock["wall"] = e.t, time.perf_counter()

        k = 0
        try:
            while viewer.is_running() and (args.episodes <= 0 or k < args.episodes):
                clock["sim"] = None
                info, _ = run_episode(env, policy, args.seed + k, episode_options(args, env, k), on_frame,
                                      frame_dt, args.pre_launch_ms / 1000)
                print(f"episode {k + 1}: speed {info['shot_speed']:.2f} m/s, start offset "
                      f"{info['start_pan'] / gk.geom.pan_max:+.2f}, outcome {info['outcome']}")
                viewer.sync()
                time.sleep(1.0)
                k += 1
        except KeyboardInterrupt:
            pass


def save_gif(args, env, policy):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=max(14, args.height // 28))
    gk = env.unwrapped
    renderer = mujoco.Renderer(gk.mj.model, args.height, args.width)
    frame_dt = args.frame_ms / 1000
    frames = []
    for k in range(args.episodes):
        ep_frames = []

        def on_frame(e, last_cam):
            renderer.update_scene(e.mj.data, camera=args.camera)
            add_markers(renderer.scene, env, last_cam)
            img = Image.fromarray(renderer.render())
            t_ms = (e.t - e.shot.prelaunch_s) * 1000
            ImageDraw.Draw(img).text((10, 8), f"shot {e.shot.speed:.2f} m/s   t = {t_ms:+5.0f} ms from launch",
                                     fill=(20, 20, 20), font=font)
            ep_frames.append(img)

        info, _ = run_episode(env, policy, args.seed + k, episode_options(args, env, k), on_frame,
                              frame_dt, args.pre_launch_ms / 1000)
        print(f"episode {k + 1}: speed {info['shot_speed']:.2f} m/s, start offset "
              f"{info['start_pan'] / gk.geom.pan_max:+.2f}, outcome {info['outcome']}")
        if ep_frames:
            last = ep_frames[-1].copy()
            ImageDraw.Draw(last).text((10, 12 + font.size), OUTCOME_TEXT[info["outcome"]], fill=(200, 60, 20),
                                      font=font)
            ep_frames += [last] * max(1, int(0.4 / frame_dt))
        frames += ep_frames
    duration = max(20, int(round(args.frame_ms / args.slowmo)))
    frames[0].save(args.save, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"Saved {args.save} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="Training run dir with final_model.zip and run_config.yaml")
    src.add_argument("--policy", choices=["center", "hold"], help="Scripted baseline")
    parser.add_argument("--params-version", default="latest")
    parser.add_argument("--episodes", type=int, default=3, help="Live: 0 = run until the window closes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--start-offset", type=float, help="Paddle start as a fraction of pan range, -1..1")
    parser.add_argument("--release-delay", type=float, default=0.0)
    parser.add_argument("--nominal", action="store_true", help="Disable domain randomization")
    parser.add_argument("--camera", default="gate", choices=["gate", "side", "top"])
    parser.add_argument("--slowmo", type=float, default=0.1, help="Playback speed (0.1 = 10x slower)")
    parser.add_argument("--frame-ms", type=float, default=4.0)
    parser.add_argument("--pre-launch-ms", type=float, default=150.0)
    parser.add_argument("--save", help="Write a GIF instead of opening a window")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    from src.slingpuck.train.common import make_goalkeeper_env
    config = load_config(args.params_version)
    policy, asymmetric, _ = load_policy(args)
    env = make_goalkeeper_env(config, asymmetric, backend="mujoco")
    if args.save:
        save_gif(args, env, policy)
    else:
        live(args, env, policy)


if __name__ == "__main__":
    main()
