"""3D MuJoCo viewer for the Phase 2 sling shot (SO-101 arm, band, puck, gate).

Each episode:
1. SlingEnv (2D) runs the policy: step 1 gives the slide and the pull, step 2
   (if correction_step) gives the corrected pull.
2. MuJoCo replays the shot with the arm: the gripper moves down and closes on the
   puck, slides it along the band, pulls it back against the band, holds it while
   the camera looks, corrects, and releases. The band pushes the puck through the
   band line and MuJoCo flies it to the gate.

Drawn: the band (red; a V around the back of the puck while it is pulled), the
opponent band, the band force on the puck (green arrow, 1 cm per N), and the
2D prediction from the same release point (orange dots), so you can compare
MuJoCo with the 2D model.

Live window (on macOS the MuJoCo viewer must run under mjpython):
    mjpython -m src.slingpuck.eval.view_mujoco_sling --run logs/sling/<run_dir>
GIF (any Python, offscreen rendering):
    python -m src.slingpuck.eval.view_mujoco_sling --policy center_correct --save sling3d.gif
Cameras: sling (default), band (close-up of the grasp), shot, gate, side, top. In the live window, press [ and ]
to switch cameras.
"""
import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from src.slingpuck.config import load_config
from src.slingpuck.envs.sling_env import launch_and_fly
from src.slingpuck.physics.mujoco_sling import MujocoSlingSim, ShotPlan

BAND_RGBA = np.array([0.80, 0.12, 0.10, 1.0], np.float32)
FORCE_RGBA = np.array([0.10, 0.70, 0.25, 1.0], np.float32)
GHOST_RGBA = np.array([0.95, 0.55, 0.10, 0.55], np.float32)
BAND_WIDTH_M = 0.0012
FORCE_M_PER_N = 0.01
FAST_PHASES = ("release", "flight", "done")
PHASE_TEXT = {
    "idle": "Arm at home: the camera sees the puck on the band",
    "approach": "Arm above the puck",
    "reach": "Arm moves down to the puck",
    "grasp": "Gripper closes on the puck",
    "slide": "Slide the puck along the band",
    "pull": "Pull back against the band",
    "look": "Camera looks at the pulled puck",
    "correct": "Correction from the camera",
    "hold": "Hold",
    "release": "Release: the band pushes the puck",
    "flight": "Puck slides to the gate",
    "retreat": "Arm goes back home",
    "done": "Result",
}


def load_sling_policy(args, config):
    if args.run:
        from src.slingpuck.train.common import load_trained_run
        model, asymmetric, _ = load_trained_run(args.run)
        return model, asymmetric, Path(args.run).name
    from src.slingpuck.policies.sling_scripted import SlideCenter, SlideCenterCorrect
    policy = SlideCenter() if args.policy == "center" else SlideCenterCorrect(config)
    return policy, False, f"scripted: {args.policy}"


def plan_shot(env, policy, seed, options):
    """Run the 2D episode. Returns (ShotPlan, 2D env info, randomized cfg, band)."""
    obs, info = env.reset(seed=seed, options=options)
    se = env.unwrapped
    action, _ = policy.predict(obs, deterministic=True)
    obs, _, terminated, _, info = env.step(action)
    slide_x, pull = se.slide_x, se.puck_pos.copy()
    corrected = None
    if not terminated:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, _, info = env.step(action)
        corrected = se.pull.pos.copy()
    return ShotPlan(se.x0, slide_x, pull, corrected), info, se.cfg, se.band


def _add(scene, gtype, size, pos, rgba, mat=None):
    if scene.ngeom >= scene.maxgeom:
        return None
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.array(size, float), np.array(pos, float),
                        np.eye(3).flatten() if mat is None else mat, rgba)
    scene.ngeom += 1
    return g


def _line(scene, gtype, a, b, width, rgba):
    g = _add(scene, gtype, [0, 0, 0], [0, 0, 0], rgba)
    if g is not None:
        mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))


def add_markers(scene, sim: MujocoSlingSim, ghost=None):
    """Bands, band force arrow and the 2D prediction."""
    cfg = sim.cfg
    r = cfg["puck"]["radius_m"]
    z = 0.5 * cfg["puck"]["thickness_m"]
    half = 0.5 * cfg["band"]["anchor_span_m"]
    y_band = sim.band.band_y - r
    left, right = np.array([-half, y_band, z]), np.array([half, y_band, z])
    vertex = sim.band_vertex()
    if vertex is None:
        _line(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, left, right, BAND_WIDTH_M, BAND_RGBA)
    else:
        v = np.array([vertex[0], vertex[1], z])
        _line(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, left, v, BAND_WIDTH_M, BAND_RGBA)
        _line(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, v, right, BAND_WIDTH_M, BAND_RGBA)
    _line(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, [-half, -y_band, z], [half, -y_band, z], BAND_WIDTH_M, BAND_RGBA)
    f = sim.band_force
    if np.linalg.norm(f) > 0.05:
        p = sim.puck_pos
        a = np.array([p[0], p[1], 2.2 * z])
        _line(scene, mujoco.mjtGeom.mjGEOM_ARROW, a, a + FORCE_M_PER_N * np.array([f[0], f[1], 0.0]), 0.0025,
              FORCE_RGBA)
    if ghost is not None:
        for _, x, y in ghost:
            _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0015, 0, 0], [x, y, 0.002], GHOST_RGBA)


def ghost_points(cfg, band, pull_pos, every_m=0.008):
    """2D prediction (no release noise) from pull_pos, thinned to one point per every_m of path."""
    fl = launch_and_fly(cfg, band, pull_pos, None, record=True, full=True)
    pts, last = [], None
    for row in fl.trajectory:
        if last is None or np.hypot(row[1] - last[1], row[2] - last[2]) >= every_m:
            pts.append(row)
            last = row
    return np.array(pts), fl


def status_text(sim: MujocoSlingSim):
    pull_mm = 1000 * max(sim.band.band_y - sim.puck_pos[1], 0.0)
    return (PHASE_TEXT.get(sim.phase, sim.phase),
            f"t {sim.t:5.2f} s   band force {np.linalg.norm(sim.band_force):4.1f} N   "
            f"pull {pull_mm:4.1f} mm   speed {np.linalg.norm(sim.puck_vel):4.2f} m/s")


def outcome(success, miss_m, rebounded=False):
    if success:
        return "THROUGH THE GATE"
    if miss_m >= 0.05:
        return "SHORT"
    return f"MISS ({1000 * miss_m:.1f} mm off center{', bounced back' if rebounded else ''})"


def result_lines(res, fl2d, env_info):
    return [f"MuJoCo:             {outcome(res.success, res.miss_m, res.rebounded)}",
            f"2D, same release:   {outcome(fl2d.success, fl2d.miss_m)}",
            f"2D env (w/ noise):  {outcome(env_info['success'], env_info['miss_m'])}"]


def print_episode(k, plan, res, fl2d, env_info):
    dev = 1000 * np.linalg.norm(res.release_pos - res.release_target)
    print(f"episode {k + 1}: x0 {1000 * plan.x0:+.1f} mm, slide {1000 * plan.slide_x:+.1f} mm, "
          f"max band force {res.max_band_force_n:.2f} N, held-puck error at release {dev:.2f} mm")
    if res.exit_vel is not None:
        print(f"  exit speed: MuJoCo {np.linalg.norm(res.exit_vel):.3f} m/s, "
              f"2D {np.linalg.norm(fl2d.exit_vel):.3f} m/s")
    for line in result_lines(res, fl2d, env_info):
        print("  " + line)


def episodes(args, config):
    """Yield (k, plan, cfg, band, env_info) for each episode."""
    from src.slingpuck.train.common import make_sling_env
    policy, asymmetric, _ = load_sling_policy(args, config)
    env = make_sling_env(config, asymmetric)
    k = 0
    while args.episodes <= 0 or k < args.episodes:
        options = {"randomize": not args.nominal}
        if args.x0 is not None:
            options["x0"] = args.x0
        plan, info, cfg, band = plan_shot(env, policy, args.seed + k, options)
        yield k, plan, cfg, band, info
        k += 1


def frame_dt(args):
    return lambda phase: args.fast_frame_ms / 1000 if phase in FAST_PHASES else args.frame_ms / 1000


def rate(args, phase):
    return args.slowmo if phase in FAST_PHASES else args.arm_rate


def live(args, config, sim):
    import mujoco.viewer
    model, data = sim.model, sim.data
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera(args.camera).id
        for k, plan, cfg, band, info in episodes(args, config):
            if not viewer.is_running():
                break
            sim.apply_params(cfg)
            final = plan.corrected if plan.corrected is not None else plan.pull
            ghost, fl2d = ghost_points(cfg, band, final)
            clock = {"wall": None, "sim": 0.0}
            extra = []

            def on_frame(s):
                if not viewer.is_running():
                    raise KeyboardInterrupt
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    add_markers(viewer.user_scn, s, None if args.no_ghost else ghost)
                title, values = status_text(s)
                viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, title, None),
                                  (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                   values, None)] + extra)
                viewer.sync()
                now = time.perf_counter()
                if clock["wall"] is not None:
                    wait = (s.t - clock["sim"]) / rate(args, s.phase) - (now - clock["wall"])
                    if wait > 0:
                        time.sleep(wait)
                clock["sim"], clock["wall"] = s.t, time.perf_counter()

            try:
                res = sim.run(plan, on_frame, frame_dt(args), observe_s=cfg["sling"]["observe_window_s"],
                              max_flight_s=cfg["sling"]["max_flight_s"])
            except KeyboardInterrupt:
                break
            print_episode(k, plan, res, fl2d, info)
            extra.append((None, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "\n".join(result_lines(res, fl2d, info)), None))
            on_frame(sim)
            time.sleep(args.pause)


def _label(draw, xy, text, fill, font):
    """Text on a light box, so it stays readable over the arm."""
    box = draw.textbbox(xy, text, font=font)
    draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(255, 255, 255, 190))
    draw.text(xy, text, fill=fill, font=font)


def save_gif(args, config, sim):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=max(13, args.height // 32))
    renderer = mujoco.Renderer(sim.model, args.height, args.width)
    frames, durations = [], []
    for k, plan, cfg, band, info in episodes(args, config):
        sim.apply_params(cfg)
        final = plan.corrected if plan.corrected is not None else plan.pull
        ghost, fl2d = ghost_points(cfg, band, final)
        ep = []  # (image, sim time, phase)

        def on_frame(s):
            renderer.update_scene(s.data, camera=args.camera)
            add_markers(renderer.scene, s, None if args.no_ghost else ghost)
            img = Image.fromarray(renderer.render())
            title, values = status_text(s)
            draw = ImageDraw.Draw(img, "RGBA")
            _label(draw, (10, 8), title, (20, 20, 20), font)
            _label(draw, (10, args.height - font.size - 12), values, (20, 20, 20), font)
            ep.append((img, s.t, s.phase))

        res = sim.run(plan, on_frame, frame_dt(args), observe_s=cfg["sling"]["observe_window_s"],
                      max_flight_s=cfg["sling"]["max_flight_s"])
        print_episode(k, plan, res, fl2d, info)
        draw = ImageDraw.Draw(ep[-1][0], "RGBA")
        for i, line in enumerate(result_lines(res, fl2d, info)):
            _label(draw, (10, 14 + (i + 1) * (font.size + 6)), line, (190, 50, 15), font)
        for (img, t, phase), (_, t_next, _) in zip(ep, ep[1:]):
            frames.append(img)
            durations.append(max(20, int(round(1000 * (t_next - t) / rate(args, phase)))))
        frames.append(ep[-1][0])
        durations.append(int(1000 * args.pause))
    frames[0].save(args.save, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"Saved {args.save} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="Sling training run dir with final_model.zip and run_config.yaml")
    src.add_argument("--policy", choices=["center", "center_correct"], help="Scripted baseline")
    parser.add_argument("--params-version", default="latest")
    parser.add_argument("--episodes", type=int, default=3, help="Live: 0 = run until the window closes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x0", type=float, help="Puck start x on the band, m (default: random)")
    parser.add_argument("--nominal", action="store_true", help="Disable domain randomization")
    parser.add_argument("--camera", default="sling", choices=["sling", "band", "shot", "gate", "side", "top"])
    parser.add_argument("--arm-rate", type=float, default=1.0, help="Playback speed of the arm motion (1 = real time)")
    parser.add_argument("--slowmo", type=float, default=0.1, help="Playback speed of the release and flight")
    parser.add_argument("--frame-ms", type=float, default=20.0, help="Frame interval during the arm motion")
    parser.add_argument("--fast-frame-ms", type=float, default=2.0, help="Frame interval during the release/flight")
    parser.add_argument("--pause", type=float, default=1.5, help="Pause on the result, s")
    parser.add_argument("--no-ghost", action="store_true", help="Do not draw the 2D prediction")
    parser.add_argument("--save", help="Write a GIF instead of opening a window")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    config = load_config(args.params_version)
    sim = MujocoSlingSim(config)
    if args.save:
        save_gif(args, config, sim)
    else:
        live(args, config, sim)


if __name__ == "__main__":
    main()
