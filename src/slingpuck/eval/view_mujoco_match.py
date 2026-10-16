"""3D MuJoCo viewer of a match: the SO-101 against a human opponent, face to face across the board.

The robot's pucks are black, the human's white. The robot guards the gate with its closed gripper and
slings while the human reloads; the human hand comes from the opponent's end. All pucks are in
the physics (physics/mujoco_match.py). Placeholder rule: the first side with no pucks on it wins.

Live window (on macOS the MuJoCo viewer must run under mjpython):
    mjpython -m src.slingpuck.eval.view_mujoco_match
GIF (any Python, offscreen rendering):
    python -m src.slingpuck.eval.view_mujoco_match --save match.gif --max-time 20
Cameras: overview (default, high over the robot's right side), opponent (the human's view).
"""
import argparse
import time

import mujoco
import numpy as np

from src.slingpuck.config import load_config
from src.slingpuck.physics.mujoco_match import MatchSim

BAND_RGBA = np.array([0.80, 0.12, 0.10, 1.0], np.float32)
BAND_WIDTH_M = 0.0012
STATE_TEXT = {"guard": "guards the gate", "fetch": "fetches and slings a puck", "idle": "waits",
              "reload": "reloads", "place": "places a puck on the band", "pull": "pulls back", "hold": "aims"}


def overview_camera(sim) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = sim.to_world([0.0, 0.0, 0.0])
    cam.distance, cam.azimuth, cam.elevation = 0.8, 120, -58   # Above the robot's right
    return cam


def add_bands(scene, sim: MatchSim):
    """Both bands; a V around the back of a puck that is pulled or being launched from behind a band line."""
    r, z = sim.r, sim.z_mid
    half = 0.5 * sim.cfg["band"]["anchor_span_m"]
    by = sim.band.band_y
    loaded = {"robot": None, "human": None}
    for side, who in (("robot", sim.robot), ("human", sim.human)):
        if who["held"] is not None:
            loaded[side] = who["held"]
    for rel in sim.releases:
        loaded[rel["side"]] = rel["p"]
    for side, sign in (("robot", 1.0), ("human", -1.0)):
        y_line = sign * (by - r)
        pts = [[-half, y_line]]
        p = loaded[side]
        if p is not None:
            pos = sim.puck_pos(p)
            if sign * pos[1] < sign * by * 1.0 and abs(pos[0]) < half and (by - sign * pos[1]) > 0:
                pts.append([pos[0], pos[1] - sign * r])
        pts.append([half, y_line])
        for a, b in zip(pts, pts[1:]):
            if scene.ngeom >= scene.maxgeom:
                return
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                                BAND_RGBA)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, BAND_WIDTH_M, sim.to_world([*a, z]),
                                 sim.to_world([*b, z]))
            scene.ngeom += 1


def status(sim: MatchSim):
    c = sim.counts()
    colours = {"robot": [0, 0], "human": [0, 0]}
    for p in sim.pucks:
        colours[sim.side_of(p)][0 if p["colour"] == "black" else 1] += 1
    title = f"Robot {STATE_TEXT.get(sim.robot['state'], sim.robot['state'])}  |  " \
            f"Human {STATE_TEXT.get(sim.human['state'], sim.human['state'])}"
    line = (f"t {sim.t:4.1f} s | robot side {c['robot']} ({colours['robot'][0]} black, {colours['robot'][1]} white)"
            f" | human side {c['human']} ({colours['human'][0]} black, {colours['human'][1]} white)"
            f" | shots {sim.shots['robot']} : {sim.shots['human']}")
    return title, line


def result_text(sim: MatchSim):
    if sim.winner is None:
        return "Time is up: no winner"
    return f"{'ROBOT' if sim.winner == 'robot' else 'HUMAN'} WINS (no pucks left on its side)"


def run(sim, on_frame, frame_dt, max_time):
    next_t = 0.0
    while sim.t < max_time and sim.winner is None:
        sim.step()
        if sim.t >= next_t:
            next_t += frame_dt
            on_frame(sim)
    on_frame(sim)


def live(args, sim):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        if args.camera == "overview":
            c = overview_camera(sim)
            viewer.cam.lookat[:], viewer.cam.distance = c.lookat, c.distance
            viewer.cam.azimuth, viewer.cam.elevation = c.azimuth, c.elevation
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = sim.model.camera(args.camera).id
        for k in range(max(args.games, 1) if args.games > 0 else 10 ** 9):
            if not viewer.is_running():
                break
            sim.rng = np.random.default_rng(args.seed + k)
            sim.reset()
            clock = {"wall": time.perf_counter(), "sim": 0.0}

            def on_frame(s):
                if not viewer.is_running():
                    raise KeyboardInterrupt
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    add_bands(viewer.user_scn, s)
                title, line = status(s)
                texts = [(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, title, None),
                         (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, line, None)]
                if s.winner is not None or s.t >= args.max_time:
                    texts.append((None, mujoco.mjtGridPos.mjGRID_TOPRIGHT, result_text(s), None))
                viewer.set_texts(texts)
                viewer.sync()
                wait = (s.t - clock["sim"]) / args.speed - (time.perf_counter() - clock["wall"])
                if wait > 0:
                    time.sleep(wait)

            try:
                run(sim, on_frame, args.frame_ms / 1000, args.max_time)
            except KeyboardInterrupt:
                break
            print(f"game {k + 1}: {result_text(sim)} at t {sim.t:.1f} s; shots robot {sim.shots['robot']}, "
                  f"human {sim.shots['human']}")
            time.sleep(args.pause)


def save_gif(args, sim):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=max(12, args.height // 34))
    renderer = mujoco.Renderer(sim.model, args.height, args.width)
    cam = overview_camera(sim) if args.camera == "overview" else args.camera
    frames = []
    sim.rng = np.random.default_rng(args.seed)
    sim.reset()

    def label(draw, xy, text, fill):
        box = draw.textbbox(xy, text, font=font)
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(255, 255, 255, 190))
        draw.text(xy, text, fill=fill, font=font)

    def on_frame(s):
        renderer.update_scene(s.data, camera=cam)
        add_bands(renderer.scene, s)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img, "RGBA")
        title, line = status(s)
        label(draw, (10, 8), title, (20, 20, 20))
        label(draw, (10, args.height - font.size - 12), line, (20, 20, 20))
        frames.append(img)

    run(sim, on_frame, args.frame_ms / 1000, args.max_time)
    label(ImageDraw.Draw(frames[-1], "RGBA"), (10, 14 + font.size + 8), result_text(sim), (190, 50, 15))
    frames[0].save(args.save, save_all=True, append_images=frames[1:] + [frames[-1]] * 15,
                   duration=int(round(args.frame_ms / args.speed)), loop=0)
    print(f"{result_text(sim)} at t {sim.t:.1f} s; shots robot {sim.shots['robot']}, human {sim.shots['human']}")
    print(f"Saved {args.save} ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params-version", default="latest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--games", type=int, default=1, help="Live: games in a row (0 = until the window closes)")
    p.add_argument("--max-time", type=float, default=60.0, help="Game time limit, s")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.5 = half speed)")
    p.add_argument("--camera", default="overview",
                   choices=["overview", "opponent"])
    p.add_argument("--frame-ms", type=float, default=33.0)
    p.add_argument("--pause", type=float, default=2.0, help="Live: pause on the result, s")
    p.add_argument("--save", help="Write a GIF instead of opening a window")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    args = p.parse_args()
    sim = MatchSim(load_config(args.params_version), seed=args.seed)
    if args.save:
        save_gif(args, sim)
    else:
        live(args, sim)


if __name__ == "__main__":
    main()
