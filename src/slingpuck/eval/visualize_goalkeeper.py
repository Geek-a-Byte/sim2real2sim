"""Top-down slow-motion viewer for the Phase 1 goalkeeper.

Shows what the sim does and what the policy sees:
- true puck (filled), latest camera frame (x), Kalman estimate (ring + velocity arrow)
- paddle on its arc, divider and gate, opponent band line
- time since launch, shot speed, paddle state (locked / free), outcome

Examples:
    # Trained run, live window
    python -m slingpuck.eval.visualize_goalkeeper --run logs/goalkeeper/goalkeeper_v0_s0_<stamp>
    # Scripted baseline, fast shot, save a GIF
    python -m slingpuck.eval.visualize_goalkeeper --policy center --speed 2.8 --save gk.gif
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import yaml

from slingpuck.config import load_config
from slingpuck.policies.scripted import CenterBlocker, HoldStart
from slingpuck.train.common import make_goalkeeper_env

# Reference palette (dataviz skill), light mode.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BOARD, WOOD_EDGE = "#f3ead9", "#b9a27c"
TRACK, PADDLE = "#2a78d6", "#eb6834"
OUTCOME_TEXT = {"save": "SAVE", "goal": "GOAL", "none": "no threat", None: ""}


def load_policy(args):
    if args.run:
        from stable_baselines3 import PPO
        run = Path(args.run)
        run_cfg = yaml.safe_load((run / "run_config.yaml").read_text())
        model = PPO.load(run / "final_model.zip", device="cpu")
        return model, run_cfg["train"]["policy"] == "asymmetric", run.name
    policy = {"center": CenterBlocker, "hold": HoldStart}[args.policy]()
    return policy, False, f"scripted: {args.policy}"


def record_episode(env, policy, seed, options, frame_dt, hold_s=0.3):
    """Run one episode and return snapshot dicts, one per frame_dt of sim time.

    The final state is repeated for hold_s, with the outcome shown.
    """
    gk = env.unwrapped
    frames, last_cam, next_t = [], [None], [0.0]

    def hook(e, cam_frames):
        if cam_frames:
            last_cam[0] = cam_frames[-1].xy.copy()
        if e.t + 1e-12 < next_t[0]:
            return
        next_t[0] = e.t + frame_dt
        est = e.tracker.estimate(e.t)
        released = e.t >= e.shot.prelaunch_s + e.release_delay - 1e-9
        frames.append({
            "t": e.t, "t_launch": e.shot.prelaunch_s, "released": released,
            "puck": e.sim.pos[0].copy(), "cam": None if last_cam[0] is None else last_cam[0].copy(),
            "est": None if est is None else (est[0].copy(), est[1].copy()),
            "pan": float(e.servo.pos), "outcome": None,
        })

    gk.substep_hook = hook
    obs, info = env.reset(seed=seed, options=options)
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    gk.substep_hook = None
    if frames:
        frames += [{**frames[-1], "outcome": info["outcome"]} for _ in range(max(1, int(hold_s / frame_dt)))]
    return frames, info


def draw_static(ax, gk):
    from matplotlib.patches import Rectangle
    cfg = gk.cfg
    hw, hl = 0.5 * cfg["board"]["width_m"], 0.5 * cfg["board"]["length_m"]
    ax.add_patch(Rectangle((-hw, -hl), 2 * hw, 2 * hl, facecolor=BOARD, edgecolor=WOOD_EDGE, linewidth=2))
    for xmin, xmax, ymin, ymax in gk.sim.board.divider_boxes():
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor=WOOD_EDGE, edgecolor="none"))
    band_y = hl - cfg["band"]["band_offset_from_end_wall_m"]
    ax.plot([-hw, hw], [band_y, band_y], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-hw - 0.004, band_y, "band", ha="right", va="center", fontsize=8, color=MUTED)
    pans = np.linspace(-gk.geom.pan_max, gk.geom.pan_max, 50)
    arc = np.array([gk.geom.paddle_center(p) for p in pans])
    ax.plot(arc[:, 0], arc[:, 1], color=GRID, linewidth=1)
    ax.text(0, -hl + 0.006, "agent side", ha="center", fontsize=8, color=MUTED)
    ax.text(0, hl - 0.012, "opponent side", ha="center", fontsize=8, color=MUTED)
    ax.set_xlim(-hw - 0.03, hw + 0.01)
    ax.set_ylim(-hl - 0.01, hl + 0.01)
    ax.set_aspect("equal")
    ax.axis("off")


def animate(episodes, gk, title, frame_dt, slowmo, save):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Polygon

    r = gk.cfg["puck"]["radius_m"]
    fig, ax = plt.subplots(figsize=(4.6, 7.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08)
    draw_static(ax, gk)
    puck = ax.add_patch(Circle((0, 0), r, facecolor=INK, edgecolor="none", zorder=4))
    est_ring = ax.add_patch(Circle((0, 0), r, facecolor="none", edgecolor=TRACK, linewidth=2, zorder=5))
    est_vel, = ax.plot([], [], color=TRACK, linewidth=2, zorder=5)
    cam_mark, = ax.plot([], [], marker="x", color=TRACK, markersize=7, markeredgewidth=1.5, zorder=6,
                        linestyle="none")
    paddle = ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, facecolor=PADDLE, edgecolor="none", zorder=3))
    header = fig.text(0.04, 0.955, title, fontsize=10, color=INK, va="top")
    status = fig.text(0.04, 0.92, "", fontsize=9, color=INK_2, va="top", family="monospace")
    result = fig.text(0.96, 0.955, "", ha="right", va="top", fontsize=16, color=INK, weight="bold")
    fig.text(0.04, 0.035, "filled = true puck   x = last camera frame   ring = Kalman estimate",
             fontsize=7.5, color=MUTED)

    flat = [(i, f, info) for i, (frames, info) in enumerate(episodes) for f in frames]

    def paddle_corners(pan):
        s = gk.geom.paddle_state(pan, 0.0, 0.5)
        c, sn = np.cos(s.angle), np.sin(s.angle)
        u, n = np.array([c, sn]), np.array([-sn, c])
        return np.array([s.center + a * s.half_width * u + b * s.half_thickness * n
                         for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

    def update(k):
        i, f, info = flat[k]
        puck.center = tuple(f["puck"])
        if f["est"] is not None:
            p, v = f["est"]
            est_ring.center = tuple(p)
            est_ring.set_visible(True)
            est_vel.set_data([p[0], p[0] + 0.03 * v[0]], [p[1], p[1] + 0.03 * v[1]])
        else:
            est_ring.set_visible(False)
            est_vel.set_data([], [])
        cam_mark.set_data(*(([f["cam"][0]], [f["cam"][1]]) if f["cam"] is not None else ([], [])))
        paddle.set_xy(paddle_corners(f["pan"]))
        paddle.set_alpha(1.0 if f["released"] else 0.45)
        t_rel = (f["t"] - f["t_launch"]) * 1000
        status.set_text(f"episode {i + 1}/{len(episodes)}   shot {info['shot_speed']:.2f} m/s\n"
                        f"t = {t_rel:+6.0f} ms from launch   paddle {'free' if f['released'] else 'locked'}")
        result.set_text(OUTCOME_TEXT[f["outcome"]])
        result.set_color(TRACK if f["outcome"] == "save" else PADDLE)
        return puck, est_ring, est_vel, cam_mark, paddle, status, result

    interval_ms = 1000 * frame_dt / slowmo
    anim = FuncAnimation(fig, update, frames=len(flat), interval=interval_ms, blit=False, repeat=not save)
    if save:
        anim.save(save, writer=PillowWriter(fps=max(1, round(1000 / interval_ms))), dpi=110)
        print(f"Saved {save} ({len(flat)} frames)")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="Training run dir with final_model.zip and run_config.yaml")
    src.add_argument("--policy", choices=["center", "hold"], help="Scripted baseline")
    parser.add_argument("--params-version", default="latest")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, help="Fix the shot speed (m/s)")
    parser.add_argument("--start-offset", type=float, help="Paddle start as a fraction of pan range, -1..1")
    parser.add_argument("--release-delay", type=float, default=0.0)
    parser.add_argument("--nominal", action="store_true", help="Disable domain randomization")
    parser.add_argument("--slowmo", type=float, default=0.1, help="Playback speed (0.1 = 10x slower)")
    parser.add_argument("--frame-ms", type=float, default=4.0, help="Sim time between frames")
    parser.add_argument("--pre-launch-ms", type=float, default=150.0, help="Show this much time before launch")
    parser.add_argument("--save", help="Write a GIF instead of opening a window")
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")
    config = load_config(args.params_version)
    policy, asymmetric, name = load_policy(args)
    env = make_goalkeeper_env(config, asymmetric)
    frame_dt = args.frame_ms / 1000

    episodes = []
    for k in range(args.episodes):
        options = {"randomize": not args.nominal, "release_delay": args.release_delay}
        if args.speed is not None:
            options["speed"] = args.speed
        if args.start_offset is not None:
            options["start_pan"] = args.start_offset * env.unwrapped.geom.pan_max
        frames, info = record_episode(env, policy, args.seed + k, options, frame_dt)
        frames = [f for f in frames if f["t"] >= f["t_launch"] - args.pre_launch_ms / 1000]
        episodes.append((frames, info))
        print(f"episode {k + 1}: speed {info['shot_speed']:.2f} m/s, start offset "
              f"{info['start_pan'] / env.unwrapped.geom.pan_max:+.2f}, outcome {info['outcome']}")
    animate(episodes, env.unwrapped, f"Goalkeeper - {name}", frame_dt, args.slowmo, args.save)


if __name__ == "__main__":
    main()
