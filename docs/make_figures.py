"""Make the report figures that are not written by the eval scripts.

    python docs/make_figures.py

Writes to docs/figures/: training curves (from tensorboard_logs/), the block
primitive table, the servo step responses (2D model vs MuJoCo arm), the band
hysteresis loop, and frames from the 2D and 3D viewers.
"""
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "docs" / "figures"

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEEDS = ["#2a78d6", "#eb6834", "#1baf7a"]  # Categorical slots 1-3 (validated)
BLUES = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#86b6ef", "#2a78d6", "#184f95", "#0d2f5c"])
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.titlelocation": "left", "axes.titlesize": 11})


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def scalars(run_dir, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def smooth(y, k=9):
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def training_curves(task, rate_tag, rate_label, out):
    runs = sorted(glob.glob(str(ROOT / "tensorboard_logs" / task / "*")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=SURFACE)
    for ax, tag, label in ((axes[0], rate_tag, rate_label), (axes[1], "rollout/ep_rew_mean", "Mean episode reward")):
        style(ax)
        for i, run in enumerate(runs):
            seed = Path(run).name.split("_s")[1][0]
            x, y = scalars(run, tag)
            ax.plot(x / 1e3, y, color=SEEDS[i], alpha=0.25, linewidth=1)
            ax.plot(x / 1e3, smooth(y), color=SEEDS[i], linewidth=2, label=f"seed {seed}")
        ax.set_xlabel("Training steps (thousands)")
        ax.set_title(label)
        if "rate" in tag:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"{task.capitalize()} PPO training, per rollout (thin) and 9-rollout moving average (thick)",
                 x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def original_training():
    run = glob.glob(str(ROOT / "tensorboard_logs" / "ppo_goalkeeper_*" / "events*"))
    if not run:
        return
    x, y = scalars(Path(run[0]).parent, "rollout/ep_rew_mean")
    fig, ax = plt.subplots(figsize=(6.5, 3.4), facecolor=SURFACE)
    style(ax)
    ax.plot(x / 1e3, y, color=SEEDS[0], linewidth=2)
    ax.set_xlabel("Training steps (thousands)")
    ax.set_title("Mean episode reward")
    fig.suptitle("Original code (before the rework): goalkeeper PPO, one seed", x=0.01, ha="left", fontsize=9,
                 color=INK_2)
    fig.tight_layout()
    fig.savefig(FIG / "train_original.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def block_table():
    from src.slingpuck.config import load_config
    from src.slingpuck.primitives.block import load_or_build_block_table
    t = load_or_build_block_table(load_config("v0"), verbose=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    style(ax, grid=None)
    im = ax.imshow(t.p_edge, origin="lower", aspect="auto", cmap=BLUES, vmin=0, vmax=1,
                   extent=[t.delays[0] * 1000 - 12.5, t.delays[-1] * 1000 + 12.5, t.speeds[0] - 0.1, t.speeds[-1] + 0.1])
    for i, s in enumerate(t.speeds):
        for j, d in enumerate(t.delays):
            v = t.p_edge[i, j]
            if v >= 0.05:
                ax.text(d * 1000, s, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if v > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, format=matplotlib.ticker.PercentFormatter(1.0))
    cb.set_label("Save probability")
    ax.set_xlabel("Arm delay after the launch (ms)")
    ax.set_ylabel("Shot speed (m/s)")
    ax.set_title("Paddle starting at the edge of its range (just back from the band)")
    fig.suptitle(f"Frozen block primitive table (Phase 1 env, scripted center blocker). "
                 f"Paddle settled at the gate: {t.p_gate.min():.0%} at every speed.",
                 x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(FIG / "block_table.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def servo_steps():
    from src.slingpuck.config import load_config
    from src.slingpuck.physics.mujoco_goalkeeper import MujocoGoalkeeperSim
    from src.slingpuck.physics.servo_model import ServoModel
    c = load_config("v0")
    sim = MujocoGoalkeeperSim(c)
    start = np.deg2rad(20)
    t = np.arange(1, 401) * 1e-3
    servo_cfg = dict(c["servo"], command_latency_s=0.0)
    two_d = ServoModel(servo_cfg, 1e-3, start)
    p2 = np.array([float(two_d.step(0.0)) for _ in t])
    sim.reset_arm(start)
    sim.reset((0.0, 0.12), (0, 0))
    sim.set_pan_target(0.0)
    pm = []
    for _ in t:
        sim.step(1e-3)
        pm.append(sim.pan)
    pm = np.array(pm)
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    style(ax)
    ax.plot(t * 1000, np.rad2deg(p2), color=SEEDS[0], linewidth=2, label="2D ServoModel (first-order lag + 300 deg/s limit)")
    ax.plot(t * 1000, np.rad2deg(pm), color=SEEDS[1], linewidth=2, label="MuJoCo SO-101 (STS3215 actuator, arm inertia)")
    ax.axhline(10, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(395, 10.6, "50% of the move", ha="right", fontsize=8, color=MUTED)
    ax.set_xlabel("Time after the command (ms), no command latency")
    ax.set_ylabel("Pan angle (deg)")
    ax.set_title("Pan step from +20 deg to 0 deg (paddle mounted)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "servo_step.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def band_loop():
    import pandas as pd
    from src.slingpuck.config import load_config
    from src.slingpuck.physics.band_model import BandModel
    from src.slingpuck.sysid.synthetic import DEFAULT_TRUTH, truth_config
    df = pd.read_csv(ROOT / "data" / "real_logs" / "synthetic_example" / "band_bench.csv")
    cfg = truth_config(load_config("v0"), DEFAULT_TRUTH)
    band = BandModel.from_config(cfg)
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    style(ax)
    top = df[df.cycle == df.cycle.max()].pull_m.max()
    d = np.linspace(0, top, 200)
    s_max = band.elongation((0.0, band.band_y - top))
    load = [float(band.tension_load(band.elongation((0, band.band_y - x)))) * band.direction_sum((0, band.band_y - x))[1]
            for x in d]
    unload = [float(band.tension_unload(band.elongation((0, band.band_y - x)), s_max)) *
              band.direction_sum((0, band.band_y - x))[1] for x in d]
    ax.fill_between(d * 1000, unload, load, color=SEEDS[0], alpha=0.12, linewidth=0)
    ax.plot(d * 1000, load, color=SEEDS[0], linewidth=2, label="Model: loading (pull back)")
    ax.plot(d * 1000, unload, color=SEEDS[1], linewidth=2, label="Model: unloading (release)")
    for phase, marker, color in (("load", "o", SEEDS[0]), ("unload", "^", SEEDS[1])):
        sel = df[(df.phase == phase) & (df.cycle == df.cycle.max())]
        ax.plot(sel.pull_m * 1000, sel.force_n, marker, color=color, markersize=4, alpha=0.7,
                markeredgecolor=SURFACE, label=f"Synthetic gauge readings: {phase}")
    ax.text(0.55 * top * 1000, 0.5 * max(load), "loop area =\nenergy lost", fontsize=8, color=INK_2)
    ax.set_xlabel("Center pull behind the band line (mm)")
    ax.set_ylabel("Band force on the puck (N)")
    ax.set_title("Band force: closed hysteresis loop (synthetic 'true' band)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "band_loop.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def viewer_frames():
    from PIL import Image
    from src.slingpuck.config import load_config
    from src.slingpuck.envs.goalkeeper_env import GoalkeeperEnv
    from src.slingpuck.eval.visualize_goalkeeper import animate, record_episode
    from src.slingpuck.policies.scripted import CenterBlocker
    tmp = ROOT / "docs" / "report" / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env = GoalkeeperEnv(load_config("v0"))
    frames, info = record_episode(env, CenterBlocker(), seed=3,
                                  options={"speed": 1.5, "start_pan": 0.9 * env.geom.pan_max, "randomize": False},
                                  frame_dt=0.004)
    frames = [f for f in frames if f["t"] >= f["t_launch"] - 0.05]
    animate([(frames, info)], env, "Goalkeeper - scripted: center", 0.004, 1.0, str(tmp / "gk2d.gif"))
    gif = Image.open(tmp / "gk2d.gif")
    picks = [int(gif.n_frames * f) for f in (0.15, 0.55, 0.98)]
    ims = []
    for k in picks:
        gif.seek(k)
        ims.append(gif.convert("RGB"))
    w, h = ims[0].size
    sheet = Image.new("RGB", (w * 3, h), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, (i * w, 0))
    sheet.save(FIG / "viewer_2d.png")
    # 3D MuJoCo frames
    import mujoco
    from src.slingpuck.envs.mujoco_goalkeeper_env import MujocoGoalkeeperEnv
    from src.slingpuck.eval.view_mujoco import add_markers, run_episode
    menv = MujocoGoalkeeperEnv(load_config("v0"))
    r = mujoco.Renderer(menv.mj.model, 300, 400)
    shots = []

    def on_frame(e, last_cam):
        r.update_scene(e.mj.data, camera="gate")
        add_markers(r.scene, menv, last_cam)
        shots.append(Image.fromarray(r.render()))

    run_episode(menv, CenterBlocker(), 3, {"speed": 1.5, "start_pan": 0.9 * menv.geom.pan_max, "randomize": False},
                on_frame, 0.02, 0.1)
    picks = [shots[int(len(shots) * f)] for f in (0.1, 0.45, 0.75, 0.99)]
    sheet = Image.new("RGB", (400 * 4, 300), "white")
    for i, im in enumerate(picks):
        sheet.paste(im, (i * 400, 0))
    sheet.save(FIG / "viewer_3d.png")


def main():
    training_curves("goalkeeper", "goalkeeper/save_rate", "Save rate (training episodes)", "train_goalkeeper.png")
    training_curves("sling", "sling/success_rate", "Gate success rate (training episodes)", "train_sling.png")
    original_training()
    block_table()
    servo_steps()
    band_loop()
    viewer_frames()
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
