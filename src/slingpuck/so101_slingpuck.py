"""SO-101 arm with the sling puck board in the lightbox workspace, in MuJoCo: the scene of this project.

    python -m src.slingpuck.so101_slingpuck               # opens the interactive viewer
    python -m src.slingpuck.so101_slingpuck --save        # also writes assets/mujoco/robotstudio_so101/so101_slingpuck.xml
    python -m src.slingpuck.so101_slingpuck --no-viewer   # build and check the scene only (e.g. on a headless machine)

Same structure as so101_lightbox.py (the course's lightbox script), for the sling puck use case. The SO-101
model is the RobotStudio MJCF already in assets/ (no download).

Frame: origin at the robot base, on the floor it stands on. x forward (away from the robot), y the robot's
left, z up. Meters. The lightbox, mat and C920 are the measured values in src/slingpuck/workspace.py. For the
game the lightbox has no back wall: the human opponent sits at the open far end, face to face with the robot,
and the longer side walls keep the game between them.

The board lies on the mat, lengthwise along x, centered left to right, with the robot behind the agent end
wall. Its sizes, the bands and the pucks come from configs/physics_params_v<N>.yaml, so this scene always
matches the simulators. The rest of the project uses the BOARD frame (origin at the board center, y toward
the opponent, agent half y < 0); board_to_robot() converts. In the viewer, drag a puck with Ctrl + right
mouse to flick it: the pucks slide with Coulomb friction and bounce off the walls, the divider and each other.
"""
import argparse
import math

import mujoco
import mujoco.viewer
import numpy as np

from src.slingpuck.config import load_config
from src.slingpuck.physics.band_model import BandModel
from src.slingpuck.physics.mujoco_goalkeeper import SO101_XML, WALL_PAIRS, add_board
from src.slingpuck.physics.mujoco_sling import add_band_pegs
from src.slingpuck.workspace import BASE_AXIS_X, FAR_X, add_lightbox, play_surface_z

GRAVITY = 9.81
PUCKS_PER_SIDE = 5
RACK_Y = 0.07                              # Board y of the puck rows on each half (|y|, m), clear of the band
RACK_PITCH = 0.03                          # Puck spacing in a row, m (puck diameter 2.54 cm)
HOME = [1.319, -0.862, 0.582, 1.658, -0.301, 0.268]  # start pose (rad): gripper over the agent-right corner, jaws open
BAND_RGBA = [0.80, 0.12, 0.10, 1]
PUCK_RGBA = [0.08, 0.08, 0.08, 1]


def board_pose(cfg: dict):
    """Board-frame origin (board center, play surface) in the robot frame, and the board's yaw.

    robot.base_xy_m is the arm's turning axis in the board frame, and the axis is BASE_AXIS_X in front of the
    base origin. The board's +y (toward the opponent) is the robot's +x.
    """
    axis_x_b, axis_y_b = cfg["robot"]["base_xy_m"]
    pos = np.array([BASE_AXIS_X - axis_y_b, axis_x_b, play_surface_z(cfg)])
    return pos, -math.pi / 2


def board_to_robot(cfg: dict, xy_board, z_board: float = 0.0) -> np.ndarray:
    pos, _ = board_pose(cfg)
    x, y = np.asarray(xy_board, float)
    return pos + np.array([y, -x, z_board])


def build(cfg: dict):
    spec = mujoco.MjSpec.from_file(str(SO101_XML))
    spec.option.timestep = cfg["sim"]["mujoco_timestep_s"]
    w = spec.worldbody
    add_lightbox(w)

    pos, yaw = board_pose(cfg)
    board = w.add_body(name="board", pos=pos.tolist(), quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    add_board(board, cfg)
    add_band_pegs(spec, cfg, board)
    band_y, r = BandModel.from_config(cfg).band_y, cfg["puck"]["radius_m"]
    half_span, z_mid = 0.5 * cfg["band"]["anchor_span_m"], 0.5 * cfg["puck"]["thickness_m"]
    for side, sy in (("agent", 1.0), ("opponent", -1.0)):   # Bands at rest (visual), one puck radius behind the line
        y = sy * (band_y - r)
        board.add_geom(name=f"band_{side}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0008, 0, 0],
                       fromto=[-half_span, y, z_mid, half_span, y, z_mid], rgba=BAND_RGBA, contype=0, conaffinity=0)

    # Pucks: planar joints in the board frame. Sliding friction = mu m g on each slide joint (Coulomb per
    # axis). The collision shape is a sphere of the puck radius at mid-height: an exact 2D circle against the
    # vertical walls (MuJoCo's cylinder-box contact gives a wrong normal at the gate corners).
    puck = cfg["puck"]
    friction = puck["friction_kinetic"] * puck["mass_kg"] * GRAVITY
    names = []
    for side, sy in (("a", -1.0), ("o", 1.0)):
        for i in range(PUCKS_PER_SIDE):
            name = f"puck_{side}{i}"
            names.append(name)
            x = (i - (PUCKS_PER_SIDE - 1) / 2) * RACK_PITCH
            b = board.add_body(name=name, pos=[x, sy * RACK_Y, z_mid])
            for jn, jt, axis in (("x", "SLIDE", (1, 0, 0)), ("y", "SLIDE", (0, 1, 0)), ("yaw", "HINGE", (0, 0, 1))):
                b.add_joint(name=f"{name}_{jn}", type=getattr(mujoco.mjtJoint, f"mjJNT_{jt}"), axis=list(axis),
                            frictionloss=friction if jt == "SLIDE" else 1e-5, damping=0.0)
            b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, z_mid, 0], mass=puck["mass_kg"],
                       rgba=PUCK_RGBA, contype=0, conaffinity=0)
            b.add_geom(name=f"{name}_col", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], density=0.0,
                       rgba=[0, 0, 0, 0], group=3, contype=0, conaffinity=0)
    # Bounces: frictionless contacts; damping ratio set so a head-on bounce keeps about the configured restitution.
    e = cfg["board"]["restitution"]
    zeta = -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)
    solref = [cfg["sim"]["mujoco_contact_time_s"], zeta]
    for k, a in enumerate(names):
        for wall in WALL_PAIRS:
            spec.add_pair(geomname1=f"{a}_col", geomname2=wall, condim=1, solref=solref)
        for b in names[k + 1:]:
            spec.add_pair(geomname1=f"{a}_col", geomname2=f"{b}_col", condim=1, solref=solref)

    n_puck_q = 3 * len(names)
    spec.add_key(name="home", qpos=HOME + [0.0] * n_puck_q, ctrl=HOME)
    spec.visual.global_.azimuth, spec.visual.global_.elevation = 20, -30   # viewer starts looking in the open front
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save", action="store_true", help="write assets/mujoco/robotstudio_so101/so101_slingpuck.xml")
    p.add_argument("--no-viewer", action="store_true", help="build and check the scene without opening a window")
    p.add_argument("--params-version", default="latest")
    a = p.parse_args()
    cfg = load_config(a.params_version)
    spec = build(cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)        # arm in the HOME pose
    mujoco.mj_forward(model, data)
    tip = data.site("gripperframe").xpos
    surface = play_surface_z(cfg)
    near = board_to_robot(cfg, (0.0, -0.5 * cfg["board"]["outer_length_m"]))[0]
    far = board_to_robot(cfg, (0.0, 0.5 * cfg["board"]["outer_length_m"]))[0]
    print(f"Scene ready: {model.nbody} bodies, {model.nu} actuators, {2 * PUCKS_PER_SIDE} pucks. "
          f"Gripper tip at HOME: {100 * tip[0]:.1f} cm forward, {100 * (tip[2] - surface):.1f} cm above the board.")
    print(f"Board: play surface {100 * surface:.2f} cm above the robot's floor; near end {100 * (near - BASE_AXIS_X):.1f} cm "
          f"in front of the turning axis; far end {100 * (FAR_X - far):.1f} cm before the open end (opponent side); "
          f"gate {100 * cfg['board']['gate_width_m']:.1f} cm.")
    if far > FAR_X:
        print("WARNING: the board runs past the side walls; check robot.base_xy_m or workspace.FAR_X.")
    if a.save:
        path = SO101_XML.parent / "so101_slingpuck.xml"   # next to so101.xml so the mesh paths resolve
        path.write_text(spec.to_xml())
        print(f"Saved {path}")
    if not a.no_viewer:
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
