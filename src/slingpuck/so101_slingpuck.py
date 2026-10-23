"""SO-101 arm with the sling puck board on a table, in MuJoCo: the scene of this project.

    python -m src.slingpuck.so101_slingpuck               # opens the interactive viewer
    python -m src.slingpuck.so101_slingpuck --save        # also writes assets/mujoco/robotstudio_so101/so101_slingpuck.xml
    python -m src.slingpuck.so101_slingpuck --no-viewer   # build and check the scene only (e.g. on a headless machine)

Same structure as so101_lightbox.py (the course's lightbox script), for the sling puck use case, without the
lightbox. The SO-101 model is the RobotStudio MJCF already in assets/ (no download).

Frame: origin at the robot base, on the table it stands on. x forward (away from the robot), y the robot's
left, z up. Meters. The human opponent sits at the far end, face to face with the robot (camera "opponent").

The board lies on the same table, lengthwise along x, centered left to right, with the robot behind the agent end
wall. Its sizes, the bands and the pucks come from configs/physics_params_v<N>.yaml, so this scene always
matches the simulators. The rest of the project uses the BOARD frame (origin at the board center, y toward
the opponent, agent half y < 0); board_to_robot() converts. The robot's 5 pucks are black, the human's 5 white. In the viewer, drag a puck with Ctrl + right
mouse to flick it: the pucks slide with Coulomb friction and bounce off the walls, the divider and each other.
"""
import argparse
import math

import mujoco
import mujoco.viewer
import numpy as np

from src.slingpuck.config import REPO_ROOT, load_config
from src.slingpuck.physics.band_model import BandModel

SO101_XML = REPO_ROOT / "assets" / "mujoco" / "robotstudio_so101" / "so101.xml"
GRAVITY = 9.81
CONTACT_SOLIMP = (0.99, 0.99, 0.001, 0.5, 2.0)  # dmin, dmax, width, midpoint, power
WALL_PAIRS = ("wall_left", "wall_right", "wall_agent", "wall_opponent", "divider_left", "divider_right")


def _pan_anchor_local() -> np.ndarray:
    """shoulder_pan axis position in the SO-101 base frame."""
    m = mujoco.MjModel.from_xml_path(str(SO101_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return d.xanchor[m.joint("shoulder_pan").id].copy()


BASE_AXIS_X = float(_pan_anchor_local()[0])   # the arm's turning axis is 3.88 cm in front of the base origin
PUCKS_PER_SIDE = 5
RACK_Y = 0.07                              # Board y of the puck rows on each half (|y|, m), clear of the band
RACK_PITCH = 0.03                          # Puck spacing in a row, m (puck diameter 2.54 cm)
HOME = [1.319, -0.862, 0.582, 1.658, -0.301, 0.268]  # start pose (rad): gripper over the agent-right corner, jaws open
BAND_RGBA = [0.80, 0.12, 0.10, 1]
PUCK_RGBA = {"a": [0.06, 0.06, 0.06, 1],    # Robot (agent) pucks: black
             "o": [0.96, 0.96, 0.94, 1]}    # Human (opponent) pucks: white


def _divider_boxes(cfg):
    hw, hg = 0.5 * cfg["board"]["width_m"], 0.5 * cfg["board"]["gate_width_m"]
    ht = 0.5 * cfg["board"]["divider_thickness_m"]
    return [(-hw, -hg, -ht, ht), (hg, hw, -ht, ht)]


def board_floor_thickness(cfg: dict) -> float:
    """Board base under the play surface: outer height minus the wall height above the surface."""
    return cfg["board"]["outer_height_m"] - cfg["board"]["wall_height_m"]


def add_table_and_light(body, table_z: float):
    """Table plane (visual) and one light straight from above, so the puck shadows fall straight down."""
    body.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.0, 1.0, 0.01], pos=[0, 0, table_z],
                  rgba=[0.42, 0.40, 0.37, 1], contype=0, conaffinity=0)
    body.add_light(name="overhead", pos=[0, 0, 1.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                   diffuse=[0.55, 0.55, 0.55], specular=[0.1, 0.1, 0.1])   # With the fill light: board not overexposed


def add_board(body, cfg: dict):
    """Play surface, walls and the divider with its gate, in the board frame of `body` (visual; see pairs)."""
    board = cfg["board"]
    hw, hl = 0.5 * board["width_m"], 0.5 * board["length_m"]
    wt, wh = board["wall_thickness_m"], board["wall_height_m"]
    wood, edge = [0.86, 0.78, 0.64, 1], [0.62, 0.50, 0.33, 1]

    def box(name, center, half, rgba):
        return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(center), size=list(half),
                             rgba=rgba, contype=0, conaffinity=0)

    ft = board_floor_thickness(cfg)
    box("surface", (0, 0, -0.5 * ft), (hw + wt, hl + wt, 0.5 * ft), wood)
    box("wall_left", (-hw - wt / 2, 0, wh / 2), (wt / 2, hl + wt, wh / 2), edge)
    box("wall_right", (hw + wt / 2, 0, wh / 2), (wt / 2, hl + wt, wh / 2), edge)
    box("wall_agent", (0, -hl - wt / 2, wh / 2), (hw, wt / 2, wh / 2), edge)
    box("wall_opponent", (0, hl + wt / 2, wh / 2), (hw, wt / 2, wh / 2), edge)
    for name, (xmin, xmax, ymin, ymax) in zip(("divider_left", "divider_right"), _divider_boxes(cfg)):
        box(name, ((xmin + xmax) / 2, (ymin + ymax) / 2, wh / 2), ((xmax - xmin) / 2, (ymax - ymin) / 2, wh / 2), edge)


def add_band_pegs(cfg: dict, body):
    """Band pegs on both sides, one puck radius behind each band line (the band touches the back of the
    puck instead of passing through it). body: its frame must be the board frame."""
    band_y = BandModel.from_config(cfg).band_y
    r, wh = cfg["puck"]["radius_m"], cfg["board"]["wall_height_m"]
    half = 0.5 * cfg["band"]["anchor_span_m"]
    for side, sy in (("agent", 1.0), ("opponent", -1.0)):
        for sx in (-1.0, 1.0):
            body.add_geom(name=f"peg_{side}_{'l' if sx < 0 else 'r'}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          size=[0.003, 0.5 * wh, 0], pos=[sx * half, sy * (band_y - r), 0.5 * wh],
                          rgba=[0.75, 0.75, 0.78, 1], contype=0, conaffinity=0)


def contact_params(cfg: dict):
    """solref (-stiffness, -damping) and solimp of the puck contacts."""
    e = cfg["board"]["restitution"]
    zeta = -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)
    k = (math.pi / cfg["sim"]["mujoco_contact_time_s"]) ** 2
    return [-k, -2.0 * zeta * math.sqrt(k)], list(CONTACT_SOLIMP)


def board_pose(cfg: dict):
    """Board-frame origin (board center, play surface) in the robot frame, and the board's yaw.

    robot.base_xy_m is the arm's turning axis in the board frame, and the axis is BASE_AXIS_X in front of the
    base origin. The board's +y (toward the opponent) is the robot's +x.
    """
    axis_x_b, axis_y_b = cfg["robot"]["base_xy_m"]
    pos = np.array([BASE_AXIS_X - axis_y_b, axis_x_b, board_floor_thickness(cfg)])   # Board on the robot's table
    return pos, -math.pi / 2


def board_to_robot(cfg: dict, xy_board, z_board: float = 0.0) -> np.ndarray:
    pos, _ = board_pose(cfg)
    x, y = np.asarray(xy_board, float)
    return pos + np.array([y, -x, z_board])


def build(cfg: dict):
    spec = mujoco.MjSpec.from_file(str(SO101_XML))
    spec.option.timestep = cfg["sim"]["mujoco_timestep_s"]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960   # GIF frames up to 1280 x 960
    w = spec.worldbody
    add_table_and_light(w, 0.0)

    pos, yaw = board_pose(cfg)
    board = w.add_body(name="board", pos=pos.tolist(), quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    add_board(board, cfg)
    add_band_pegs(cfg, board)
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
                       rgba=PUCK_RGBA[side], contype=0, conaffinity=0)
            b.add_geom(name=f"{name}_col", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], density=0.0,
                       rgba=[0, 0, 0, 0], group=3, contype=0, conaffinity=0)
    # Bounces: frictionless contacts as a direct spring-damper (the (timeconst,
    # dampratio) form with a low damping ratio gained energy on every bounce). Damping ratio from the
    # configured restitution: a head-on bounce keeps about that fraction of the normal speed.
    solref, solimp = contact_params(cfg)
    for k, a in enumerate(names):
        for wall in WALL_PAIRS:
            spec.add_pair(geomname1=f"{a}_col", geomname2=wall, condim=1, solref=solref, solimp=solimp)
        for b in names[k + 1:]:
            spec.add_pair(geomname1=f"{a}_col", geomname2=f"{b}_col", condim=1, solref=solref, solimp=solimp)

    # The human's view: seated at the far end, eyes about 40 cm above the board, facing the robot.
    far_end = board_to_robot(cfg, (0.0, 0.5 * cfg["board"]["outer_length_m"]))
    w.add_body(name="board_center", pos=(pos + [0.0, 0.0, 0.0]).tolist())
    w.add_camera(name="opponent", pos=(far_end + [0.20, 0.0, 0.42]).tolist(), fovy=55,
                 mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="board_center")

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
    surface = board_floor_thickness(cfg)
    near = board_to_robot(cfg, (0.0, -0.5 * cfg["board"]["outer_length_m"]))[0]
    far = board_to_robot(cfg, (0.0, 0.5 * cfg["board"]["outer_length_m"]))[0]
    print(f"Scene ready: {model.nbody} bodies, {model.nu} actuators, {2 * PUCKS_PER_SIDE} pucks. "
          f"Gripper tip at HOME: {100 * tip[0]:.1f} cm forward, {100 * (tip[2] - surface):.1f} cm above the board.")
    print(f"Board: play surface {100 * surface:.2f} cm above the table; near end {100 * (near - BASE_AXIS_X):.1f} cm "
          f"in front of the turning axis; far end {100 * far:.1f} cm in front of the robot base; "
          f"gate {100 * cfg['board']['gate_width_m']:.1f} cm.")
    if a.save:
        path = SO101_XML.parent / "so101_slingpuck.xml"   # next to so101.xml so the mesh paths resolve
        path.write_text(spec.to_xml())
        print(f"Saved {path}")
    if not a.no_viewer:
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
