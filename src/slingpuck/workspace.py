"""The real workspace around the robot: the SO-101 lightbox (NVIDIA Sim-to-Real SO-101 build), opened at
the far end for the human opponent.

Measurements from so101_lightbox.py (the course's lightbox script). Changed for the game: a human plays
face to face with the robot, across the board. So the back wall is removed, and the floor and the side
walls run on past the board's far end (to FAR_X), so the game stays between the side walls. Frame of this
module (the "robot frame"): origin at the robot base origin, on the floor it stands
on; x forward (away from the robot, toward the back wall), y the robot's left,
z up. Meters.

The sling puck board lies on the mat, lengthwise along x, centered left to right,
with the robot behind the agent end wall (robot.base_xy_m in the physics params).
The board scenes (mujoco_goalkeeper, mujoco_sling, twin) use the board frame, so
they add the lightbox as one body placed at the robot base: add_lightbox(that body).

Notes from the lightbox page:
- Mat height matters most (a policy there was sensitive to a 1 cm error). Here it
  sets the play-surface height, which sets the grasp height for the sling.
- Model the wrist camera mount and penalize self-contact: their policy pressed the
  mount against the upper arm and broke it on the real arm.
- The C920 records at 30 fps. The twin tests showed that sysid needs 120 fps or
  more for the band-to-divider part of a shot (wall restitution, energy transfer).
"""
import mujoco

# Lightbox: white foam board, 76.2 wide x 50.8 deep x 50.8 high (cm), open at the front.
BOARD = 0.0048                    # 3/16" foam board
LIGHTBOX_X = (-0.031, 0.477)      # the robot's back edge is flush with the floor's front (open) edge
# Game version: no back wall (the opponent sits there), floor and side walls longer. FAR_X leaves 21 cm of
# floor past the board's far end (0.44 m with the current robot.base_xy_m) for the opponent's hands.
FAR_X = 0.65                      # [design] TODO: measure the extended side walls
FLOOR_X = (LIGHTBOX_X[0], FAR_X)
FLOOR_Y = 0.381                   # half of the 76.2 cm width; the robot is centered
WALL_H = 0.508
# Black EVA mat, 45.7 wide x 30.5 deep x 0.5 thick (cm), on an extra foam board; centered left to right.
BASE_AXIS_X = 0.0388              # the arm's turning axis is 3.88 cm in front of the base origin
MAT_X = (BASE_AXIS_X + 0.079, BASE_AXIS_X + 0.079 + 0.305)   # near edge 7.9 cm in front of the turning axis
MAT_Y = 0.2285
MAT_T = 0.005
MAT_TOP = BOARD + MAT_T           # ~1 cm above the floor the robot stands on
# Logitech C920: through the left wall, 27 cm in front of the back wall, 40 cm up, aimed at the mat (~45 deg down).
CAM_POS = (LIGHTBOX_X[1] - BOARD - 0.27, FLOOR_Y - BOARD - 0.01, 0.40)   # Measured from the original back wall
CAM_FOVY = 52                     # degrees, vertical, at 4:3 (the C920's 78 deg is diagonal)
CAM_FPS = 30
INNER_Y = FLOOR_Y - BOARD         # Side walls, inside faces

WHITE, BLACK = [1, 1, 1, 1], [0.05, 0.05, 0.05, 1]


def box(body, name, x, y, z, rgba):
    """Box from its extents: x, y, z are (min, max) pairs in meters (visual only)."""
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, rgba=rgba, contype=0, conaffinity=0,
                  size=[(x[1] - x[0]) / 2, (y[1] - y[0]) / 2, (z[1] - z[0]) / 2],
                  pos=[(x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2])


def add_lightbox(body, camera: bool = True):
    """Table, floor and the two side walls (open at both ends), mat, overhead light and the C920, in the robot
    frame of `body`."""
    body.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], pos=[0, 0, -BOARD],
                  rgba=[0.55, 0.47, 0.38, 1], contype=0, conaffinity=0)
    box(body, "lb_floor", FLOOR_X, (-FLOOR_Y, FLOOR_Y), (-BOARD, 0), WHITE)
    box(body, "lb_wall_left", FLOOR_X, (FLOOR_Y - BOARD, FLOOR_Y), (0, WALL_H), WHITE)
    box(body, "lb_wall_right", FLOOR_X, (-FLOOR_Y, -FLOOR_Y + BOARD), (0, WALL_H), WHITE)
    box(body, "mat_board", MAT_X, (-MAT_Y, MAT_Y), (0, BOARD), WHITE)
    box(body, "mat", MAT_X, (-MAT_Y, MAT_Y), (BOARD, MAT_TOP), BLACK)
    body.add_light(name="lightbox_overhead", pos=[0.5 * (FLOOR_X[0] + FLOOR_X[1]), 0, 0.5], dir=[0, 0, -1],
                   diffuse=[0.8, 0.8, 0.8])
    # The human opponent's view: seated at the open end, eyes about 40 cm above the board, facing the robot.
    body.add_body(name="opponent_look", pos=[0.20, 0, MAT_TOP])
    body.add_camera(name="opponent", pos=[FAR_X + 0.05, 0, 0.42], fovy=55,
                    mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="opponent_look")
    if camera:
        body.add_body(name="mat_center", pos=[(MAT_X[0] + MAT_X[1]) / 2, 0, MAT_TOP])
        body.add_camera(name="c920", pos=list(CAM_POS), fovy=CAM_FOVY,
                        mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="mat_center")


def board_floor_thickness(cfg: dict) -> float:
    """Board base under the play surface: outer height minus the wall height above the surface."""
    return cfg["board"]["outer_height_m"] - cfg["board"]["wall_height_m"]


def play_surface_z(cfg: dict) -> float:
    """Play-surface height above the floor the robot stands on (board lying on the mat)."""
    return MAT_TOP + board_floor_thickness(cfg)
