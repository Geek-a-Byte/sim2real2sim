"""MuJoCo physics for the goalkeeper: real-size board, puck, SO-101 holding a paddle.

The scene is built in code from the merged config, so the MuJoCo board always
matches physics_params_v<N>.yaml. The SO-101 model is the RobotStudio MJCF in
assets/mujoco/robotstudio_so101 (STS3215 position actuators, kp 998, +-2.94 N m).

Arm setup:
- The base is turned +90 deg about z, so the arm reaches along +y (toward the
  gate) at shoulder_pan = 0, and the pan axis is at robot.base_xy_m. A positive
  pan then moves the paddle toward +x, as in GoalkeeperGeometry.
- shoulder_lift, elbow_flex and wrist_flex hold a pose found by inverse
  kinematics, so that the gripper fingertips are at the paddle center. The
  paddle is a box fixed to the gripper body (a custom mount), placed so that at
  pan = 0 it matches the 2D paddle pose exactly.
- The hold targets are corrected for gravity sag, so the settled paddle is at
  its target position.

Puck: planar (slide x, slide y, spin hinge), so it cannot tumble or hop.
Sliding friction is an applied Coulomb force (mu * m * g against the velocity,
stop without reversal), the same model as the 2D sim. MuJoCo contact friction
on a soft floor made the puck hop by ~0.5 mm, which switched friction off for
~10 ms after each wall impact.

Contacts: only explicit contact pairs collide (puck-walls, puck-dividers,
puck-paddle); all automatic arm collisions are off. Wall, divider
and paddle contacts are frictionless (condim 1), as in the 2D sim. They use a
direct spring-damper (solref = -stiffness, -damping). The stiffness sets the
contact time. The damping is calibrated in this scene (calibrate_contacts):
restitution is measured on a side wall and on the arm-held paddle over a grid of
damping values, and the config restitution is interpolated. So
board.paddle_restitution is the *effective* restitution of the paddle on the
compliant arm, which is what sysid measures from video.
"""
import math
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

from src.slingpuck.config import REPO_ROOT
from src.slingpuck.kinematics import GoalkeeperGeometry
from src.slingpuck.physics.backend import GateCrossing, PaddleState, PuckPhysicsBackend, StepEvents
from src.slingpuck.physics.servo_model import latency_steps

SO101_XML = REPO_ROOT / "assets" / "mujoco" / "robotstudio_so101" / "so101.xml"
HOLD_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
PUCK_JOINTS = ("puck_x", "puck_y", "puck_yaw")
GRAVITY = 9.81
BASE_YAW = math.pi / 2           # Arm reaches +y at pan = 0
CONTACT_SOLIMP = (0.99, 0.99, 0.001, 0.5, 2.0)  # dmin, dmax, width, midpoint, power
STOP_SPEED_M_S = 1e-3            # Below this the puck counts as stopped


def _rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pan_anchor_local() -> np.ndarray:
    """shoulder_pan axis position in the SO-101 base frame."""
    m = mujoco.MjModel.from_xml_path(str(SO101_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return d.xanchor[m.joint("shoulder_pan").id].copy()


@dataclass(frozen=True)
class SceneLayout:
    """Board-frame positions derived from the config."""
    base_pos: np.ndarray
    base_quat: np.ndarray
    paddle_center: np.ndarray    # World paddle center at pan = 0
    geom: GoalkeeperGeometry

    @classmethod
    def from_config(cls, cfg: dict) -> "SceneLayout":
        geom = GoalkeeperGeometry.from_config(cfg)
        robot = cfg["robot"]
        anchor = _pan_anchor_local()
        base_xy = geom.base_xy - (_rotz(BASE_YAW) @ anchor)[:2]
        base_pos = np.array([base_xy[0], base_xy[1], robot["base_z_m"]])
        base_quat = np.array([math.cos(BASE_YAW / 2), 0.0, 0.0, math.sin(BASE_YAW / 2)])
        z = robot["paddle_clearance_m"] + 0.5 * robot["paddle_height_m"]
        center = np.array([0.0, geom.paddle_center(0.0)[1], z])
        return cls(base_pos, base_quat, center, geom)


@dataclass(frozen=True)
class ArmHoldPose:
    ik_joints: np.ndarray        # HOLD_JOINTS angles that put the fingertips at the paddle center
    hold_ctrl: np.ndarray        # Actuator targets for HOLD_JOINTS, corrected for gravity sag
    settled_qpos: np.ndarray     # HOLD_JOINTS angles after settling under gravity
    paddle_pos: np.ndarray       # Paddle geom position in the gripper body frame
    paddle_quat: np.ndarray
    ik_error_m: float            # Fingertip error (absorbed by the paddle mount)
    settled_error_m: float       # Paddle center error after sag compensation


def _divider_boxes(cfg):
    hw, hg = 0.5 * cfg["board"]["width_m"], 0.5 * cfg["board"]["gate_width_m"]
    ht = 0.5 * cfg["board"]["divider_thickness_m"]
    return [(-hw, -hg, -ht, ht), (hg, hw, -ht, ht)]


WALL_PAIRS = ("wall_left", "wall_right", "wall_agent", "wall_opponent", "divider_left", "divider_right")


def _build_spec(cfg: dict, layout: SceneLayout, pose: ArmHoldPose | None) -> mujoco.MjSpec:
    board, puck, robot = cfg["board"], cfg["puck"], cfg["robot"]
    spec = mujoco.MjSpec.from_file(str(SO101_XML))
    spec.option.timestep = cfg["sim"]["mujoco_timestep_s"]
    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0
    base = spec.body("base")
    base.pos = layout.base_pos.tolist()
    base.quat = layout.base_quat.tolist()

    world = spec.worldbody
    hw, hl = 0.5 * board["width_m"], 0.5 * board["length_m"]
    wt, wh = board["wall_thickness_m"], board["wall_height_m"]
    wood, edge = [0.86, 0.78, 0.64, 1], [0.62, 0.50, 0.33, 1]

    def box(name, center, half, rgba):
        return world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(center), size=list(half),
                              rgba=rgba, contype=0, conaffinity=0)

    box("surface", (0, 0, -0.005), (hw + wt, hl + wt, 0.005), wood)
    box("wall_left", (-hw - wt / 2, 0, wh / 2), (wt / 2, hl + wt, wh / 2), edge)
    box("wall_right", (hw + wt / 2, 0, wh / 2), (wt / 2, hl + wt, wh / 2), edge)
    box("wall_agent", (0, -hl - wt / 2, wh / 2), (hw, wt / 2, wh / 2), edge)
    box("wall_opponent", (0, hl + wt / 2, wh / 2), (hw, wt / 2, wh / 2), edge)
    for name, (xmin, xmax, ymin, ymax) in zip(("divider_left", "divider_right"), _divider_boxes(cfg)):
        box(name, ((xmin + xmax) / 2, (ymin + ymax) / 2, wh / 2), ((xmax - xmin) / 2, (ymax - ymin) / 2, wh / 2), edge)

    pb = world.add_body(name="puck", pos=[0.0, 0.0, 0.5 * puck["thickness_m"]])
    for name, jtype, axis in zip(PUCK_JOINTS, ("slide", "slide", "hinge"), ((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        pb.add_joint(name=name, type=getattr(mujoco.mjtJoint, f"mjJNT_{jtype.upper()}"), axis=list(axis))
    pb.add_geom(name="puck", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[puck["radius_m"], 0.5 * puck["thickness_m"], 0], mass=puck["mass_kg"],
                rgba=[0.08, 0.08, 0.08, 1], contype=0, conaffinity=0)

    if pose is not None:
        spec.body("gripper").add_geom(
            name="paddle", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.5 * robot["paddle_width_m"], 0.5 * robot["paddle_thickness_m"], 0.5 * robot["paddle_height_m"]],
            pos=pose.paddle_pos.tolist(), quat=pose.paddle_quat.tolist(), mass=robot["paddle_mass_kg"],
            rgba=[0.92, 0.41, 0.20, 1], contype=0, conaffinity=0)
        _add_pairs(spec, cfg)

    # Visual only: table, sky, lights, cameras.
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.0, 1.0, 0.01], pos=[0, 0, -0.0101],
                   rgba=[0.42, 0.40, 0.37, 1], contype=0, conaffinity=0)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.86, 0.89, 0.93], rgb2=[0.55, 0.60, 0.68], width=512, height=3072)
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    world.add_light(pos=[0, -0.1, 1.2], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    diffuse=[0.6, 0.6, 0.6])
    world.add_light(pos=[0.6, 0.6, 0.8], dir=[-0.5, -0.5, -0.7], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    diffuse=[0.35, 0.35, 0.35], castshadow=0)
    world.add_camera(name="top", pos=[0, -0.02, 0.75], quat=[1, 0, 0, 0], fovy=45)
    world.add_camera(name="side", pos=[0.42, -0.42, 0.32], xyaxes=[0.707, 0.707, 0, -0.33, 0.33, 0.88], fovy=50)
    # From the opponent side, above the board, looking at the gate and the paddle.
    world.add_camera(name="gate", pos=[0, 0.30, 0.24], xyaxes=[-1, 0, 0, 0, -0.62, 0.785], fovy=45)
    return spec


def _stiffness(cfg) -> float:
    return (math.pi / cfg["sim"]["mujoco_contact_time_s"]) ** 2


def _add_pairs(spec: mujoco.MjSpec, cfg: dict):
    """Contact pairs. Wall/paddle damping is a placeholder until apply_params() sets the calibrated value."""
    k = _stiffness(cfg)
    for name in WALL_PAIRS + ("paddle",):
        spec.add_pair(name=f"puck_{name}", geomname1="puck", geomname2=name, condim=1,
                      solref=[-k, -0.1 * math.sqrt(k)], solimp=list(CONTACT_SOLIMP))


@dataclass(frozen=True)
class ContactCalibration:
    """Measured restitution vs contact damping, for walls and for the arm-held paddle."""
    stiffness: float
    damping: np.ndarray
    wall_e: np.ndarray
    paddle_e: np.ndarray

    def solref(self, kind: str, restitution: float) -> np.ndarray:
        e_grid = self.wall_e if kind == "wall" else self.paddle_e
        order = np.argsort(e_grid)
        lo, hi = e_grid.min(), e_grid.max()
        if not lo <= restitution <= hi:
            raise ValueError(f"{kind} restitution {restitution} outside the calibrated range [{lo:.3f}, {hi:.3f}]")
        b = float(np.interp(restitution, e_grid[order], self.damping[order]))
        return np.array([-self.stiffness, -b])


def _qadr(model, names):
    return [int(model.joint(n).qposadr[0]) for n in names]


def _park_puck(model, data, cfg, xy=(0.0, 0.1)):
    """Put the puck at rest (used while the arm settles)."""
    data.qpos[_qadr(model, PUCK_JOINTS)] = [xy[0], xy[1], 0.0]


def solve_hold_pose(cfg: dict, layout: SceneLayout) -> ArmHoldPose:
    """Inverse kinematics for the paddle pose at pan = 0, then gravity-sag compensation."""
    model = _build_spec(cfg, layout, None).compile()
    data = mujoco.MjData(model)
    site = model.site("gripperframe").id
    adr = _qadr(model, HOLD_JOINTS)
    pan_adr = _qadr(model, ("shoulder_pan",))[0]
    lo = np.array([model.joint(n).range[0] for n in HOLD_JOINTS])
    hi = np.array([model.joint(n).range[1] for n in HOLD_JOINTS])
    mid, span = (lo + hi) / 2, hi - lo
    target = layout.paddle_center
    link_ids = [model.body(b).id for b in ("upper_arm", "lower_arm", "wrist", "gripper")]
    min_link_z = cfg["board"]["wall_height_m"] + 0.02

    def fk(q):
        data.qpos[:] = 0.0
        data.qpos[pan_adr] = layout.geom.pan_to_joint(0.0)
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        return data.site_xpos[site].copy()

    def residual(q):
        p = fk(q)
        low = np.minimum(data.xpos[link_ids, 2] - min_link_z, 0.0)
        return np.concatenate([p - target, 1e-4 * (q - mid) / span, 10.0 * low])

    starts = np.random.default_rng(0).uniform(lo, hi, (40, 3))
    sols = [least_squares(residual, q0, bounds=(lo, hi)) for q0 in starts]
    # The pitch joints move the fingertips in a plane ~1 mm beside the pan axis, so the
    # fingertips cannot reach x = 0 exactly. The paddle mount below absorbs this offset:
    # it places the paddle exactly at the target from the actual gripper pose.
    ok = [s for s in sols if np.linalg.norm(fk(s.x) - target) < 5e-3]
    if not ok:
        raise RuntimeError("SO-101 cannot reach the paddle position; check robot.base_xy_m / base_z_m")
    q_ik = min(ok, key=lambda s: np.abs((s.x - mid) / span).max()).x
    ik_error = float(np.linalg.norm(fk(q_ik) - target))

    # Paddle pose in the gripper frame: world-aligned box at the target when the arm is at q_ik.
    g = model.body("gripper").id
    R_g, p_g = data.xmat[g].reshape(3, 3), data.xpos[g].copy()
    paddle_pos = R_g.T @ (target - p_g)
    paddle_quat = np.zeros(4)
    mujoco.mju_mat2Quat(paddle_quat, R_g.T.flatten())

    # Gravity-sag compensation: shift the hold targets by the settled joint error.
    pose = ArmHoldPose(q_ik, q_ik.copy(), q_ik.copy(), paddle_pos, paddle_quat, ik_error, float("nan"))
    full = _build_spec(cfg, layout, pose).compile()
    fdata = mujoco.MjData(full)
    fadr = _qadr(full, HOLD_JOINTS)
    fpan = _qadr(full, ("shoulder_pan",))[0]
    acts = [full.actuator(n).id for n in HOLD_JOINTS]
    pan_act = full.actuator("shoulder_pan").id
    paddle_geom = full.geom("paddle").id
    ctrl = q_ik.copy()
    for _ in range(5):
        mujoco.mj_resetData(full, fdata)
        fdata.qpos[fadr] = q_ik
        fdata.qpos[fpan] = layout.geom.pan_to_joint(0.0)
        fdata.ctrl[acts] = ctrl
        fdata.ctrl[pan_act] = layout.geom.pan_to_joint(0.0)
        _park_puck(full, fdata, cfg)
        mujoco.mj_step(full, fdata, nstep=int(0.6 / full.opt.timestep))
        settled = fdata.qpos[fadr].copy()
        ctrl = ctrl + (q_ik - settled)
    settled_error = float(np.linalg.norm(fdata.geom_xpos[paddle_geom] - target))
    return ArmHoldPose(q_ik, ctrl, settled, paddle_pos, paddle_quat, ik_error, settled_error)


def calibrate_contacts(sim: "MujocoGoalkeeperSim", impact_speed: float = 2.0, n: int = 24,
                       n_phases: int = 5) -> ContactCalibration:
    """Measure restitution vs damping in the real scene (no friction during the test).

    Each value is the mean over n_phases impacts that start a fraction of a
    timestep apart, because the discrete solver makes restitution depend on
    where the impact falls between steps (about +-1.5% at the default settings).
    """
    m, cfg = sim.model, sim._cfg
    d = mujoco.MjData(m)
    k = _stiffness(cfg)
    w = math.sqrt(k)
    damping = 2.0 * w * np.geomspace(0.003, 2.0, n)   # Damping ratio 0.003 .. 2
    saved_solref = m.pair_solref.copy()
    hw = 0.5 * cfg["board"]["width_m"]
    r = cfg["puck"]["radius_m"]
    q, v = sim._puck_qadr, sim._puck_vadr

    def bounce(start, vel, axis):
        mujoco.mj_resetData(m, d)
        d.qpos[sim._hold_qadr] = sim.pose.settled_qpos
        d.qpos[sim._pan_qadr] = sim.layout.geom.pan_to_joint(0.0)
        d.ctrl[sim._hold_act] = sim.pose.hold_ctrl
        d.ctrl[sim._pan_act] = sim.layout.geom.pan_to_joint(0.0)
        d.qpos[q] = [start[0], start[1], 0.0]
        d.qvel[v] = [vel[0], vel[1], 0.0]
        sign = np.sign(vel[axis])
        for _ in range(int(0.25 / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if np.sign(d.qvel[v[axis]]) == -sign:
                for _ in range(int(0.012 / m.opt.timestep)):  # Let the contact finish
                    mujoco.mj_step(m, d)
                return -d.qvel[v[axis]] / vel[axis]
        raise RuntimeError("calibration impact did not happen")

    wall_e, paddle_e = [], []
    for b in damping:
        for name in WALL_PAIRS:
            m.pair_solref[m.pair(f"puck_{name}").id] = (-k, -b)
        m.pair_solref[m.pair("puck_paddle").id] = (-k, -b)
        shifts = np.arange(n_phases) / n_phases * impact_speed * m.opt.timestep
        wall_e.append(np.mean([bounce((hw - r - 0.02 + ds, 0.10), (impact_speed, 0.0), 0) for ds in shifts]))
        paddle_e.append(np.mean([bounce((0.0, 0.06 - ds), (0.0, -impact_speed), 1) for ds in shifts]))
    m.pair_solref[:] = saved_solref
    return ContactCalibration(k, damping, np.array(wall_e), np.array(paddle_e))


_POSE_CACHE: dict = {}
_CALIB_CACHE: dict = {}


def _pose_key(cfg):
    keys = (("robot",), ("board", "length_m"), ("board", "width_m"), ("board", "divider_thickness_m"),
            ("board", "wall_height_m"), ("goalkeeper", "paddle_standoff_m"), ("goalkeeper", "pan_range_deg"),
            ("sim", "mujoco_timestep_s"))
    return tuple(repr(cfg[k[0]] if len(k) == 1 else cfg[k[0]][k[1]]) for k in keys)


class MujocoGoalkeeperSim(PuckPhysicsBackend):
    """PuckPhysicsBackend on MuJoCo. The paddle is moved by the arm, not by set_paddle()."""

    n_pucks = 1

    def __init__(self, cfg: dict):
        self.layout = SceneLayout.from_config(cfg)
        key = _pose_key(cfg)
        if key not in _POSE_CACHE:
            _POSE_CACHE[key] = solve_hold_pose(cfg, self.layout)
        self.pose = _POSE_CACHE[key]
        self.model = _build_spec(cfg, self.layout, self.pose).compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.dt = m.opt.timestep
        self._puck_qadr = _qadr(m, PUCK_JOINTS)
        self._puck_vadr = [int(m.joint(n).dofadr[0]) for n in PUCK_JOINTS]
        self._pan_qadr = _qadr(m, ("shoulder_pan",))[0]
        self._pan_vadr = int(m.joint("shoulder_pan").dofadr[0])
        self._hold_qadr = _qadr(m, HOLD_JOINTS)
        self._hold_act = [m.actuator(n).id for n in HOLD_JOINTS]
        self._pan_act = m.actuator("shoulder_pan").id
        self._puck_geom = m.geom("puck").id
        self._paddle_geom = m.geom("paddle").id
        self._puck_body = m.body("puck").id
        self._nominal_inertia = m.body_inertia[self._puck_body].copy()
        self._nominal_mass = float(m.body_mass[self._puck_body])
        self._cfg = cfg
        self.paddle = None
        self._mu = cfg["puck"]["friction_kinetic"]
        calib_key = key + (repr(cfg["sim"]["mujoco_contact_time_s"]), repr(cfg["puck"]))
        if calib_key not in _CALIB_CACHE:
            _CALIB_CACHE[calib_key] = calibrate_contacts(self)
        self.calibration = _CALIB_CACHE[calib_key]
        self.apply_params(cfg)
        mujoco.mj_forward(m, self.data)

    # -------------------------------------------------- domain randomization
    def apply_params(self, cfg: dict):
        """Update friction, restitution, puck mass and gate width in place (no recompile)."""
        m = self.model
        self._mu = cfg["puck"]["friction_kinetic"]
        wall = self.calibration.solref("wall", cfg["board"]["restitution"])
        for name in WALL_PAIRS:
            m.pair_solref[m.pair(f"puck_{name}").id] = wall
        m.pair_solref[m.pair("puck_paddle").id] = self.calibration.solref("paddle", cfg["board"]["paddle_restitution"])
        scale = cfg["puck"]["mass_kg"] / self._nominal_mass
        m.body_mass[self._puck_body] = cfg["puck"]["mass_kg"]
        m.body_inertia[self._puck_body] = self._nominal_inertia * scale
        for name, (xmin, xmax, _, _) in zip(("divider_left", "divider_right"), _divider_boxes(cfg)):
            gid = m.geom(name).id
            m.geom_pos[gid][0] = 0.5 * (xmin + xmax)
            m.geom_size[gid][0] = 0.5 * (xmax - xmin)
        self._cfg = cfg

    # ------------------------------------------------------------- the arm
    def reset_arm(self, pan: float):
        d = self.data
        d.qvel[:] = 0.0
        d.qpos[self._hold_qadr] = self.pose.settled_qpos
        d.qpos[self._pan_qadr] = self.layout.geom.pan_to_joint(pan)
        d.ctrl[:] = 0.0
        d.ctrl[self._hold_act] = self.pose.hold_ctrl
        d.ctrl[self._pan_act] = self.layout.geom.pan_to_joint(pan)
        mujoco.mj_forward(self.model, d)

    def set_pan_target(self, pan: float):
        self.data.ctrl[self._pan_act] = self.layout.geom.pan_to_joint(pan)

    @property
    def pan(self) -> float:
        return self.layout.geom.joint_to_pan(float(self.data.qpos[self._pan_qadr]))

    @property
    def pan_vel(self) -> float:
        return float(self.data.qvel[self._pan_vadr])

    def paddle_center(self) -> np.ndarray:
        return self.data.geom_xpos[self._paddle_geom].copy()

    # ------------------------------------------------------ the puck backend
    def reset(self, pos, vel):
        pos = np.asarray(pos, dtype=float).reshape(2)
        vel = np.asarray(vel, dtype=float).reshape(2)
        self.data.qpos[self._puck_qadr] = [pos[0], pos[1], 0.0]
        self.data.qvel[self._puck_vadr] = [vel[0], vel[1], 0.0]
        mujoco.mj_forward(self.model, self.data)

    @property
    def pos(self) -> np.ndarray:
        return self.data.qpos[self._puck_qadr[:2]].copy()[None]

    @property
    def vel(self) -> np.ndarray:
        v = self.data.qvel[self._puck_vadr[:2]].copy()
        return (np.zeros(2) if np.hypot(*v) < STOP_SPEED_M_S else v)[None]

    def get_state(self):
        return self.pos, self.vel

    def set_paddle(self, paddle: PaddleState | None):
        pass  # The arm moves the paddle; see set_pan_target().

    def kinetic_energy(self) -> float:
        v = self.data.qvel[self._puck_vadr[:2]]
        return 0.5 * float(self.model.body_mass[self._puck_body]) * float(v @ v)

    def step(self, dt: float) -> StepEvents:
        d = self.data
        y_adr = self._puck_qadr[1]
        side_before = d.qpos[y_adr] > 0.0
        touched = False
        vx_adr, vy_adr = self._puck_vadr[0], self._puck_vadr[1]
        for _ in range(max(1, int(round(dt / self.dt)))):
            # Coulomb sliding friction as an applied force; stop without reversal.
            vx, vy = d.qvel[vx_adr], d.qvel[vy_adr]
            speed = math.hypot(vx, vy)
            decel = self._mu * GRAVITY
            if 0.0 < speed <= decel * self.dt:
                d.qvel[vx_adr] = d.qvel[vy_adr] = 0.0
                speed = 0.0
            f = self.model.body_mass[self._puck_body] * decel / speed if speed > 0.0 else 0.0
            d.qfrc_applied[vx_adr] = -f * vx
            d.qfrc_applied[vy_adr] = -f * vy
            mujoco.mj_step(self.model, d)
            if d.ncon:
                geoms = d.contact.geom[:d.ncon]
                hit = (((geoms[:, 0] == self._puck_geom) & (geoms[:, 1] == self._paddle_geom))
                       | ((geoms[:, 1] == self._puck_geom) & (geoms[:, 0] == self._paddle_geom)))
                touched |= bool(np.any(hit & (d.contact.dist[:d.ncon] < 0.0)))
        x, y = d.qpos[self._puck_qadr[0]], d.qpos[y_adr]
        crossings = []
        if (y > 0.0) != side_before:
            crossings.append(GateCrossing(0, 1 if y > 0.0 else -1, float(x)))
        return StepEvents(crossings, [0] if touched else [])


class MujocoPanServo:
    """Servo adapter: command latency in software, then the MuJoCo STS3215 actuator.

    Rate limit, lag and overshoot come from the MuJoCo actuator and arm dynamics,
    not from ServoModel. Has the .pos/.vel/.step/.reset interface of ServoModel.
    """

    def __init__(self, sim: MujocoGoalkeeperSim, servo_cfg: dict, dt: float):
        self.sim = sim
        self.latency_steps = latency_steps(servo_cfg["command_latency_s"], dt)
        self._queue = []

    def reset(self, pan: float):
        self.sim.reset_arm(pan)
        self._queue = [pan] * (self.latency_steps + 1)

    def step(self, target_pan: float) -> float:
        self._queue.append(float(target_pan))
        self.sim.set_pan_target(self._queue.pop(0))
        return self.sim.pan

    @property
    def pos(self) -> float:
        return self.sim.pan

    @property
    def vel(self) -> float:
        return self.sim.pan_vel
