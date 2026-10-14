"""MuJoCo physics for the Phase 2 sling shot: the SO-101 grasps the puck, slides it
along the band, pulls it back against the band, and releases it toward the gate.

Scene: the same board, puck, walls, dividers and SO-101 as the goalkeeper scene
(mujoco_goalkeeper._build_spec), plus band pegs and a grasp site between the
gripper jaw tips. There is no paddle.

Grasp: the gripper points straight down and the jaws close on the left and right
sides of the puck. While the arm holds the puck, the puck is attached rigidly to
the grasp site (its position is set from the site every step), and the band
force on the puck is applied to the gripper body at the grasp site. So the arm
actuators carry the band load, and any arm deflection moves the puck.

Band: BandModel gives the force. While held: loading curve (as SlingEnv). On
release: the puck is detached, the unloading curve (with hysteresis) pushes it
until it crosses the band line, then its speed is scaled by sqrt(energy_transfer),
as in BandModel.release. The release runs with a 0.1 ms timestep, near the 2D
model's step. After that the puck flies on MuJoCo with Coulomb sliding friction
and calibrated wall contacts, as in the goalkeeper scene.

Visual band: the model puts the band vertex at the puck center. The viewer draws
the band one puck radius behind that line, so the band touches the back of the
puck instead of passing through it.

Arm control: the grasp-site path is a list of segments with minimum-jerk
timing. Every control tick, inverse kinematics (5 joints, gripper down, jaws
along x) turns the path point into position-actuator targets.
"""
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.optimize import least_squares

from src.slingpuck.physics.band_model import BandModel
from src.slingpuck.physics.mujoco_goalkeeper import (
    CONTACT_SOLIMP, WALL_PAIRS, SceneLayout, SO101_XML, _build_spec, _divider_boxes, _qadr, _stiffness,
    apply_puck_friction)

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
PUCK_JOINTS = ("puck_x", "puck_y", "puck_yaw")
RELEASE_TIMESTEP_S = 1e-4
CONTROL_DT_S = 0.01
LIFT_M = 0.05                 # Grasp-site height above the puck center before the grasp and after the release
JAW_OPEN_EXTRA_M = 0.012      # Jaw tip gap beyond the puck diameter when open
SHORT_MISS_M = 0.05           # Same meaning as in sling_env
HOME_XYZ = (0.11, -0.19, 0.08)  # Grasp-site rest position: over the agent-right corner, clear of the band area
SPEEDS = {"slide": 0.08, "pull": 0.05, "correct": 0.03}  # Grasp-site speeds, m/s


def _look_at(pos, target):
    """xyaxes for a camera at pos looking at target, with world z up in the image."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.concatenate([x, y]).tolist()


def _orbit(lookat, azimuth_deg, elevation_deg, distance):
    """Camera position as the MuJoCo free camera (azimuth, elevation, distance around lookat)."""
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.asarray(lookat, float) - distance * forward


def min_jerk(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau * tau)


@dataclass(frozen=True)
class GraspGeometry:
    q_grasp: float          # Gripper joint angle with the jaw tips one puck diameter apart
    q_open: float
    site_pos: np.ndarray    # Grasp point (jaw tip midpoint at q_grasp) in the gripper body frame
    site_quat: np.ndarray   # Same axes as gripperframe: x = approach, z = jaw opening direction


def grasp_geometry(puck_diameter: float) -> GraspGeometry:
    m = mujoco.MjModel.from_xml_path(str(SO101_XML))
    d = mujoco.MjData(m)
    g_adr = int(m.joint("gripper").qposadr[0])
    tip_f, tip_m = m.geom("fixed_jaw_sph_tip1").id, m.geom("moving_jaw_sph_tip1").id
    lo, hi = m.jnt_range[m.joint("gripper").id]

    def gap(q):
        d.qpos[g_adr] = q
        mujoco.mj_kinematics(m, d)
        return float(np.linalg.norm(d.geom_xpos[tip_m] - d.geom_xpos[tip_f]))

    def solve(width):
        a, b = lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            a, b = (mid, b) if gap(mid) < width else (a, mid)
        return 0.5 * (a + b)

    q_grasp = solve(puck_diameter)
    q_open = solve(puck_diameter + JAW_OPEN_EXTRA_M)
    gap(q_grasp)
    body = m.body("gripper").id
    R, p = d.xmat[body].reshape(3, 3), d.xpos[body]
    mid = 0.5 * (d.geom_xpos[tip_f] + d.geom_xpos[tip_m])
    site = m.site("gripperframe").id
    return GraspGeometry(q_grasp, q_open, R.T @ (mid - p), m.site_quat[site].copy())


def add_grasp_site(spec: mujoco.MjSpec, grasp: GraspGeometry):
    spec.body("gripper").add_site(name="grasp", pos=grasp.site_pos.tolist(), quat=grasp.site_quat.tolist(),
                                  size=[0.002, 0, 0], rgba=[0.1, 0.8, 0.2, 1], group=3)


def add_band_pegs(spec: mujoco.MjSpec, cfg: dict, body=None):
    """Band pegs on both sides, one puck radius behind each band line (see the module docstring).
    body: where to add them (default the world); its frame must be the board frame."""
    body = spec.worldbody if body is None else body
    band_y = BandModel.from_config(cfg).band_y
    r, wh = cfg["puck"]["radius_m"], cfg["board"]["wall_height_m"]
    half = 0.5 * cfg["band"]["anchor_span_m"]
    for side, sy in (("agent", 1.0), ("opponent", -1.0)):
        for sx in (-1.0, 1.0):
            body.add_geom(name=f"peg_{side}_{'l' if sx < 0 else 'r'}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                    size=[0.003, 0.5 * wh, 0], pos=[sx * half, sy * (band_y - r), 0.5 * wh],
                                    rgba=[0.75, 0.75, 0.78, 1], contype=0, conaffinity=0)


def add_view_cameras(spec: mujoco.MjSpec):
    """sling: band, puck and gate together. band: close-up of the grasp and the band V.
    shot: from behind the opponent band, looking back at the gate."""
    world = spec.worldbody
    # All inside the lightbox (its back wall is 3 cm behind the board's far end).
    for name, lookat, az, el, dist, fovy in (("sling", (0.0, -0.05, 0.02), -120, -40, 0.36, 60),
                                             ("band", (0.0, -0.12, 0.01), -130, -22, 0.22, 45)):
        pos = _orbit(lookat, az, el, dist)
        world.add_camera(name=name, pos=pos.tolist(), xyaxes=_look_at(pos, lookat), fovy=fovy)
    world.add_camera(name="shot", pos=[0.0, 0.19, 0.26], xyaxes=_look_at([0.0, 0.19, 0.26], [0.0, -0.08, 0.0]),
                     fovy=60)


def _build_sling_spec(cfg: dict, layout: SceneLayout, grasp: GraspGeometry) -> mujoco.MjSpec:
    spec = _build_spec(cfg, layout, None)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960  # GIF frames up to 1280 x 960
    add_grasp_site(spec, grasp)
    add_band_pegs(spec, cfg)
    # Collision shape of the puck: a sphere of the puck radius at mid-height. The puck
    # cannot move in z, so against the vertical walls this is an exact 2D circle.
    # MuJoCo's cylinder-box contact gives a wrong normal (along x) at the gate corners.
    r = cfg["puck"]["radius_m"]
    spec.body("puck").add_geom(name="puck_collide", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0],
                               density=0.0, rgba=[0, 0, 0, 0], group=3, contype=0, conaffinity=0)
    k = _stiffness(cfg)
    for name in WALL_PAIRS:
        spec.add_pair(name=f"puck_{name}", geomname1="puck_collide", geomname2=name, condim=1,
                      solref=[-k, -0.1 * math.sqrt(k)], solimp=list(CONTACT_SOLIMP))
    add_view_cameras(spec)
    return spec


@dataclass(frozen=True)
class ShotPlan:
    """Grasp-site targets in the board frame (puck center positions, from SlingEnv)."""
    x0: float                    # Puck rest position on the band line
    slide_x: float               # Slide target on the band line
    pull: np.ndarray             # First held position (step 1)
    corrected: np.ndarray | None = None  # Held position after the correction (step 2), if any


@dataclass
class MujocoShotResult:
    success: bool = False
    miss_m: float = float("nan")
    release_target: np.ndarray | None = None   # Commanded held position at the release
    release_pos: np.ndarray | None = None      # Actual puck position at the release (arm deflection, tracking)
    exit_pos: np.ndarray | None = None
    exit_vel: np.ndarray | None = None
    max_band_force_n: float = 0.0
    launched: bool = False
    rebounded: bool = False                    # Turned back before the gate (first pass only counts, as SlingEnv)
    rebound_t: float | None = None
    events: dict = field(default_factory=dict)  # Segment name -> sim start time (robot command log)
    log: list = field(default_factory=list)    # (t, x, y) puck path after the release


@dataclass(frozen=True)
class Segment:
    name: str
    duration: float
    start: np.ndarray            # Grasp-site start (x, y, z)
    end: np.ndarray
    grip: tuple                  # Gripper target (from, to)


class MujocoSlingSim:
    def __init__(self, cfg: dict):
        self.layout = SceneLayout.from_config(cfg)
        self.grasp = grasp_geometry(2.0 * cfg["puck"]["radius_m"])
        self.model = _build_sling_spec(cfg, self.layout, self.grasp).compile()
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        m = self.model
        self.dt = m.opt.timestep
        self._arm_qadr = _qadr(m, ARM_JOINTS)
        self._arm_act = [m.actuator(n).id for n in ARM_JOINTS]
        self._grip_qadr = _qadr(m, ("gripper",))[0]
        self._grip_act = m.actuator("gripper").id
        self._arm_lo = m.jnt_range[[m.joint(n).id for n in ARM_JOINTS], 0]
        self._arm_hi = m.jnt_range[[m.joint(n).id for n in ARM_JOINTS], 1]
        self._puck_qadr = _qadr(m, PUCK_JOINTS)
        self._puck_vadr = [int(m.joint(n).dofadr[0]) for n in PUCK_JOINTS]
        self._puck_body = m.body("puck").id
        self._gripper_body = m.body("gripper").id
        self._site = m.site("grasp").id
        self._nominal_mass = float(m.body_mass[self._puck_body])
        self._nominal_inertia = m.body_inertia[self._puck_body].copy()
        self.z_grasp = 0.5 * cfg["puck"]["thickness_m"]
        self.wall_calibration = calibrate_walls(self, cfg)
        self.apply_params(cfg)
        self.held = False
        self.phase = "idle"
        self.t = 0.0
        self.band_force = np.zeros(2)
        self.q_arm = None

    # -------------------------------------------------- domain randomization
    def apply_params(self, cfg: dict):
        """Band, friction, wall restitution, puck mass and gate width (no recompile)."""
        m = self.model
        self.cfg = cfg
        self.band = BandModel.from_config(cfg)
        self._mu = cfg["puck"]["friction_kinetic"]
        damping = self.wall_calibration(cfg["board"]["restitution"])
        for name in WALL_PAIRS:
            m.pair_solref[m.pair(f"puck_{name}").id] = (-_stiffness(cfg), -damping)
        scale = cfg["puck"]["mass_kg"] / self._nominal_mass
        m.body_mass[self._puck_body] = cfg["puck"]["mass_kg"]
        m.body_inertia[self._puck_body] = self._nominal_inertia * scale
        for name, (xmin, xmax, _, _) in zip(("divider_left", "divider_right"), _divider_boxes(cfg)):
            gid = m.geom(name).id
            m.geom_pos[gid][0] = 0.5 * (xmin + xmax)
            m.geom_size[gid][0] = 0.5 * (xmax - xmin)

    # ------------------------------------------------------------------ IK
    def solve_ik(self, target, q0=None) -> np.ndarray:
        """Arm joints that put the grasp site at target, gripper down, jaws along x."""
        m, d = self.model, self._ik_data
        target = np.asarray(target, float)

        def residual(q):
            d.qpos[self._arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            R = d.site_xmat[self._site].reshape(3, 3)
            return np.concatenate([d.site_xpos[self._site] - target,
                                   0.05 * (R[:, 0] - (0.0, 0.0, -1.0)), [0.05 * R[1, 2]]])

        d.qpos[:] = 0.0
        if q0 is not None:
            return least_squares(residual, q0, bounds=(self._arm_lo, self._arm_hi), xtol=1e-10).x
        starts = np.random.default_rng(0).uniform(self._arm_lo, self._arm_hi, (30, len(ARM_JOINTS)))
        sols = [least_squares(residual, s, bounds=(self._arm_lo, self._arm_hi)) for s in starts]
        good = [s for s in sols if s.cost < 1e-10] or [min(sols, key=lambda s: s.cost)]
        return min(good, key=lambda s: abs(s.x[4])).x  # Least wrist roll

    def site_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._site].copy()

    # ------------------------------------------------------------- state
    @property
    def puck_pos(self) -> np.ndarray:
        return self.data.qpos[self._puck_qadr[:2]].copy()

    @property
    def puck_vel(self) -> np.ndarray:
        return self.data.qvel[self._puck_vadr[:2]].copy()

    def band_vertex(self):
        """Visual band V vertex (back of the puck), or None if the band is straight."""
        p = self.puck_pos
        if p[1] >= self.band.band_y:
            return None
        return np.array([p[0], p[1] - self.cfg["puck"]["radius_m"]])

    def reset(self, x0: float):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        m.opt.timestep = self.dt
        d.qpos[self._puck_qadr] = [x0, self.band.band_y, 0.0]
        self.q_arm = self.solve_ik(HOME_XYZ)
        d.qpos[self._arm_qadr] = self.q_arm
        d.qpos[self._grip_qadr] = self.grasp.q_open
        d.ctrl[self._arm_act] = self.q_arm
        d.ctrl[self._grip_act] = self.grasp.q_open
        self.held, self.phase, self.t = False, "idle", 0.0
        self.band_force = np.zeros(2)
        mujoco.mj_forward(m, d)

    # -------------------------------------------------------------- step
    def _step(self, release_state=None):
        """One MuJoCo step with the band and friction forces."""
        m, d = self.model, self.data
        d.xfrc_applied[self._gripper_body] = 0.0
        vx, vy = self._puck_vadr[0], self._puck_vadr[1]
        if self.held:
            pos = self.puck_pos
            self.band_force = self.band.force_load(pos) if pos[1] < self.band.band_y else np.zeros(2)
            f3 = np.array([*self.band_force, 0.0])
            arm = d.site_xpos[self._site] - d.xipos[self._gripper_body]
            d.xfrc_applied[self._gripper_body, :3] = f3
            d.xfrc_applied[self._gripper_body, 3:] = np.cross(arm, f3)
            d.qfrc_applied[[vx, vy]] = 0.0
        else:
            apply_puck_friction(m, d, vx, vy, self._puck_body, self._mu)
            if release_state is not None:
                pos = self.puck_pos
                u = self.band.direction_sum(pos)
                self.band_force = float(self.band.tension_unload(self.band.elongation(pos),
                                                                 release_state["s_max"])) * u
                d.qfrc_applied[vx] += self.band_force[0]
                d.qfrc_applied[vy] += self.band_force[1]
            else:
                self.band_force = np.zeros(2)
        old = self.puck_pos
        mujoco.mj_step(m, d)
        if self.held:  # Rigid grasp: the puck follows the grasp site
            new = d.site_xpos[self._site][:2]
            d.qpos[self._puck_qadr[:2]] = new
            d.qvel[[vx, vy]] = (new - old) / m.opt.timestep
        self.t += m.opt.timestep

    # ----------------------------------------------------------- episode
    def segments(self, plan: ShotPlan, observe_s: float, rest_s: float = 0.3) -> list:
        """Grasp-site path. The arm waits at HOME_XYZ (clear of the overhead camera's view of the
        puck) for rest_s, then comes to the puck."""
        by, z, zu = self.band.band_y, self.z_grasp, self.z_grasp + LIFT_M
        g = self.grasp
        P = lambda xy, h=z: np.array([xy[0], xy[1], h])  # noqa: E731
        rest, slid = P((plan.x0, by)), P((plan.slide_x, by))
        pull = P(plan.pull)

        def move(name, a, b, speed, min_s, grip=(g.q_grasp, g.q_grasp)):
            return Segment(name, max(min_s, np.linalg.norm(b - a) / speed), a, b, grip)

        home = np.array(HOME_XYZ)
        segs = [Segment("idle", rest_s, home, home, (g.q_open, g.q_open)),
                Segment("approach", 0.6, home, P((plan.x0, by), zu), (g.q_open, g.q_open)),
                Segment("reach", 0.5, P((plan.x0, by), zu), rest, (g.q_open, g.q_open)),
                Segment("grasp", 0.3, rest, rest, (g.q_open, g.q_grasp)),
                move("slide", rest, slid, SPEEDS["slide"], 0.3),
                move("pull", slid, pull, SPEEDS["pull"], 0.4),
                Segment("look", observe_s, pull, pull, (g.q_grasp, g.q_grasp))]
        last = pull
        if plan.corrected is not None:
            corr = P(plan.corrected)
            segs += [move("correct", pull, corr, SPEEDS["correct"], 0.25),
                     Segment("hold", 0.2, corr, corr, (g.q_grasp, g.q_grasp))]
            last = corr
        segs.append(Segment("release", 0.4, last, last + [0, 0, LIFT_M], (g.q_grasp, g.q_open)))
        segs.append(Segment("retreat", 0.6, last + [0, 0, LIFT_M], home, (g.q_open, g.q_open)))
        return segs

    def run(self, plan: ShotPlan, on_frame=None, frame_dt: float = 0.01, observe_s: float = 0.3,
            max_flight_s: float = 2.0, after_gate_s: float = 0.4, rest_s: float = 0.3) -> MujocoShotResult:
        """Run one shot. on_frame(sim) is called every frame_dt of sim time.

        frame_dt can be a function of the phase name (finer frames for the release).
        """
        self.reset(plan.x0)
        res = MujocoShotResult()
        next_frame = [0.0]
        dt_of = frame_dt if callable(frame_dt) else (lambda _phase: frame_dt)

        def frame():
            if on_frame is not None and self.t + 1e-12 >= next_frame[0]:
                # Frame clock on a fixed grid (a camera does not drift by the step size each frame).
                next_frame[0] = max(next_frame[0] + dt_of(self.phase), self.t)
                on_frame(self)

        def control(seg, t_seg):
            s = min_jerk(t_seg / seg.duration)
            target = seg.start + s * (seg.end - seg.start)
            self.q_arm = self.solve_ik(target, self.q_arm)
            self.data.ctrl[self._arm_act] = self.q_arm
            self.data.ctrl[self._grip_act] = seg.grip[0] + s * (seg.grip[1] - seg.grip[0])
            return target

        release_state = None
        face_y = -(0.5 * self.cfg["board"]["divider_thickness_m"] + self.cfg["puck"]["radius_m"])
        t_cross = None
        released = False
        for seg in self.segments(plan, observe_s, rest_s):
            if not released or seg.name == "release":
                self.phase = seg.name
            t0 = self.t
            res.events.setdefault(seg.name, t0)
            n_ctrl = -1
            if seg.name == "release":
                released = True
                self.held = False
                res.release_target = seg.start[:2].copy()
                res.release_pos = self.puck_pos
                self.data.qvel[self._puck_vadr] = 0.0
                if res.release_pos[1] < self.band.band_y - 1e-6:
                    release_state = {"s_max": self.band.elongation(res.release_pos)}
                    self.model.opt.timestep = RELEASE_TIMESTEP_S
            while self.t - t0 < seg.duration - 1e-9:
                k = int((self.t - t0) / CONTROL_DT_S)
                if k != n_ctrl:
                    n_ctrl = k
                    control(seg, self.t - t0)
                self._step(release_state)
                res.max_band_force_n = max(res.max_band_force_n, float(np.linalg.norm(self.band_force)))
                if release_state is not None and self.puck_pos[1] >= self.band.band_y:
                    self._exit(res)
                    self.phase = "flight"
                    release_state = None
                    self.model.opt.timestep = self.dt
                if released:
                    t_cross = self._flight_step(res, face_y, t_cross)
                frame()
            if seg.name == "grasp":
                self.held = True
        if release_state is not None:  # Band could not push the puck over the band line
            release_state = None
            self.model.opt.timestep = self.dt
        # Flight: arm parked above, puck free.
        self.phase = "flight"
        t_rel = self.t
        while self.t - t_rel < max_flight_s:
            self._step()
            t_cross = self._flight_step(res, face_y, t_cross)
            frame()
            t_end = t_cross if t_cross is not None else res.rebound_t
            if not np.any(self.puck_vel) or (t_end is not None and self.t - t_end > after_gate_s):
                break
        if np.isnan(res.miss_m):
            res.miss_m = SHORT_MISS_M + (face_y - self.puck_pos[1])
        self.phase = "done"
        if on_frame is not None:
            on_frame(self)
        return res

    def _exit(self, res: MujocoShotResult):
        """Puck crosses the band line: energy_transfer losses, as BandModel.release."""
        v = self.data.qvel[self._puck_vadr[:2]] * math.sqrt(self.band.p.energy_transfer)
        self.data.qvel[self._puck_vadr[:2]] = v
        res.exit_pos, res.exit_vel, res.launched = self.puck_pos, v.copy(), True

    def _flight_step(self, res, face_y, t_cross):
        x, y = self.puck_pos
        res.log.append((self.t, x, y))
        if np.isnan(res.miss_m) and y >= face_y:
            res.miss_m = abs(x)
        if res.launched and not res.rebounded and t_cross is None and self.puck_vel[1] < 0.0:
            res.rebounded, res.rebound_t = True, self.t
        if t_cross is None and y > 0.0:
            res.success = not res.rebounded
            t_cross = self.t
        return t_cross


class WallCalibration:
    """Wall restitution vs contact damping, measured in the sling scene."""

    def __init__(self, damping: np.ndarray, restitution: np.ndarray):
        order = np.argsort(restitution)
        self.damping, self.restitution = damping[order], restitution[order]

    def __call__(self, e: float) -> float:
        if not self.restitution[0] <= e <= self.restitution[-1]:
            raise ValueError(f"wall restitution {e} outside the calibrated range")
        return float(np.interp(e, self.restitution, self.damping))


_WALL_CACHE: dict = {}


def calibrate_walls(sim: MujocoSlingSim, cfg: dict, impact_speed: float = 2.0, n: int = 24,
                    n_phases: int = 5) -> WallCalibration:
    """Same method as mujoco_goalkeeper.calibrate_contacts, side wall only (no friction)."""
    key = (repr(cfg["sim"]), repr(cfg["puck"]), repr(cfg["board"]["width_m"]))
    if key in _WALL_CACHE:
        return _WALL_CACHE[key]
    m = sim.model
    d = mujoco.MjData(m)
    k = _stiffness(cfg)
    damping = 2.0 * math.sqrt(k) * np.geomspace(0.003, 2.0, n)
    pid = m.pair("puck_wall_right").id
    saved = m.pair_solref[pid].copy()
    hw, r = 0.5 * cfg["board"]["width_m"], cfg["puck"]["radius_m"]
    q, v = sim._puck_qadr, sim._puck_vadr
    out = []
    for b in damping:
        m.pair_solref[pid] = (-k, -b)
        es = []
        for ds in np.arange(n_phases) / n_phases * impact_speed * m.opt.timestep:
            mujoco.mj_resetData(m, d)
            d.qpos[q] = [hw - r - 0.02 + ds, 0.10, 0.0]
            d.qvel[v] = [impact_speed, 0.0, 0.0]
            for _ in range(int(0.25 / m.opt.timestep)):
                mujoco.mj_step(m, d)
                if d.qvel[v[0]] < 0.0:
                    mujoco.mj_step(m, d, nstep=int(0.012 / m.opt.timestep))
                    break
            es.append(-d.qvel[v[0]] / impact_speed)
        out.append(np.mean(es))
    m.pair_solref[pid] = saved
    _WALL_CACHE[key] = WallCalibration(damping, np.array(out))
    return _WALL_CACHE[key]
