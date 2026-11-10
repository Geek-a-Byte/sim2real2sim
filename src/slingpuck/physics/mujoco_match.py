"""MuJoCo match: the SO-101 against a human opponent, face to face across the board.

Scene: so101_slingpuck.build (table, board, bands, 5 black robot pucks, 5 white human pucks) plus a grasp
site between the jaw tips and a simple hand (mocap) that comes from the opponent's end.
All 10 pucks are in the physics: Coulomb friction, bounces off the walls, the divider and each other.

Robot (scripted, the "sling while the opponent reloads" rule, as policies.match_scripted.ReloadRule):
  guard  the closed gripper stands in the gate (blocks a shot: gate 3.4 cm, puck 2.54 cm)
  fetch  when the human is reloading and a black or white puck rests on the robot's half: move above
         it, grasp, drag it to the band, pull back, hold, release; then back to the guard pose.
         A fetch that has not grasped yet is dropped when the human starts to pull.
Human (scripted, timings and aim from env.yaml `opponent`): reload (hand to a puck on its half),
place it on its band, pull back, hold, release. threat_prob of the shots aim at the gate.

The human's hand holds a puck by a stiff connect constraint. The robot holds a puck by its jaws
(skills.grasp_model): "friction" = jaw contacts with Coulomb friction under a constant gripper torque, so the
puck can slip or be missed; "constraint" = the same stiff constraint as the hand. While a puck is held behind
its band, the band force acts on it, and the grip passes it to the arm (or the hand). On release, the band
(unloading curve) pushes the puck to its band line, then the puck keeps energy_transfer of its energy (as
BandModel.release). The human's band is the same model, mirrored.

Robot skills (layer 1 of docs/PLAN.md, scripted): skill_pick, skill_seat, skill_draw, skill_hold, skill_release,
skill_free, skill_block each return a list of moves. A puck in the touching start row has no room for a jaw at its
sides: skill_free first pushes it out of the row with the two jaw tips (a fork, so that it goes straight). A shot (launch x, aim, pull) comes from the shot model when a
ShotSelector is given; else it is the fixed straight pull. fetch_log has one record per fetch (times, errors,
exit state, outcome).

Grasp: the gripper points straight down and the jaws close on the left and right sides of the puck. The
grasp site is the jaw-tip midpoint at the grasp opening (grasp_geometry).

Rule (placeholder, as match_env): the first side with no pucks on it wins.
Frames: the scene is in the robot frame (so101_slingpuck); game logic is in the board frame.
"""
import math
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

from src.slingpuck.opponent.scripted_opponent import LAUNCH_X_MAX_M, MISS_MIN_M, threat_half_width
from src.slingpuck.physics.band_model import BandModel
from src.slingpuck.so101_slingpuck import GRAVITY, PUCKS_PER_SIDE, SO101_XML, board_pose, build, coulomb_along_velocity

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
JAW_OPEN_EXTRA_M = 0.012        # Jaw tip gap beyond the puck diameter when open

CONTROL_DT_S = 0.01
LIFT_M = 0.035                  # Jaw tips clear the 1.5 cm walls and the pucks
REST_SPEED = 0.02               # Below this a puck counts as resting
HOLD_TIMECONST_S = 0.002        # Grip stiffness: about 1 mm of give under the 5 N band load
SKIN = [0.87, 0.70, 0.58, 1]
HUMAN_HOME = (0.0, 0.26, 0.09)  # Board frame: above the open end, behind the opponent's end wall
SPEED_MOVE, SPEED_DRAG = 0.25, 0.12   # Grasp-site / hand speeds, m/s
ROBOT_PULL_M = 0.025            # Straight pull for the robot's shot (a pre-stretched band launches almost straight)
CLOSE_PAST_RAD = 0.2            # Friction grip: the jaw target is this far past the contact, so the torque limit sets the grip
GRIP_LOST_S = 0.02              # Friction grip: no contact on one jaw for this long = the puck left the jaws
JAW_CONTACT_TIME_S = 0.002      # Jaw-puck contact time constant (friction grip)
JAW_CLOSE_S = 0.25              # Jaw target ramp. Not scaled with skills.speed_scale
ARRIVE_TOL_M, ARRIVE_MAX_S = 0.003, 0.3   # Before a descent the arm must be at the target: it lags at high speed
MAX_DESCENT_SPEED = 1.25        # Descents are not faster than this speed_scale: the jaws overshot into the board
DESCENT_S = 0.25
SPREAD_S = 0.3
RISE_S = 0.12                   # Up from the gate pose
GRIP_SETTLE_S = 0.15            # The jaw lags its target under the torque limit: wait for it, then check the grip
PUSH_OUT_M = 0.026              # skill_free: how far the puck moves out of the row
FORK_Q = 0.0                    # Gripper joint for the push: the two jaw tips stand apart, a fork on the puck
RELEASE_GIVE_UP_S = 0.5         # A released puck that stays behind the band line this long did not launch


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


@dataclass
class Move:
    name: str
    start: np.ndarray            # Board frame (x, y, z)
    end: np.ndarray
    duration: float
    grip: tuple = (None, None)   # Gripper joint target (from, to); None = keep
    event: str | None = None     # "attach" or "release" at the end of the move
    arrive: bool = False         # Robot: at the end, wait until the grasp site is at the end point

    def at(self, t):
        s = min_jerk(t / self.duration) if self.duration > 0 else 1.0
        return self.start + s * (self.end - self.start), s


def _timed(name, a, b, speed, min_s, **kw):
    return Move(name, a, b, max(min_s, float(np.linalg.norm(b - a)) / speed), **kw)


class MatchSim:
    def __init__(self, cfg: dict, seed: int = 0, opponent: bool = True, selector=None):
        """opponent=False: no human shots (the skill tests). selector: primitives.sling.ShotSelector for the shots."""
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.has_opponent, self.selector = opponent, selector
        self.friction_grip = cfg["skills"]["grasp_model"] == "friction"
        self.speed = cfg["skills"]["speed_scale"]
        self.joint_offset = np.zeros(len(ARM_JOINTS))    # Calibration error: real joint = IK solution + offset
        self._guard_q = None
        self.grasp = grasp_geometry(2.0 * cfg["puck"]["radius_m"])
        spec = build(cfg)
        add_grasp_site(spec, self.grasp)
        # The jaws touch the puck's true shape. The sphere collider (for the walls) reaches 7 mm above the
        # puck, and an open jaw that came down near a puck was pushed off by it.
        for side in ("a", "o"):
            for i in range(PUCKS_PER_SIDE):
                spec.body(f"puck_{side}{i}").add_geom(
                    name=f"puck_{side}{i}_jaw", type=mujoco.mjtGeom.mjGEOM_CYLINDER, density=0.0,
                    size=[cfg["puck"]["radius_m"], 0.5 * cfg["puck"]["thickness_m"], 0], rgba=[0, 0, 0, 0], group=3,
                    contype=0, conaffinity=2)
        hand = spec.worldbody.add_body(name="hand", mocap=True, pos=[0.6, 0, 0.1])
        hand.add_geom(name="palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.04, 0.012], pos=[0.03, 0, 0.02],
                      rgba=SKIN, contype=0, conaffinity=0)
        for k, y in enumerate((-0.02, 0.0, 0.02)):
            hand.add_geom(name=f"finger{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.008, 0, 0],
                          fromto=[-0.01, y, 0.018, -0.03, y, 0.004], rgba=SKIN, contype=0, conaffinity=0)
        hand.add_geom(name="thumb", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.009, 0, 0],
                      fromto=[0.0, 0.035, 0.015, -0.02, 0.03, 0.004], rgba=SKIN, contype=0, conaffinity=0)
        # Holding: a stiff connect constraint per puck and holder, off until the grasp. Moving a held puck by
        # position instead pushed it ever deeper into the other pucks (unbounded contact forces).
        for side in ("a", "o"):
            for i in range(PUCKS_PER_SIDE):
                for holder in ("gripper", "hand"):
                    spec.add_equality(name=f"hold_{side}{i}_{holder}", type=mujoco.mjtEq.mjEQ_CONNECT,
                                      objtype=mujoco.mjtObj.mjOBJ_BODY, name1=f"puck_{side}{i}", name2=holder,
                                      active=False, solref=[HOLD_TIMECONST_S, 1.0])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._ik = mujoco.MjData(self.model)
        m = self.model
        self.dt = m.opt.timestep
        self.P, _ = board_pose(cfg)
        self.band = BandModel.from_config(cfg)
        self.r = cfg["puck"]["radius_m"]
        self.z_mid = 0.5 * cfg["puck"]["thickness_m"]
        self.face_y = 0.5 * cfg["board"]["divider_thickness_m"] + self.r
        self._arm_q = [int(m.joint(n).qposadr[0]) for n in ARM_JOINTS]
        self._arm_act = [m.actuator(n).id for n in ARM_JOINTS]
        self._grip_act = m.actuator("gripper").id
        self._grip_q = int(m.joint("gripper").qposadr[0])
        self._lo = m.jnt_range[[m.joint(n).id for n in ARM_JOINTS], 0]
        self._hi = m.jnt_range[[m.joint(n).id for n in ARM_JOINTS], 1]
        self._site = m.site("grasp").id
        self._hand = int(m.body_mocapid[m.body("hand").id])
        # Pucks: names, owner colour, rest position (board), joint addresses.
        self.pucks = []
        for side in ("a", "o"):
            for i in range(PUCKS_PER_SIDE):
                name = f"puck_{side}{i}"
                b = m.body(name).id
                self.pucks.append({
                    "name": name, "colour": "black" if side == "a" else "white",
                    "rest": m.body_pos[b][:2].copy(),
                    "q": [int(m.joint(f"{name}_{j}").qposadr[0]) for j in ("x", "y")],
                    "v": [int(m.joint(f"{name}_{j}").dofadr[0]) for j in ("x", "y")],
                    "col": m.geom(f"{name}_jaw").id, "body": b,
                    "eq": {h: m.equality(f"hold_{name[5:]}_{h}").id for h in ("gripper", "hand")}})
        # Contacts with the gripper: its collision boxes (contype 2) touch puck colliders (conaffinity 2).
        # Every other arm geom touches nothing. A held puck is taken out (conaffinity 0).
        base = m.body("base").id
        jaw_bodies = {m.body("gripper").id, m.body("moving_jaw_so101_v1").id}
        solref, solimp = m.pair_solref[0].copy(), m.pair_solimp[0].copy()
        for g in range(m.ngeom):
            if m.body_rootid[m.geom_bodyid[g]] != base:
                continue
            on_jaw = m.geom_bodyid[g] in jaw_bodies and m.geom_group[g] == 3
            m.geom_contype[g], m.geom_conaffinity[g] = (2, 0) if on_jaw else (0, 0)
            if on_jaw:   # Friction grip: a soft contact. A stiff one flings the arm when a jaw lands on a puck
                if not self.friction_grip:
                    m.geom_condim[g], m.geom_solref[g], m.geom_solimp[g] = 1, solref, solimp
        for p in self.pucks:
            m.geom_conaffinity[p["col"]] = 2
            m.geom_solref[p["col"]], m.geom_solimp[p["col"]] = solref, solimp
        self._fixed_jaw, self._moving_jaw = jaw_bodies_ids = (m.body("gripper").id, m.body("moving_jaw_so101_v1").id)
        # MuJoCo filters collisions by body first, with masks fixed at compile time: without these lines the
        # jaws never touch a puck, whatever the geom masks say.
        for b in range(m.nbody):
            if m.body_rootid[b] == base:
                m.body_contype[b], m.body_conaffinity[b] = (2, 0) if b in jaw_bodies_ids else (0, 0)
        for p in self.pucks:
            m.body_conaffinity[p["body"]] = 2
        self._jaw_geoms = {g: m.geom_bodyid[g] for g in range(m.ngeom)
                           if m.geom_bodyid[g] in jaw_bodies_ids and m.geom_group[g] == 3}
        if self.friction_grip:
            grip = cfg["gripper"]
            m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL   # The elliptic cone let a gripped object slip
            m.opt.iterations = 50
            m.actuator_forcerange[self._grip_act] = (-grip["grip_torque_nm"], grip["grip_torque_nm"])
            for g in self._jaw_geoms:
                m.geom_condim[g] = 3
                m.geom_solref[g] = (JAW_CONTACT_TIME_S, 1.0)
                m.geom_friction[g] = (grip["jaw_friction"], 0.005, 0.0001)
        # How far the open jaws' center lies from the grasp site (see skill_pick).
        ik = self._ik
        ik.qpos[:] = self.model.key_qpos[0]
        ik.qpos[self._grip_q] = self.grasp.q_open
        mujoco.mj_kinematics(m, ik)
        self._tips = [m.geom(f"{j}_jaw_sph_tip1").id for j in ("fixed", "moving")]
        self.open_offset_m = 0.5 * float(np.linalg.norm(ik.geom_xpos[self._tips[0]] - ik.geom_xpos[self._tips[1]])) - self.r
        ik.qpos[self._grip_q] = FORK_Q
        mujoco.mj_kinematics(m, ik)
        self.fork_half_m = 0.5 * float(np.linalg.norm(ik.geom_xpos[self._tips[0]] - ik.geom_xpos[self._tips[1]]))
        self.puck_friction_n = cfg["puck"]["friction_kinetic"] * cfg["puck"]["mass_kg"] * GRAVITY
        self.see_noise_m = 0.0     # Camera error on a puck position when the arm plans a pick
        self.reset()

    # ------------------------------------------------------------- frames
    def to_world(self, b) -> np.ndarray:
        b = np.asarray(b, float)
        z = b[2] if len(b) > 2 else 0.0
        return self.P + np.array([b[1], -b[0], z])

    def to_board(self, w) -> np.ndarray:
        d = np.asarray(w, float) - self.P
        return np.array([-d[1], d[0], d[2]])

    def puck_pos(self, p) -> np.ndarray:
        return p["rest"] + self.data.qpos[p["q"]]

    def puck_vel(self, p) -> np.ndarray:
        return self.data.qvel[p["v"]].copy()

    def side_of(self, p) -> str:
        return "robot" if self.puck_pos(p)[1] < 0.0 else "human"

    def counts(self) -> dict:
        c = {"robot": 0, "human": 0}
        for p in self.pucks:
            c[self.side_of(p)] += 1
        return c

    # ---------------------------------------------------------------- IK
    def solve_ik(self, target_board, q0=None) -> np.ndarray:
        """Arm joints: grasp site at the target, gripper pointing down, jaws across the board."""
        m, d = self.model, self._ik
        target = self.to_world(target_board)

        def residual(q):
            d.qpos[self._arm_q] = q
            mujoco.mj_kinematics(m, d)
            R = d.site_xmat[self._site].reshape(3, 3)
            return np.concatenate([d.site_xpos[self._site] - target, 0.05 * (R[:, 0] - (0, 0, -1)),
                                   [0.05 * R[0, 2]]])

        if q0 is not None:
            return least_squares(residual, q0, bounds=(self._lo, self._hi), xtol=1e-10).x
        starts = np.random.default_rng(0).uniform(self._lo, self._hi, (30, len(ARM_JOINTS)))
        sols = [least_squares(residual, s, bounds=(self._lo, self._hi)) for s in starts]
        return min(sols, key=lambda s: s.cost).x

    def site_board(self) -> np.ndarray:
        return self.to_board(self.data.site_xpos[self._site])

    # -------------------------------------------------------------- reset
    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        self.t = 0.0
        self.guard = np.array([0.0, -(self.face_y + 0.004), self.z_mid])
        if self._guard_q is None:
            self._guard_q = self.solve_ik(self.guard)     # Global IK: 30 starts, once
        self.q = self._guard_q.copy()
        d.qpos[self._arm_q] = self.q + self.joint_offset
        d.ctrl[self._arm_act] = self.q + self.joint_offset
        d.ctrl[self._grip_act] = d.qpos[int(m.joint("gripper").qposadr[0])] = 0.0
        self.robot = {"state": "guard", "moves": [], "k": 0, "t0": 0.0, "held": None, "grip": 0.0,
                      "target": self.guard.copy()}
        self.auto_fetch = True     # False: the caller commands the skills (eval/skill_eval.py)
        self.human = {"state": "idle", "moves": [], "k": 0, "t0": 0.0, "held": None,
                      "pos": np.array(HUMAN_HOME, float), "aim": None, "threat": None, "next_t": 0.3}
        self.releases = []        # Active band releases: {"p", "side", "s_max"}
        self.band_force = {"robot": np.zeros(2), "human": np.zeros(2)}
        self.shots = {"robot": 0, "human": 0}
        self.winner = None
        self.fetch_log = []       # One record per fetch
        self._flights = []        # Launched robot pucks whose outcome is open
        self._grip_lost_s = 0.0
        self._ctrl_k = -1
        self._set_hand(self.human["pos"])
        mujoco.mj_forward(m, d)

    def _set_hand(self, pos_board):
        self.data.mocap_pos[self._hand] = self.to_world(pos_board)

    # ------------------------------------------------------------ choices
    def _resting_pucks(self, side) -> list:
        out = []
        for p in self.pucks:
            if p is self.robot["held"] or p is self.human["held"] or any(r["p"] is p for r in self.releases):
                continue
            pos = self.puck_pos(p)
            if (pos[1] < -self.face_y) == (side == "robot") and abs(pos[1]) > self.face_y \
                    and np.linalg.norm(self.puck_vel(p)) < REST_SPEED:
                out.append(p)
        return out

    def human_phase(self) -> str:
        return self.human["state"]

    # ------------------------------------------------------------- skills
    def _timed_robot(self, name, a, b, speed, min_s, **kw):
        return _timed(name, a, b, speed * self.speed, min_s / self.speed, **kw)

    def _close_target(self) -> float:
        return self.grasp.q_grasp - CLOSE_PAST_RAD if self.friction_grip else self.grasp.q_grasp

    def _rise(self, here) -> list:
        """Up first if the jaws are low (the gate pose): a straight line to the next target hit the puck at speed."""
        top = self.z_mid + LIFT_M
        if here[2] >= top - 1e-3:
            return []
        return [Move("rise", here, np.array([here[0], here[1], top]), RISE_S / self.speed)]

    def skill_pick(self, here, puck_xy) -> list:
        """From `here`: above the puck with open jaws, down, close."""
        z, g = self.z_mid, self.grasp
        # The grasp site is the jaw midpoint at the grasp opening, so the fixed jaw stands one puck radius from
        # it. Center the open jaws on the puck instead; closing then moves the puck to the grasp site.
        fixed_side = np.sign(self.to_board(self.data.geom_xpos[self._tips[0]])[0] - self.site_board()[0])
        x = puck_xy[0] + fixed_side * self.open_offset_m
        above = np.array([x, puck_xy[1], z + LIFT_M])
        at = np.array([x, puck_xy[1], z])
        rise = self._rise(here)
        start = rise[-1].end if rise else here
        return [*rise, self._timed_robot("to puck", start, above, SPEED_MOVE, 0.3, grip=(None, g.q_open), arrive=True),
                Move("down", above, at, DESCENT_S / min(self.speed, MAX_DESCENT_SPEED), arrive=True),
                Move("grasp", at, at, JAW_CLOSE_S, grip=(g.q_open, self._close_target())),   # The jaw has its own speed
                Move("settle", at, at, GRIP_SETTLE_S, event="attach")]

    def _fixed_side(self) -> float:
        return float(np.sign(self.to_board(self.data.geom_xpos[self._tips[0]])[0] - self.site_board()[0]))

    def jaw_blockers(self, p) -> list:
        """Other pucks on the board that leave no room for an open jaw at the sides of puck p."""
        pos = self.puck_pos(p)
        return [q for q in self.pucks if q is not p and abs(self.puck_pos(q)[0] - pos[0]) < self.cfg["shot"]["jaw_room_x_m"]
                and abs(self.puck_pos(q)[1] - pos[1]) < self.cfg["shot"]["jaw_room_y_m"]]

    def skill_free(self, here, puck_xy) -> list:
        """Push a puck straight back (toward the band) out of its row. The fork comes down in front of it."""
        z = self.z_mid
        x = puck_xy[0] - self._fixed_side() * (self.r - self.fork_half_m)    # Fork center on the puck's center
        y0 = puck_xy[1] + self.r + 0.006                                     # In front of the puck, clear of it
        y1 = puck_xy[1] + math.sqrt(self.r ** 2 - self.fork_half_m ** 2) - PUSH_OUT_M
        above, start, end = (np.array([x, y0, z + LIFT_M]), np.array([x, y0, z]), np.array([x, y1, z]))
        rise = self._rise(here)
        return [*rise, self._timed_robot("to front", rise[-1].end if rise else here, above, SPEED_MOVE, 0.3, grip=(None, FORK_Q),
                                         arrive=True),
                Move("down", above, start, DESCENT_S / min(self.speed, MAX_DESCENT_SPEED), arrive=True),
                self._timed_robot("push", start, end, SPEED_DRAG, 0.3),
                Move("clear", end, end + [0.0, 0.0, LIFT_M], 0.2 / self.speed)]

    def skill_spread(self, here, hole_xy) -> list:
        """Open the jaws in the hole that a pushed-out puck left: the pucks at its sides move apart."""
        z, side = self.z_mid, self._fixed_side()
        start_x = hole_xy[0] - side * (self.r - self.fork_half_m)            # Fork center on the hole
        end_x = hole_xy[0] + side * self.open_offset_m                       # Open jaws centered on the hole
        above = np.array([start_x, hole_xy[1], z + LIFT_M])
        at, end = np.array([start_x, hole_xy[1], z]), np.array([end_x, hole_xy[1], z])
        return [self._timed_robot("to hole", here, above, SPEED_MOVE, 0.2, grip=(None, FORK_Q), arrive=True),
                Move("down", above, at, DESCENT_S / min(self.speed, MAX_DESCENT_SPEED), arrive=True),
                Move("spread", at, end, SPREAD_S / min(self.speed, MAX_DESCENT_SPEED), grip=(FORK_Q, self.grasp.q_open)),
                Move("clear", end, end + [0.0, 0.0, LIFT_M], 0.2 / self.speed)]

    def skill_seat(self, here, launch_x: float) -> list:
        """Drag the held puck to the band line at launch_x."""
        seat = np.array([launch_x, self.band.band_y, self.z_mid])
        return [self._timed_robot("to band", here, seat, SPEED_DRAG, 0.3, event="seated")]

    def skill_hold(self, here, seconds: float) -> list:
        """Keep the puck pressed where it is (on the band, or pulled back)."""
        return [Move("hold", here, here, seconds)]

    def pull_point(self, launch_x: float, aim: float, pull: float) -> np.ndarray:
        return np.array([launch_x + pull * math.sin(aim), self.band.band_y - pull * math.cos(aim), self.z_mid])

    def skill_draw(self, here, launch_x: float, aim: float, pull: float) -> list:
        """Pull the held puck back from the seat, `pull` in the direction `aim` (as ShotSim.pull_point)."""
        return [Move("pull", here, self.pull_point(launch_x, aim, pull), max(0.4 * pull / ROBOT_PULL_M, 0.1) / self.speed,
                     event="drawn")]

    def skill_release(self, here) -> list:
        """Open the jaws in place (the band launches the puck), then lift."""
        g = self.grasp
        lift = here + [0.0, 0.0, LIFT_M]
        open_s = self.cfg["skills"]["open_time_s"]
        return [Move("hold", here, here, 0.2 / self.speed, event="release"),
                Move("open", here, here, open_s, grip=(self._close_target(), g.q_open)),
                Move("lift", here, lift, 0.25 / self.speed)]

    def skill_block(self, here, x: float = 0.0) -> list:
        """Closed gripper in the gate at x. x is limited so that the jaws stay in front of the gate."""
        target = np.array([x, self.guard[1], self.guard[2]])
        up = np.array([here[0], here[1], max(here[2], self.z_mid + LIFT_M)])
        over = np.array([x, self.guard[1], self.z_mid + LIFT_M])
        return [Move("up", here, up, 0.15 / self.speed),
                self._timed_robot("to gate", up, over, SPEED_MOVE, 0.3, grip=(None, 0.0), arrive=True),
                Move("down", over, target, DESCENT_S / min(self.speed, MAX_DESCENT_SPEED))]

    def _robot_fetch(self, p, shot=None, seen_xy=None):
        """One shot: pick, seat, draw, release, back to the gate. shot = (launch x, aim, pull). seen_xy: where the
        camera sees the puck (default: its true position)."""
        pos = self.puck_pos(p) if seen_xy is None else np.asarray(seen_xy, float)
        if seen_xy is None and self.see_noise_m:
            pos = pos + self.rng.normal(0.0, self.see_noise_m, 2)
        if shot is None:
            shot = (self.rng.normal(0.0, 0.001), 0.0, ROBOT_PULL_M)
        launch_x, aim, pull = (float(v) for v in shot)
        here = self.robot["target"]
        if not self.robot.pop("freed", False):
            self.fetch_log.append({"puck": p["name"], "shot": (launch_x, aim, pull), "start": pos.copy(), "t0": self.t,
                                   "grasped": False, "slip": False, "outcome": None, "freed": False})
            blockers = self.jaw_blockers(p) if self.friction_grip else []
            if blockers:
                moves = self.skill_free(here, pos)
                sides = {np.sign(self.puck_pos(q)[0] - self.puck_pos(p)[0]) for q in blockers}
                if self.cfg["skills"]["spread_row"] and len(sides) == 2:      # A hole with a puck at each side
                    moves += self.skill_spread(moves[-1].end, pos)
                self.robot.update(state="free", k=0, t0=self.t, puck=p, after=(launch_x, aim, pull), moves=moves)
                return
        seat = np.array([launch_x, self.band.band_y, self.z_mid])
        pulled = self.pull_point(launch_x, aim, pull)
        if not self.friction_grip:
            self.model.geom_conaffinity[p["col"]] = 0   # The open jaws come down around it: no contact
        moves = [*self.skill_pick(here, pos), *self.skill_seat(np.array([pos[0], pos[1], self.z_mid]), launch_x),
                 *self.skill_draw(seat, launch_x, aim, pull), *self.skill_release(pulled)]
        if self.cfg["skills"]["game_rule"] != "greedy_sling":
            moves += self.skill_block(moves[-1].end)
        self.robot.update(state="fetch", k=0, t0=self.t, puck=p, moves=moves)

    def _robot_guard(self):
        p = self.robot.get("puck")
        if p is not None and self.robot["held"] is None:
            self.model.geom_conaffinity[p["col"]] = 2   # Fetch dropped before the grasp
        self.robot.update(state="guard", k=0, t0=self.t, moves=self.skill_block(self.robot["target"]))

    def _human_cycle(self, p):
        c = self.cfg["opponent"]
        pos = self.puck_pos(p)
        threat = bool(self.rng.random() < c["threat_prob"])
        half = threat_half_width(self.cfg)
        aim = self.rng.uniform(-half, half) if threat else \
            self.rng.choice([-1, 1]) * self.rng.uniform(MISS_MIN_M, LAUNCH_X_MAX_M)
        u = self.rng.random()
        pull = c["max_pull_m"] * (0.3 + 0.7 * u)
        reload = max(0.3, float(self.rng.normal(c["reload_time_mu_s"], c["reload_time_std_s"])))
        hold = float(self.rng.uniform(*c["hold_time_range_s"]))
        zc = self.z_mid + 0.012      # Fingertips at the puck's side, palm above
        here = self.human["pos"]
        above = np.array([pos[0], pos[1], zc + 0.04])
        at = np.array([pos[0], pos[1], zc])
        band = np.array([aim, -self.band.band_y, zc])
        pulled = band + [0.0, pull, 0.0]
        self.human.update(state="reload", k=0, t0=self.t, puck=p, aim=aim, threat=threat, moves=[
            Move("reload", here, above, 0.7 * reload), Move("reload", above, at, 0.3 * reload, event="attach"),
            Move("place", at, band, c["place_time_s"]), Move("pull", band, pulled, c["pull_time_s"]),
            Move("hold", pulled, pulled, hold, event="release"),
            Move("idle", pulled, np.array(HUMAN_HOME), 0.5)])

    # ------------------------------------------------------------ scripts
    def _run_script(self, who: dict, t: float):
        """Advance a move list; returns (target, grip, events fired)."""
        events = []
        while who["moves"]:
            mv = who["moves"][who["k"]]
            local = t - who["t0"]
            if mv.arrive and mv.duration <= local < mv.duration + ARRIVE_MAX_S \
                    and np.linalg.norm(self.site_board() - mv.end) > ARRIVE_TOL_M:
                who["waited"] = local - mv.duration
                return mv.end, None, events
            if local < mv.duration:
                pos, s = mv.at(local)
                grip = None if mv.grip[1] is None else (
                    mv.grip[1] if mv.grip[0] is None else mv.grip[0] + s * (mv.grip[1] - mv.grip[0]))
                return pos, grip, events
            if mv.event:
                events.append(mv.event)
            who["t0"] += mv.duration + who.pop("waited", 0.0)
            who["k"] += 1
            if who["k"] >= len(who["moves"]):
                who["moves"] = []
                return mv.end, mv.grip[1], events
        return None, None, events

    def _attach(self, who, p, holder: str):
        """Switch on the connect constraint, anchored where the puck is now (no jump)."""
        m, d = self.model, self.data
        if holder == "gripper" and self.friction_grip:
            if all(self.jaw_contacts(p)):
                who["held"] = p
                self._grip_lost_s = 0.0
                self.fetch_log[-1].update(grasped=True, t_grasp=self.t)
            else:
                self.fetch_log[-1].update(outcome="missed_grasp", t_end=self.t)
                self._robot_guard()
            return
        who["held"] = p
        hb = m.body(holder).id
        e = p["eq"][holder]
        m.eq_data[e][:3] = 0.0
        m.eq_data[e][3:6] = d.xmat[hb].reshape(3, 3).T @ (d.xpos[p["body"]] - d.xpos[hb])
        d.eq_active[e] = 1
        who["eq"] = e
        m.geom_conaffinity[p["col"]] = 0

    def _release(self, who, side):
        p = who["held"]
        who["held"] = None
        pos = self.puck_pos(p)
        if not (side == "robot" and self.friction_grip):    # A friction grip lets go by itself as the jaws open
            self.data.eq_active[who["eq"]] = 0
            self.model.geom_conaffinity[p["col"]] = 2
            self.data.qvel[p["v"]] = 0.0
        local = pos if side == "robot" else pos * [1, -1]
        log = self.fetch_log[-1] if side == "robot" else None
        if log is not None:
            log.update(t_release=self.t, release_pos=pos.copy())
        if local[1] < self.band.band_y - 1e-6:
            self.releases.append({"p": p, "side": side, "s_max": self.band.elongation(local), "t0": self.t, "log": log})
        elif log is not None:
            log["outcome"] = "not_drawn"
        self.shots[side] += 1

    def jaw_contacts(self, p) -> tuple:
        """(fixed jaw touches puck p, moving jaw touches it)."""
        d, fixed, moving = self.data, False, False
        for k in range(d.ncon):
            c = d.contact[k]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a == p["col"] and b in self._jaw_geoms:
                    fixed |= self._jaw_geoms[b] == self._fixed_jaw
                    moving |= self._jaw_geoms[b] == self._moving_jaw
        return fixed, moving

    def _check_grip(self):
        """A friction grip can lose the puck: then the band (if it is behind the line) launches it now."""
        rb = self.robot
        p = rb["held"]
        if p is None or not self.friction_grip:
            return
        self._grip_lost_s = 0.0 if all(self.jaw_contacts(p)) else self._grip_lost_s + CONTROL_DT_S
        if self._grip_lost_s >= GRIP_LOST_S:
            self.fetch_log[-1]["slip"] = True
            self._release(rb, "robot")
            self.shots["robot"] -= 1
            rb.update(state="guard", k=0, t0=self.t, moves=self.skill_block(rb["target"]))

    def _track_flights(self):
        """Outcome of the launched robot pucks, as physics/shot_sim.py."""
        half_t = 0.5 * self.cfg["board"]["divider_thickness_m"]
        for f in list(self._flights):
            pos, vel, log = self.puck_pos(f["p"]), self.puck_vel(f["p"]), f["log"]
            if "divider_x" not in log and pos[1] >= -self.face_y - 1e-3:
                log["divider_x"] = float(pos[0])
            outcome = None
            if pos[1] > half_t + self.r:
                outcome = "through"
            elif np.linalg.norm(vel) < 1e-3 or ("divider_x" in log and vel[1] < 0):
                if "divider_x" not in log:
                    outcome = "weak"
                else:
                    near = abs(log["divider_x"]) < 0.5 * self.cfg["board"]["gate_width_m"] + self.r
                    outcome = "edge" if near else "miss"
            if outcome:
                log.update(outcome=outcome, flight_s=self.t - log["t_exit"])
                self._flights.remove(f)

    def _control_human(self):
        """Every physics step: the hand is a mocap body, and a held puck must move smoothly with it."""
        t = self.t
        h = self.human
        if self.has_opponent and not h["moves"] and t >= h["next_t"] and self.winner is None:
            mine = self._resting_pucks("human")
            if mine:
                self._human_cycle(min(mine, key=lambda p: abs(self.puck_pos(p)[1] + self.band.band_y)))
        target, _, events = self._run_script(h, t)
        if target is not None:
            h["pos"] = target
            h["state"] = h["moves"][h["k"]].name if h["moves"] else "idle"
        elif h["state"] != "idle":
            h["state"] = "idle"
            h["next_t"] = t + 0.1
        for ev in events:
            if ev == "attach":
                self._attach(h, h["puck"], "hand")
            elif ev == "release":
                self._release(h, "human")
        self._set_hand(h["pos"])

    def _control_robot(self):
        """At CONTROL_DT_S: decisions, the arm script and the IK (the jaws move physically in between)."""
        t = self.t
        rb = self.robot
        if rb["state"] == "guard" and not rb["moves"] and self.winner is None:
            mine = self._resting_pucks("robot")
            reachable = [p for p in mine if self.puck_pos(p)[1] > -0.5 * self.cfg["board"]["length_m"] + self.r + 1e-3]
            greedy = self.cfg["skills"]["game_rule"] == "greedy_sling"
            if reachable and (greedy or self.human_phase() in ("reload", "idle")) and self.auto_fetch:
                if self.selector is not None:
                    c = self.selector.choose([self.puck_pos(p) for p in reachable], rb["target"][:2], self.rng)
                    self._robot_fetch(reachable[c.index], c.shot)
                else:
                    self._robot_fetch(min(reachable, key=lambda p: np.linalg.norm(
                        self.puck_pos(p) - [0.0, self.band.band_y])))
        if self.cfg["skills"]["game_rule"] != "greedy_sling" and rb["state"] in ("fetch", "free") \
                and rb["held"] is None and self.human_phase() in ("pull", "hold") \
                and rb["moves"] and rb["moves"][rb["k"]].name in ("rise", "to puck", "to front") and rb["k"] < 2:   # Once it descends, it commits
            self._robot_guard()   # Too late to shoot: guard the gate
        target, grip, events = self._run_script(rb, t)
        if target is not None:
            rb["target"] = target
            if grip is not None:
                rb["grip"] = grip
        if not rb["moves"]:
            if rb["state"] == "free":
                if np.linalg.norm(self.puck_vel(rb["puck"])) < REST_SPEED:    # A pushed puck slides on after the push
                    self.fetch_log[-1]["freed"] = True
                    rb["freed"] = True
                    self._robot_fetch(rb["puck"], rb["after"])   # The camera sees the puck again
            else:
                rb["state"] = "guard"
        for ev in events:
            if ev == "attach":
                self._attach(rb, rb["puck"], "gripper")
            elif ev == "release" and rb["held"] is not None:
                self._release(rb, "robot")
            elif ev in ("seated", "drawn") and rb["held"] is not None:
                self.fetch_log[-1].update({f"t_{ev}": self.t, f"pos_{ev}": self.puck_pos(rb["puck"]).copy()})
        self.q = self.solve_ik(rb["target"], self.q)
        self.data.ctrl[self._arm_act] = self.q + self.joint_offset
        self.data.ctrl[self._grip_act] = rb["grip"]
        self._check_grip()
        self._track_flights()

    # --------------------------------------------------------------- step
    def step(self):
        m, d = self.model, self.data
        self._control_human()
        k = int(self.t / CONTROL_DT_S)
        if k != self._ctrl_k:
            self._ctrl_k = k
            self._control_robot()
        for p in self.pucks:
            d.qfrc_applied[p["v"]] = 0.0
            coulomb_along_velocity(m, d, p["v"], self.puck_friction_n)
        # A held puck behind its band: the loading band force acts on the puck, and the grip carries it.
        self.band_force = {"robot": np.zeros(2), "human": np.zeros(2)}
        for side, who, sign in (("robot", self.robot, 1.0), ("human", self.human, -1.0)):
            p = who["held"]
            if p is None:
                continue
            local = self.puck_pos(p) * [1, sign]
            if local[1] < self.band.band_y:
                f = self.band.force_load(local) * [1, sign]
                d.qfrc_applied[p["v"]] += f
                self.band_force[side] = f
        # Releases: unloading band pushes the puck to its band line.
        for r in list(self.releases):
            sign = 1.0 if r["side"] == "robot" else -1.0
            local = self.puck_pos(r["p"]) * [1, sign]
            if local[1] >= self.band.band_y:
                d.qvel[r["p"]["v"]] *= math.sqrt(self.band.p.energy_transfer)
                self.releases.remove(r)
                if r["log"] is not None:
                    v = self.puck_vel(r["p"])
                    r["log"].update(t_exit=self.t, exit_pos=self.puck_pos(r["p"]).copy(), exit_vel=v)
                    self._flights.append({"p": r["p"], "log": r["log"]})
                continue
            if self.t - r["t0"] > RELEASE_GIVE_UP_S and np.linalg.norm(self.puck_vel(r["p"])) < REST_SPEED:
                self.releases.remove(r)                     # The band cannot beat friction: not launched
                if r["log"] is not None:
                    r["log"].update(outcome="weak", exit_vel=np.zeros(2))
                continue
            f = float(self.band.tension_unload(self.band.elongation(local), r["s_max"])) * \
                self.band.direction_sum(local) * [1, sign]
            d.qfrc_applied[r["p"]["v"]] += f
            self.band_force[r["side"]] = f
        mujoco.mj_step(m, d)
        self.t += m.opt.timestep
        self._check_end()

    def _check_end(self):
        if self.winner is not None:
            return
        busy = self.robot["held"] is not None or self.human["held"] is not None or self.releases or \
            any(np.linalg.norm(self.puck_vel(p)) > REST_SPEED for p in self.pucks)
        if busy:
            return
        c = self.counts()
        for side in ("robot", "human"):
            if c[side] == 0:
                self.winner = side
