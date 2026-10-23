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

A held puck is tied to the jaws or the hand by a stiff connect constraint. While a puck is held behind
its band, the band force acts on it, and the constraint passes it to the arm (or the hand). On release, the band (unloading curve) pushes the puck to
its band line, then the puck keeps energy_transfer of its energy (as BandModel.release). The human's band
is the same model, mirrored.

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
from src.slingpuck.so101_slingpuck import PUCKS_PER_SIDE, SO101_XML, board_pose, build

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
JAW_OPEN_EXTRA_M = 0.012        # Jaw tip gap beyond the puck diameter when open

CONTROL_DT_S = 0.01
LIFT_M = 0.05
REST_SPEED = 0.02               # Below this a puck counts as resting
HOLD_TIMECONST_S = 0.002        # Grip stiffness: about 1 mm of give under the 5 N band load
SKIN = [0.87, 0.70, 0.58, 1]
HUMAN_HOME = (0.0, 0.26, 0.09)  # Board frame: above the open end, behind the opponent's end wall
SPEED_MOVE, SPEED_DRAG = 0.25, 0.12   # Grasp-site / hand speeds, m/s
ROBOT_PULL_M = 0.025            # Straight pull for the robot's shot (a pre-stretched band launches almost straight)


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

    def at(self, t):
        s = min_jerk(t / self.duration) if self.duration > 0 else 1.0
        return self.start + s * (self.end - self.start), s


def _timed(name, a, b, speed, min_s, **kw):
    return Move(name, a, b, max(min_s, float(np.linalg.norm(b - a)) / speed), **kw)


class MatchSim:
    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.grasp = grasp_geometry(2.0 * cfg["puck"]["radius_m"])
        spec = build(cfg)
        add_grasp_site(spec, self.grasp)
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
                    "col": m.geom(f"{name}_col").id, "body": b,
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
            if on_jaw:
                m.geom_condim[g], m.geom_solref[g], m.geom_solimp[g] = 1, solref, solimp
        for p in self.pucks:
            m.geom_conaffinity[p["col"]] = 2
            m.geom_solref[p["col"]], m.geom_solimp[p["col"]] = solref, solimp
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
        self.q = self.solve_ik(self.guard)
        d.qpos[self._arm_q] = self.q
        d.ctrl[self._arm_act] = self.q
        d.ctrl[self._grip_act] = d.qpos[int(m.joint("gripper").qposadr[0])] = 0.0
        self.robot = {"state": "guard", "moves": [], "k": 0, "t0": 0.0, "held": None, "grip": 0.0,
                      "target": self.guard.copy()}
        self.human = {"state": "idle", "moves": [], "k": 0, "t0": 0.0, "held": None,
                      "pos": np.array(HUMAN_HOME, float), "aim": None, "threat": None, "next_t": 0.3}
        self.releases = []        # Active band releases: {"p", "side", "s_max"}
        self.band_force = {"robot": np.zeros(2), "human": np.zeros(2)}
        self.shots = {"robot": 0, "human": 0}
        self.winner = None
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

    def _robot_fetch(self, p):
        pos = self.puck_pos(p)
        z, zu = self.z_mid, self.z_mid + LIFT_M
        g = self.grasp
        here = self.robot["target"]
        above = np.array([pos[0], pos[1], zu])
        at = np.array([pos[0], pos[1], z])
        band = np.array([self.rng.normal(0.0, 0.001), self.band.band_y, z])
        pulled = band + [0.0, -ROBOT_PULL_M, 0.0]
        lift = pulled + [0.0, 0.0, LIFT_M]
        self.model.geom_conaffinity[p["col"]] = 0   # The open jaws come down around it: no contact
        self.robot.update(state="fetch", k=0, t0=self.t, puck=p, moves=[
            _timed("to puck", here, above, SPEED_MOVE, 0.3, grip=(None, g.q_open)),
            Move("down", above, at, 0.25),
            Move("grasp", at, at, 0.25, grip=(g.q_open, g.q_grasp), event="attach"),
            _timed("to band", at, band, SPEED_DRAG, 0.3),
            Move("pull", band, pulled, 0.4),
            Move("hold", pulled, pulled, 0.2, event="release"),
            Move("lift", pulled, lift, 0.25, grip=(g.q_grasp, g.q_open)),
            _timed("to gate", lift, self.guard, SPEED_MOVE, 0.4, grip=(g.q_open, 0.0))])

    def _robot_guard(self):
        p = self.robot.get("puck")
        if p is not None and self.robot["held"] is None:
            self.model.geom_conaffinity[p["col"]] = 2   # Fetch dropped before the grasp
        here = self.robot["target"]
        up = np.array([here[0], here[1], max(here[2], self.z_mid + LIFT_M)])
        self.robot.update(state="guard", k=0, t0=self.t, moves=[
            Move("up", here, up, 0.15), _timed("to gate", up, self.guard, SPEED_MOVE, 0.4, grip=(None, 0.0))])

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
            if local < mv.duration:
                pos, s = mv.at(local)
                grip = None if mv.grip[1] is None else (
                    mv.grip[1] if mv.grip[0] is None else mv.grip[0] + s * (mv.grip[1] - mv.grip[0]))
                return pos, grip, events
            if mv.event:
                events.append(mv.event)
            who["t0"] += mv.duration
            who["k"] += 1
            if who["k"] >= len(who["moves"]):
                who["moves"] = []
                return mv.end, mv.grip[1], events
        return None, None, events

    def _attach(self, who, p, holder: str):
        """Switch on the connect constraint, anchored where the puck is now (no jump)."""
        m, d = self.model, self.data
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
        self.data.eq_active[who["eq"]] = 0
        self.model.geom_conaffinity[p["col"]] = 2
        self.data.qvel[p["v"]] = 0.0
        pos = self.puck_pos(p)
        local = pos if side == "robot" else pos * [1, -1]
        if local[1] < self.band.band_y - 1e-6:
            self.releases.append({"p": p, "side": side, "s_max": self.band.elongation(local)})
        self.shots[side] += 1

    def _control_human(self):
        """Every physics step: the hand is a mocap body, and a held puck must move smoothly with it."""
        t = self.t
        h = self.human
        if not h["moves"] and t >= h["next_t"] and self.winner is None:
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
            if reachable and self.human_phase() in ("reload", "idle"):
                self._robot_fetch(min(reachable, key=lambda p: np.linalg.norm(
                    self.puck_pos(p) - [0.0, self.band.band_y])))
        if rb["state"] == "fetch" and rb["held"] is None and rb["k"] <= 2 and self.human_phase() in ("pull", "hold"):
            self._robot_guard()   # Too late to shoot: guard the gate
        target, grip, events = self._run_script(rb, t)
        if target is not None:
            rb["target"] = target
            if grip is not None:
                rb["grip"] = grip
        if not rb["moves"]:
            rb["state"] = "guard"
        for ev in events:
            if ev == "attach":
                self._attach(rb, rb["puck"], "gripper")
            elif ev == "release":
                self._release(rb, "robot")
        self.q = self.solve_ik(rb["target"], self.q)
        self.data.ctrl[self._arm_act] = self.q
        self.data.ctrl[self._grip_act] = rb["grip"]

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
