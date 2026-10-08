"""Phase 1: goalkeeper.

The opponent's puck rests at the opponent band line, then launches toward the
gate. The agent commands the SO-101 pan joint, which swings a paddle along an
arc in front of the gate. The policy sees only the Kalman-tracked puck (camera
latency, noise, dropouts), the pan encoder, and its previous action. The true
state is in info["privileged"] for an asymmetric critic (see envs/wrappers.py).

Recovery task (paddle_locked_until_launch: true): the paddle is held at its
start offset, as if the arm is busy with a sling, until the puck launches plus
release_delay_s. reset() runs this locked period internally, so the first agent
step is at the release time. The camera and tracker run during the lock.

Outcomes (reward):
  save (+1): the puck touched the paddle and then moved back toward the opponent.
  goal (-1): the puck got past the paddle line, or stopped in the agent half.
  none (0):  the puck never threatened (only when threats_only is false).
A smoothness penalty of smoothness_weight * |a_t - a_{t-1}| is added every step.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from slingpuck.config import sample_randomized
from slingpuck.kinematics import GoalkeeperGeometry
from slingpuck.physics.puck_dynamics import Fast2DPuckSim
from slingpuck.physics.servo_model import ServoModel
from slingpuck.sensing.camera_model import CameraModel
from slingpuck.sensing.kalman_tracker import KalmanTracker

SAVE, GOAL, NONE = "save", "goal", "none"
REWARD = {SAVE: 1.0, GOAL: -1.0, NONE: 0.0}
OBS_CLIP = 5.0
VEL_SCALE_M_S = 3.0


@dataclass(frozen=True)
class Shot:
    start: np.ndarray
    vel: np.ndarray
    speed: float
    prelaunch_s: float
    threat: bool   # Direct shot passes the gate when no paddle is present
    tries: int


class GoalkeeperEnv(gym.Env):
    metadata = {"render_modes": []}
    POLICY_OBS_NAMES = ("trk_x", "trk_y", "trk_vx", "trk_vy", "trk_valid", "pan", "pan_vel", "prev_action")
    PRIVILEGED_OBS_NAMES = ("x", "y", "vx", "vy", "launched", "time_to_launch", "pan", "pan_vel", "touched",
                            "friction", "paddle_e", "wall_e", "cam_latency", "servo_latency", "gate_width",
                            "servo_rate")

    def __init__(self, config: dict):
        super().__init__()
        self.base_config = config
        self.task = config["goalkeeper"]
        self.physics_dt = config["sim"]["physics_dt_s"]
        self.control_dt = 1.0 / config["sim"]["control_hz"]
        self.n_substeps = max(1, int(round(self.control_dt / self.physics_dt)))
        self.max_steps = int(np.ceil(self.task["max_episode_s"] / self.control_dt))
        self.geom = GoalkeeperGeometry.from_config(config)
        self.cfg = config
        self.substep_hook = None  # Optional callable(env, frames) after each physics step (viewer)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(len(self.POLICY_OBS_NAMES),),
                                            dtype=np.float32)
        self.privileged_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(len(self.PRIVILEGED_OBS_NAMES),),
                                           dtype=np.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        """options: speed (m/s), start_pan (rad), randomize (bool)."""
        super().reset(seed=seed)
        options = options or {}
        cfg = self.base_config
        if self.task["randomize"] and cfg["randomization"]["enabled"] and options.get("randomize", True):
            cfg = sample_randomized(cfg, self.np_random)
        self.cfg = cfg
        self.geom = GoalkeeperGeometry.from_config(cfg)
        self.sim = Fast2DPuckSim.from_config(cfg)
        self._cf_sim = Fast2DPuckSim.from_config(cfg)
        self.servo = ServoModel(cfg["servo"], self.physics_dt)
        self.camera = CameraModel(cfg["camera"], rng=self.np_random)
        self.tracker = KalmanTracker.from_config(cfg)
        self.paddle_e = cfg["board"]["paddle_restitution"]
        r = cfg["puck"]["radius_m"]
        # Puck center is past the paddle when it is a full radius behind the paddle's back face.
        self.goal_line_y = self.geom.paddle_center(0.0)[1] - self.geom.paddle_half_thickness - r - 1e-3

        self.shot = self._sample_shot(options)
        if "start_pan" in options:
            pan0 = float(np.clip(options["start_pan"], -self.geom.pan_max, self.geom.pan_max))
        elif self.task["paddle_start"] == "center":
            pan0 = 0.0
        else:
            pan0 = float(self.np_random.uniform(-self.geom.pan_max, self.geom.pan_max))
        self.start_pan = pan0
        self.servo.reset(pan0)
        self.sim.reset(self.shot.start[None], np.zeros((1, 2)))
        self.sim.set_paddle(self.geom.paddle_state(pan0, 0.0, self.paddle_e))

        self.t = 0.0
        self.steps = 0
        self.launched = False
        self.touched = False
        self.entered = False
        self.prev_action = self.geom.pan_to_action(pan0)
        self._pending_outcome = None

        self.release_delay = 0.0
        if self.task["paddle_locked_until_launch"]:
            self.release_delay = float(options.get("release_delay",
                                                   self.np_random.uniform(*self.task["release_delay_s"])))
            t_release = self.shot.prelaunch_s + self.release_delay
            while self.t < t_release - 1e-12 and self._pending_outcome is None:
                self._pending_outcome = self._substep(pan0)
        return self._obs(), self._info(None)

    def _sample_shot(self, options) -> Shot:
        cfg, task = self.cfg, self.task
        hw, hl = 0.5 * cfg["board"]["width_m"], 0.5 * cfg["board"]["length_m"]
        r = cfg["puck"]["radius_m"]
        y0 = min(hl - cfg["band"]["band_offset_from_end_wall_m"], hl - r)
        tries = task["max_threat_tries"] if task["threats_only"] else 1
        for k in range(1, tries + 1):
            speed = float(options["speed"]) if "speed" in options else float(self.np_random.uniform(*task["speed_range_m_s"]))
            start = np.array([self.np_random.uniform(-(hw - r), hw - r), y0])
            aim = np.array([self.np_random.uniform(-task["aim_spread_m"], task["aim_spread_m"]), 0.0])
            direction = (aim - start) / np.linalg.norm(aim - start)
            vel = speed * direction
            threat = self._would_score(start, vel)
            if threat:
                break
        prelaunch = float(self.np_random.uniform(*task["prelaunch_s"]))
        return Shot(start, vel, speed, prelaunch, threat, k)

    def _would_score(self, start, vel) -> bool:
        """Direct shot with no paddle: True if it enters the agent half before it turns back or stops."""
        sim = self._cf_sim
        sim.set_paddle(None)
        sim.reset(start[None], vel[None])
        for _ in range(int(self.task["max_episode_s"] / self.physics_dt)):
            ev = sim.step(self.physics_dt)
            if any(c.direction == -1 for c in ev.crossings):
                return True
            if sim.vel[0, 1] >= 0.0:
                return False
        return False

    # ------------------------------------------------------------------- step
    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        target = self.geom.action_to_pan(a)
        reward = -self.task["smoothness_weight"] * abs(a - self.prev_action)
        self.prev_action = a

        # An outcome reached during the locked period ends the episode at the first step.
        outcome, self._pending_outcome = self._pending_outcome, None
        if outcome is None:
            for _ in range(self.n_substeps):
                outcome = self._substep(target)
                if outcome is not None:
                    break

        self.steps += 1
        terminated = outcome is not None
        truncated = not terminated and self.steps >= self.max_steps
        if truncated:
            outcome = self._final_outcome()
        if outcome is not None:
            reward += REWARD[outcome]
        return self._obs(), reward, terminated, truncated, self._info(outcome)

    def _substep(self, target_pan: float):
        """Advance one physics step. Returns the outcome if the episode ended, else None."""
        self.t += self.physics_dt
        if not self.launched and self.t >= self.shot.prelaunch_s - 1e-12:
            self.sim.reset(self.sim.pos, self.shot.vel[None])
            self.launched = True
        pan = float(self.servo.step(target_pan))
        self.sim.set_paddle(self.geom.paddle_state(pan, float(self.servo.vel), self.paddle_e))
        events = self.sim.step(self.physics_dt)
        self.touched |= bool(events.paddle_contacts)
        self.entered |= any(c.direction == -1 for c in events.crossings)
        frames = self.camera.observe(self.t, self.sim.pos[0])
        for frame in frames:
            self.tracker.update(frame)
        if self.substep_hook is not None:
            self.substep_hook(self, frames)
        return self._check_outcome(events) if self.launched else None

    def _check_outcome(self, events):
        y, vy = self.sim.pos[0, 1], self.sim.vel[0, 1]
        if y < self.goal_line_y:
            return GOAL
        if self.touched and (any(c.direction == 1 for c in events.crossings) or (y > 0.0 and vy > 0.0)):
            return SAVE
        if not self.touched and y > 0.0 and vy > 0.0:
            return NONE  # Turned back by the divider without a paddle touch
        if not np.any(self.sim.vel):
            return self._final_outcome()
        return None

    def _final_outcome(self):
        if self.sim.pos[0, 1] < 0.0:
            return GOAL
        return SAVE if self.touched else NONE

    # ------------------------------------------------------------ observation
    def _obs(self):
        hl = 0.5 * self.base_config["board"]["length_m"]
        est = self.tracker.estimate(self.t)
        if est is None:
            pos, vel, valid = np.zeros(2), np.zeros(2), 0.0
        else:
            pos, vel, valid = est[0], est[1], 1.0
        nominal_rate = np.deg2rad(self.base_config["servo"]["max_rate_deg_s"])
        # TODO(sim2real): add pan encoder quantization/noise once measured.
        obs = np.array([pos[0] / hl, pos[1] / hl, vel[0] / VEL_SCALE_M_S, vel[1] / VEL_SCALE_M_S, valid,
                        float(self.servo.pos) / self.geom.pan_max, float(self.servo.vel) / nominal_rate,
                        self.prev_action])
        return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def privileged_obs(self):
        base, cfg = self.base_config, self.cfg
        hl = 0.5 * base["board"]["length_m"]
        pos, vel = self.sim.pos[0], self.sim.vel[0]

        def rel(path_a, path_b):
            return cfg[path_a][path_b] / base[path_a][path_b]

        obs = np.array([
            pos[0] / hl, pos[1] / hl, vel[0] / VEL_SCALE_M_S, vel[1] / VEL_SCALE_M_S,
            float(self.launched), max(self.shot.prelaunch_s - self.t, 0.0),
            float(self.servo.pos) / self.geom.pan_max,
            float(self.servo.vel) / np.deg2rad(base["servo"]["max_rate_deg_s"]),
            float(self.touched),
            rel("puck", "friction_kinetic"), cfg["board"]["paddle_restitution"], cfg["board"]["restitution"],
            rel("camera", "latency_s"), rel("servo", "command_latency_s"), rel("board", "gate_width_m"),
            rel("servo", "max_rate_deg_s"),
        ])
        return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def _info(self, outcome):
        return {
            "outcome": outcome,
            "shot_speed": self.shot.speed,
            "threat": self.shot.threat,
            "start_pan": self.start_pan,
            "release_delay": self.release_delay,
            "privileged": self.privileged_obs(),
        }
