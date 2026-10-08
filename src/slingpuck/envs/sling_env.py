"""Phase 2: targeted shot through the gate.

The puck rests against the agent's band at a random x. A pre-stretched band
launches the puck almost straight forward from where it is released, so the
arm aims mainly by *placement*: it slides the puck along the band, then pulls
it back at an angle and distance. The pull angle and the small asymmetry of the
band V give fine steering (about +-3 mm at the gate).

Episode (correction_step: true):
  step 1  obs: tracked puck at rest.          action: (slide x, pull angle, pull distance)
          The arm slides and pulls (placement, angle and pull noise; arm deflection).
  step 2  obs: tracked pulled puck and band depth from the camera.
          action: correction (lateral shift, angle, pull), then release.
With correction_step: false the puck is released after step 1.

Release: band unloading dynamics (physics/band_model.py), then puck flight on
Fast2DPuckSim (walls, divider, gate corners). Reward: +1 if the puck crosses
the gate into the opponent half, plus shaping_weight * exp(-(miss/scale)^2),
where miss is the |x| at which the puck reaches the divider (or a large value
if it falls short).
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.slingpuck.config import sample_randomized
from src.slingpuck.physics.arm_deflection import PullResult, solve_pull
from src.slingpuck.physics.band_model import BandModel
from src.slingpuck.physics.puck_dynamics import Fast2DPuckSim
from src.slingpuck.sensing.camera_model import CameraModel

OBS_CLIP = 5.0
SHORT_MISS_M = 0.05  # Miss value added when the puck never reaches the divider


@dataclass(frozen=True)
class FlightResult:
    success: bool
    miss_m: float                 # |x| at the divider face, or SHORT_MISS_M + shortfall
    exit_pos: np.ndarray
    exit_vel: np.ndarray
    trajectory: np.ndarray | None  # (n, 3) rows of [t, x, y] at physics_dt after release, if recorded


def slide_limit(cfg: dict) -> float:
    """Largest |x| the puck can be slid to along the band."""
    return 0.5 * cfg["board"]["width_m"] - cfg["puck"]["radius_m"] - 0.002


def realize_pull(cfg, band: BandModel, x_start: float, angle: float, pull: float,
                 rng: np.random.Generator | None, noise_scale: float = 1.0) -> PullResult:
    """Arm pull with execution noise and deflection, kept inside the board."""
    sl = cfg["sling"]
    if rng is not None and noise_scale > 0:
        angle += rng.normal(0.0, np.deg2rad(sl["angle_noise_deg"]) * noise_scale)
        pull += rng.normal(0.0, sl["pull_noise_m"] * noise_scale)
    hl = 0.5 * cfg["board"]["length_m"]
    r = cfg["puck"]["radius_m"]
    # The pull may not push the puck into the end wall.
    max_pull = (band.band_y - (-hl + r)) / max(np.cos(angle), 1e-3)
    pull = float(np.clip(pull, 0.0, max_pull))
    x_start = float(np.clip(x_start, -slide_limit(cfg), slide_limit(cfg)))
    return solve_pull(band, (x_start, band.band_y), angle, pull, cfg["arm"]["deflection_compliance_m_per_n"])


def launch_and_fly(cfg, band: BandModel, pull_pos, rng: np.random.Generator | None, record: bool = False,
                   physics_dt: float = 0.001) -> FlightResult:
    puck = cfg["puck"]
    if pull_pos[1] >= band.band_y - 1e-6:  # No pull: nothing launches
        return FlightResult(False, SHORT_MISS_M + abs(band.band_y), np.asarray(pull_pos), np.zeros(2),
                            np.zeros((0, 3)) if record else None)
    rel = band.release(pull_pos, puck["mass_kg"], puck["friction_kinetic"])
    vel = rel.exit_vel
    if rng is not None:
        a = rng.normal(0.0, np.deg2rad(cfg["sling"]["release_angle_noise_deg"]))
        c, s = np.cos(a), np.sin(a)
        vel = np.array([c * vel[0] - s * vel[1], s * vel[0] + c * vel[1]])
    sim = Fast2DPuckSim.from_config(cfg)
    lim = 0.5 * cfg["board"]["width_m"] - puck["radius_m"]
    sim.reset([[float(np.clip(rel.exit_pos[0], -lim, lim)), rel.exit_pos[1]]], vel[None])
    face_y = -(0.5 * cfg["board"]["divider_thickness_m"] + puck["radius_m"])
    miss, success, traj = None, False, []
    for k in range(int(cfg["sling"]["max_flight_s"] / physics_dt)):
        ev = sim.step(physics_dt)
        x, y = sim.pos[0]
        if record:
            traj.append((rel.release_time_s + (k + 1) * physics_dt, x, y))
        if miss is None and y >= face_y:
            miss = abs(x)
        if any(c.direction == 1 for c in ev.crossings):
            success = True
            break
        if not np.any(sim.vel) or (sim.vel[0, 1] < 0.0 and miss is not None):
            break
    if miss is None:
        miss = SHORT_MISS_M + (face_y - sim.pos[0, 1])
    return FlightResult(success, float(miss), rel.exit_pos, vel,
                        np.array(traj).reshape(-1, 3) if record else None)


class SlingEnv(gym.Env):
    metadata = {"render_modes": []}
    POLICY_OBS_NAMES = ("trk_x", "trk_y", "band_depth", "stage", "cmd_slide", "cmd_angle", "cmd_pull")
    PRIVILEGED_OBS_NAMES = ("x0", "slide_x", "pull_x", "pull_y", "band_k", "natural_length", "energy_transfer",
                            "hysteresis", "compliance", "friction", "wall_e", "gate_width")

    def __init__(self, config: dict):
        super().__init__()
        self.base_config = config
        self.task = config["sling"]
        self.physics_dt = config["sim"]["physics_dt_s"]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(len(self.POLICY_OBS_NAMES),),
                                            dtype=np.float32)
        self.privileged_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(len(self.PRIVILEGED_OBS_NAMES),),
                                           dtype=np.float32)
        self.cfg = config

    # ----------------------------------------------------- action scaling
    def scale_aim(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(3), -1.0, 1.0)
        slide = a[0] * slide_limit(self.cfg)
        angle = a[1] * np.deg2rad(self.task["max_angle_deg"])
        pull = 0.5 * (a[2] + 1.0) * self.cfg["band"]["max_pull_m"]
        return slide, angle, pull

    def scale_correction(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(3), -1.0, 1.0)
        return (a[0] * self.task["correction_slide_m"], a[1] * np.deg2rad(self.task["correction_angle_deg"]),
                a[2] * self.task["correction_pull_frac"] * self.cfg["band"]["max_pull_m"])

    # ---------------------------------------------------------------- API
    def reset(self, seed=None, options=None):
        """options: x0 (puck start x), randomize (bool)."""
        super().reset(seed=seed)
        options = options or {}
        cfg = self.base_config
        if cfg["randomization"]["enabled"] and options.get("randomize", True):
            cfg = sample_randomized(cfg, self.np_random)
        self.cfg = cfg
        self.band = BandModel.from_config(cfg)
        self.camera = CameraModel(cfg["camera"], rng=self.np_random)
        self.t = 0.0
        lo, hi = self.task["puck_x_range_m"]
        self.x0 = float(options["x0"]) if "x0" in options else float(self.np_random.uniform(lo, hi))
        self.puck_pos = np.array([self.x0, self.band.band_y])
        self.stage = 0
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_action = np.zeros(3)
        self.slide_x = self.x0
        self.pull: PullResult | None = None
        self.est = self._observe_still(self.puck_pos)
        return self._obs(), self._info(None)

    def step(self, action):
        if self.stage == 0:
            slide, angle, pull = self.scale_aim(action)
            self.cmd = (slide, angle, pull)
            self.cmd_action = np.clip(np.asarray(action, dtype=float).reshape(3), -1, 1)
            self.slide_x = slide + self.np_random.normal(0.0, self.task["place_noise_m"])
            self.pull = realize_pull(self.cfg, self.band, self.slide_x, angle, pull, self.np_random)
            self.puck_pos = self.pull.pos
            if self.task["correction_step"]:
                self.stage = 1
                self.est = self._observe_still(self.puck_pos)
                return self._obs(), 0.0, False, False, self._info(None)
        else:
            d_slide, d_angle, d_pull = self.scale_correction(action)
            slide, angle, pull = self.cmd
            self.cmd = (slide + d_slide, angle + d_angle, pull + d_pull)
            self.slide_x += d_slide
            self.pull = realize_pull(self.cfg, self.band, self.slide_x, self.cmd[1], self.cmd[2], self.np_random,
                                     noise_scale=self.task["correction_noise_scale"])
            self.puck_pos = self.pull.pos

        self.flight = launch_and_fly(self.cfg, self.band, self.pull.pos, self.np_random, physics_dt=self.physics_dt)
        reward = float(self.flight.success) + self.task["shaping_weight"] * float(
            np.exp(-(self.flight.miss_m / self.task["shaping_scale_m"]) ** 2))
        self.stage = 2
        return self._obs(), reward, True, False, self._info(self.flight)

    # ------------------------------------------------------------ sensing
    def _observe_still(self, pos):
        """Camera watches the still puck for observe_window_s; returns the mean of the frames.

        The puck is known to be still (it rests on the band or the arm holds it),
        so the frame mean is the right estimator: error ~ noise / sqrt(frames).
        The constant-velocity Kalman tracker allows motion and gives ~2 mm error
        here, which is larger than the placement error a correction must fix.
        """
        self.camera.reset(t0=self.t)
        frames = []
        for _ in range(int(round(self.task["observe_window_s"] / self.physics_dt))):
            self.t += self.physics_dt
            frames += [f.xy for f in self.camera.observe(self.t, pos)]
        return np.mean(frames, axis=0) if frames else None

    def _obs(self):
        hl = 0.5 * self.base_config["board"]["length_m"]
        max_pull = self.base_config["band"]["max_pull_m"]
        if self.est is None:
            trk = np.zeros(2)
        else:
            trk = self.est
        depth = 0.0 if self.stage == 0 else (self.band.band_y - trk[1]) / max_pull
        obs = np.array([trk[0] / hl, trk[1] / hl, depth, float(self.stage >= 1), *self.cmd_action])
        return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def privileged_obs(self):
        base, cfg = self.base_config, self.cfg
        hl = 0.5 * base["board"]["length_m"]
        pos = self.puck_pos

        def rel(a, b):
            return cfg[a][b] / base[a][b]

        obs = np.array([self.x0 / hl, self.slide_x / hl, pos[0] / hl, pos[1] / hl,
                        rel("band", "stiffness_k_n_m"), rel("band", "natural_length_m"), rel("band", "energy_transfer"),
                        cfg["band"]["hysteresis_loss_factor"], rel("arm", "deflection_compliance_m_per_n"),
                        rel("puck", "friction_kinetic"), cfg["board"]["restitution"], rel("board", "gate_width_m")])
        return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def _info(self, flight: FlightResult | None):
        info = {"x0": self.x0, "stage": self.stage, "privileged": self.privileged_obs(), "outcome": None}
        if flight is not None:
            info.update(outcome="success" if flight.success else "miss", success=flight.success,
                        miss_m=flight.miss_m, exit_speed=float(np.linalg.norm(flight.exit_vel)),
                        cmd_slide=self.cmd[0], cmd_angle=self.cmd[1], cmd_pull=self.cmd[2],
                        actual_pull=self.pull.pull_m)
        return info
