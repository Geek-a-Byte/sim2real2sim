"""Phase 1 goalkeeper on MuJoCo physics.

Same task, observation, action, reward and sensing as GoalkeeperEnv; only the
physics hooks change. The puck slides on a MuJoCo board, and the SO-101 arm
(STS3215 actuators) swings the paddle. A policy trained on the fast 2D sim runs
here unchanged, which makes this a sim-to-sim transfer test.

Differences from the 2D env that matter when you compare results:
- Pan dynamics come from the MuJoCo actuator (second-order, with overshoot),
  not from ServoModel. Only the command latency is shared.
- Contacts are soft (MuJoCo solver); restitution is set through damping.
- The threat check at reset still uses the 2D sim, so a few "threats" may not
  score in MuJoCo even without a paddle.
- Randomization updates friction, restitution, puck mass, gate width, latency
  and camera values. Servo rate and lag are not randomized here.
"""
from src.slingpuck.envs.goalkeeper_env import GoalkeeperEnv
from src.slingpuck.physics.mujoco_goalkeeper import MujocoGoalkeeperSim, MujocoPanServo


class MujocoGoalkeeperEnv(GoalkeeperEnv):
    def __init__(self, config: dict):
        super().__init__(config)
        self.mj = MujocoGoalkeeperSim(config)

    def _build_physics(self, cfg: dict):
        self.mj.apply_params(cfg)
        self.sim = self.mj
        self.servo = MujocoPanServo(self.mj, cfg["servo"], self.physics_dt)

    def _reset_physics(self, pan0, puck_start):
        self.servo.reset(pan0)
        self.sim.reset(puck_start, (0.0, 0.0))

    def _physics_step(self, target_pan: float):
        self.servo.step(target_pan)
        return self.sim.step(self.physics_dt)
