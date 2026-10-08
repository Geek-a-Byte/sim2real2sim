import gymnasium as gym
from gymnasium import spaces
import numpy as np

from slingpuck.physics.servo_model import ServoModel
from slingpuck.sensing.camera_model import CameraModel
from slingpuck.sensing.kalman_tracker import KalmanTracker

class GoalkeeperEnv(gym.Env):
    """
    Phase 1: Goalkeeper Environment.
    The agent controls a single pan joint to block a puck sliding towards the gate.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.control_dt = 1.0 / config['sim']['control_hz']
        self.physics_dt = 0.005 # Finer resolution for collision/physics
        self.physics_steps_per_control = int(self.control_dt / self.physics_dt)
        
        # Geometry & Physics Limits
        self.gate_width = config['board']['gate_width_m']
        self.puck_radius = config['puck']['radius_m']
        self.board_length = config['board']['length_m']
        self.board_width = config['board']['width_m']
        
        # Assume paddle swings on an arc of radius R from the base
        self.arm_radius = 0.20 
        self.max_pan_angle = np.arcsin((self.gate_width / 2.0) / self.arm_radius) * 1.5 # 50% wider than gate
        self.paddle_width = 0.04
        
        # RL Spaces
        # Action: [-1.0, 1.0] mapping to [-max_pan_angle, max_pan_angle]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation: [track_x, track_y, track_vx, track_vy, paddle_angle, paddle_vel]
        high_obs = np.array([1.0, 1.0, 10.0, 10.0, 3.14, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Re-initialize M1 models
        self.servo = ServoModel(self.config['servo'], self.physics_dt)
        self.camera = CameraModel(self.config['camera'], rng=self.np_random)
        self.tracker = KalmanTracker.from_config(self.config)
        self.t = 0.0
        
        # Spawn puck at the far side, aiming towards the gate
        self.puck_y = self.board_length / 2.0
        self.puck_x = self.np_random.uniform(-self.board_width/3, self.board_width/3)
        
        # Sample incoming speed [1.0 to 4.0 m/s]
        speed = self.np_random.uniform(1.0, 4.0)
        
        # Aim roughly at the gate
        target_x = self.np_random.uniform(-self.gate_width/2, self.gate_width/2)
        target_y = 0.0
        
        angle = np.arctan2(target_y - self.puck_y, target_x - self.puck_x)
        self.puck_vx = speed * np.cos(angle)
        self.puck_vy = speed * np.sin(angle)
        
        self.last_action = 0.0
        self.steps = 0
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        target_angle = float(np.clip(action[0], -1.0, 1.0) * self.max_pan_angle)
        
        reward = 0.0
        terminated = False
        truncated = self.steps > 150 # Timeout safety
        
        # Smoothness penalty
        action_diff = abs(action[0] - self.last_action)
        reward -= 0.01 * action_diff
        self.last_action = action[0]

        # Physics loop
        for _ in range(self.physics_steps_per_control):
            # 1. Update Servo
            paddle_angle = self.servo.step(target_angle)
            paddle_x = self.arm_radius * np.sin(paddle_angle)
            
            # 2. Update Puck Kinematics (with simple kinetic friction)
            friction = self.config['puck']['friction_kinetic']
            speed = np.hypot(self.puck_vx, self.puck_vy)
            if speed > 0:
                decel = friction * 9.81 * self.physics_dt
                new_speed = max(0.0, speed - decel)
                self.puck_vx = self.puck_vx * (new_speed / speed)
                self.puck_vy = self.puck_vy * (new_speed / speed)
                
            self.puck_x += self.puck_vx * self.physics_dt
            self.puck_y += self.puck_vy * self.physics_dt
            
            # 3. Wall Bounces
            if abs(self.puck_x) > (self.board_width / 2.0 - self.puck_radius):
                self.puck_vx *= -self.config['board']['restitution']
                self.puck_x = np.sign(self.puck_x) * (self.board_width / 2.0 - self.puck_radius)

            # 4. Camera Step: the tracker gets each frame once, at its arrival time
            self.t += self.physics_dt
            for frame in self.camera.observe(self.t, (self.puck_x, self.puck_y)):
                self.tracker.update(frame)

            # 5. Gate & Block Collision Check (y <= puck_radius)
            if self.puck_y <= self.puck_radius and not terminated:
                # Did it hit the paddle?
                dist_to_paddle = abs(self.puck_x - paddle_x)
                if dist_to_paddle <= (self.paddle_width / 2.0 + self.puck_radius):
                    reward += 1.0 # Save!
                    terminated = True
                elif abs(self.puck_x) <= (self.gate_width / 2.0):
                    reward -= 1.0 # Conceded goal
                    terminated = True
                elif self.puck_y < -0.1:
                    # Missed gate entirely, bounced back up or stopped
                    terminated = True 

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        est = self.tracker.estimate(self.t)
        track_pos, track_vel = (est[0], est[1]) if est is not None else (np.zeros(2), np.zeros(2))
        return np.array([
            track_pos[0], track_pos[1],
            track_vel[0], track_vel[1],
            float(self.servo.pos),
            float(self.servo.vel)
        ], dtype=np.float32)