import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.slingpuck.physics.band_model import ElasticBand
from src.slingpuck.physics.arm_deflection import DeflectionModel
from src.slingpuck.sensing.camera_model import CameraModel

class SlingEnv(gym.Env):
    """
    Phase 2: Targeted Shot Environment.
    This is effectively a contextual bandit (single-step episode). 
    The agent observes the puck, chooses an angle and pullback distance, 
    and receives a reward based on whether the puck passes the gate.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dt = 1.0 / config.get('control_hz', 30.0)
        
        self.gate_width = config['board']['gate_width_m']
        self.puck_radius = config['puck']['radius_m']
        self.puck_mass = config['puck']['mass_kg']
        self.max_stretch = config['band']['max_stretch_m']
        
        self.band = ElasticBand(config['band'])
        self.arm_model = DeflectionModel(config)
        self.camera = CameraModel(config['camera'], self.dt)
        
        # Action: [strike_angle (rad), pull_back_distance (m)]
        # Scaled to [-1, 1] for RL stability
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation: [puck_x, puck_y, band_disp_x, band_disp_y]
        high_obs = np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Spawn puck randomly on the agent's side
        self.puck_x = self.np_random.uniform(-0.15, 0.15)
        self.puck_y = self.np_random.uniform(0.1, 0.25) 
        
        # Reset visual band displacement
        self.band_disp_x = 0.0
        self.band_disp_y = 0.0
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Un-scale actions
        # Angle maps from [-1, 1] to [-pi/4, pi/4] relative to gate center
        strike_angle = action[0] * (np.pi / 4.0) 
        # Pullback maps from [-1, 1] to [0, max_stretch]
        cmd_pullback = ((action[1] + 1.0) / 2.0) * self.max_stretch
        
        # 2. Physics: Deflection & Launch
        # Estimate initial band force to calculate arm deflection
        theoretical_force = self.band.get_force(cmd_pullback)
        actual_pullback = self.arm_model.get_actual_pullback(cmd_pullback, theoretical_force)
        
        # Calculate exit velocity using the realized pullback
        velocity_mag = self.band.calculate_launch_velocity(actual_pullback, self.puck_mass)
        
        vx = velocity_mag * np.sin(strike_angle)
        vy = -velocity_mag * np.cos(strike_angle) # Negative Y is towards the gate
        
        # 3. Simulate forward trajectory to check gate success
        # (Using a simplified straight-line check for the M3 bandit setup)
        time_to_gate = abs(self.puck_y / (vy + 1e-6))
        x_at_gate = self.puck_x + (vx * time_to_gate)
        
        # 4. Reward Logic
        margin = (self.gate_width / 2.0) - self.puck_radius
        if abs(x_at_gate) <= margin and velocity_mag > 0.5:
            reward = 1.0
        else:
            reward = 0.0
            
        # Single-step episode
        terminated = True
        
        return self._get_obs(), reward, terminated, False, {"x_at_gate": x_at_gate, "exit_vel": velocity_mag}

    def _get_obs(self):
        # Simulate camera measurement with noise/latency
        meas = self.camera.step(self.puck_x, self.puck_y, self.dt)
        meas_x = meas[0] if meas else self.puck_x
        meas_y = meas[1] if meas else self.puck_y
        
        return np.array([meas_x, meas_y, self.band_disp_x, self.band_disp_y], dtype=np.float32)