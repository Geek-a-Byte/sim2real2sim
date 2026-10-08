import numpy as np
from collections import deque

class ServoModel:
    def __init__(self, config, dt):
        self.dt = dt
        self.max_rate = config['max_rate_rad_s']
        self.tau = config['lag_time_constant_s']
        self.deadband = config['deadband_rad']
        
        # Latency buffer
        latency_steps = max(1, int(config['command_latency_s'] / dt))
        self.command_queue = deque([0.0] * latency_steps, maxlen=latency_steps)
        
        self.current_pos = 0.0
        self.current_vel = 0.0

    def step(self, target_pos):
        # 1. Apply command latency
        self.command_queue.append(target_pos)
        delayed_target = self.command_queue[0]
        
        # 2. Apply deadband
        if abs(delayed_target - self.current_pos) < self.deadband:
            delayed_target = self.current_pos
            
        # 3. First-order lag (low-pass filter)
        alpha = self.dt / (self.tau + self.dt)
        desired_pos = self.current_pos + alpha * (delayed_target - self.current_pos)
        
        # 4. Rate limiting (Speed constraints)
        pos_diff = desired_pos - self.current_pos
        max_step = self.max_rate * self.dt
        
        step_val = np.clip(pos_diff, -max_step, max_step)
        
        self.current_vel = step_val / self.dt
        self.current_pos += step_val
        
        return self.current_pos