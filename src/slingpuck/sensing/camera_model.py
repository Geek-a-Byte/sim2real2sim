import numpy as np
from collections import deque

class CameraModel:
    def __init__(self, config, sim_dt):
        self.fps = config['fps']
        self.frame_time = 1.0 / self.fps
        self.noise_std = config['noise_std_dev_m']
        self.dropout_prob = config['dropout_prob']
        
        latency_steps = max(1, int(config['latency_s'] / sim_dt))
        self.latency_buffer = deque(maxlen=latency_steps)
        self.time_since_last_frame = 0.0
        self.last_output = None

    def step(self, ground_truth_x, ground_truth_y, dt):
        self.time_since_last_frame += dt
        
        # Only capture at camera FPS
        if self.time_since_last_frame >= self.frame_time:
            self.time_since_last_frame -= self.frame_time
            
            # Simulate dropout
            if np.random.rand() < self.dropout_prob:
                measurement = None 
            else:
                noise_x = np.random.normal(0, self.noise_std)
                noise_y = np.random.normal(0, self.noise_std)
                measurement = (ground_truth_x + noise_x, ground_truth_y + noise_y)
                
            self.latency_buffer.append(measurement)
        else:
            # Not a new frame, pad buffer with None
            self.latency_buffer.append(None)
            
        # Return oldest frame (simulating latency)
        latest_delayed = self.latency_buffer[0] if len(self.latency_buffer) > 0 else None
        
        if latest_delayed is not None:
            self.last_output = latest_delayed
            
        return self.last_output