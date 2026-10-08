import numpy as np

class ScriptedOpponent:
    """
    State machine: IDLE -> RELOADING -> AIMING -> FIRED
    """
    def __init__(self, config):
        self.reload_mu = config['reload_time_mu_s']
        self.reload_std = config['reload_time_std_s']
        self.aim_time = config['aim_time_s']
        self.tell_strength = config['tell_strength']
        self.shot_speed = config['shot_speed_mu']
        self.max_angle = np.pi / 4.0
        
        self.reset()

    def reset(self):
        self.state = "RELOADING"
        self.timer = max(0.5, np.random.normal(self.reload_mu, self.reload_std))
        self.true_target_angle = 0.0
        self.observed_tell_angle = 0.0
        self.is_reloading = True
        return self._get_obs()

    def step(self, dt):
        self.timer -= dt
        fired_puck = None
        
        if self.state == "RELOADING" and self.timer <= 0:
            self.state = "AIMING"
            self.timer = self.aim_time
            self.is_reloading = False
            
            # Decide where to shoot
            self.true_target_angle = np.random.uniform(-self.max_angle, self.max_angle)
            
            # Generate the "tell" 
            # tell_strength 1.0 -> std=0; tell_strength 0.0 -> std=max_angle
            noise_std = self.max_angle * (1.0 - self.tell_strength)
            noise = np.random.normal(0, noise_std)
            self.observed_tell_angle = np.clip(self.true_target_angle + noise, -self.max_angle, self.max_angle)
            
        elif self.state == "AIMING" and self.timer <= 0:
            self.state = "RELOADING"
            self.timer = max(0.5, np.random.normal(self.reload_mu, self.reload_std))
            self.is_reloading = True
            
            # Fire the puck (velocity vector)
            vx = self.shot_speed * np.sin(self.true_target_angle)
            vy = self.shot_speed * np.cos(self.true_target_angle) # Positive Y is towards agent
            fired_puck = (vx, vy)
            
            self.observed_tell_angle = 0.0 # Reset tell after firing
            
        return fired_puck, self._get_obs()

    def _get_obs(self):
        # Return [is_reloading (0 or 1), observed_tell_angle]
        return np.array([float(self.is_reloading), self.observed_tell_angle], dtype=np.float32)