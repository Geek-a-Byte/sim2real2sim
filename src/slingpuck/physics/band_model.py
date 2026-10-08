import numpy as np

class ElasticBand:
    def __init__(self, config):
        self.k = config['stiffness_k_n_m']
        self.hysteresis = config['hysteresis_loss_factor']
        self.last_stretch = 0.0

    def get_force(self, current_stretch):
        """Returns force with a simple hysteresis branch."""
        if current_stretch < 0:
            return 0.0
            
        # Loading (stretching further)
        if current_stretch >= self.last_stretch:
            force = self.k * current_stretch
        # Unloading (releasing energy)
        else:
            force = self.k * current_stretch * (1.0 - self.hysteresis)
            
        self.last_stretch = current_stretch
        return force
        
    def calculate_launch_velocity(self, stretch, puck_mass):
        """Maps stored potential energy to kinetic energy minus losses."""
        # E = 0.5 * k * x^2
        energy_stored = 0.5 * self.k * (stretch ** 2)
        energy_released = energy_stored * (1.0 - self.hysteresis)
        
        # KE = 0.5 * m * v^2 -> v = sqrt((2 * KE) / m)
        if energy_released <= 0:
            return 0.0
        return np.sqrt((2 * energy_released) / puck_mass)