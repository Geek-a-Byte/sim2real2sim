import pytest
import numpy as np
from src.slingpuck.physics.servo_model import ServoModel
from src.slingpuck.physics.band_model import ElasticBand

@pytest.fixture
def mock_config():
    return {
        'max_rate_rad_s': 5.0,
        'lag_time_constant_s': 0.05,
        'deadband_rad': 0.01,
        'command_latency_s': 0.05,
        'stiffness_k_n_m': 50.0,
        'hysteresis_loss_factor': 0.15
    }

def test_servo_latency(mock_config):
    dt = 0.01
    servo = ServoModel(mock_config, dt)
    
    # Send a step command
    servo.step(1.0)
    
    # Because latency is 0.05s and dt is 0.01s, it should take 5 steps 
    # before the internal command starts moving from 0.0
    for _ in range(4):
        pos = servo.step(1.0)
        assert pos == 0.0, "Servo moved before latency period ended"
        
    # 5th step should initiate movement
    pos = servo.step(1.0)
    assert pos > 0.0, "Servo failed to move after latency period"

def test_band_hysteresis(mock_config):
    band = ElasticBand(mock_config)
    
    # Load the band
    force_load_1 = band.get_force(0.1)
    force_load_2 = band.get_force(0.2)
    
    # Unload the band to the exact same position
    force_unload_1 = band.get_force(0.1)
    
    assert force_load_2 > force_load_1, "Force should increase with stretch"
    assert force_unload_1 < force_load_1, "Unloading force must be lower than loading force due to hysteresis"
    assert np.isclose(force_unload_1, force_load_1 * (1 - mock_config['hysteresis_loss_factor']))