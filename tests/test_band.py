import numpy as np

from src.slingpuck.physics.band_model import ElasticBand

# TODO(M3): replace with nonlinear-stiffness tests and a closed-hysteresis-loop test.


def test_band_hysteresis():
    params = {"stiffness_k_n_m": 50.0, "hysteresis_loss_factor": 0.15}
    band = ElasticBand(params)
    force_load_1 = band.get_force(0.1)
    force_load_2 = band.get_force(0.2)
    force_unload_1 = band.get_force(0.1)
    assert force_load_2 > force_load_1
    assert force_unload_1 < force_load_1
    assert np.isclose(force_unload_1, force_load_1 * (1 - params["hysteresis_loss_factor"]))
