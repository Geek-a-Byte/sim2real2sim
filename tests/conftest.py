import pytest

from src.slingpuck.config import load_config


@pytest.fixture(scope="session")
def cfg():
    """Params v0 with the fixed sling model (shot.selection: fixed), so these tests need no shot model file."""
    return load_config("v0", overrides={"shot": {"selection": "fixed"}})


@pytest.fixture(scope="session")
def greedy_cfg():
    return load_config("v0", overrides={"shot": {"selection": "greedy"}})


@pytest.fixture(scope="session")
def shot_model(greedy_cfg):
    """A small shot model fitted on 600 simulated shots (the real one uses 4000: python -m src.slingpuck.shot.model)."""
    pytest.importorskip("mujoco")
    from src.slingpuck.shot.model import ShotModel, sample_shots

    X, y, _ = sample_shots(greedy_cfg, 600, seed=0)
    return ShotModel.fit(X, y, greedy_cfg, members=3, epochs=200)
