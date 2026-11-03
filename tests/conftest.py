import pytest

from src.slingpuck.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config("v0")


@pytest.fixture(scope="session")
def gcfg():
    """configs/general.yaml with a coarse reachability map (fast to build)."""
    from src.slingpuck.general.common import load_general_config

    return load_general_config(overrides={"reach": {"voxel_m": 0.08}})


@pytest.fixture(scope="session")
def reach_map(gcfg):
    from src.slingpuck.general.reachability import ReachabilityMap

    return ReachabilityMap.build(gcfg)


@pytest.fixture(scope="session")
def shapes(gcfg):
    from src.slingpuck.general.common import objects_from_config

    return objects_from_config(gcfg)
