import pytest

from src.slingpuck.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config("v0")


@pytest.fixture(scope="session")
def gcfg():
    """configs/general.yaml with a coarse reachability map (fast to build), and no assisted starts unless a test
    asks for them (grasped_start_prob / pregrasp_start_prob arguments)."""
    from src.slingpuck.general.common import load_general_config

    return load_general_config(overrides={"reach": {"voxel_m": 0.08},
                                          "env": {"grasped_start_prob": 0.0, "pregrasp_start_prob": 0.0}})


@pytest.fixture(scope="session")
def reach_map(gcfg):
    from src.slingpuck.general.reachability import ReachabilityMap

    return ReachabilityMap.build(gcfg)


@pytest.fixture(scope="session")
def shapes(gcfg):
    from src.slingpuck.general.common import objects_from_config

    return objects_from_config(gcfg)
