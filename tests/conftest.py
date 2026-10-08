import pytest

from slingpuck.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config("v0")
