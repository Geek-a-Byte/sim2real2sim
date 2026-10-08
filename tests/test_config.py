import shutil

import numpy as np
import pytest
import yaml

from slingpuck.config import (DEFAULT_CONFIG_DIR, ConfigError, load_config, sample_randomized,
                              save_run_config)


@pytest.fixture
def cfg_dir(tmp_path):
    for name in ("servo.yaml", "env.yaml", "physics_params_v0.yaml"):
        shutil.copy(DEFAULT_CONFIG_DIR / name, tmp_path / name)
    return tmp_path


def write_calibrated(cfg_dir, n):
    params = yaml.safe_load((cfg_dir / "physics_params_v0.yaml").read_text())
    params.update(version=n, calibrated=True, placeholders=[])
    params["servo"] = {"max_rate_deg_s": 280.0, "lag_tau_s": 0.04, "deadband_deg": 0.3, "command_latency_s": 0.02}
    (cfg_dir / f"physics_params_v{n}.yaml").write_text(yaml.safe_dump(params))


def test_v0_loads_and_lists_placeholders(cfg):
    assert cfg["meta"]["params_version"] == "v0"
    assert cfg["meta"]["calibrated"] is False
    assert "board.gate_width_m" in cfg["meta"]["placeholders"]
    assert "servo.max_rate_deg_s" in cfg["meta"]["placeholders"]
    assert cfg["servo"]["max_rate_deg_s"] == 300.0
    assert cfg["sim"]["control_hz"] == 30.0


def test_strict_rejects_placeholders():
    with pytest.raises(ConfigError, match="board.gate_width_m"):
        load_config("v0", strict=True)


def test_latest_and_servo_override(cfg_dir):
    write_calibrated(cfg_dir, 2)
    cfg = load_config("latest", cfg_dir=cfg_dir)
    assert cfg["meta"]["params_version"] == "v2"
    assert cfg["servo"]["max_rate_deg_s"] == 280.0
    assert cfg["meta"]["placeholders"] == []
    load_config("latest", strict=True, cfg_dir=cfg_dir)  # Must not raise


def test_version_mismatch_rejected(cfg_dir):
    shutil.copy(cfg_dir / "physics_params_v0.yaml", cfg_dir / "physics_params_v5.yaml")
    with pytest.raises(ConfigError, match="expected 5"):
        load_config("v5", cfg_dir=cfg_dir)


def test_missing_version_rejected():
    with pytest.raises(ConfigError):
        load_config("v999")


def test_randomization_in_range_reproducible_and_pure(cfg):
    before = cfg["puck"]["friction_kinetic"]
    a = sample_randomized(cfg, np.random.default_rng(0))
    b = sample_randomized(cfg, np.random.default_rng(0))
    assert a["meta"]["randomized"] == b["meta"]["randomized"]
    assert cfg["puck"]["friction_kinetic"] == before
    for _ in range(200):
        s = sample_randomized(cfg, np.random.default_rng())
        assert 0.7 * before <= s["puck"]["friction_kinetic"] <= 1.3 * before
        assert 0.0 <= s["board"]["restitution"] <= 1.0
        assert 0.0 <= s["camera"]["dropout_prob"] <= 0.5


def test_save_run_config_records_version_seed_and_git(cfg, tmp_path):
    path = save_run_config(cfg, tmp_path / "run", seed=123)
    saved = yaml.safe_load(path.read_text())
    assert saved["meta"]["params_version"] == "v0"
    assert saved["meta"]["seed"] == 123
    assert "git_hash" in saved["meta"]
