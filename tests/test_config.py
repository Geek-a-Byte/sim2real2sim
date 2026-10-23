import shutil

import pytest
import yaml

from src.slingpuck.config import DEFAULT_CONFIG_DIR, ConfigError, load_config, save_run_config


@pytest.fixture
def cfg_dir(tmp_path):
    for name in ("env.yaml", "physics_params_v0.yaml"):
        shutil.copy(DEFAULT_CONFIG_DIR / name, tmp_path / name)
    return tmp_path


def write_calibrated(cfg_dir, n):
    params = yaml.safe_load((cfg_dir / "physics_params_v0.yaml").read_text())
    params.update(version=n, calibrated=True, placeholders=[])
    params["puck"]["friction_kinetic"] = 0.18
    (cfg_dir / f"physics_params_v{n}.yaml").write_text(yaml.safe_dump(params))


def test_v0_loads_and_lists_placeholders(cfg):
    assert cfg["meta"]["params_version"] == "v0"
    assert cfg["meta"]["calibrated"] is False
    assert "board.divider_thickness_m" in cfg["meta"]["placeholders"]
    assert "board.gate_width_m" not in cfg["meta"]["placeholders"]  # Measured by the user
    assert cfg["match"]["pucks_per_side"] == 5   # env.yaml is merged in


def test_strict_rejects_placeholders():
    with pytest.raises(ConfigError, match="board.divider_thickness_m"):
        load_config("v0", strict=True)


def test_latest_loads_the_newest_version(cfg_dir):
    write_calibrated(cfg_dir, 2)
    cfg = load_config("latest", cfg_dir=cfg_dir)
    assert cfg["meta"]["params_version"] == "v2"
    assert cfg["puck"]["friction_kinetic"] == 0.18
    assert cfg["meta"]["placeholders"] == []
    load_config("latest", strict=True, cfg_dir=cfg_dir)  # Must not raise


def test_version_mismatch_rejected(cfg_dir):
    shutil.copy(cfg_dir / "physics_params_v0.yaml", cfg_dir / "physics_params_v5.yaml")
    with pytest.raises(ConfigError, match="expected 5"):
        load_config("v5", cfg_dir=cfg_dir)


def test_missing_version_rejected():
    with pytest.raises(ConfigError):
        load_config("v999")


def test_save_run_config_records_version_seed_and_git(cfg, tmp_path):
    path = save_run_config(cfg, tmp_path / "run", seed=123)
    saved = yaml.safe_load(path.read_text())
    assert saved["meta"]["params_version"] == "v0"
    assert saved["meta"]["seed"] == 123
    assert "git_hash" in saved["meta"]
