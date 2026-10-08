import numpy as np
import pandas as pd
import pytest
import yaml

from src.slingpuck.config import DEFAULT_CONFIG_DIR, load_config
from src.slingpuck.eval.hole_rate import _crossing_time
from src.slingpuck.sysid import run_sysid
from src.slingpuck.sysid.fit import fit_band_bench, fit_servo_steps, fit_shots, with_values
from src.slingpuck.sysid.schema import LogError, load_log
from src.slingpuck.sysid.synthetic import DEFAULT_TRUTH, band_bench_log, generate, servo_step_log, truth_config


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    truth = generate(out, seed=0, n_gate=30, n_divider=20)
    return out, truth


@pytest.fixture(scope="module")
def base(synthetic):
    _, truth = synthetic
    # The natural band length is measured directly (ruler), so the fit gets the true value.
    return with_values(load_config("v0"), {"band.natural_length_m": truth["band.natural_length_m"]})


def test_schema_rejects_bad_logs(tmp_path):
    pd.DataFrame({"cycle": [0], "phase": ["load"], "pull_m": [0.01]}).to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(LogError, match="force_n"):
        load_log(tmp_path / "b.csv", "band_bench")
    pd.DataFrame({"cycle": [0], "phase": ["pull"], "pull_m": [0.01], "force_n": [1.0]}).to_csv(
        tmp_path / "c.csv", index=False)
    with pytest.raises(LogError, match="phase"):
        load_log(tmp_path / "c.csv", "band_bench")


def test_band_bench_recovers_truth(synthetic, base):
    out, truth = synthetic
    fit = fit_band_bench(load_log(out / "band_bench.csv", "band_bench"), base)
    assert fit["band.stiffness_k_n_m"].value == pytest.approx(truth["band.stiffness_k_n_m"], rel=0.05)
    assert fit["band.stiffness_exponent"].value == pytest.approx(truth["band.stiffness_exponent"], abs=0.06)
    assert fit["band.hysteresis_loss_factor"].value == pytest.approx(truth["band.hysteresis_loss_factor"], abs=0.01)


def test_servo_steps_recover_truth(base):
    cfg = truth_config(base, DEFAULT_TRUTH)
    df = servo_step_log(cfg, np.random.default_rng(1))
    fit = fit_servo_steps(df, base, max_latency_s=0.08)
    assert fit["servo.command_latency_s"].value == pytest.approx(DEFAULT_TRUTH["servo.command_latency_s"], abs=1e-3)
    assert fit["servo.lag_tau_s"].value == pytest.approx(DEFAULT_TRUTH["servo.lag_tau_s"], rel=0.1)
    assert fit["servo.max_rate_deg_s"].value == pytest.approx(DEFAULT_TRUTH["servo.max_rate_deg_s"], rel=0.05)
    assert fit["servo.deadband_deg"].value == pytest.approx(DEFAULT_TRUTH["servo.deadband_deg"], abs=0.1)


def test_shots_recover_truth(synthetic, base):
    out, truth = synthetic
    band = fit_band_bench(load_log(out / "band_bench.csv", "band_bench"), base)
    fit = fit_shots(load_log(out / "shots.csv", "shots"), with_values(base, band))
    assert fit["arm.deflection_compliance_m_per_n"].value == pytest.approx(
        truth["arm.deflection_compliance_m_per_n"], rel=0.1)
    assert fit["puck.friction_kinetic"].value == pytest.approx(truth["puck.friction_kinetic"], rel=0.1)
    assert fit["board.restitution"].value == pytest.approx(truth["board.restitution"], abs=0.08)
    assert fit["band.energy_transfer"].value == pytest.approx(truth["band.energy_transfer"], rel=0.06)


def test_band_bench_log_is_a_closed_loop():
    cfg = truth_config(load_config("v0"), DEFAULT_TRUTH)
    df = band_bench_log(cfg, np.random.default_rng(0))
    for _, g in df.groupby("cycle"):
        load, unload = g[g.phase == "load"], g[g.phase == "unload"]
        assert load.force_n.iloc[-1] == pytest.approx(unload.force_n.iloc[0], abs=0.25)  # Same top point
        assert np.trapezoid(unload.force_n[::-1], unload.pull_m[::-1]) < np.trapezoid(load.force_n, load.pull_m)


def test_run_sysid_writes_versioned_params(synthetic, tmp_path):
    out, truth = synthetic
    fit_dir = tmp_path / "fit"
    path = run_sysid.run(out, base_version="v0", out_dir=fit_dir, natural_length=truth["band.natural_length_m"])
    assert path.name == "physics_params_v1.yaml"
    raw = yaml.safe_load(path.read_text())
    assert raw["version"] == 1 and raw["sysid"]["parent_version"] == "v0" and raw["sysid"]["synthetic"] is True
    assert raw["calibrated"] is False  # Synthetic fits are never marked calibrated
    assert set(raw["sysid"]["logs"]) == {"band_bench", "servo_step", "shots"}
    cfg = load_config("v1", cfg_dir=fit_dir)
    assert cfg["meta"]["params_version"] == "v1"
    for fitted in ("puck.friction_kinetic", "band.energy_transfer", "servo.command_latency_s",
                   "band.natural_length_m"):
        assert fitted not in cfg["meta"]["placeholders"]
    assert cfg["servo"]["command_latency_s"] == pytest.approx(truth["servo.command_latency_s"], abs=1e-3)


def test_synthetic_fit_never_written_to_configs(synthetic):
    out, _ = synthetic
    with pytest.raises(ValueError, match="synthetic"):
        run_sysid.run(out, base_version="v0", out_dir=DEFAULT_CONFIG_DIR)


def test_crossing_time_interpolates():
    t = np.array([0.0, 0.01, 0.02])
    y = np.array([0.0, 0.1, 0.2])
    assert _crossing_time(t, y, 0.15) == pytest.approx(0.015)
    assert _crossing_time(t, y, 0.5) is None
