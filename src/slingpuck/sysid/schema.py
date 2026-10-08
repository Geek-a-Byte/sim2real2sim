"""CSV schemas for real (or synthetic) logs used by sysid. See data/real_logs/schema.md."""
from pathlib import Path

import pandas as pd

SCHEMAS = {
    # Force gauge on the band center, pulled straight back from the band line.
    "band_bench": {
        "cycle": "int, load/unload cycle number",
        "phase": "str, 'load' or 'unload'",
        "pull_m": "float, center pull distance behind the band line, m",
        "force_n": "float, gauge force along the pull direction, N",
    },
    # Step commands to one joint (normally shoulder_pan) and the encoder reading.
    "servo_step": {
        "test_id": "int, step test number",
        "t_s": "float, time since the test start, s",
        "cmd_rad": "float, commanded joint angle, rad",
        "meas_rad": "float, measured joint angle (encoder), rad",
    },
    # Camera frames of sling shots: still puck, held puck after the pull, flight.
    "shots": {
        "shot_id": "int, shot number",
        "phase": "str, 'rest' (before the slide), 'pulled' (held by the arm), 'flight' (after release)",
        "t_s": "float, frame capture time; for 'flight', seconds since the release command",
        "puck_x_m": "float, tracked puck x in the board frame, m",
        "puck_y_m": "float, tracked puck y in the board frame, m",
        "cmd_slide_m": "float, commanded slide x along the band, m",
        "cmd_angle_rad": "float, commanded pull angle (0 = straight back), rad",
        "cmd_pull_m": "float, commanded pull distance, m",
        "success": "int, 1 if the puck went through the gate, else 0",
    },
}

PHASES = {"band_bench": {"load", "unload"}, "shots": {"rest", "pulled", "flight"}}


class LogError(ValueError):
    pass


def load_log(path, kind: str) -> pd.DataFrame:
    """Load and validate one log CSV."""
    df = pd.read_csv(path)
    missing = set(SCHEMAS[kind]) - set(df.columns)
    if missing:
        raise LogError(f"{path}: missing columns {sorted(missing)}")
    if df[list(SCHEMAS[kind])].isna().any().any():
        raise LogError(f"{path}: empty values in required columns")
    if kind in PHASES and not set(df["phase"]).issubset(PHASES[kind]):
        raise LogError(f"{path}: phase must be one of {sorted(PHASES[kind])}")
    return df


def find_logs(log_dir) -> dict:
    """Return {kind: path} for the log files present in log_dir (<kind>*.csv)."""
    out = {}
    for kind in SCHEMAS:
        files = sorted(Path(log_dir).glob(f"{kind}*.csv"))
        if len(files) > 1:
            raise LogError(f"more than one {kind} log in {log_dir}: {[f.name for f in files]}")
        if files:
            out[kind] = files[0]
    return out
