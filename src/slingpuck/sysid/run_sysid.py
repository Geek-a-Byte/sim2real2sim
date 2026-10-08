"""Fit physics parameters from a log folder and write a new versioned params file.

    python -m src.slingpuck.sysid.run_sysid --logs data/real_logs/<session> --natural-length 0.152

The folder may hold band_bench*.csv, servo_step*.csv and shots*.csv (see
data/real_logs/schema.md); fits run for the logs that are present. The result is
physics_params_v<N+1>.yaml: the parent params with the fitted values replaced,
those values removed from `placeholders`, servo values in its `servo:` section,
and a `sysid:` record (parent version, log checksums, git hash, each fit with
its standard error, suggested randomization ranges).

Synthetic logs (a truth.yaml in the folder, or --synthetic) are never written to
configs/, so load_config("latest") cannot pick up a fit of fake data. By default
they go to <logs>/fit/, with copies of servo.yaml and env.yaml so that
load_config(..., cfg_dir=<logs>/fit) works.
"""
import argparse
import copy
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from src.slingpuck.config import (DEFAULT_CONFIG_DIR, _PARAMS_RE, available_param_versions, git_state,
                                  load_config, resolve_params_file, set_path)
from src.slingpuck.sysid.fit import fit_band_bench, fit_servo_steps, fit_shots, with_values
from src.slingpuck.sysid.schema import find_logs, load_log


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def run(log_dir, base_version="latest", out_dir=None, natural_length=None, synthetic=None,
        cfg_dir=None) -> Path:
    log_dir = Path(log_dir)
    cfg_dir = Path(cfg_dir) if cfg_dir else DEFAULT_CONFIG_DIR
    logs = find_logs(log_dir)
    if not logs:
        raise FileNotFoundError(f"no band_bench/servo_step/shots CSV in {log_dir}")
    if synthetic is None:
        synthetic = (log_dir / "truth.yaml").exists()
    if out_dir is None:
        out_dir = log_dir / "fit" if synthetic else cfg_dir
    out_dir = Path(out_dir)
    if synthetic and out_dir.resolve() == DEFAULT_CONFIG_DIR.resolve():
        raise ValueError("refusing to write a fit of synthetic logs into configs/")

    base_file = resolve_params_file(base_version, cfg_dir)
    base_raw = yaml.safe_load(base_file.read_text())
    cfg = load_config(base_version, cfg_dir=cfg_dir)
    measured = {}
    if natural_length is not None:
        measured["band.natural_length_m"] = natural_length
        cfg = with_values(cfg, measured)

    fits = {}
    if "band_bench" in logs:
        fits.update(fit_band_bench(load_log(logs["band_bench"], "band_bench"), cfg))
    if "servo_step" in logs:
        fits.update(fit_servo_steps(load_log(logs["servo_step"], "servo_step"), cfg))
    if "shots" in logs:
        fits.update(fit_shots(load_log(logs["shots"], "shots"), with_values(cfg, fits)))

    new = copy.deepcopy(base_raw)
    new.setdefault("servo", {})
    for path, value in {**measured, **{k: v.value for k, v in fits.items()}}.items():
        if path.startswith("servo."):
            new["servo"][path.split(".", 1)[1]] = float(value)
        else:
            set_path(new, path, float(value))
    done = set(measured) | set(fits)
    new["placeholders"] = [p for p in base_raw.get("placeholders", []) if p not in done]

    versions = set(available_param_versions(cfg_dir))
    if out_dir.exists():
        versions |= {int(_PARAMS_RE.match(p.name).group(1)) for p in out_dir.iterdir() if _PARAMS_RE.match(p.name)}
    version = max(versions | {int(base_raw["version"])}) + 1
    new["version"] = version
    servo_placeholders = yaml.safe_load((cfg_dir / "servo.yaml").read_text()).get("placeholders", [])
    servo_left = [p for p in servo_placeholders if p.split(".", 1)[1] not in new["servo"]]
    new["calibrated"] = bool(not synthetic and not new["placeholders"] and not servo_left)

    new["sysid"] = {
        "parent_version": f"v{base_raw['version']}",
        "created": datetime.now().isoformat(timespec="seconds"),
        "synthetic": bool(synthetic),
        "log_dir": str(log_dir),
        "logs": {kind: {"file": Path(p).name, "sha256": _sha256(p)} for kind, p in logs.items()},
        "measured": {k: float(v) for k, v in measured.items()},
        "fits": {k: v.as_dict() for k, v in fits.items()},
        "suggested_randomization": {
            k: {"range": [float(v.value - 2 * v.stderr), float(v.value + 2 * v.stderr)]}
            for k, v in fits.items() if v.stderr == v.stderr and v.stderr > 0},
        **git_state(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"physics_params_v{version}.yaml"
    header = (f"# Written by sysid/run_sysid.py from {log_dir} ({'SYNTHETIC' if synthetic else 'real'} logs).\n"
              f"# Parent: {base_file.name}. Do not edit fitted values by hand; run sysid again.\n")
    out_file.write_text(header + yaml.safe_dump(new, sort_keys=False))
    if out_dir.resolve() != cfg_dir.resolve():
        for name in ("servo.yaml", "env.yaml"):
            shutil.copy(cfg_dir / name, out_dir / name)
        base_copy = out_dir / base_file.name
        if not base_copy.exists():
            shutil.copy(base_file, base_copy)
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True, help="Folder with band_bench*.csv, servo_step*.csv, shots*.csv")
    parser.add_argument("--base-version", default="latest")
    parser.add_argument("--out-dir", help="Default: configs/ (real logs) or <logs>/fit (synthetic)")
    parser.add_argument("--natural-length", type=float,
                        help="Measured natural (unstretched) band length, m. Needed for the band bench fit")
    parser.add_argument("--synthetic", action="store_true", help="Treat the logs as synthetic")
    args = parser.parse_args()
    out = run(args.logs, args.base_version, args.out_dir, args.natural_length, args.synthetic or None)
    rec = yaml.safe_load(out.read_text())["sysid"]
    print(f"Wrote {out}")
    for k, v in rec["fits"].items():
        print(f"  {k:36s} {v['value']:.6g} +- {v['stderr']:.2g}   ({v['note']})")


if __name__ == "__main__":
    main()
