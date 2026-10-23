"""Config loading and physics-param versioning.

The merged config is built from two files in ``configs/``:

1. ``physics_params_v<N>.yaml``  - measured / fitted physics (versioned)
2. ``env.yaml``                  - sim timing, opponent, match, scoring

The later file overrides the earlier one key by key. ``cfg["meta"]`` records which
params version was used, which values are still uncalibrated placeholders, and
the git state, so every run can be traced to its physics fit.
"""
from __future__ import annotations

import copy
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
_PARAMS_RE = re.compile(r"^physics_params_v(\d+)\.yaml$")


class ConfigError(ValueError):
    pass


def config_dir(path: str | os.PathLike | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("SLINGPUCK_CONFIG_DIR", DEFAULT_CONFIG_DIR))


def available_param_versions(cfg_dir: str | os.PathLike | None = None) -> list[int]:
    versions = []
    for p in config_dir(cfg_dir).iterdir():
        m = _PARAMS_RE.match(p.name)
        if m:
            versions.append(int(m.group(1)))
    return sorted(versions)


def resolve_params_file(version: str | int = "latest", cfg_dir: str | os.PathLike | None = None) -> Path:
    """Map "latest", "v3", "3" or 3 to configs/physics_params_v3.yaml."""
    d = config_dir(cfg_dir)
    if version == "latest":
        versions = available_param_versions(d)
        if not versions:
            raise ConfigError(f"No physics_params_v*.yaml in {d}")
        n = versions[-1]
    else:
        n = int(str(version).lstrip("v"))
    path = d / f"physics_params_v{n}.yaml"
    if not path.exists():
        raise ConfigError(f"Physics params version v{n} not found: {path}")
    return path


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_path(cfg: dict, dotted: str) -> Any:
    node = cfg
    for key in dotted.split("."):
        node = node[key]
    return node


def set_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def _find_nulls(node: Any, prefix: str = "") -> list[str]:
    if isinstance(node, dict):
        out = []
        for k, v in node.items():
            out += _find_nulls(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    return [prefix] if node is None else []


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()

    try:
        return {"git_hash": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_hash": "unknown", "git_dirty": None}


def load_config(
    params_version: str | int = "latest",
    strict: bool = False,
    cfg_dir: str | os.PathLike | None = None,
    overrides: dict | None = None,
) -> dict:
    """Load and merge configs.

    strict=True raises ConfigError if any value is null or still listed as a
    placeholder. Use it for runs whose results are reported against real data.
    """
    d = config_dir(cfg_dir)
    params_file = resolve_params_file(params_version, d)
    params = _read_yaml(params_file)
    env = _read_yaml(d / "env.yaml")

    expected = int(_PARAMS_RE.match(params_file.name).group(1))
    if params.get("version") != expected:
        raise ConfigError(f"{params_file.name} has version {params.get('version')!r}, expected {expected}")

    placeholders = list(params.pop("placeholders", []) or [])
    env.pop("placeholders", None)

    cfg = deep_merge(params, env)
    if overrides:
        cfg = deep_merge(cfg, overrides)

    for p in placeholders:
        try:
            get_path(cfg, p)
        except (KeyError, TypeError):
            raise ConfigError(f"Placeholder {p!r} does not exist in the merged config") from None

    nulls = _find_nulls({k: v for k, v in cfg.items() if k != "meta"})
    if strict and (placeholders or nulls):
        raise ConfigError(
            "strict=True but uncalibrated values remain.\n"
            f"  placeholders: {placeholders}\n  null values: {nulls}"
        )

    cfg["meta"] = {
        "params_version": f"v{expected}",
        "params_file": str(params_file),
        "calibrated": bool(params.get("calibrated", False)),
        "placeholders": placeholders,
        "null_values": nulls,
        "strict": strict,
        **git_state(),
    }
    return cfg


def save_run_config(cfg: dict, run_dir: str | os.PathLike, seed: int | None = None) -> Path:
    """Write the full merged config, params version, git state and seed to run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = copy.deepcopy(cfg)
    out.setdefault("meta", {}).update(git_state())
    if seed is not None:
        out["meta"]["seed"] = int(seed)
    path = run_dir / "run_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return path
