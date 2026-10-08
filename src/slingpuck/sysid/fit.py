"""Fit physics parameters to logs (see sysid/schema.py).

Each fit returns {config_path: FitValue}. Standard errors come from the
least-squares Jacobian (linearized), so they are approximate.

- fit_band_bench: band stiffness, exponent, hysteresis loss. The bench pull
  changes the band length only a little, so stiffness, exponent and natural
  length trade off (only the pre-tension and slope are fixed by the data).
  Measure the natural length directly (unhooked band, ruler); it is held fixed.
- fit_servo_steps: command latency (grid, 1 ms), lag, max rate, deadband.
- fit_shots: compliance, sliding friction, wall restitution, energy transfer.
  Hysteresis and energy transfer both scale the launch speed, so shots alone
  cannot separate them; the band bench gives hysteresis.
"""
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

from src.slingpuck.config import set_path
from src.slingpuck.physics.arm_deflection import pull_direction, solve_pull
from src.slingpuck.physics.band_model import GRAVITY, BandModel, BandParams
from src.slingpuck.physics.servo_model import ServoModel


@dataclass(frozen=True)
class FitValue:
    value: float
    stderr: float
    n: int            # Number of data points (or shots) behind the fit
    note: str = ""

    def as_dict(self):
        return {"value": float(self.value), "stderr": float(self.stderr), "n": int(self.n), "note": self.note}


def _stderr(result, n_params):
    """Linearized standard errors from a scipy least_squares result."""
    J = result.jac
    dof = max(len(result.fun) - n_params, 1)
    s2 = 2.0 * result.cost / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        return np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        return np.full(n_params, np.nan)


def with_values(cfg: dict, values: dict) -> dict:
    out = copy.deepcopy(cfg)
    for path, v in values.items():
        set_path(out, path, float(v.value if isinstance(v, FitValue) else v))
    return out


# ------------------------------------------------------------------ band
def fit_band_bench(df: pd.DataFrame, cfg: dict) -> dict:
    """natural_length_m is taken from cfg (measure it directly) and held fixed."""
    b = cfg["band"]
    natural = b["natural_length_m"]
    band_y = 0.0
    pulls = df["pull_m"].to_numpy()
    forces = df["force_n"].to_numpy()
    unload = (df["phase"] == "unload").to_numpy()
    top = df.groupby("cycle")["pull_m"].transform("max").to_numpy()

    def model(p):
        k, a, loss = p
        band = BandModel(BandParams(b["anchor_span_m"], natural, k, a, loss, 1.0), band_y)
        out = np.empty(len(pulls))
        for i, (d, u, d_top) in enumerate(zip(pulls, unload, top)):
            pos = (0.0, band_y - d)
            s = band.elongation(pos)
            t = band.tension_unload(s, band.elongation((0.0, band_y - d_top))) if u else band.tension_load(s)
            out[i] = float(t) * band.direction_sum(pos)[1]
        return out

    x0 = [b["stiffness_k_n_m"], b["stiffness_exponent"], b["hysteresis_loss_factor"]]
    res = least_squares(lambda p: model(p) - forces, x0, bounds=([1.0, 0.5, 0.0], [5000.0, 3.0, 0.9]),
                        x_scale="jac")
    se = _stderr(res, 3)
    rms = float(np.sqrt(np.mean(res.fun ** 2)))
    note = f"rms force residual {rms:.3f} N; natural_length fixed at {natural:.4f} m"
    names = ["band.stiffness_k_n_m", "band.stiffness_exponent", "band.hysteresis_loss_factor"]
    return {n: FitValue(v, s, len(df), note) for n, v, s in zip(names, res.x, se)}


# ----------------------------------------------------------------- servo
def _servo_sim(df, dt, latency_s, tau, rate_deg, deadband_deg):
    tests = [g.sort_values("t_s") for _, g in df.groupby("test_id")]
    n = int(round(max(g["t_s"].iloc[-1] for g in tests) / dt)) + 1
    cmd = np.empty((n, len(tests)))
    for j, g in enumerate(tests):
        idx = np.searchsorted(g["t_s"].to_numpy(), np.arange(n) * dt + 1e-9, side="right") - 1
        cmd[:, j] = g["cmd_rad"].to_numpy()[np.clip(idx, 0, len(g) - 1)]
    servo = ServoModel({"command_latency_s": latency_s, "lag_tau_s": tau, "max_rate_deg_s": rate_deg,
                        "deadband_deg": deadband_deg}, dt,
                       init_pos=np.array([g["meas_rad"].iloc[0] for g in tests]))
    traj = np.array([servo.step(cmd[k]) for k in range(n)])
    pred = [traj[np.clip(np.round(g["t_s"].to_numpy() / dt).astype(int), 0, n - 1), j] for j, g in enumerate(tests)]
    meas = [g["meas_rad"].to_numpy() for g in tests]
    return np.concatenate(pred), np.concatenate(meas)


def fit_servo_steps(df: pd.DataFrame, cfg: dict, dt: float = 0.001, max_latency_s: float = 0.15) -> dict:
    """Latency (1 ms grid) and deadband (0.02 deg grid) by search; lag and rate by least squares.

    A deadband gives no useful gradient, so it is searched, not fitted.
    """
    s = cfg["servo"]
    x0 = [s["lag_tau_s"], s["max_rate_deg_s"]]
    bounds = ([1e-3, 10.0], [0.5, 3000.0])

    def fit_at(latency, deadband):
        r = least_squares(lambda p: np.subtract(*_servo_sim(df, dt, latency, p[0], p[1], deadband)), x0,
                          bounds=bounds, x_scale=[0.05, 100.0], diff_step=1e-3)
        return r.cost, r

    def search(grid, fn):
        results = {g: fn(g) for g in grid}
        return min(results, key=lambda g: results[g][0]), results

    deadband = s["deadband_deg"]
    center, _ = search(np.arange(0.0, max_latency_s + 1e-9, 0.005), lambda lat: fit_at(lat, deadband))
    latency, _ = search(np.arange(max(center - 0.005, 0.0), center + 0.0051, dt), lambda lat: fit_at(lat, deadband))
    coarse_db, _ = search(np.arange(0.0, 2.0001, 0.1), lambda db: fit_at(latency, db))
    deadband, results = search(np.arange(max(coarse_db - 0.1, 0.0), coarse_db + 0.1001, 0.02),
                               lambda db: fit_at(latency, db))
    res = results[deadband][1]
    se = _stderr(res, 2)
    rms = float(np.sqrt(np.mean(res.fun ** 2)))
    note = f"rms joint residual {np.rad2deg(rms):.3f} deg"
    return {
        "servo.command_latency_s": FitValue(latency, dt / 2, len(df), note + "; 1 ms grid"),
        "servo.lag_tau_s": FitValue(res.x[0], se[0], len(df), note),
        "servo.max_rate_deg_s": FitValue(res.x[1], se[1], len(df), note),
        "servo.deadband_deg": FitValue(deadband, 0.01, len(df), note + "; 0.02 deg grid"),
    }


# ----------------------------------------------------------------- shots
@dataclass
class _Shot:
    sid: int
    rest: np.ndarray
    pulled: np.ndarray
    flight: pd.DataFrame
    slide: float
    angle: float
    pull: float


def _shots(df):
    out = []
    for sid, g in df.groupby("shot_id"):
        first = g.iloc[0]
        rest = g[g.phase == "rest"][["puck_x_m", "puck_y_m"]].to_numpy()
        pulled = g[g.phase == "pulled"][["puck_x_m", "puck_y_m"]].to_numpy()
        if len(rest) == 0 or len(pulled) == 0:
            continue
        out.append(_Shot(int(sid), rest.mean(axis=0), pulled.mean(axis=0),
                         g[g.phase == "flight"].sort_values("t_s"),
                         float(first.cmd_slide_m), float(first.cmd_angle_rad), float(first.cmd_pull_m)))
    return out


def _path(t, p, theta, s0, a):
    """Straight sliding path from p at speed s0 (time t >= 0), constant deceleration a, stops at 0."""
    t = np.asarray(t, dtype=float)
    tau = np.clip(t, None, s0 / a) if a > 0 else t
    dist = s0 * tau - 0.5 * a * tau ** 2
    return p[0] + np.sin(theta) * dist, p[1] + np.cos(theta) * dist


@dataclass
class _Segment:
    shot: int          # Index into the shot list
    kind: str          # "exit" (band line -> divider), "through" (after the gate), "return" (after a divider bounce)
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t0: float          # Reference time of the segment parameters


MIN_FRAMES = 4


def _raw_segments(i, shot, band_y, face_y, half_length, radius):
    """Free sliding segments of one shot, cut by the measured y (first pass)."""
    f = shot.flight
    t, x, y = f.t_s.to_numpy(), f.puck_x_m.to_numpy(), f.puck_y_m.to_numpy()
    if len(t) < MIN_FRAMES:
        return []
    far = half_length - radius - 0.01
    segs = []
    # Exit: frames after the band line, before the divider. A bounce can fall between two
    # frames, so for a puck that never crossed, the exit also ends at the highest frame.
    crossed = np.flatnonzero(y > -face_y + 0.004)
    k_stop = len(y) if len(crossed) else int(np.argmax(y)) + 1
    idx = []
    for k in range(k_stop):
        if y[k] >= face_y - 0.004:
            break
        if y[k] > band_y + 0.005:
            idx.append(k)
    if len(idx) >= MIN_FRAMES:
        segs.append(("exit", idx))
    after = idx[-1] + 1 if idx else 0
    if len(crossed):  # Through the gate: free path in the opponent half until the far wall
        k0 = int(crossed[0])
        k1 = k0 + int(np.argmax(y[k0:]))
        idx2 = [k for k in range(k0, k1 + 1) if -face_y + 0.004 < y[k] < far]
        if len(idx2) >= MIN_FRAMES:
            segs.append(("through", idx2))
    elif after < len(y):  # Bounced off the divider: free path back toward the agent end wall
        rest = np.arange(after, len(y))
        k_top = after + int(np.argmax(y[after:]))
        back = rest[rest > k_top]
        if len(back):
            k_end = int(back[np.argmin(y[back])])
            idx3 = [k for k in back if k <= k_end and -half_length + radius + 0.01 < y[k] < face_y - 0.004]
            if len(idx3) >= MIN_FRAMES:
                segs.append(("return", idx3))
    return [_Segment(i, kind, t[k], x[k], y[k], float(t[k][0])) for kind, k in segs]


def _seg_params_guess(seg):
    theta = np.arctan2(seg.x[-1] - seg.x[0], seg.y[-1] - seg.y[0])
    speed = np.hypot(seg.x[-1] - seg.x[0], seg.y[-1] - seg.y[0]) / max(seg.t[-1] - seg.t[0], 1e-3)
    return [seg.x[0], seg.y[0], theta, speed]


def _fit_segments(segs, a0):
    """Shared deceleration + per segment (x, y at t0, heading, speed at t0)."""
    x0 = [a0]
    for sg in segs:
        x0 += _seg_params_guess(sg)

    def resid(p):
        a = p[0]
        out = []
        for i, sg in enumerate(segs):
            px, py, th, s0 = p[1 + 4 * i: 5 + 4 * i]
            mx, my = _path(sg.t - sg.t0, (px, py), th, s0, a)
            out += [mx - sg.x, my - sg.y]
        return np.concatenate(out)

    lo = [0.0] + [-np.inf, -np.inf, -2 * np.pi, 0.01] * len(segs)
    hi = [50.0] + [np.inf, np.inf, 2 * np.pi, 20.0] * len(segs)
    return least_squares(resid, x0, bounds=(lo, hi), x_scale="jac")


def _reselect(segs, fit, shots, band_y, face_y, half_length, radius, margin=0.006):
    """Second pass: keep frames by the *fitted* y with a margin larger than the camera noise.

    Selecting by the measured y keeps frames whose noise points inward at both window ends,
    which makes each path look shorter and biases speeds and deceleration low.
    """
    out = []
    for i, sg in enumerate(segs):
        px, py, th, s0 = fit.x[1 + 4 * i: 5 + 4 * i]
        f = shots[sg.shot].flight
        t_all, x_all, y_all = f.t_s.to_numpy(), f.puck_x_m.to_numpy(), f.puck_y_m.to_numpy()
        window = (t_all >= sg.t[0] - 0.05) & (t_all <= sg.t[-1] + 0.05)
        _, yp = _path(t_all - sg.t0, (px, py), th, s0, fit.x[0])
        if sg.kind == "exit":
            ok = (yp > band_y + margin) & (yp < face_y - margin)
        elif sg.kind == "through":
            ok = (yp > -face_y + margin) & (yp < half_length - radius - margin)
        else:
            ok = (yp < face_y - margin) & (yp > -half_length + radius + margin)
        ok &= window & (t_all >= sg.t0 - 1e-9) if sg.kind != "exit" else window & ok
        if sg.kind == "exit":
            ok &= t_all <= sg.t[-1] + 1e-9 + 0.02  # Do not run past the divider contact
        if ok.sum() >= MIN_FRAMES:
            out.append(_Segment(sg.shot, sg.kind, t_all[ok], x_all[ok], y_all[ok], sg.t0))
    return out


def _time_at_y(px, py, th, s0, a, y_target):
    """Time (relative to t0, may be negative) when the path reaches y_target, or None."""
    c = np.cos(th)
    if abs(c) < 1e-6:
        return None
    # py + c (s0 tau - a tau^2 / 2) = y_target
    A, B, C = -0.5 * a * c, s0 * c, py - y_target
    if abs(A) < 1e-12:
        return -C / B
    disc = B * B - 4 * A * C
    if disc < 0:
        return None
    roots = sorted([(-B - np.sqrt(disc)) / (2 * A), (-B + np.sqrt(disc)) / (2 * A)])
    # The physical root is the one where the speed s0 - a tau is still positive.
    valid = [r for r in roots if s0 - a * r > 0]
    return min(valid, key=abs) if valid else None


def fit_shots(df: pd.DataFrame, cfg: dict) -> dict:
    shots = _shots(df)
    if not shots:
        raise ValueError("no usable shots")
    hl, r = 0.5 * cfg["board"]["length_m"], cfg["puck"]["radius_m"]
    face_y = -(0.5 * cfg["board"]["divider_thickness_m"] + r)
    band_y = float(np.mean([s.rest[1] for s in shots]))  # Band line seen by the camera
    out = {}

    # 1) Compliance: measured pull depth vs commanded pull through the band model.
    band = BandModel(BandParams.from_config(cfg), band_y)

    def depth_pred(c, s):
        u = pull_direction(s.angle)
        start_x = s.pulled[0] + s.pull * u[0] * -1.0
        res = solve_pull(band, (start_x, band_y), s.angle, s.pull, c, tol=1e-7)
        return band_y - res.pos[1]

    meas = np.array([band_y - s.pulled[1] for s in shots])

    def sse(c):
        return float(np.sum((meas - np.array([depth_pred(c, s) for s in shots])) ** 2))

    best = minimize_scalar(sse, bounds=(0.0, 0.01), method="bounded", options={"xatol": 1e-6})
    c = float(best.x)
    resid = meas - np.array([depth_pred(c, s) for s in shots])
    h = 1e-5
    deriv = (np.array([depth_pred(c + h, s) for s in shots]) - np.array([depth_pred(max(c - h, 0), s) for s in shots])) \
        / (c + h - max(c - h, 0))
    se_c = float(np.sqrt(np.var(resid, ddof=1) / max(np.sum(deriv ** 2), 1e-18)))
    out["arm.deflection_compliance_m_per_n"] = FitValue(c, se_c, len(shots),
                                                        f"rms depth residual {1000 * np.sqrt(np.mean(resid ** 2)):.2f} mm")

    # 2) Friction: one deceleration shared by every free sliding segment of every shot
    #    (exit, through the gate, back from the divider). Two passes; see _reselect.
    segs = [sg for i, s in enumerate(shots) for sg in _raw_segments(i, s, band_y, face_y, hl, r)]
    if not segs:
        raise ValueError("no usable flight segments")
    fit = _fit_segments(segs, cfg["puck"]["friction_kinetic"] * GRAVITY)
    segs = _reselect(segs, fit, shots, band_y, face_y, hl, r)
    fit = _fit_segments(segs, fit.x[0])
    decel = float(fit.x[0])
    se_decel = float(_stderr(fit, len(fit.x))[0])
    kinds = {k: sum(sg.kind == k for sg in segs) for k in ("exit", "through", "return")}
    out["puck.friction_kinetic"] = FitValue(
        decel / GRAVITY, se_decel / GRAVITY, len(segs),
        f"segments {kinds}; rms position residual {1000 * np.sqrt(np.mean(fit.fun ** 2)):.2f} mm")
    params = {(sg.shot, sg.kind): fit.x[1 + 4 * i: 5 + 4 * i] for i, sg in enumerate(segs)}
    t0 = {(sg.shot, sg.kind): sg.t0 for sg in segs}

    # 3) Wall restitution: normal speed into and out of the divider face at the bounce time.
    es = []
    for (i, kind), (px, py, th, s0) in params.items():
        if kind != "exit" or (i, "return") not in params:
            continue
        tau_b = _time_at_y(px, py, th, s0, decel, face_y)
        if tau_b is None:
            continue
        t_b = t0[(i, "exit")] + tau_b
        v_in = (s0 - decel * tau_b) * np.cos(th)
        qx, qy, th2, s2 = params[(i, "return")]
        v_out = (s2 - decel * (t_b - t0[(i, "return")])) * np.cos(th2)
        if v_in > 0 and v_out < 0:
            es.append(-v_out / v_in)
    if es:
        es = np.array(es)
        se_e = float(1.2533 * np.std(es, ddof=1) / np.sqrt(len(es))) if len(es) > 1 else float("nan")
        out["board.restitution"] = FitValue(float(np.median(es)), se_e, len(es), "median over divider bounces")

    # 4) Energy transfer: exit speed^2 at the band line vs band-model exit speed^2 (energy_transfer = 1).
    fitted = with_values(cfg, {"puck.friction_kinetic": decel / GRAVITY})
    band1 = BandModel(BandParams(**{**BandParams.from_config(fitted).__dict__, "energy_transfer": 1.0}), band_y)
    v_meas2, v_model2 = [], []
    for (i, kind), (px, py, th, s0) in params.items():
        if kind != "exit":
            continue
        tau_e = _time_at_y(px, py, th, s0, decel, band_y)
        if tau_e is None:
            continue
        v_meas2.append((s0 - decel * tau_e) ** 2)
        rel = band1.release(shots[i].pulled, fitted["puck"]["mass_kg"], fitted["puck"]["friction_kinetic"])
        v_model2.append(float(rel.exit_vel @ rel.exit_vel))
    v_meas2, v_model2 = np.array(v_meas2), np.array(v_model2)
    kappa = float(np.sum(v_meas2 * v_model2) / np.sum(v_model2 ** 2))
    res_k = v_meas2 - kappa * v_model2
    se_k = float(np.sqrt(np.var(res_k, ddof=1) / np.sum(v_model2 ** 2)))
    out["band.energy_transfer"] = FitValue(kappa, se_k, len(v_meas2), "least squares on exit speed^2")
    return out
