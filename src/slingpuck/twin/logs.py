"""Twin logs: what the twin writes, and the conversion to the sysid shots format.

Inputs from the robot (one file each, see data/real_logs/schema.md, section 4):
  joints*.csv       t_s + one column per joint (rad), as read from the servos
  robot_shots*.csv  one row per sling shot: command values and the event times
                    t_rest_s (arm at home, puck free on the band), t_approach_s
                    (arm starts to move to the puck), t_grasp_s (jaws closed on the
                    puck), t_release_s (release command)

Outputs of the twin:
  tracks.csv  one row per frame per puck: filter state and the raw measurement
  shots.csv   sysid format (schema.md, section 3), built per robot shot:
      rest:    raw camera positions of the shot puck while the arm is at home
      pulled:  the grasp point from forward kinematics of the joint readings,
               0.3 s before the release (the gripper hides the puck from the
               overhead camera while it holds it)
      flight:  raw camera positions of the shot puck after the release command
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["t_s", "puck_id", "colour", "x_m", "y_m", "vx_m_s", "vy_m_s", "hidden", "meas_x_m", "meas_y_m"]
ROBOT_SHOT_COLUMNS = ["shot_id", "t_rest_s", "t_approach_s", "t_grasp_s", "t_release_s", "cmd_slide_m", "cmd_angle_rad",
                      "cmd_pull_m"]
PULLED_WINDOW_S = 0.3
FLIGHT_WINDOW_S = 1.0


def tracks_frame(tracker) -> pd.DataFrame:
    rows = [(t, tr.id, tr.colour, x, y, vx, vy, int(h), mx, my)
            for tr in tracker.tracks if tr.confirmed for (t, x, y, vx, vy, h, mx, my) in tr.history]
    return pd.DataFrame(rows, columns=TRACK_COLUMNS).sort_values(["t_s", "puck_id"]).reset_index(drop=True)


@dataclass
class ShotMatch:
    shot_id: int
    puck_id: int | None
    n_rest: int
    n_pulled: int
    n_flight: int
    success: bool


def build_shots(tracks: pd.DataFrame, robot_shots: pd.DataFrame, joints: pd.DataFrame, grasp_fk) -> tuple:
    """tracks.csv + robot logs -> sysid shots.csv rows.

    grasp_fk(q) gives the board xy of the grasp point for one row of joint angles
    (TwinScene.set_arm + grasp_xy). Returns (DataFrame, list of ShotMatch).
    """
    meas = tracks.dropna(subset=["meas_x_m"])
    jt = joints["t_s"].to_numpy()
    jq = joints.drop(columns=["t_s"]).to_numpy()
    rows, matches = [], []
    for _, s in robot_shots.iterrows():
        sid = int(s.shot_id)
        meta = (float(s.cmd_slide_m), float(s.cmd_angle_rad), float(s.cmd_pull_m))
        # Pulled: forward kinematics of the joint readings before the release.
        sel = (jt >= s.t_release_s - PULLED_WINDOW_S) & (jt < s.t_release_s)
        pulled = [(t, *grasp_fk(q)) for t, q in zip(jt[sel], jq[sel])]
        grasp_at_release = np.mean([p[1:] for p in pulled], axis=0) if pulled else None
        # Rest: the puck that rests nearest the grasp point when the jaws close.
        rest_win = meas[(meas.t_s >= s.t_rest_s) & (meas.t_s < s.t_approach_s)]
        k = min(int(np.searchsorted(jt, s.t_grasp_s)), len(jt) - 1)
        grasp_xy = grasp_fk(jq[k])
        puck_id = None
        if len(rest_win):
            means = rest_win.groupby("puck_id")[["meas_x_m", "meas_y_m"]].mean()
            dist = np.hypot(means.meas_x_m - grasp_xy[0], means.meas_y_m - grasp_xy[1])
            if dist.min() < 0.02:
                puck_id = int(dist.idxmin())
        flight = meas[(meas.t_s >= s.t_release_s) & (meas.t_s < s.t_release_s + FLIGHT_WINDOW_S)]
        if puck_id is not None:
            fl = flight[flight.puck_id == puck_id]
            if len(fl) < 4 and grasp_at_release is not None:  # Identity lost: the puck that leaves the release point
                first = flight.groupby("puck_id").first()
                near = np.hypot(first.meas_x_m - grasp_at_release[0], first.meas_y_m - grasp_at_release[1])
                if len(near) and near.min() < 0.05:
                    fl = flight[flight.puck_id == int(near.idxmin())]
        else:
            fl = flight.iloc[0:0]
        ys = fl.meas_y_m.to_numpy()
        success = bool(len(ys) and ys.max() > 0.0 and np.argmax(ys > 0.0) <= np.argmax(ys))
        rest = rest_win[rest_win.puck_id == puck_id] if puck_id is not None else rest_win.iloc[0:0]
        for _, r in rest.iterrows():
            rows.append((sid, "rest", r.t_s - s.t_release_s, r.meas_x_m, r.meas_y_m, *meta, int(success)))
        for t, x, y in pulled:
            rows.append((sid, "pulled", t - s.t_release_s, x, y, *meta, int(success)))
        for _, r in fl.iterrows():
            rows.append((sid, "flight", r.t_s - s.t_release_s, r.meas_x_m, r.meas_y_m, *meta, int(success)))
        matches.append(ShotMatch(sid, puck_id, len(rest), len(pulled), len(fl), success))
    df = pd.DataFrame(rows, columns=["shot_id", "phase", "t_s", "puck_x_m", "puck_y_m", "cmd_slide_m",
                                     "cmd_angle_rad", "cmd_pull_m", "success"])
    return df, matches
