"""Multi-puck tracker for the twin: one Kalman filter per puck, identities kept by assignment.

Each camera frame:
1. Predict every track to the frame time.
2. Match detections to tracks with the Hungarian method on distance. The gate
   (largest match distance) grows with the time a track has been hidden, because
   a hidden puck can be moved by the hand or the arm.
3. Matched tracks get a Kalman update. A track that reappears after a long hide
   restarts its filter, so an old velocity does not carry over.
4. Unmatched detections start tentative tracks. A tentative track becomes a
   puck after `confirm_frames` hits in a row; a single false detection (an arm
   part) dies. When all `n_pucks` pucks have tracks, an unmatched detection is
   given to the nearest hidden puck instead (the board holds a fixed number of
   pucks, so a new puck is an old puck that reappeared).
5. Unmatched pucks are hidden: they coast at their last velocity for `coast_s`,
   then stay where they were last seen (pucks under a hand do not move far on
   their own).

Every frame, the raw matched detection of each puck is kept as well. Sysid fits
its own motion model to raw positions, so the logs use those, not the filter.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.slingpuck.sensing.camera_model import Frame
from src.slingpuck.sensing.kalman_tracker import KalmanTracker


@dataclass
class Track:
    id: int
    kf: KalmanTracker
    confirmed: bool = False
    hits: int = 0
    last_seen: float = 0.0
    hidden: bool = False
    last_meas: np.ndarray | None = None
    history: list = field(default_factory=list)  # (t, x, y, vx, vy, hidden, meas_x, meas_y)


@dataclass(frozen=True)
class PuckState:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    hidden: bool
    hidden_s: float


class MultiPuckTracker:
    def __init__(self, cfg: dict, n_pucks: int, meas_noise_m: float = 0.002, accel_std_m_s2: float = 8.0,
                 gate_m: float = 0.03, gate_growth_m_s: float = 0.5, max_gate_m: float = 0.15,
                 coast_s: float = 0.1, confirm_frames: int = 3, restart_after_s: float = 0.15):
        self.cfg, self.n_pucks = cfg, n_pucks
        self.meas_noise, self.accel_std = meas_noise_m, accel_std_m_s2
        self.gate, self.gate_growth, self.max_gate = gate_m, gate_growth_m_s, max_gate_m
        self.coast_s, self.confirm_frames, self.restart_after_s = coast_s, confirm_frames, restart_after_s
        self.tracks: list[Track] = []
        self._next_id = 0
        self.t = 0.0

    def _new_kf(self):
        return KalmanTracker(self.meas_noise, self.accel_std)

    def _predict(self, tr: Track, t: float) -> np.ndarray:
        est = tr.kf.estimate(t)
        if est is None:
            return tr.last_meas
        pos, vel, _ = est
        hidden_s = t - tr.last_seen
        if hidden_s > self.coast_s:  # Stop coasting: the puck stays where it was last seen
            pos = tr.kf.x[:2] + tr.kf.x[2:] * self.coast_s
        return pos

    def _gate(self, tr: Track, t: float) -> float:
        return min(self.max_gate, self.gate + self.gate_growth * max(t - tr.last_seen, 0.0))

    def _hit(self, tr: Track, t: float, z: np.ndarray):
        if t - tr.last_seen > self.restart_after_s:
            tr.kf = self._new_kf()
        tr.kf.update(Frame(t, t, z))
        tr.hits += 1
        tr.last_seen, tr.hidden, tr.last_meas = t, False, z.copy()
        if tr.hits >= self.confirm_frames:
            tr.confirmed = True

    def update(self, t: float, detections: list[np.ndarray]) -> list[PuckState]:
        self.t = t
        dets = [np.asarray(d, float) for d in detections]
        tracks = self.tracks
        matched_t, matched_d = set(), set()
        if tracks and dets:
            pred = np.array([self._predict(tr, t) for tr in tracks])
            cost = np.linalg.norm(pred[:, None, :] - np.array(dets)[None, :, :], axis=2)
            # Confirmed tracks first: give them a lower cost than tentative ones at the same distance.
            biased = cost + np.array([0.0 if tr.confirmed else 0.01 for tr in tracks])[:, None]
            rows, cols = linear_sum_assignment(biased)
            for i, j in zip(rows, cols):
                if cost[i, j] <= self._gate(tracks[i], t):
                    self._hit(tracks[i], t, dets[j])
                    matched_t.add(i)
                    matched_d.add(j)
        free = [j for j in range(len(dets)) if j not in matched_d]
        n_confirmed = sum(tr.confirmed for tr in tracks)
        for j in free:
            hidden = [i for i, tr in enumerate(tracks) if tr.confirmed and i not in matched_t]
            if n_confirmed >= self.n_pucks and hidden:
                # Fixed puck count: a detection with no track is a hidden puck that reappeared.
                i = min(hidden, key=lambda k: np.linalg.norm(self._predict(tracks[k], t) - dets[j]))
                tracks[i].kf = self._new_kf()
                tracks[i].last_seen = -1e9  # Force a filter restart
                self._hit(tracks[i], t, dets[j])
                matched_t.add(i)
            elif n_confirmed < self.n_pucks:
                tr = Track(self._next_id, self._new_kf(), last_seen=t)
                self._next_id += 1
                self._hit(tr, t, dets[j])
                tracks.append(tr)
                matched_t.add(len(tracks) - 1)
        # Tentative tracks die on their first miss; confirmed ones become hidden.
        keep = []
        for i, tr in enumerate(tracks):
            if i not in matched_t:
                if not tr.confirmed:
                    continue
                tr.hidden = True
            keep.append(tr)
        self.tracks = keep
        states = self.states(t)
        for tr, s in zip([tr for tr in self.tracks if tr.confirmed], states):
            meas = tr.last_meas if not tr.hidden else (np.nan, np.nan)
            tr.history.append((t, *s.pos, *s.vel, s.hidden, *meas))
        return states

    def states(self, t: float | None = None) -> list[PuckState]:
        t = self.t if t is None else t
        out = []
        for tr in self.tracks:
            if not tr.confirmed:
                continue
            pos = self._predict(tr, t)
            est = tr.kf.estimate(t)
            vel = est[1] if est is not None and t - tr.last_seen <= self.coast_s else np.zeros(2)
            out.append(PuckState(tr.id, np.asarray(pos, float), np.asarray(vel, float), tr.hidden, t - tr.last_seen))
        return out
