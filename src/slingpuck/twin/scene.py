"""MuJoCo scene of the digital twin: the board, the SO-101 and N pucks, all set from measurements.

Nothing here is simulated. The pucks are mocap bodies put at the tracked
positions, and the arm joints are set from the robot's joint readings, so the
twin shows the real game as measured. A hidden puck (under a hand or the arm)
is drawn faded at its last known place.

Overlays (drawn per frame, not part of the model): both bands (a V around the
back of a puck pressed behind a band line), the predicted path of each moving
puck from the 2D sim (orange dots), and the gate-arrival point of a puck
heading for the agent gate (red ring).

The scene has the cameras of the sling scene (sling, band, shot, gate, side,
top) plus "overhead": the fixed camera above the board that the real tracking
camera stands for (straight down, image x along the board length).
"""
import mujoco
import numpy as np

from src.slingpuck.physics.band_model import BandModel
from src.slingpuck.physics.mujoco_goalkeeper import SceneLayout, _build_spec, _qadr
from src.slingpuck.physics.mujoco_sling import (
    ARM_JOINTS, add_band_pegs, add_grasp_site, add_view_cameras, grasp_geometry)
from src.slingpuck.physics.puck_dynamics import Fast2DPuckSim

JOINTS = ARM_JOINTS + ("gripper",)
PUCK_RGBA = {"black": (0.06, 0.06, 0.06, 1.0), "white": (0.96, 0.96, 0.94, 1.0)}  # Robot's / human's pucks
HIDDEN_ALPHA = 0.3
PARK_Z = -1.0
BAND_RGBA = np.array([0.80, 0.12, 0.10, 1.0], np.float32)
PATH_RGBA = np.array([0.95, 0.55, 0.10, 0.6], np.float32)
ARRIVAL_RGBA = np.array([0.90, 0.10, 0.10, 0.8], np.float32)
BAND_WIDTH_M = 0.0012


def overhead_xyaxes():
    """Image x = board +y (length), image up = board -x; the camera looks straight down."""
    return [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]


class TwinScene:
    def __init__(self, cfg: dict, n_pucks: int, overhead_height_m: float = 0.6, overhead_nadir=(0.0, 0.0),
                 overhead_fovy_deg: float = 35.0):
        self.cfg, self.n_pucks = cfg, n_pucks
        self.layout = SceneLayout.from_config(cfg)
        spec = _build_spec(cfg, self.layout, None)
        spec.delete(spec.body("puck"))
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
        add_grasp_site(spec, grasp_geometry(2.0 * cfg["puck"]["radius_m"]))
        add_band_pegs(spec, cfg)
        add_view_cameras(spec)
        r, th = cfg["puck"]["radius_m"], cfg["puck"]["thickness_m"]
        for i in range(n_pucks):
            b = spec.worldbody.add_body(name=f"twin_puck_{i}", mocap=True, pos=[0.0, 0.0, PARK_Z])
            b.add_geom(name=f"twin_puck_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, 0.5 * th, 0],
                       rgba=list(PUCK_RGBA["black"]), contype=0, conaffinity=0)
        self.overhead_pos = np.array([overhead_nadir[0], overhead_nadir[1], overhead_height_m])
        spec.worldbody.add_camera(name="overhead", pos=self.overhead_pos.tolist(), xyaxes=overhead_xyaxes(),
                                  fovy=overhead_fovy_deg)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._qadr = _qadr(m, JOINTS)
        self._mocap = [int(m.body_mocapid[m.body(f"twin_puck_{i}").id]) for i in range(n_pucks)]
        self._geom = [m.geom(f"twin_puck_{i}").id for i in range(n_pucks)]
        self._site = m.site("grasp").id
        base = m.body("base").id
        self._arm_mesh_geoms = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == base
                                and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] < 3]
        self.band = BandModel.from_config(cfg)
        self.slots: dict[int, int] = {}   # Track id -> puck slot
        self.pucks = []                    # Last PuckState list
        self.predictions = {}              # Track id -> (n, 2) predicted path
        self.arrivals = {}                 # Track id -> x where it reaches the agent gate line
        self.set_arm(np.zeros(len(JOINTS)))

    # ---------------------------------------------------------------- state
    def set_arm(self, q):
        self.data.qpos[self._qadr] = np.asarray(q, float)[:len(JOINTS)]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def arm_mask(self, calib, height: int, width: int, grow_px: int = 4) -> np.ndarray:
        """Pixels of the arm in the real camera image, from the joint readings and the calibration only.

        For any pinhole camera, a point at height z appears at the board point where the line from the
        lens through the point meets the board: nadir + (p - nadir) * H / (H - z). So each arm mesh is
        projected to the board plane that way, then to pixels with the board homography, and filled.
        Needs calib.camera_height_m and camera_nadir_xy (no lens values needed).
        """
        import cv2
        m, d = self.model, self.data
        H, nadir = calib.camera_height_m, calib.camera_nadir_xy
        mask = np.zeros((height, width), np.uint8)
        for g in self._arm_mesh_geoms:
            mid = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            verts = m.mesh_vert[v0:v0 + nv:max(1, nv // 300)]
            world = verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            z = np.minimum(world[:, 2], 0.95 * H)
            xy = nadir + (world[:, :2] - nadir) * (H / (H - z))[:, None]
            px = calib.plane_to_px(xy)
            hull = cv2.convexHull(px.astype(np.float32)).astype(np.int32)
            cv2.fillConvexPoly(mask, hull, 1)
        if grow_px:
            k = 2 * grow_px + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        return mask > 0

    def grasp_xy(self) -> np.ndarray:
        """Board xy of the point between the jaw tips (forward kinematics of the joint readings)."""
        return self.data.site_xpos[self._site][:2].copy()

    def set_pucks(self, states):
        """states: PuckState list from MultiPuckTracker (or anything with id, pos, hidden, optional colour)."""
        z = 0.5 * self.cfg["puck"]["thickness_m"]
        live = {s.id for s in states}
        for tid in [t for t in self.slots if t not in live]:
            del self.slots[tid]
        for s in states:
            if s.id not in self.slots:
                free = sorted(set(range(self.n_pucks)) - set(self.slots.values()))
                if not free:
                    continue
                self.slots[s.id] = free[0]
        used = set()
        for s in states:
            k = self.slots.get(s.id)
            if k is None:
                continue
            used.add(k)
            self.data.mocap_pos[self._mocap[k]] = [s.pos[0], s.pos[1], z]
            rgba = list(PUCK_RGBA[getattr(s, "colour", "black")])
            rgba[3] = HIDDEN_ALPHA if s.hidden else 1.0
            self.model.geom_rgba[self._geom[k]] = rgba
        for k in set(range(self.n_pucks)) - used:
            self.data.mocap_pos[self._mocap[k]] = [0.0, 0.0, PARK_Z]
        self.pucks = list(states)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def predict(self, min_speed: float = 0.2, horizon_s: float = 1.0, dt: float = 0.002):
        """2D-sim path of each moving puck (free flight, other pucks ignored) and its agent-gate arrival x."""
        self.predictions, self.arrivals = {}, {}
        lim = np.array([0.5 * self.cfg["board"]["width_m"], 0.5 * self.cfg["board"]["length_m"]]) \
            - self.cfg["puck"]["radius_m"]
        for s in self.pucks:
            if s.hidden or np.linalg.norm(s.vel) < min_speed:
                continue
            sim = Fast2DPuckSim.from_config(self.cfg)
            sim.reset(np.clip(s.pos, -lim, lim)[None], s.vel[None])
            path = []
            for k in range(int(horizon_s / dt)):
                ev = sim.step(dt)
                if k % 10 == 0:
                    path.append(sim.pos[0].copy())
                for c in ev.crossings:
                    if c.direction == -1 and s.id not in self.arrivals:
                        self.arrivals[s.id] = c.x
                if not np.any(sim.vel):
                    break
            self.predictions[s.id] = np.array(path)

    # -------------------------------------------------------------- overlays
    def add_overlays(self, scene: mujoco.MjvScene, paths: bool = True):
        r = self.cfg["puck"]["radius_m"]
        z = 0.5 * self.cfg["puck"]["thickness_m"]
        half = 0.5 * self.cfg["band"]["anchor_span_m"]
        by = self.band.band_y
        for sign in (1.0, -1.0):  # Agent band (y < 0), then the opponent band (mirror)
            y_line = sign * (by - r)
            def depth(p):  # How far a puck is behind this band line (toward its end wall)
                return by - sign * p[1]

            pressed = [s for s in self.pucks if abs(s.pos[0]) < half and 1e-4 < depth(s.pos) < 0.045]
            pts = [np.array([-half, y_line, z])]
            if pressed:
                p = max(pressed, key=lambda s: depth(s.pos)).pos
                pts.append(np.array([p[0], p[1] - sign * r, z]))
            pts.append(np.array([half, y_line, z]))
            for a, b in zip(pts, pts[1:]):
                _connector(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, a, b, BAND_WIDTH_M, BAND_RGBA)
        if not paths:
            return
        for path in self.predictions.values():
            for x, y in path[::2]:
                _geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0015, 0, 0], [x, y, 0.002], PATH_RGBA)
        for x in self.arrivals.values():
            _geom(scene, mujoco.mjtGeom.mjGEOM_CYLINDER, [r, 0.0005, 0], [x, 0.0, 0.016], ARRIVAL_RGBA)


def _geom(scene, gtype, size, pos, rgba):
    if scene.ngeom >= scene.maxgeom:
        return None
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.array(size, float), np.array(pos, float), np.eye(3).flatten(), rgba)
    scene.ngeom += 1
    return g


def _connector(scene, gtype, a, b, width, rgba):
    g = _geom(scene, gtype, [0, 0, 0], [0, 0, 0], rgba)
    if g is not None:
        mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))


def project(model, data, camera: str, pts3d, width: int, height: int) -> np.ndarray:
    """Pixel coordinates of world points in a MuJoCo camera image (pinhole, as the renderer draws it)."""
    cam = model.camera(camera).id
    pos, R = data.cam_xpos[cam], data.cam_xmat[cam].reshape(3, 3)
    f = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)
    pc = (np.asarray(pts3d, float) - pos) @ R  # Camera frame: x right, y up, looking along -z
    u = 0.5 * width + f * pc[:, 0] / -pc[:, 2]
    v = 0.5 * height - f * pc[:, 1] / -pc[:, 2]
    return np.c_[u, v]
