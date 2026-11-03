import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from src.slingpuck.general.goal_env import GoalEnv, TaskSpec  # noqa: E402
from src.slingpuck.general.reachability import SEED_POSES, approach_dir  # noqa: E402


@pytest.fixture(scope="module")
def env(gcfg, reach_map):
    return GoalEnv(gcfg, reach_map=reach_map)


def test_check_env(gcfg, reach_map):
    check_env(GoalEnv(gcfg, reach_map=reach_map), skip_render_check=True)


def test_paired_points_pair_each_point_with_its_target(env):
    obs, info = env.reset(seed=0, options={"task": TaskSpec("bar", (0.25, 0.0), 0.0, [(0.30, 0.05, 0.012, 0.5)])})
    pts = obs["points"]
    assert pts.shape == (env.e["n_points"], 7) and np.all(pts[:, 6] == 1)
    d = np.linalg.norm(pts[:, :3] - pts[:, 3:6], axis=1)
    assert d.mean() == pytest.approx(info["d_goal"], rel=1e-5)
    # The same local point: current and target differ by the goal's rigid transform
    c, s = np.cos(0.5), np.sin(0.5)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    np.testing.assert_allclose(pts[:, 3:6], (pts[:, :3] - [0.25, 0, 0.012]) @ R.T + [0.30, 0.05, 0.012], atol=1e-5)


def test_goal_at_the_start_pose_counts_as_reached_and_chains_a_new_goal(env):
    env.reset(seed=0, options={"task": TaskSpec("cube", (0.25, 0.0), 0.0, [(0.25, 0.0, 0.015, 0.0)])})
    _, r, _, _, info = env.step(np.r_[np.zeros(5), 0.0])
    assert info["goals_reached"] == 1 and r > env.w["success"] - 1
    g = np.array(info["goal"])
    assert np.linalg.norm(g[:2] - [0.25, 0.0]) <= env.e["chain_radius_m"] + 1e-9


def test_chained_goals_pass_the_reachable_goal_filter(env):
    env.reset(seed=3)
    shape = env.shape
    for k in range(10):
        g = env.sample_goal_near(*env.object_pose())
        assert env.checker.check(shape, g[:3], g[3], exact=False).ok


def test_random_tasks_are_seeded(gcfg, reach_map):
    a, b = GoalEnv(gcfg, reach_map=reach_map), GoalEnv(gcfg, reach_map=reach_map)
    ia, ib = a.reset(seed=7)[1], b.reset(seed=7)[1]
    assert ia["task"] == ib["task"] and ia["goal"] == ib["goal"]


def test_task_pool_is_used(env):
    pool = [TaskSpec("puck", (0.2, 0.1), 0.0, [(0.22, 0.1, 0.0051, 0.0)])]
    env.set_task_pool(pool)
    _, info = env.reset(seed=1)
    env.set_task_pool(None)
    assert info["task"]["obj"] == "puck" and info["goal"] == list(pool[0].goals[0])


def test_student_points_are_masked_and_noisy(gcfg, reach_map):
    st = GoalEnv(gcfg, student=True, reach_map=reach_map)
    seen = []
    for seed in range(10):
        obs, _ = st.reset(seed=seed)
        v = obs["points"][:, 6]
        assert np.all(obs["points"][v == 0, :3] == 0)       # Masked points carry no position
        seen.append(v.mean())
    assert 0.2 < np.mean(seen) < 0.9


def test_scripted_grasp_lifts_the_cube(gcfg, reach_map):
    """Physics check: the gripper can hold and lift an object (reach, grasp and lift reward terms fire)."""
    env = GoalEnv(gcfg, reach_map=reach_map)
    env.reset(seed=0, options={"task": TaskSpec("cube", (0.25, 0.0), 0.0, [(0.25, 0.0, 0.08, 0.0)])})
    k = env.checker.kin

    def go(p, grip, n):
        for q0 in (env.d.qpos[env.qadr[:5]].copy(), *SEED_POSES):
            ok, q, *_ = k.ik(p, approach_dir(p, 0), np.array([0, 1.0, 0]), q0=q0)
            if ok:
                break
        for _ in range(n):
            env.d.ctrl[:5], env.d.ctrl[5] = q, grip
            mujoco.mj_step(env.m, env.d, nstep=env.n_sub)

    go((0.25, 0, 0.06), k.gripper_for_opening(0.05), 30)
    go((0.25, 0, 0.015), k.gripper_for_opening(0.05), 30)
    go((0.25, 0, 0.015), k.gripper_for_opening(0.02), 20)
    go((0.25, 0, 0.08), k.gripper_for_opening(0.02), 40)
    _, _, _, _, info = env.step(np.r_[np.zeros(5), -0.8])
    assert env.object_pose()[0][2] > 0.06 and all(env._jaw_contacts())


def test_object_out_of_the_workspace_terminates(env):
    env.reset(seed=0, options={"task": TaskSpec("cube", (0.25, 0.0), 0.0, [(0.3, 0.0, 0.015, 0.0)])})
    env._set_object("cube", (0.25, 0.6, 0.015), 0.0)      # Pushed far to the side
    _, _, terminated, _, info = env.step(np.zeros(6))
    assert terminated and info["failure"] == "object_lost"


# ------------------------------------------------------------------ regression tests (codebase review)
def test_no_goal_is_ever_the_current_pose(gcfg, reach_map):
    """A goal at the current pose counts as reached on every step: a reward exploit."""
    env = GoalEnv(gcfg, reach_map=reach_map)
    env.reset(seed=1, options={"task": TaskSpec("cube", (0.25, 0.0), 0.0)})
    rng = np.random.default_rng(0)
    for p in rng.uniform([-0.02, -0.42], [0.54, 0.42], (60, 2)):
        env._set_object("cube", (p[0], p[1], 0.015), 0.0)
        mujoco.mj_forward(env.m, env.d)
        g = env.sample_goal_near(*env.object_pose())
        assert g is None or not np.allclose(g[:2], p)


def test_object_out_of_reach_ends_the_episode(gcfg, reach_map):
    env = GoalEnv(gcfg, reach_map=reach_map)
    env.reset(seed=1, options={"task": TaskSpec("cube", (0.25, 0.0), 0.0)})
    env._set_object("cube", (0.5, 0.38, 0.015), 0.0)     # Inside the workspace margin, out of reach
    _, r, terminated, _, info = env.step(np.zeros(6))
    assert terminated and info["failure"] == "object_lost" and r < 1.0


def test_no_reachable_next_goal_truncates_without_reward(gcfg, reach_map):
    env = GoalEnv(gcfg, reach_map=reach_map)
    env.reset(seed=0, options={"task": TaskSpec("cube", (0.25, 0.0), 0.0, [(0.25, 0.0, 0.015, 0.0)])})
    env.sample_goal_near = lambda pos, yaw: None             # As if no reachable goal existed
    _, r1, _, truncated, info = env.step(np.zeros(6))        # Reaches the given goal; no next goal
    assert info["goals_reached"] == 1 and truncated
    _, r2, *_ = env.step(np.zeros(6))
    assert r2 < env.w["success"] and env.goals_reached == 1     # No success bonus again


@pytest.mark.parametrize("obj, turn, same", [("cube", np.pi / 2, True), ("bar", np.pi, True),
                                             ("bar", np.pi / 2, False), ("tube", 1.3, True)])
def test_symmetric_objects_reach_turned_goals(gcfg, reach_map, obj, turn, same):
    env = GoalEnv(gcfg, reach_map=reach_map)
    z = env.shapes[obj].half_height
    _, info = env.reset(seed=0, options={"task": TaskSpec(obj, (0.25, 0.0), 0.2, [(0.25, 0.0, z, 0.2 + turn)])})
    assert (info["d_goal"] < 1e-6) == same


def test_jaw_contacts_use_only_jaw_geoms(env):
    m = env.m
    for geoms in env.jaw_geoms:
        assert geoms and all(m.geom_priority[g] == 1 for g in geoms)
    wrist_box = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.geom_bodyid[m.geom("fixed_jaw_box1").id]
                 and m.geom_priority[g] == 0 and m.geom_contype[g]]
    assert wrist_box and not set(wrist_box) & env.jaw_geoms[0]


def test_lift_reward_needs_a_grasp(gcfg, reach_map):
    env = GoalEnv(gcfg, reach_map=reach_map)
    env.reset(seed=0, options={"task": TaskSpec("cube", (0.25, 0.1), 0.0, [(0.25, 0.1, 0.08, 0.0)])})
    env._set_object("cube", (0.25, 0.1, 0.06), 0.0)          # In the air, not held
    mujoco.mj_forward(env.m, env.d)
    w = dict(env.w)
    env.w = {**w, "lift": 1000.0}
    _, r, *_ = env.step(np.zeros(6))
    env.w = w
    assert r < 100.0


# ------------------------------------------------------------- end-effector actions (training run 2)
@pytest.mark.parametrize("u, axis", [((1, 0, 0), 0), ((-1, 0, 0), 0), ((0, 1, 0), 1), ((0, -1, 0), 1)])
def test_ee_actions_move_the_gripper_as_commanded(gcfg, reach_map, u, axis):
    env = GoalEnv(gcfg, reach_map=reach_map)
    env.reset(seed=0, options={"task": TaskSpec("cube", (0.3, 0.15), 0.0)})
    for _ in range(8):           # Out from the home pose first: near the base at 12 cm, -x is out of reach
        env.step(np.r_[1, 0, 0, 0, 0, 0])
    p0 = env.d.site_xpos[env.site].copy()
    for _ in range(5):
        env.step(np.r_[u, 0.0, 0.0, 0.0])
    dp = env.d.site_xpos[env.site] - p0
    assert dp[axis] == pytest.approx(5 * gcfg["robot"]["ee_pos_step_m"] * u[axis], abs=0.006)
    assert np.linalg.norm(np.delete(dp, axis)) < 0.008


def test_ee_hold_does_not_drift(env):
    env.reset(seed=0, options={"task": TaskSpec("cube", (0.3, 0.15), 0.0)})
    p0 = env.d.site_xpos[env.site].copy()
    for _ in range(40):
        env.step(np.zeros(6))
    assert np.linalg.norm(env.d.site_xpos[env.site] - p0) < 0.001


def test_pushing_into_the_table_has_no_windup_and_respects_the_floor(gcfg, reach_map):
    env = GoalEnv(gcfg, reach_map=reach_map)
    env.reset(seed=0, options={"task": TaskSpec("cube", (0.3, 0.15), 0.0)})
    for _ in range(3):
        env.step(np.r_[0, 0, 0, 1, 0, 0])                 # Tilt to top-down
    for _ in range(60):
        env.step(np.r_[0, 0, -1, 0, 0, 0])
    z = env.d.site_xpos[env.site][2]
    assert gcfg["robot"]["table_z_m"] < z < 0.012           # Low enough for a 1 cm puck
    lead = np.abs(env.d.ctrl[:5] - env.d.qpos[env.qadr[:5]]).max()
    assert lead < 0.35                                      # Targets stay near the joints (windup was 2.2 rad)


def test_home_gripper_is_above_every_object(env, shapes):
    env.reset(seed=0)
    tips = [env.m.geom(n).id for n in ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1")]
    lowest = min(env.d.geom_xpos[g][2] for g in tips)
    assert lowest > max(2 * s.half_height for s in shapes.values()) + 0.02


def test_is_success_at_the_episode_end_reports_reached_goals(gcfg, reach_map):
    """SB3's rollout/success_rate reads is_success from the last step only."""
    cfg = {**gcfg, "env": {**gcfg["env"], "max_steps": 3}}
    env = GoalEnv(cfg, reach_map=reach_map)
    env.reset(seed=0, options={"task": TaskSpec("cube", (0.25, 0.0), 0.0, [(0.25, 0.0, 0.015, 0.0),
                                                                            (0.35, 0.1, 0.015, 0.0)])})
    infos = [env.step(np.zeros(6))[4] for _ in range(3)]
    assert infos[0]["goal_reached_now"] and not infos[1]["goal_reached_now"]
    assert infos[-1]["is_success"]
    env.reset(seed=0, options={"task": TaskSpec("cube", (0.25, 0.0), 0.0, [(0.35, 0.1, 0.015, 0.0)])})
    infos = [env.step(np.zeros(6))[4] for _ in range(3)]
    assert not infos[-1]["is_success"]


# --------------------------------------------------- reward stages and grasped starts (training run 3)
def test_home_pose_follows_the_tallest_training_object(gcfg, reach_map):
    puck = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"])
    allobj = GoalEnv(gcfg, reach_map=reach_map)
    assert puck.home_tilt == 0.0 and puck.home_point[2] < allobj.home_point[2]   # Low and top-down for a puck
    assert allobj.home_point[2] > 2 * allobj.shapes["tube"].half_height


def test_grasped_starts_hold_the_object_through_a_lift(gcfg, reach_map):
    env = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"], grasped_start_prob=1.0)
    grasped = lifted = 0
    for seed in range(10):
        _, info = env.reset(seed=seed)
        if not info["grasped_start"]:
            continue
        grasped += 1
        for _ in range(5):
            env.step(np.r_[0, 0, 0, 0, 0, -1.0])
        for _ in range(10):
            env.step(np.r_[0, 0, 1, 0, 0, -1.0])
        lifted += env.object_pose()[0][2] > env.shape.half_height + 0.04
    assert grasped >= 8 and lifted >= grasped - 2       # Elliptic cones (so101.xml) held 2 of 18


def test_a_grasp_pays_more_than_hovering(gcfg, reach_map):
    task = TaskSpec("puck", (0.25, 0.0), 0.0, [(0.25, 0.0, 0.06, 0.0)])
    held = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"], grasped_start_prob=1.0)
    _, info = held.reset(seed=0, options={"task": task})
    assert info["grasped_start"]
    r_held = np.mean([held.step(np.r_[0, 0, 0, 0, 0, -1.0])[1] for _ in range(5)])
    hover = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"], grasped_start_prob=0.0)
    hover.reset(seed=0, options={"task": task})
    for _ in range(15):                                  # Down to 2 cm above the puck, jaws open
        e = np.array([0.25, 0.0, 0.02]) - hover.d.site_xpos[hover.site]
        r_hover = hover.step(np.r_[np.clip(e / 0.01, -1, 1), 0, 0, 0.3])[1]
    assert r_held > r_hover + 0.1                       # Hovering pays about 0; a held object pays the grasp term


def test_grasped_starts_are_left_out_of_the_success_rate(gcfg, reach_map):
    cfg = {**gcfg, "env": {**gcfg["env"], "max_steps": 2}}
    env = GoalEnv(cfg, reach_map=reach_map, objects=["puck"], grasped_start_prob=1.0)
    _, info = env.reset(seed=0)
    assert info["grasped_start"]
    info = [env.step(np.zeros(6))[4] for _ in range(2)][-1]
    assert info["is_success"] is None                    # SB3 skips None in rollout/success_rate


def test_ik_does_not_depend_on_earlier_calls(gcfg):
    from src.slingpuck.general.reachability import SO101Kinematics
    kin = SO101Kinematics()
    target = ((0.25, 0.05, 0.03), approach_dir((0.25, 0.05, 0.03), 0.0))
    q1 = kin.ik(*target, None)[1]
    kin.jaw_gap(1.5)                                      # Leaves the gripper joint at 1.5 in kin.d
    q2 = kin.ik(*target, None)[1]
    assert np.array_equal(q1, q2)


# ------------------------------------------------------------------- binary gripper (after run 3)
def test_binary_gripper_closes_on_negative_and_opens_on_positive(env):
    env.reset(seed=0, options={"task": TaskSpec("cube", (0.3, 0.15), 0.0)})
    env.step(np.r_[np.zeros(5), -0.05])
    assert env.d.ctrl[5] == pytest.approx(env.ctrl_lo[5])                 # Any negative value: fully closed
    env.step(np.r_[np.zeros(5), 0.05])
    gap = env.checker.kin.jaw_gap(env.d.ctrl[5])
    assert gap == pytest.approx(env.cfg["robot"]["gripper_open_m"], abs=1e-3)


def test_puck_can_be_grasped_and_lifted_with_env_actions_only(gcfg, reach_map):
    """The task is possible in the policy's own action space (a simple scripted controller)."""
    env = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"], grasped_start_prob=0.0)
    d = env.d

    def to(target, grip, n=80):
        for _ in range(n):
            e = np.asarray(target) - d.site_xpos[env.site]
            R = d.site_xmat[env.site].reshape(3, 3)
            if np.linalg.norm(e) < 0.002 and abs(R[0, 0]) < 0.05:
                break
            env.step(np.r_[np.clip(e / 0.01, -1, 1), np.clip(5 * R[0, 0], -1, 1), 0, grip])

    lifted = 0
    for xy in ((0.25, 0.0), (0.22, 0.08), (0.30, -0.06), (0.20, -0.10), (0.28, 0.10), (0.18, 0.0)):
        env.reset(seed=0, options={"task": TaskSpec("puck", xy, 0.0, [(xy[0], xy[1], 0.08, 0.0)])})
        to((xy[0], xy[1], 0.04), 1.0)
        to((xy[0], xy[1], 0.006), 1.0)
        for _ in range(6):
            env.step(np.r_[0, 0, 0, 0, 0, -1.0])
        to((xy[0], xy[1], 0.08), -1.0)
        lifted += env.object_pose()[0][2] > env.shape.half_height + 0.04
    assert lifted >= 3


# ------------------------------------------------- pregrasp starts and height-based lift (after run 4)
def test_pregrasp_starts_leave_open_jaws_around_the_object(gcfg, reach_map):
    env = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"], pregrasp_start_prob=1.0)
    n = closed_grasp = 0
    for seed in range(10):
        _, info = env.reset(seed=seed)
        if info["start"] != "pregrasp":
            continue
        n += 1
        assert not any(env._jaw_contacts())                         # Open: the policy must close
        p, _ = env.object_pose()
        assert np.linalg.norm(env.d.site_xpos[env.site][:2] - p[:2]) < 0.01
        for _ in range(6):
            env.step(np.r_[0, 0, 0, 0, 0, -1.0])
        closed_grasp += all(env._jaw_contacts())
    assert n >= 7 and closed_grasp >= n - 2                         # Closing there gives a grasp


# ------------------------------------------------------------------- progress reward (after run 5)
def _only_progress(env):
    w = env.w
    env.w = {**w, "reach": 0.0, "grasp": 0.0, "time": 0.0, "action_rate": 0.0, "joint_vel": 0.0}
    return w


def test_progress_pays_nothing_for_holding_still(gcfg, reach_map):
    env = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"], grasped_start_prob=1.0)
    w = _only_progress(env)
    _, info = env.reset(seed=1, options={"task": TaskSpec("puck", (0.25, 0.0), 0.0, [(0.30, 0.05, 0.08, 0.0)])})
    assert info["start"] == "grasped"
    r = [env.step(np.r_[0, 0, 0, 0, 0, -1.0])[1] for _ in range(10)]
    env.w = w
    assert abs(sum(r[2:])) < 0.5                                    # Held still: no progress


def test_progress_pays_for_a_lift_toward_a_lifted_goal(gcfg, reach_map):
    env = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"], grasped_start_prob=1.0)
    w = _only_progress(env)
    env.reset(seed=1, options={"task": TaskSpec("puck", (0.25, 0.0), 0.0, [(0.25, 0.0, 0.10, 0.0)])})
    d0 = env._info()["d_goal"]
    r = [env.step(np.r_[0, 0, 1, 0, 0, -1.0])[1] for _ in range(4)]
    d1 = env._info()["d_goal"]
    env.w = w
    assert all(x > 0 for x in r[1:])                                # Every step up pays
    assert sum(r) == pytest.approx(w["progress"] * (d0 - d1) / 0.01, rel=1e-6)   # Potential-based: sums exactly


def test_reaching_a_goal_does_not_lower_the_reward(gcfg, reach_map):
    env = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"])
    w = _only_progress(env)
    env.reset(seed=0, options={"task": TaskSpec("puck", (0.25, 0.0), 0.0, [(0.25, 0.0, 0.0051, 0.0),
                                                                            (0.35, 0.1, 0.0051, 0.0)])})
    r_goal = env.step(np.zeros(6))[1]                               # First goal reached on this step
    r_next = env.step(np.zeros(6))[1]                               # The far next goal costs no reward
    env.w = w
    assert r_goal == pytest.approx(w["success"], abs=0.1) and abs(r_next) < 0.1   # Settling: < 0.1



def test_returns_rank_behaviours_in_the_right_order(gcfg, reach_map):
    """The reward must rank idle < hover < hold still < carry to goals (runs 2-5 each broke this order)."""
    task = TaskSpec("puck", (0.25, 0.0), 0.0, [(0.28, 0.04, 0.06, 0.0)])

    def episode(start, policy, n=150):
        env = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"],
                      grasped_start_prob=1.0 if start == "grasped" else 0.0)
        env.reset(seed=1, options={"task": task})
        total = 0.0
        for _ in range(n):
            _, r, te, tr, _ = env.step(policy(env))
            total += r
            if te or tr:
                break
        return total

    def hover(env):
        e = np.array([0.25, 0.0, 0.02]) - env.d.site_xpos[env.site]
        return np.r_[np.clip(e / 0.01, -1, 1), 0, 0, 0.3]

    def carry(env):
        e = np.array(env.goal[:3]) - env.object_pose()[0]
        return np.r_[np.clip(e / 0.01, -1, 1), 0, 0, -1.0] if np.linalg.norm(e) > 0.003 else np.r_[0, 0, 0, 0, 0, -1.0]

    idle = episode("normal", lambda env: np.zeros(6))
    hov = episode("normal", hover)
    hold = episode("grasped", lambda env: np.r_[0, 0, 0, 0, 0, -1.0])
    car = episode("grasped", carry)
    assert idle < hov < hold < car
