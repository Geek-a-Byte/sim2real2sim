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
