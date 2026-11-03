import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.slingpuck.general.expert import ScriptedExpert, run_expert  # noqa: E402
from src.slingpuck.general.goal_env import GoalEnv, TaskSpec  # noqa: E402


@pytest.mark.parametrize("obj, min_ok", [("puck", 3), ("cube", 6)])
def test_expert_reaches_goals(gcfg, reach_map, obj, min_ok):
    """With the coarse 8 cm test map; with the real 2 cm map the expert reaches a goal in 95% of puck episodes."""
    env = GoalEnv(gcfg, reach_map=reach_map, objects=[obj])
    stats = run_expert(env, 10, seed=100, max_steps=150)
    assert sum(s["goals"] >= 1 for s in stats) >= min_ok


def test_expert_acts_through_the_policy_action_space(gcfg, reach_map):
    env = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"])
    env.reset(seed=0)
    ex = ScriptedExpert(env)
    for _ in range(30):
        a = ex.act()
        assert a.shape == (6,) and np.all(np.abs(a) <= 1.0)
        env.step(a)


def test_expert_starts_in_the_right_phase(gcfg, reach_map):
    for kw, phase in ((dict(grasped_start_prob=1.0), "carry"), (dict(pregrasp_start_prob=1.0), "close"),
                      (dict(), "approach")):
        env = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"], **kw)
        env.reset(seed=1, options={"task": TaskSpec("puck", (0.22, 0.0), 0.0)})
        assert ScriptedExpert(env).phase == phase


def test_recorded_demos_pair_observations_with_clean_actions(gcfg, reach_map):
    env = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"])
    stats, data = run_expert(env, 2, seed=0, noise=0.3, max_steps=40, record=True)
    n = sum(s["steps"] for s in stats)
    assert len(data["action"]) == len(data["points"]) == len(data["proprio"]) == n
    assert data["points"].shape[1:] == env.observation_space["points"].shape
    assert np.all(np.isin(np.sign(data["action"][:, 5]), (-1, 1)))     # Clean gripper commands: open or close
