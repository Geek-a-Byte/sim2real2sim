import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.slingpuck.general.curriculum import Curriculum, task_vector  # noqa: E402
from src.slingpuck.general.failures import (TrialRecord, load_jsonl, rebuild_task, save_jsonl,  # noqa: E402
                                            summarize, tag_unreachable)
from src.slingpuck.general.goal_env import TaskSpec  # noqa: E402
from src.slingpuck.general.reachability import ReachabilityChecker  # noqa: E402


def failure(stage="grasp", goal=(0.30, -0.10, 0.08, 0.0), obj="cube", pose=(0.28, -0.12, 0.015, 0.4)):
    task = TaskSpec(obj, (0.25, -0.1), 0.2, [goal]).to_dict()
    return TrialRecord("real-1", "real", task, 0, list(goal), stage, list(pose), stage == "success")


# ------------------------------------------------------------------ trial records
def test_records_round_trip(tmp_path):
    recs = [failure(), failure("success")]
    save_jsonl(recs, tmp_path / "t.jsonl", append=False)
    back = load_jsonl(tmp_path / "t.jsonl")
    assert [r.stage for r in back] == ["grasp", "success"] and back[0].task == recs[0].task
    assert summarize(back)["success_rate"] == 0.5


def test_bad_records_are_rejected():
    with pytest.raises(ValueError):
        failure(stage="exploded")
    with pytest.raises(ValueError):
        TrialRecord("x", "lab", {}, 0, [], "grasp", [], False)


def test_unreachable_goal_is_not_a_policy_failure(gcfg, shapes):
    ch = ReachabilityChecker(gcfg)
    rec = tag_unreachable(failure(goal=(0.7, 0.0, 0.015, 0.0)), ch, shapes)
    assert rec.stage == "unreachable" and not rec.is_policy_failure
    assert tag_unreachable(failure(), ch, shapes).is_policy_failure


def test_rebuilt_task_starts_where_the_object_lay(gcfg, shapes):
    t = rebuild_task(failure(), gcfg["robot"]["table_z_m"], shapes["cube"].half_height)
    assert t.start_xy == (0.28, -0.12) and t.start_yaw == 0.4 and t.goals[0] == (0.30, -0.10, 0.08, 0.0)
    lifted = rebuild_task(failure(pose=(0.3, 0.0, 0.07, 0.0)), 0.0, shapes["cube"].half_height)
    assert lifted.start_xy == (0.25, -0.1)          # Not on the table: start from the original pose


# ---------------------------------------------------------------------- curriculum
@pytest.fixture(scope="module")
def fast_checker(gcfg, reach_map):
    return ReachabilityChecker(gcfg, reach_map)


def test_candidates_are_reachable(gcfg, shapes, fast_checker):
    cur = Curriculum(gcfg, fast_checker, shapes, "uniform", seed=0)
    pool = cur.sample_pool(40)
    assert len(pool) == 40 and all(cur.reachable(t) for t in pool)


def test_failure_curriculum_oversamples_tasks_near_real_failures(gcfg, shapes, fast_checker):
    f = failure()
    near = lambda pool, cur: np.mean([np.linalg.norm(cur.vec(t) - cur.failure_vecs[0]) < 1.0 for t in pool])  # noqa
    fail = Curriculum(gcfg, fast_checker, shapes, "failure", [f], seed=0)
    atr = Curriculum(gcfg, fast_checker, shapes, "atr", [f], seed=0)
    assert near(fail.sample_pool(200), fail) > 2 * near(atr.sample_pool(200), fail)


def test_successes_and_unreachable_goals_do_not_steer_the_curriculum(gcfg, shapes, fast_checker):
    cur = Curriculum(gcfg, fast_checker, shapes, "failure", [failure("success"), failure("unreachable")], seed=0)
    assert len(cur.failure_tasks) == 0


def test_feasibility_follows_history_and_diversity_prefers_rare_tasks(gcfg, shapes, fast_checker):
    cur = Curriculum(gcfg, fast_checker, shapes, "atr", seed=0)
    easy, hard = cur.random_task(), cur.random_task()
    cur.update([(easy, True)] * 10 + [(hard, False)] * 10)
    V = np.array([cur.vec(easy), cur.vec(hard)])
    f = cur.feasibility(V)
    assert f[0] > f[1]
    far = TaskSpec(easy.obj, (easy.start_xy[0] + 0.3, easy.start_xy[1]), easy.start_yaw, easy.goals)
    assert cur.diversity(np.array([cur.vec(far)]))[0] > cur.diversity(V[:1])[0]


def test_uniform_fraction_keeps_every_candidate_possible(gcfg, shapes, fast_checker):
    cur = Curriculum(gcfg, fast_checker, shapes, "failure", [failure()], seed=0)
    V = np.zeros((4, len(task_vector(cur.random_task(), cur.names))))
    assert np.all(np.isfinite(cur.score(V)))
    assert gcfg["curriculum"]["uniform_frac"] > 0
