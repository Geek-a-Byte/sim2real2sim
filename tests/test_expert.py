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


def test_bc_learns_the_expert_and_warmup_keeps_the_actor(gcfg, reach_map):
    """A short BC run lowers the copy error; a critic warm-up with the actor frozen leaves the actions unchanged."""
    import torch as th
    from stable_baselines3 import PPO

    from src.slingpuck.general.bc import train_bc
    from src.slingpuck.general.policy import PairedPointPolicy
    from src.slingpuck.general.train_teacher import set_actor_trainable

    env = GoalEnv(gcfg, reach_map=reach_map, objects=["puck"])
    stats, data = run_expert(env, 4, seed=0, max_steps=60, record=True)
    model = PPO(PairedPointPolicy, env, n_steps=64, batch_size=64, n_epochs=2, seed=0,
                policy_kwargs={"net_arch": [64, 64], "log_std_init": -1.0})
    losses = train_bc(model, data, epochs=15, batch=128, log=lambda *a: None)
    assert losses[-1] < 0.6 * losses[0]
    obs, _ = env.reset(seed=3)
    before = model.predict(obs, deterministic=True)[0]
    v_before = model.policy.predict_values({k: th.as_tensor(v[None]) for k, v in obs.items()}).item()
    set_actor_trainable(model.policy, False)
    model.learn(128)
    set_actor_trainable(model.policy, True)
    np.testing.assert_allclose(model.predict(obs, deterministic=True)[0], before, atol=1e-6)
    v_after = model.policy.predict_values({k: th.as_tensor(v[None]) for k, v in obs.items()}).item()
    assert v_after != v_before                                       # The critic did train
