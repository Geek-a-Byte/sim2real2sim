import copy

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.slingpuck.envs.match_env import BLOCK, SLING, WAIT, MatchEnv  # noqa: E402
from src.slingpuck.physics.shot_sim import ShotSim  # noqa: E402
from src.slingpuck.policies.match_scripted import GreedyRule, GreedySling  # noqa: E402
from src.slingpuck.primitives.sling import ShotSelector  # noqa: E402
from src.slingpuck.shot.model import ShotModel, fingerprint  # noqa: E402


@pytest.fixture(scope="module")
def sim(greedy_cfg):
    return ShotSim(greedy_cfg)


# ------------------------------------------------------------------- shot simulator
def test_straight_center_shot_goes_through(sim):
    r = sim.shoot(0.0, 0.0, 0.02)
    assert r.through and r.exit_speed > 0.5 and abs(r.divider_x) < 1e-3


def test_off_center_shot_hits_the_divider(sim):
    assert sim.shoot(0.01, 0.0, 0.02).outcome == "edge"      # Within one radius of the opening
    assert sim.shoot(0.05, 0.0, 0.02).outcome == "miss"


def test_small_pull_is_weak(sim):
    assert sim.shoot(0.0, 0.0, 0.001).outcome == "weak"
    assert sim.shoot(0.0, 0.0, 0.005).outcome == "weak"


def test_angled_pull_still_launches_almost_straight(sim):
    """Pre-stretched band: a 0.3 rad pull moves the puck only a few mm at the divider."""
    r = sim.shoot(0.0, 0.3, 0.02)
    assert r.through and abs(r.divider_x) < 0.008


def test_release_noise_is_seeded(sim):
    a = [sim.shoot(0.004, 0.0, 0.02, np.random.default_rng(5)).divider_x for _ in range(3)]
    b = [sim.shoot(0.004, 0.0, 0.02, np.random.default_rng(5)).divider_x for _ in range(3)]
    assert a == b and a[0] != sim.shoot(0.004, 0.0, 0.02).divider_x


# ---------------------------------------------------------------------- shot model
def test_shot_model_learns_where_shots_go_through(shot_model):
    p, std = shot_model.predict([[0.0, 0.0, 0.025], [0.04, 0.0, 0.025], [0.0, 0.0, 0.006]])
    assert p[0] > 0.7 and p[1] < 0.1 and p[2] < 0.1 and np.all(std >= 0)


def test_shot_model_save_load_and_fingerprint(shot_model, greedy_cfg, tmp_path):
    shot_model.save(tmp_path / "m.npz")
    loaded = ShotModel.load(tmp_path / "m.npz", greedy_cfg)
    X = [[0.0, 0.0, 0.02], [0.006, 0.1, 0.015]]
    np.testing.assert_allclose(loaded.predict(X)[0], shot_model.predict(X)[0])
    other = copy.deepcopy(greedy_cfg)
    other["board"]["gate_width_m"] = 0.036
    assert fingerprint(other) != fingerprint(greedy_cfg)
    with pytest.raises(ValueError, match="Fit it again"):
        ShotModel.load(tmp_path / "m.npz", other)


# ------------------------------------------------------------------- shot selection
def test_greedy_choice_takes_the_near_puck_and_jammed_pucks_go_last(greedy_cfg, shot_model):
    sel = ShotSelector(greedy_cfg, shot_model)
    near, far = np.array([0.0, -0.09]), np.array([0.05, -0.02])
    c = sel.choose([far, near], from_xy=(0.0, -0.1176))
    assert c.index == 1 and c.p == pytest.approx(sel.p.max(), abs=0.15)
    wall = np.array([sel.wall_x - 0.001, -0.09])
    assert sel.jammed(wall)
    assert sel.choose([wall, far], from_xy=(0.0, -0.1176)).index == 1
    assert sel.choose([wall], from_xy=(0.0, -0.1176)).index == 0      # The last puck is taken anyway


def test_shot_time_follows_the_time_model(greedy_cfg, shot_model):
    sel = ShotSelector(greedy_cfg, shot_model)
    s = greedy_cfg["shot"]
    puck, start = np.array([0.0, -0.07]), np.array([0.0, sel.band_y])
    T = sel.times(puck, start)
    k = int(np.argmin(np.abs(sel.shots[:, 0])) + 0)
    carry = np.linalg.norm(puck - start) + np.linalg.norm(sel.seats[k] - puck)
    assert T[k] == pytest.approx(s["pick_time_s"] + carry / s["arm_speed_m_s"] + s["seat_release_s"]
                                 + sel.shots[k, 2] / s["draw_speed_m_s"])


def test_random_selection_varies_the_puck(greedy_cfg, shot_model):
    cfg = {**greedy_cfg, "shot": {**greedy_cfg["shot"], "selection": "random"}}
    sel = ShotSelector(cfg, shot_model)
    pucks = [np.array([x, -0.07]) for x in (-0.06, -0.03, 0.0, 0.03, 0.06)]
    rng = np.random.default_rng(0)
    assert len({sel.choose(pucks, (0.0, sel.band_y), rng).index for _ in range(40)}) >= 4


# --------------------------------------------------------------- match env (greedy)
def test_greedy_match_conserves_pucks_and_removes_the_chosen_puck(greedy_cfg, shot_model):
    env = MatchEnv(greedy_cfg, shot_model)
    rng = np.random.default_rng(0)
    total = 2 * greedy_cfg["match"]["pucks_per_side"]
    for seed in range(5):
        env.reset(seed=seed)
        while True:
            before = [p.copy() for p in env.agent_pucks]
            chosen = None if env.pending is None else env.agent_pucks[env.pending.index].copy()
            _, _, te, tr, _ = env.step(int(rng.integers(3)))
            assert env.agent_side + env.opp_side + len(env.incoming) + (env.own is not None) == total
            if chosen is not None and env.pending is None and env.arm == "at_band" and len(env.agent_pucks) < len(before):
                assert not any(np.array_equal(chosen, p) for p in env.agent_pucks)
            if te or tr:
                break


def test_best_shot_observation(greedy_cfg, shot_model):
    env = MatchEnv(greedy_cfg, shot_model)
    obs, _ = env.reset(seed=0)
    n = MatchEnv.POLICY_OBS_NAMES.index
    p, t = env.best_shot()
    assert obs[n("shot_p")] == pytest.approx(p, abs=1e-6) and 0.5 < p <= 1.0
    assert t > greedy_cfg["match"]["move_gate_to_band_s"]          # From the gate: leave it first
    env.step(SLING)
    while env.arm != "at_band":
        env.step(WAIT)
    assert env.best_shot()[1] < t                                  # Already at the band: no move out


def test_greedy_shot_choice_beats_random_puck_order(greedy_cfg, shot_model):
    def net(selection):
        cfg = {**greedy_cfg, "shot": {**greedy_cfg["shot"], "selection": selection}}
        env, pol, out = MatchEnv(cfg, shot_model), GreedySling(), []
        for seed in range(40):
            obs, _ = env.reset(seed=seed)
            while True:
                obs, _, te, tr, info = env.step(pol.predict(obs)[0])
                if te or tr:
                    break
            out.append(info["stats"]["sent"] - info["stats"]["received"])
        return np.mean(out)
    assert net("greedy") > net("random")


# ---------------------------------------------------------------------- greedy rule
def test_greedy_rule_decisions(greedy_cfg):
    rule = GreedyRule(greedy_cfg)
    n = MatchEnv.POLICY_OBS_NAMES.index
    hl = rule.half_length
    obs = np.zeros(len(MatchEnv.POLICY_OBS_NAMES), np.float32)
    obs[n("arm_gate")], obs[n("gate_covered")], obs[n("agent_pucks")] = 1.0, 1.0, 1.0
    obs[n("hand_y")] = (rule.band_y - 0.05) / hl                   # Opponent reloading (hand away from its band)
    obs[n("shot_p")], obs[n("shot_time")] = 0.9, 1.0 / 3.0         # A good shot that takes 1 s
    assert rule.act(obs, None) == SLING
    obs[n("puck_valid")], obs[n("puck_x")], obs[n("puck_y")], obs[n("puck_vy")] = 1.0, 0.0, 0.05 / hl, -1.0
    obs[n("shot_p")] = 0.5                                         # A threat arrives in 50 ms; the shot is poor
    assert rule.act(obs, None) == BLOCK
    obs[n("agent_pucks")] = 0.0
    assert rule.act(obs, None) == BLOCK
