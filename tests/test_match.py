import copy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.slingpuck.envs.match_env import BLOCK, HOLD, SLING, MatchEnv
from src.slingpuck.envs.wrappers import PrivilegedObsWrapper
from src.slingpuck.opponent.scripted_opponent import THREAT_HALF_WIDTH_M, Opponent
from src.slingpuck.policies.match_scripted import AlwaysBlock, GreedySling, ReloadRule
from src.slingpuck.primitives.block import BlockTable

# Small hand-made table: a settled paddle saves everything; from the edge only slow, early shots.
TABLE = BlockTable(np.array([0.8, 3.0]), np.array([0.0, 0.3]), np.array([1.0, 1.0]),
                   np.array([[1.0, 0.0], [0.0, 0.0]]))


def env_for(cfg, **match):
    c = copy.deepcopy(cfg)
    c["match"].update(match)
    return MatchEnv(c, TABLE)


def play(env, policy, seed):
    obs, info = env.reset(seed=seed)
    total = 0.0
    while True:
        obs, r, terminated, truncated, info = env.step(policy.predict(obs)[0])
        total += r
        if terminated or truncated:
            return info, total


# ----------------------------------------------------------------- opponent
def cycles(cfg, tell, n=300, seed=0):
    c = copy.deepcopy(cfg)
    c["opponent"]["tell_strength"] = tell
    opp = Opponent(c, np.random.default_rng(seed))
    opp.reset(0.0, True)
    out = []
    t = 0.0
    for _ in range(n):
        shot = opp.next_shot
        t_aim = 0.5 * (opp.cycle.times[3] + opp.cycle.times[4])  # Middle of the hold
        out.append((opp.hand_true(t_aim)[0], shot.x_launch, shot.threat))
        t = shot.t_launch + 1e-6
        opp.advance(t, True)
    return np.array(out)


def test_tell_one_shows_launch_x_and_zero_hides_it(cfg):
    full = cycles(cfg, 1.0)
    np.testing.assert_allclose(full[:, 0], full[:, 1], atol=1e-12)
    none = cycles(cfg, 0.0)
    assert abs(np.corrcoef(none[:, 0], none[:, 1])[0, 1]) < 0.15


def test_threat_shots_aim_inside_the_gate(cfg):
    c = cycles(cfg, 1.0)
    threats = c[c[:, 2] == 1]
    assert np.all(np.abs(threats[:, 1]) <= THREAT_HALF_WIDTH_M)
    assert np.all(np.abs(c[c[:, 2] == 0][:, 1]) > THREAT_HALF_WIDTH_M)
    assert abs(len(threats) / len(c) - cfg["opponent"]["threat_prob"]) < 0.08


def test_opponent_phases_in_order_and_hand_continuous(cfg):
    opp = Opponent(cfg, np.random.default_rng(1))
    opp.reset(0.0, True)
    ts = np.linspace(0, opp.next_shot.t_launch - 1e-6, 400)
    phases = [opp.phase(t) for t in ts]
    order = [p for k, p in enumerate(phases) if k == 0 or p != phases[k - 1]]
    assert order == ["reload", "place", "pull", "hold"]
    pos = np.array([opp.hand_true(t)[:2] for t in ts])
    assert np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1)) < 0.01  # No jumps


# --------------------------------------------------------------- match env
def test_check_env(cfg):
    check_env(MatchEnv(cfg, TABLE), skip_render_check=True)
    check_env(PrivilegedObsWrapper(MatchEnv(cfg, TABLE)), skip_render_check=True)


def test_pucks_are_conserved_and_rewards_add_up(cfg):
    env = env_for(cfg)
    rng = np.random.default_rng(0)
    total_pucks = 2 * cfg["match"]["pucks_per_side"]
    for seed in range(20):
        env.reset(seed=seed)
        total = 0.0
        while True:
            _, r, te, tr, info = env.step(int(rng.integers(3)))
            total += r
            assert env.agent_side + env.opp_side + len(env.incoming) + (env.own is not None) == total_pucks
            if te or tr:
                break
        s, sc = info["stats"], cfg["scoring"]
        expected = s["sent"] * sc["reward_puck_sent"] + s["received"] * sc["reward_puck_received"]
        expected += {"win": sc["reward_win"], "loss": sc["reward_loss"]}.get(info["outcome"], 0.0)
        assert total == pytest.approx(expected)


def test_settled_blocker_saves_every_threat(cfg):
    info, total = play(env_for(cfg), AlwaysBlock(), seed=3)
    s = info["stats"]
    assert s["threats"] > 0 and s["saves"] == s["threats"] and s["received"] == 0
    assert info["outcome"] == "time" and total == 0.0


def test_greedy_slinger_concedes_every_threat_while_away(cfg):
    info, _ = play(env_for(cfg), GreedySling(), seed=3)
    s = info["stats"]
    assert s["saves"] == 0 and s["conceded_while_away"] == s["received"] == s["threats"]


def test_sling_timing(cfg):
    env = env_for(cfg)
    m = cfg["match"]
    env.reset(seed=0)
    env.step(SLING)
    assert env.arm == "to_band" and env.gate_ready_since is None
    t_release = m["move_gate_to_band_s"] + m["fetch_time_s"] + m["sling_core_s"]
    while env.arm != "at_band":
        env.step(HOLD)
    assert env.t == pytest.approx(t_release, abs=m["decision_dt_s"] + 1e-9)
    assert env.own is not None and env.agent_side == m["pucks_per_side"] - 1
    env.step(BLOCK)
    assert env.arm == "to_gate"
    while env.arm != "gate":
        env.step(HOLD)
    assert env.gate_ready_since == pytest.approx(env.t, abs=m["decision_dt_s"])


def test_seeded_matches_repeat(cfg):
    a = play(env_for(cfg), ReloadRule(cfg), seed=11)
    b = play(env_for(cfg), ReloadRule(cfg), seed=11)
    assert a[0]["stats"] == b[0]["stats"] and a[1] == b[1]


def test_hand_ablation_zeros_hand_features(cfg):
    env = env_for(cfg, observe_hand=False)
    obs, _ = env.reset(seed=0)
    idx = [MatchEnv.POLICY_OBS_NAMES.index(n) for n in ("hand_x", "hand_y", "hand_vx", "hand_vy")]
    for _ in range(30):
        assert np.all(obs[idx] == 0.0)
        obs, *_ = env.step(BLOCK)


def test_incoming_puck_hidden_before_camera_latency(cfg):
    env = env_for(cfg)
    env.reset(seed=0)
    valid = MatchEnv.POLICY_OBS_NAMES.index("puck_valid")
    while not env.incoming:
        obs, *_ = env.step(BLOCK)
    shot = env.incoming[0]["shot"]
    seen = env.t - shot.t_launch >= cfg["camera"]["latency_s"]
    assert (obs[valid] == 1.0) == seen


def test_reload_detection_from_the_hand(cfg):
    """The hand shows reliably when the opponent aims (pull, hold). Most reload frames count as
    reloading; the start of reload (hand coming back from behind the band) and place do not match."""
    env = env_for(cfg)
    rule = ReloadRule(cfg)
    obs, _ = env.reset(seed=0)
    seen = {"aim": [], "reload": []}
    for _ in range(3000):
        phase = env.opponent.phase(env.t)
        if phase in ("pull", "hold"):
            seen["aim"].append(rule.reloading(obs, None))
        elif phase == "reload":
            seen["reload"].append(rule.reloading(obs, None))
        obs, *_ = env.step(BLOCK)
    assert np.mean(seen["aim"]) < 0.02
    assert np.mean(seen["reload"]) > 0.7


def test_block_table_interpolation():
    assert TABLE.p_save(0.8, at_gate=True) == 1.0
    assert TABLE.p_save(0.8, at_gate=False, delay=0.0) == 1.0
    assert TABLE.p_save(0.8, at_gate=False, delay=0.15) == pytest.approx(0.5)
    assert TABLE.p_save(1.9, at_gate=False, delay=0.0) == pytest.approx(0.5)
    assert TABLE.p_save(5.0, at_gate=False, delay=0.0) == 0.0  # Clipped to the fastest speed
