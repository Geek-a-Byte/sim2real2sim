import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.slingpuck.eval import skill_eval as se  # noqa: E402
from src.slingpuck.eval.timing_sweep import scaled  # noqa: E402
from src.slingpuck.physics.mujoco_match import MatchSim  # noqa: E402
from src.slingpuck.primitives.sling import ShotSelector  # noqa: E402

STRAIGHT = (0.0, 0.0, 0.025)


@pytest.fixture(scope="module")
def sim(cfg):
    return MatchSim(cfg, opponent=False)


def shot(sim, seed=0, **kw):
    return se.run_shot(sim, kw.pop("shot", STRAIGHT), np.random.default_rng(seed), **kw)


def test_jaws_collide_with_pucks(sim):
    m = sim.model
    p = sim.pucks[0]
    assert m.body_contype[sim._fixed_jaw] & m.body_conaffinity[p["body"]]
    assert m.body_contype[sim._moving_jaw] & m.body_conaffinity[p["body"]]


def test_scripted_shot_grips_and_goes_through(sim):
    r = shot(sim)
    assert r["grasped"] and not r["slip"]
    assert r["outcome"] == "through"
    assert 0 < r["t_pick"] < r["t_to_release"] < r["t_cycle"]


def test_start_row_pucks_touch(sim, cfg):
    sim.reset()
    a, b = sim.puck_pos(sim.pucks[1]), sim.puck_pos(sim.pucks[2])
    assert np.linalg.norm(a - b) == pytest.approx(2 * cfg["puck"]["radius_m"], abs=1e-6)


def test_middle_puck_is_pushed_out_of_the_row_then_gripped(cfg):
    sim = MatchSim({**cfg, "skills": {**cfg["skills"], "spread_row": False}}, opponent=False)
    sim.auto_fetch = False
    middle, neighbours = sim.pucks[2], (sim.pucks[1], sim.pucks[3])
    before = [sim.puck_pos(p).copy() for p in neighbours]
    assert len(sim.jaw_blockers(middle)) == 2
    sim._robot_fetch(middle, STRAIGHT)
    while sim.t < 4.0 and not (sim.robot["moves"] and sim.robot["moves"][sim.robot["k"]].name == "to band"):
        sim.step()
    assert sim.fetch_log[-1]["freed"] and sim.fetch_log[-1]["grasped"]
    assert not sim.jaw_blockers(middle)
    for p, b in zip(neighbours, before):
        assert np.linalg.norm(sim.puck_pos(p) - b) < 2e-3


def test_row_shots_keep_the_jaws_above_the_board(cfg):
    r = se.run_row(MatchSim(cfg, opponent=False), 1, True, max_t=4.0)
    assert r["freed"] >= 1 and r["grasped"] >= 1
    assert r["min_tip_z_m"] > 0.0


def test_fast_arm_does_not_knock_the_puck_before_the_grasp(cfg):
    fast = {**cfg, "skills": {**cfg["skills"], "speed_scale": 2.0}}
    r = se.run_shot(MatchSim(fast, opponent=False), STRAIGHT, np.random.default_rng(1))
    assert r["grasped"] and r["min_tip_z_m"] > 0.0


def test_spread_widens_the_hole_in_the_row(cfg):
    sim = MatchSim(cfg, opponent=False)
    sim.auto_fetch = False
    middle, left, right = sim.pucks[2], sim.pucks[1], sim.pucks[3]
    gap0 = abs(sim.puck_pos(right)[0] - sim.puck_pos(left)[0])
    sim._robot_fetch(middle, STRAIGHT)
    while sim.t < 4.0 and sim.robot["state"] == "free":
        sim.step()
    assert sim.fetch_log[-1]["freed"]
    assert abs(sim.puck_pos(right)[0] - sim.puck_pos(left)[0]) > gap0 + 0.005


def test_greedy_sling_rule_does_not_go_back_to_the_gate(cfg):
    greedy = {**cfg, "skills": {**cfg["skills"], "game_rule": "greedy_sling"}}
    sim = MatchSim(greedy, opponent=False)
    sim.step()
    assert not any(m.name == "to gate" for m in sim.robot["moves"])


def test_low_jaw_friction_loses_the_puck_in_the_pull(cfg):
    low = {**cfg, "gripper": {**cfg["gripper"], "jaw_friction": 0.02}}
    r = se.run_hold(MatchSim(low, opponent=False), 0.03)
    assert r["grasped"] and r["slip"]


def test_hold_keeps_the_puck_under_the_band_force(sim):
    r = se.run_hold(sim, 0.03, seconds=0.5)
    assert r["grasped"] and r["held"] and not r["slip"]
    assert r["drift_m"] < 0.003


def test_grasp_of_an_empty_spot_is_a_missed_grasp(cfg):
    sim = MatchSim(cfg, opponent=False)
    sim.auto_fetch = False
    p = sim.pucks[0]
    se.park_others(sim, p)
    se.place(sim, p, (0.0, -0.07))
    sim._robot_fetch(p, STRAIGHT, seen_xy=(0.0, -0.11))
    while sim.t < 1.5 and sim.fetch_log[-1]["outcome"] is None:
        sim.step()
    assert sim.fetch_log[-1]["outcome"] == "missed_grasp"
    assert sim.robot["held"] is None


def test_closed_gripper_blocks_a_shot_and_an_open_gate_does_not(sim):
    half = 0.0043
    assert not se.run_block(sim, 0.0, half, 3.0)["passed"]
    assert not se.run_block(sim, 0.0, -half, 3.0)["passed"]
    assert se.run_block(sim, 0.06, 0.0, 3.0)["passed"]      # The control: the test can fail


def test_calibration_error_moves_the_arm(sim):
    sim.joint_offset = np.full(5, np.radians(1.0))
    sim.reset()
    moved = sim.site_board().copy()
    sim.joint_offset = np.zeros(5)
    sim.reset()
    assert np.linalg.norm(moved - sim.site_board()) > 1e-3


def test_constraint_grasp_model_still_works(cfg):
    c = {**cfg, "skills": {**cfg["skills"], "grasp_model": "constraint"}}
    sim = MatchSim(c, seed=0)
    while sim.t < 3.0 and sim.robot["held"] is None:
        sim.step()
    assert sim.robot["held"] is not None


def test_selector_gives_the_shot(greedy_cfg, shot_model):
    sim = MatchSim(greedy_cfg, opponent=False, selector=ShotSelector(greedy_cfg, shot_model))
    sim.step()
    log = sim.fetch_log[-1]
    lo, hi = np.array(greedy_cfg["shot"]["pull_range_m"])
    assert lo <= log["shot"][2] <= hi


def test_timing_sweep_scales_only_the_robot_times(cfg):
    c = scaled(cfg, 2.0, 0.5, 0.8)
    assert c["match"]["fetch_time_s"] == 2 * cfg["match"]["fetch_time_s"]
    assert c["shot"]["arm_speed_m_s"] == cfg["shot"]["arm_speed_m_s"] / 2
    assert c["match"]["move_gate_to_band_s"] == 0.5 * cfg["match"]["move_gate_to_band_s"]
    assert c["opponent"]["threat_prob"] == 0.8
    assert c["opponent"]["reload_time_mu_s"] == cfg["opponent"]["reload_time_mu_s"]
    assert cfg["opponent"]["threat_prob"] == 0.5                 # The input is not changed
