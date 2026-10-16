import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.slingpuck.physics.mujoco_match import MatchSim  # noqa: E402


@pytest.fixture(scope="module")
def game(cfg):
    sim = MatchSim(cfg, seed=0)
    vmax, held = 0.0, {"robot": False, "human": False}
    while sim.t < 12.0:
        sim.step()
        vmax = max(vmax, max(np.linalg.norm(sim.puck_vel(p)) for p in sim.pucks))
        for side, who in (("robot", sim.robot), ("human", sim.human)):
            held[side] |= who["held"] is not None
    return sim, vmax, held


def test_colours_split_five_and_five(cfg):
    sim = MatchSim(cfg)
    assert sorted(p["colour"] for p in sim.pucks) == ["black"] * 5 + ["white"] * 5
    assert all(sim.side_of(p) == ("robot" if p["colour"] == "black" else "human") for p in sim.pucks)


def test_both_players_shoot_and_physics_stays_sane(game, cfg):
    sim, vmax, held = game
    assert held["robot"] and held["human"]
    assert sim.shots["robot"] >= 1 and sim.shots["human"] >= 1
    assert vmax < 6.0  # Hand-launched shots are 1-3 m/s; a blow-up gives tens to thousands of m/s
    lim = np.array([0.5 * cfg["board"]["width_m"], 0.5 * cfg["board"]["length_m"]])
    assert all(np.all(np.abs(sim.puck_pos(p)) <= lim) for p in sim.pucks)
    assert sum(sim.counts().values()) == 10


def test_robot_guard_pose_blocks_the_gate(cfg):
    sim = MatchSim(cfg)  # Starts in the guard pose: the closed jaws stand in the gate
    np.testing.assert_allclose(sim.site_board()[:2], sim.guard[:2], atol=3e-3)
    assert abs(sim.guard[0]) < 1e-9 and sim.guard[1] < 0.0
