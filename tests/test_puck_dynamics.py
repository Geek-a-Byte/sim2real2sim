import numpy as np
import pytest

from src.slingpuck.physics.puck_dynamics import GRAVITY, BoardGeometry, Fast2DPuckSim, PuckParams

DT = 0.001


def make_sim(cfg, n=1, mu=None, e_wall=None, e_puck=None):
    board = BoardGeometry.from_config(cfg)
    puck = PuckParams.from_config(cfg)
    if e_wall is not None:
        board = BoardGeometry(board.length_m, board.width_m, board.gate_width_m, board.divider_thickness_m, e_wall)
    if mu is not None or e_puck is not None:
        puck = PuckParams(puck.radius_m, puck.mass_kg,
                          puck.friction_kinetic if mu is None else mu,
                          puck.restitution if e_puck is None else e_puck)
    return Fast2DPuckSim(board, puck, n)


def test_friction_decelerates_and_stops_without_reversal(cfg):
    sim = make_sim(cfg)
    mu = sim.puck.friction_kinetic
    sim.reset([[0.0, -0.12]], [[0.3, 0.0]])  # Short slide, away from walls
    for _ in range(50):
        sim.step(DT)
    assert sim.vel[0, 0] == pytest.approx(0.3 - mu * GRAVITY * 50 * DT)
    for _ in range(1000):
        sim.step(DT)
    assert np.all(sim.vel == 0.0)
    stop_x = 0.3**2 / (2 * mu * GRAVITY)
    assert sim.pos[0, 0] == pytest.approx(stop_x, abs=0.3 * DT)


def test_wall_bounce_scales_normal_velocity(cfg):
    e = 0.8
    sim = make_sim(cfg, mu=0.0, e_wall=e)
    hw = sim.board.half_width
    sim.reset([[-hw + 0.03, -0.1]], [[-1.0, 0.2]])
    for _ in range(100):
        sim.step(DT)
    np.testing.assert_allclose(sim.vel[0], [e * 1.0, 0.2])


@pytest.mark.parametrize("seed", range(5))
def test_energy_never_increases(cfg, seed):
    rng = np.random.default_rng(seed)
    n = 4
    sim = make_sim(cfg, n=n)
    # Non-overlapping starts on a grid, random velocities up to 3 m/s.
    xs = np.linspace(-0.06, 0.06, n)
    pos = np.column_stack([xs, rng.choice([-0.1, 0.1], n)])
    sim.reset(pos, rng.uniform(-3, 3, (n, 2)))
    energy = sim.kinetic_energy()
    for _ in range(3000):
        sim.step(DT)
        new_energy = sim.kinetic_energy()
        assert new_energy <= energy + 1e-12
        energy = new_energy


def test_energy_conserved_without_losses(cfg):
    sim = make_sim(cfg, n=2, mu=0.0, e_wall=1.0, e_puck=1.0)
    sim.reset([[-0.04, -0.08], [0.04, -0.08]], [[1.5, 0.7], [-1.2, 0.4]])
    e0 = sim.kinetic_energy()
    for _ in range(2000):
        sim.step(DT)
    assert sim.kinetic_energy() == pytest.approx(e0, rel=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_pucks_stay_on_board(cfg, seed):
    rng = np.random.default_rng(seed)
    sim = make_sim(cfg, n=3)
    sim.reset([[-0.05, -0.1], [0.0, 0.1], [0.05, -0.1]], rng.uniform(-4, 4, (3, 2)))
    r = sim.puck.radius_m
    for _ in range(3000):
        sim.step(DT)
        assert np.all(np.abs(sim.pos[:, 0]) <= sim.board.half_width - r + 1e-9)
        assert np.all(np.abs(sim.pos[:, 1]) <= sim.board.half_length - r + 1e-9)


def test_divider_blocks_outside_gate(cfg):
    sim = make_sim(cfg, mu=0.0)
    x = 0.5 * sim.board.gate_width_m + sim.puck.radius_m + 0.01
    sim.reset([[x, -0.08]], [[0.0, 1.0]])
    crossings = []
    for _ in range(120):  # Hits the divider at ~60 ms, before the end wall at ~220 ms
        crossings += sim.step(DT).crossings
    assert crossings == []
    assert sim.pos[0, 1] < 0.0
    assert sim.vel[0, 1] < 0.0


def test_puck_passes_through_gate(cfg):
    sim = make_sim(cfg, mu=0.0)
    sim.reset([[0.0, -0.08]], [[0.0, 1.0]])
    crossings = []
    for _ in range(150):
        crossings += sim.step(DT).crossings
    assert len(crossings) == 1
    assert crossings[0].direction == +1 and crossings[0].puck == 0
    assert abs(crossings[0].x) < 0.5 * sim.board.gate_width_m


def test_head_on_elastic_collision_swaps_velocities(cfg):
    sim = make_sim(cfg, n=2, mu=0.0, e_puck=1.0)
    sim.reset([[-0.05, -0.1], [0.05, -0.1]], [[1.0, 0.0], [-1.0, 0.0]])
    for _ in range(40):
        sim.step(DT)
    np.testing.assert_allclose(sim.vel, [[-1.0, 0.0], [1.0, 0.0]], atol=1e-12)


def test_rejects_invalid_setup(cfg):
    with pytest.raises(ValueError):
        make_sim(cfg, e_wall=1.2)
    sim = make_sim(cfg)
    with pytest.raises(ValueError):
        sim.reset([[0.0, 1.0]], [[0.0, 0.0]])


def paddle(center, vel=(0.0, 0.0), omega=0.0, angle=0.0, e=0.5):
    from src.slingpuck.physics.backend import PaddleState
    return PaddleState(np.array(center, float), angle, np.array(vel, float), omega, 0.015, 0.003, e)


def test_stationary_paddle_reflects_with_restitution(cfg):
    sim = make_sim(cfg, mu=0.0)
    sim.set_paddle(paddle([0.0, -0.10], e=0.5))
    sim.reset([[0.0, -0.05]], [[0.0, -1.0]])
    contacts = []
    for _ in range(60):
        contacts += sim.step(DT).paddle_contacts
    assert contacts and set(contacts) == {0}
    np.testing.assert_allclose(sim.vel[0], [0.0, 0.5], atol=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_energy_never_increases_with_stationary_paddle(cfg, seed):
    rng = np.random.default_rng(seed)
    sim = make_sim(cfg, n=2)
    sim.set_paddle(paddle([0.01, -0.05], angle=0.3))
    sim.reset([[-0.05, -0.12], [0.05, 0.1]], rng.uniform(-3, 3, (2, 2)))
    energy = sim.kinetic_energy()
    for _ in range(2000):
        sim.step(DT)
        assert sim.kinetic_energy() <= energy + 1e-12
        energy = sim.kinetic_energy()


def test_moving_paddle_pushes_puck(cfg):
    sim = make_sim(cfg, mu=0.0)
    sim.set_paddle(paddle([0.0, -0.10], vel=(0.0, 1.0), e=0.0))
    sim.reset([[0.0, -0.10 + 0.003 + sim.puck.radius_m - 0.0005]], [[0.0, 0.0]])  # Small overlap
    for _ in range(5):
        sim.step(DT)
    # A perfectly inelastic contact leaves the puck with the paddle's normal velocity.
    assert sim.vel[0, 1] == pytest.approx(1.0)


def test_rotating_paddle_surface_velocity(cfg):
    sim = make_sim(cfg, mu=0.0)
    # Paddle rotates counter-clockwise about its center; its +x tip moves toward +y.
    sim.set_paddle(paddle([0.0, -0.10], omega=20.0, e=0.0))
    tip = 0.012
    sim.reset([[tip, -0.10 + 0.003 + sim.puck.radius_m - 0.0005]], [[0.0, 0.0]])  # Small overlap
    sim.step(DT)
    assert sim.vel[0, 1] == pytest.approx(20.0 * tip, rel=1e-6)
