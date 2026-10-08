import pytest

from src.slingpuck.eval.stats import t_ci, wilson_ci


def test_wilson_known_values():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4)
    assert hi == pytest.approx(0.5962, abs=1e-4)
    lo, hi = wilson_ci(60, 60)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.9398, abs=1e-4)


def test_t_ci():
    mean, lo, hi = t_ci([0.8, 0.9, 1.0])
    assert mean == pytest.approx(0.9)
    assert hi - mean == pytest.approx(4.303 * 0.1 / 3**0.5, rel=1e-3)
