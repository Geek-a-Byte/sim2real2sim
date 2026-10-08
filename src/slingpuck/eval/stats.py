"""Confidence intervals for evaluation results."""
import numpy as np
from scipy import stats


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def t_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and Student-t interval across independent runs (e.g. training seeds)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if len(v) == 1:
        return mean, float("nan"), float("nan")
    half = stats.t.ppf(0.5 + confidence / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return mean, mean - half, mean + half
