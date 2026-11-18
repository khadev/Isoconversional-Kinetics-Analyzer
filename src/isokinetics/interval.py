"""
interval.py
===========
Pure helpers for the *stabilization-phase* feature: the user drags a range
``[alpha_min, alpha_max]`` on the Ea(alpha) plot and the application reports
the mean activation energy of every method **inside that range only**.

Nothing here touches Tk or matplotlib, so every rule (clipping to the data,
the minimum number of points, NaN handling) is plain, unit-tested Python.
The full-range results are never modified: these functions only *read*
the ``{alpha: Ea}`` dictionaries produced by :mod:`isokinetics.kinetics`.

Rules
-----
* A selection is first **clipped** to the data bounds (:func:`clip_interval`);
  a selection entirely outside the data gives ``None``.
* An interval statistic needs at least :data:`MIN_POINTS` finite grid points
  of a method; with fewer, that method's mean is ``NaN`` ("not enough
  points") rather than a misleading number.
* Both interval ends are inclusive (with a 1e-9 tolerance so that a drag
  ending at exactly 0.2 includes the grid point 0.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .formatting import fmt_interval
from .kinetics import calculate_interval_mean

#: Fewest grid points an interval must contain for a mean to be reported.
MIN_POINTS = 3

#: Tolerance when comparing a grid alpha with an interval end.
_EPS = 1e-9


@dataclass(frozen=True)
class AlphaInterval:
    """A closed conversion range ``[lo, hi]`` (always ``lo <= hi``)."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:  # accept a right-to-left drag
            lo, hi = self.hi, self.lo
            object.__setattr__(self, "lo", lo)
            object.__setattr__(self, "hi", hi)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, alpha: float) -> bool:
        """True if ``alpha`` lies in the closed interval."""
        return self.lo - _EPS <= alpha <= self.hi + _EPS

    @property
    def label(self) -> str:
        """Short text such as ``0.2\u20130.8`` (used in the legend and tables)."""
        return fmt_interval(self.lo, self.hi)


def clip_interval(
    lo: float, hi: float, data_min: float, data_max: float
) -> Optional[AlphaInterval]:
    """
    Order and clip a raw selection to the alpha range of the data.

    Returns ``None`` when the input is not finite or the selection lies
    entirely outside ``[data_min, data_max]`` (e.g. a drag started and ended
    beyond the last grid point).  A selection that only *partly* overhangs is
    shortened to the data bounds.
    """
    values = (lo, hi, data_min, data_max)
    if not all(v is not None and math.isfinite(float(v)) for v in values):
        return None
    a, b = sorted((float(lo), float(hi)))
    a = max(a, float(data_min))
    b = min(b, float(data_max))
    if a > b + _EPS:
        return None
    return AlphaInterval(a, max(a, b))


def values_in_interval(result: Dict[float, float], interval: AlphaInterval) -> List[float]:
    """The finite Ea values of one method whose alpha lies in ``interval``."""
    return [
        float(e)
        for a, e in result.items()
        if interval.contains(a) and e is not None and math.isfinite(e)
    ]


def count_in_interval(result: Dict[float, float], interval: AlphaInterval) -> int:
    """Number of usable (finite) grid points of one method inside ``interval``."""
    return len(values_in_interval(result, interval))


def mean_ea_in_interval(
    result: Dict[float, float], interval: AlphaInterval, min_points: int = MIN_POINTS
) -> float:
    """
    Mean Ea (kJ/mol) of one method over ``interval``; ``NaN`` if fewer than
    ``min_points`` finite points fall inside it.  A thin wrapper around
    :func:`isokinetics.kinetics.calculate_interval_mean`, the single place
    the "stabilization-phase mean" is computed.
    """
    return calculate_interval_mean(result, interval.lo, interval.hi, min_points)


def interval_means(
    results: Dict[str, Dict[float, float]],
    interval: AlphaInterval,
    min_points: int = MIN_POINTS,
) -> Dict[str, float]:
    """``{method: mean Ea inside interval}`` for every method (NaN if too few points)."""
    return {name: mean_ea_in_interval(res, interval, min_points) for name, res in results.items()}


def interval_statistics(
    results: Dict[str, Dict[float, float]],
    interval: AlphaInterval,
    min_points: int = MIN_POINTS,
) -> Dict[str, Dict[str, float]]:
    """
    Per-method ``mean / std / min / max / n`` inside ``interval`` (the
    interval analogue of :func:`isokinetics.exporter.method_statistics`).
    Statistics are NaN when the method has fewer than ``min_points`` points;
    ``n`` always reports the true count.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for name, res in results.items():
        vals = values_in_interval(res, interval)
        if len(vals) < min_points:
            stats[name] = {"mean": math.nan, "std": math.nan, "min": math.nan,
                           "max": math.nan, "n": len(vals)}
        else:
            arr = np.asarray(vals, dtype=float)
            stats[name] = {"mean": float(arr.mean()), "std": float(arr.std()),
                           "min": float(arr.min()), "max": float(arr.max()), "n": int(arr.size)}
    return stats


def max_point_count(results: Dict[str, Dict[float, float]], interval: AlphaInterval) -> int:
    """Largest number of usable points any method has inside ``interval``
    (0 for no results).  A selection is only worth applying if this reaches
    :data:`MIN_POINTS`."""
    return max((count_in_interval(res, interval) for res in results.values()), default=0)
