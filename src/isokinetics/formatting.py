"""
formatting.py
=============
One place that defines how numbers are written, so the tables, the plots,
the status bar and the exported files all agree.

Display convention (GUI tables, plots, summary cards)
------------------------------------------------------
=====================  ==============  =====================
Quantity               Format          Example
=====================  ==============  =====================
conversion alpha       3 decimals      ``0.250``
activation energy Ea   2 decimals      ``97.15``  (kJ/mol)
temperature            2 decimals      ``551.36`` (K)
pre-exponential A      sci, 3 decimals ``1.234e+10`` (s^-1)
f(alpha), scores       4 decimals      ``0.8123``
R^2                    4 decimals      ``0.9924``
percentages            1 decimal       ``12.5 %``
heating rate beta      shortest form   ``10`` / ``7.5`` (K/min)
=====================  ==============  =====================

Not-a-number values are always shown as ``N/A`` (the calculation was
attempted but is undefined) and values that were not computed at all as
``-``.  Exported *data* files use the same conventions but with extra
digits (see the ``EXPORT_*`` constants) so no precision is lost.
"""

from __future__ import annotations

import math
from typing import Optional

#: Placeholder for "attempted, but undefined".
NA = "N/A"
#: Placeholder for "not computed / not applicable".
EMPTY = "-"

# --- display precision ------------------------------------------------------
ALPHA_DECIMALS = 3
EA_DECIMALS = 2
TEMP_DECIMALS = 2
SCI_DECIMALS = 3
SCORE_DECIMALS = 4
R2_DECIMALS = 4
PCT_DECIMALS = 1

# --- export precision (data files keep more digits than the screen) ---------
EXPORT_ALPHA_DECIMALS = 4
EXPORT_EA_DECIMALS = 4
EXPORT_SCI_DECIMALS = 6
EXPORT_SCORE_DECIMALS = 6

#: Unit strings (Unicode, as used in plot labels and table headers).
UNIT_EA = "kJ mol\u207b\u00b9"
UNIT_EA_PLAIN = "kJ/mol"
UNIT_A = "s\u207b\u00b9"
UNIT_TEMP = "K"
UNIT_BETA = "K/min"


def _finite(value: Optional[float]) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt_fixed(value: Optional[float], decimals: int, na: str = NA) -> str:
    """``value`` with a fixed number of decimals, or ``na`` if not finite."""
    return f"{float(value):.{decimals}f}" if _finite(value) else na


def fmt_alpha(value: Optional[float], decimals: int = ALPHA_DECIMALS) -> str:
    """Conversion alpha, e.g. ``0.250``."""
    return fmt_fixed(value, decimals)


def fmt_ea(value: Optional[float], decimals: int = EA_DECIMALS, unit: bool = False) -> str:
    """Activation energy in kJ/mol, e.g. ``97.15`` (or ``97.15 kJ/mol``)."""
    text = fmt_fixed(value, decimals)
    return f"{text} {UNIT_EA_PLAIN}" if unit and text != NA else text


def fmt_temp(value: Optional[float], decimals: int = TEMP_DECIMALS) -> str:
    """Temperature in kelvin, e.g. ``551.36``."""
    return fmt_fixed(value, decimals)


def fmt_sci(value: Optional[float], decimals: int = SCI_DECIMALS, na: str = NA) -> str:
    """Scientific notation, e.g. ``1.234e+10``."""
    return f"{float(value):.{decimals}e}" if _finite(value) else na


def fmt_score(value: Optional[float], decimals: int = SCORE_DECIMALS, na: str = NA) -> str:
    """Dimensionless quantity such as f(alpha) or a match score."""
    return fmt_fixed(value, decimals, na)


def fmt_r2(value: Optional[float]) -> str:
    """Coefficient of determination, e.g. ``0.9924``."""
    return fmt_fixed(value, R2_DECIMALS)


def fmt_pct(value: Optional[float], decimals: int = PCT_DECIMALS, signed: bool = False) -> str:
    """Percentage given as a *number of percent* (12.5 -> ``12.5 %``)."""
    if not _finite(value):
        return NA
    return f"{float(value):+.{decimals}f} %" if signed else f"{float(value):.{decimals}f} %"


def fmt_beta(value: Optional[float]) -> str:
    """Heating rate in the shortest faithful form (``10`` not ``10.0``)."""
    if not _finite(value):
        return NA
    return f"{float(value):g}"


def fmt_duration(seconds: Optional[float]) -> str:
    """Elapsed/remaining time: ``0.8 s``, ``12 s``, ``1 min 05 s``."""
    if not _finite(seconds) or seconds < 0:
        return EMPTY
    seconds = float(seconds)
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, rem = divmod(int(round(seconds)), 60)
    return f"{minutes} min {rem:02d} s"


def alpha_range_text(alphas) -> str:
    """``"0.010 \u2013 0.990"`` for an iterable of alpha values."""
    values = [float(a) for a in alphas if _finite(a)]
    if not values:
        return EMPTY
    return f"{fmt_alpha(min(values))} \u2013 {fmt_alpha(max(values))}"


# --- lifetime prediction ----------------------------------------------------
#: Values at or above this are shown in scientific notation.
_BIG_TIME = 1e12


def fmt_time_value(value: Optional[float], decimals: int) -> str:
    """A duration with thousands separators (``1,234.57``); scientific
    notation for >= 1e12; ``∞`` for infinity; ``N/A`` for NaN/None."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return NA
    if math.isnan(v):
        return NA
    if math.isinf(v):
        return "∞"
    if abs(v) >= _BIG_TIME:
        return f"{v:.{SCI_DECIMALS}e}"
    return f"{v:,.{decimals}f}"


def fmt_years(value: Optional[float]) -> str:
    """Lifetime in years, 2 decimals: ``12.34``."""
    return fmt_time_value(value, 2)


def fmt_days(value: Optional[float]) -> str:
    """Lifetime in days, 2 decimals: ``4,506.10``."""
    return fmt_time_value(value, 2)


def fmt_minutes(value: Optional[float]) -> str:
    """Lifetime in minutes, whole numbers: ``6,488,640``."""
    return fmt_time_value(value, 0)
