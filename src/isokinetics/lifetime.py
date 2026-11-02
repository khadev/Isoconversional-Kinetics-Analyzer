"""
lifetime.py
===========
Arrhenius shelf-life extrapolation: how long does a material last at its
*service* temperature, given how long it lasts at a (usually hotter)
*storage / ageing* temperature?

Model
-----
For a thermally activated degradation with activation energy ``Ea`` the
rate constant follows ``k = A exp(-Ea / (R T))``.  Reaching the same degree
of degradation at two temperatures takes times inversely proportional to
the rate constants, and the pre-exponential factor A cancels::

    AF = exp[ (Ea / R) * (1 / T_service - 1 / T_storage) ]
    t_service = t_storage * AF

``AF`` (the acceleration factor) is > 1 when the service temperature is
lower than the storage temperature (the usual case: accelerated ageing is
hotter than real use), and < 1 when it is higher.

Units
-----
* Temperatures are handled in **kelvin** internally (the GUI converts from
  degrees Celsius).
* ``Ea`` is in **kJ/mol**; ``R = 8.314462618 J/(mol K)``.
* Times are converted to **days** internally and reported in years
  (1 year = 365.25 days), days and minutes (1 day = 1440 min).

Assumptions (also reported to the user as warnings when they look violated)
---------------------------------------------------------------------------
1. The *same* single degradation mechanism governs both temperatures.
2. ``Ea`` is constant over the extrapolated range.  Ea from non-isothermal
   thermal analysis is measured at high temperature; extrapolating to
   ambient conditions is exactly the classical weakness of this method.
3. The extrapolation is exponentially sensitive to Ea, so the result is
   accompanied by the lifetime obtained with Ea +/- 10 %.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .kinetics import R

ZERO_CELSIUS_K = 273.15
DAYS_PER_YEAR = 365.25
MINUTES_PER_DAY = 1440.0

#: Time units accepted for the storage time, expressed in days.
TIME_UNITS_IN_DAYS: Dict[str, float] = {
    "years": DAYS_PER_YEAR,
    "days": 1.0,
    "hours": 1.0 / 24.0,
    "minutes": 1.0 / MINUTES_PER_DAY,
}

#: Predictions beyond this many years get a plausibility warning.
_IMPLAUSIBLE_YEARS = 100.0
#: Above this Ea (kJ/mol) a room-temperature ageing prediction gets a warning.
_HIGH_SHELF_LIFE_EA_KJ = 150.0

#: Above this exponent exp() overflows a double; the result is reported as
#: infinite instead of raising.
_MAX_LN = 700.0


def celsius_to_kelvin(temp_c: float) -> float:
    """Degrees Celsius -> kelvin."""
    return float(temp_c) + ZERO_CELSIUS_K


def to_kelvin(value: float, unit: str) -> float:
    """Convert a temperature given in ``"C"``/``"\u00b0C"`` or ``"K"`` to kelvin."""
    unit = unit.strip().upper().replace("\u00b0", "")
    if unit == "C":
        return celsius_to_kelvin(value)
    if unit == "K":
        return float(value)
    raise ValueError(f"Unknown temperature unit {unit!r} (use 'C' or 'K').")


def time_to_days(value: float, unit: str) -> float:
    """Convert a duration in years/days/hours/minutes to days."""
    try:
        return float(value) * TIME_UNITS_IN_DAYS[unit.strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unknown time unit {unit!r}; choose one of {sorted(TIME_UNITS_IN_DAYS)}."
        ) from None


def acceleration_factor(ea_kj_per_mol: float, storage_temp_k: float, service_temp_k: float) -> float:
    """
    ``AF = exp[(Ea/R)(1/T_service - 1/T_storage)]``  (dimensionless).

    ``AF`` multiplies a time at the storage temperature to give the
    equivalent time at the service temperature.  Returns ``inf`` instead of
    overflowing for absurdly large exponents.
    """
    _validate(ea_kj_per_mol, storage_temp_k, service_temp_k)
    ln_af = (ea_kj_per_mol * 1000.0 / R) * (1.0 / service_temp_k - 1.0 / storage_temp_k)
    if ln_af > _MAX_LN:
        return math.inf
    return math.exp(ln_af)


def _validate(ea_kj: float, t_storage_k: float, t_service_k: float) -> None:
    for name, value in (("Activation energy", ea_kj), ("Storage temperature", t_storage_k),
                        ("Service temperature", t_service_k)):
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number.")
    if ea_kj <= 0:
        raise ValueError("Activation energy must be positive (kJ/mol).")
    if t_storage_k <= 0 or t_service_k <= 0:
        raise ValueError("Temperatures must be above absolute zero (0 K = -273.15 \u00b0C).")


@dataclass
class LifetimeResult:
    """Outcome of :func:`predict_lifetime`."""

    ea_kj: float
    storage_temp_k: float
    service_temp_k: float
    storage_time_days: float
    acceleration_factor: float
    lifetime_days: float
    lifetime_years: float
    lifetime_minutes: float
    #: Lifetime in days with Ea lowered / raised by ``ea_uncertainty_pct``.
    lifetime_days_low_ea: float = math.nan
    lifetime_days_high_ea: float = math.nan
    ea_uncertainty_pct: float = 10.0
    warnings: List[str] = field(default_factory=list)

    @property
    def is_finite(self) -> bool:
        return math.isfinite(self.lifetime_days)


def predict_lifetime(
    ea_kj: float,
    storage_temp_k: float,
    storage_time_days: float,
    service_temp_k: float,
    ea_uncertainty_pct: float = 10.0,
    measured_range_k: Optional[Tuple[float, float]] = None,
    ea_variation_pct: Optional[float] = None,
) -> LifetimeResult:
    """
    Predict the service lifetime by Arrhenius extrapolation.

    Parameters
    ----------
    ea_kj : float
        Activation energy in kJ/mol (e.g. the Coats-Redfern recommended
        model's Ea).
    storage_temp_k : float
        Storage / ageing temperature in kelvin.
    storage_time_days : float
        Time (days) after which the material reaches the end-of-life
        criterion when kept at ``storage_temp_k`` -- e.g. "failed after
        90 days at 60 degC" in an accelerated-ageing test.  It is **not**
        the time the material has merely been stored.
    service_temp_k : float
        Service (use) temperature in kelvin.
    ea_uncertainty_pct : float
        Ea is varied by +/- this percentage to show the sensitivity.
    measured_range_k : (float, float), optional
        Temperature range (K) over which the kinetic data were measured;
        used only to warn about extrapolation far outside it.
    ea_variation_pct : float, optional
        Relative variation of Ea(alpha) over the conversion range (percent);
        used only to warn when a single Ea is a poor description.

    Returns
    -------
    LifetimeResult

    Raises
    ------
    ValueError
        For non-positive Ea / time or temperatures at or below 0 K.
    """
    _validate(ea_kj, storage_temp_k, service_temp_k)
    if storage_time_days is None or not math.isfinite(storage_time_days) or storage_time_days <= 0:
        raise ValueError("Storage time must be a positive number.")

    af = acceleration_factor(ea_kj, storage_temp_k, service_temp_k)
    days = storage_time_days * af

    def lifetime_with(ea: float) -> float:
        return storage_time_days * acceleration_factor(ea, storage_temp_k, service_temp_k)

    low = lifetime_with(ea_kj * (1.0 - ea_uncertainty_pct / 100.0)) if ea_uncertainty_pct < 100 else math.nan
    high = lifetime_with(ea_kj * (1.0 + ea_uncertainty_pct / 100.0))

    notes: List[str] = []
    if not math.isfinite(days):
        notes.append("The predicted lifetime exceeds the numerical range (effectively unlimited).")
    if service_temp_k > storage_temp_k:
        notes.append(
            "The service temperature is higher than the storage temperature: the "
            "lifetime is shorter than the storage time (acceleration factor < 1)."
        )
    if abs(service_temp_k - storage_temp_k) < 1e-9:
        notes.append("Storage and service temperatures are equal: the lifetime equals the storage time.")
    if measured_range_k is not None:
        lo, hi = measured_range_k
        gap = lo - min(service_temp_k, storage_temp_k)
        if gap > 50.0:
            notes.append(
                f"Extrapolation: the kinetic data were measured at {lo:.0f}\u2013{hi:.0f} K, "
                f"{gap:.0f} K above the lowest temperature used here. Ea is assumed constant "
                "down to that temperature; treat the result as an estimate."
            )
    if math.isfinite(days) and days / DAYS_PER_YEAR > _IMPLAUSIBLE_YEARS:
        notes.append(
            f"Plausibility: a lifetime of more than {_IMPLAUSIBLE_YEARS:.0f} years is far beyond what any "
            "ageing test can confirm. Treat it as \"very long\" rather than as a precise number."
        )
    if ea_kj > _HIGH_SHELF_LIFE_EA_KJ and service_temp_k < 373.15:
        notes.append(
            f"Ea = {ea_kj:.0f} kJ/mol is high for slow ageing near room temperature (typically "
            "40\u2013120 kJ/mol). Ea from high-temperature decomposition usually describes a different, "
            "higher-energy process than the one that limits shelf life at ambient conditions "
            "(oxidation, hydrolysis, plasticiser loss...), so this prediction is most likely an "
            "over-estimate (an upper bound). For a realistic shelf life use an Ea measured from "
            "ageing data at moderate temperatures."
        )
    if ea_variation_pct is not None and ea_variation_pct > 20.0:
        notes.append(
            f"Ea varies by {ea_variation_pct:.0f} % with conversion in your data (multi-step "
            "kinetics), so a single Ea is only an approximation; the result depends on the "
            "conversion range that matters for your end-of-life criterion."
        )
    if math.isfinite(days) and math.isfinite(high) and low > 0 and high / max(low, 1e-300) > 100:
        notes.append(
            f"The result is very sensitive to Ea: \u00b1{ea_uncertainty_pct:g} % in Ea changes "
            "the lifetime by more than two orders of magnitude."
        )

    return LifetimeResult(
        ea_kj=float(ea_kj),
        storage_temp_k=float(storage_temp_k),
        service_temp_k=float(service_temp_k),
        storage_time_days=float(storage_time_days),
        acceleration_factor=af,
        lifetime_days=days,
        lifetime_years=days / DAYS_PER_YEAR,
        lifetime_minutes=days * MINUTES_PER_DAY,
        lifetime_days_low_ea=low,
        lifetime_days_high_ea=high,
        ea_uncertainty_pct=float(ea_uncertainty_pct),
        warnings=notes,
    )


# --------------------------------------------------------------------------- #
# Where does Ea come from?  (integration with the existing results)
# --------------------------------------------------------------------------- #
MANUAL_SOURCE = "Manual entry"


@dataclass
class EaSource:
    """One selectable origin for the activation energy."""

    label: str
    ea_kj: Optional[float]  # None for manual entry
    note: str = ""


def available_ea_sources(
    averages: Optional[Dict[str, float]] = None,
    cr_result: Optional[Dict[str, object]] = None,
) -> List[EaSource]:
    """
    Build the list of Ea sources, best first:

    1. the **recommended Coats-Redfern model** (when one was found);
    2. the mean Ea of every method that was run (model-free; these do not
       assume a reaction model);
    3. every other Coats-Redfern model, flagged as *not recommended*;
    4. manual entry.

    When no model is recommended the first entry is therefore a model-free
    mean, and the note says why.
    """
    sources: List[EaSource] = []
    cr_results = (cr_result or {}).get("results") or {}
    recommended = (cr_result or {}).get("recommended_model")

    if recommended and recommended in cr_results:
        fit = cr_results[recommended]
        sources.append(EaSource(
            f"Coats-Redfern recommended ({recommended})", float(fit["Ea"]),
            "Ea of the reaction model that best matches the model-free reference.",
        ))

    for name, value in (averages or {}).items():
        if value is not None and math.isfinite(value) and value > 0:
            sources.append(EaSource(
                f"{name} mean", float(value),
                "Mean of Ea(\u03b1) over the conversion range (model-free)."
                + ("" if recommended else "  No Coats-Redfern model was recommended."),
            ))

    for name, fit in sorted(cr_results.items(), key=lambda kv: -(kv[1].get("r2") or 0)):
        if name == recommended:
            continue
        ea = fit.get("Ea")
        if ea is not None and math.isfinite(ea) and ea > 0:
            sources.append(EaSource(
                f"Coats-Redfern {name} (not recommended)", float(ea),
                "This model did not pass the screening; shown for comparison only.",
            ))

    sources.append(EaSource(MANUAL_SOURCE, None, "Type an activation energy in kJ/mol."))
    return sources


def ea_variation_percent(
    results: Dict[str, Dict[float, float]], lo: float = 0.10, hi: float = 0.90
) -> Optional[float]:
    """Relative spread (max-min)/mean of Ea(alpha) over ``lo..hi`` for the
    first available differential/advanced method, in percent."""
    for name in ("Friedman", "Vyazovkin", "KTH", *results.keys()):
        res = results.get(name)
        if not res:
            continue
        vals = [e for a, e in res.items() if lo <= a <= hi and math.isfinite(e)]
        if len(vals) >= 3 and sum(vals) > 0:
            mean = sum(vals) / len(vals)
            return (max(vals) - min(vals)) / mean * 100.0
    return None
