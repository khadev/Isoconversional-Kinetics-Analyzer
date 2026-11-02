"""
validation.py
=============
Self-check of every kinetic method against **synthetic data with a known
activation energy**.

How the ground truth is generated
----------------------------------
The synthetic temperatures are produced here *independently of*
:mod:`isokinetics.kinetics` (different code, different quadrature, its own
copy of the gas constant), so a mistake in a method cannot cancel itself
out:

* **Constant Ea** -- the integral form of the rate law is solved exactly:
  ``g(alpha) = (A / beta) * integral_{T0}^{T} exp(-Ea/(R T')) dT'``.  The
  integral is accumulated with the trapezoidal rule on a 0.02 K grid
  (error < 1e-8 relative) and inverted by interpolation, giving the
  temperature ``T_i(alpha)`` for each heating rate ``beta_i``.
* **Variable Ea(alpha)** -- ``d alpha / dT = (A / beta) f(alpha)
  exp(-Ea(alpha)/(R T))`` is integrated with a fixed-step RK4 scheme.

What is checked
---------------
For every scenario and method the mean recovered Ea over
``0.10 <= alpha <= 0.90`` (constant Ea) or the mean absolute error versus
the true Ea(alpha) (variable Ea) is compared with a tolerance.  The
tolerances reflect each method's *theoretical* accuracy -- notably the
Doyle approximation behind FWO (bias of a few percent when
Ea/RT is far from ~35) and the known bias of integral methods when Ea
varies with alpha -- not slack in the implementation.

Run it from a terminal with ``python -m isokinetics.validation`` or from the
GUI's *Validation* tab.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import kinetics
from .formatting import fmt_ea, fmt_pct

#: Gas constant used **only** to generate the synthetic data (CODATA 2018).
#: Deliberately a separate literal from ``kinetics.R``.
_R_TRUE = 8.314462618

DEFAULT_BETAS: Sequence[float] = (5.0, 10.0, 20.0, 40.0)  # K/min

# Integral reaction models g(alpha) and differential f(alpha) used for the data.
_G = {
    "F1": lambda a: -np.log(1.0 - a),
    "F2": lambda a: 1.0 / (1.0 - a) - 1.0,
    "R3": lambda a: 1.0 - (1.0 - a) ** (1.0 / 3.0),
    "A2": lambda a: (-np.log(1.0 - a)) ** 0.5,
    "D3": lambda a: (1.0 - (1.0 - a) ** (1.0 / 3.0)) ** 2,
}
_F = {
    "F1": lambda a: 1.0 - a,
    "F2": lambda a: (1.0 - a) ** 2,
}

ALPHA_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


# --------------------------------------------------------------------------- #
# Synthetic data generators
# --------------------------------------------------------------------------- #
def simulate_constant_ea(
    ea_kj: float,
    pre_exp: float,
    model: str = "F1",
    betas_k_per_min: Sequence[float] = DEFAULT_BETAS,
    alphas: Optional[np.ndarray] = None,
    t_start: float = 100.0,
    t_end: float = 1500.0,
    d_t: float = 0.02,
) -> pd.DataFrame:
    """
    Exact synthetic dataset (columns alpha, T1..T4, in K) for a reaction
    ``g(alpha) = (A/beta) * I(T)`` with a *constant* activation energy.

    Parameters
    ----------
    ea_kj : float
        True activation energy (kJ/mol).
    pre_exp : float
        Pre-exponential factor A (s^-1).
    model : str
        One of ``F1, F2, R3, A2, D3``.
    betas_k_per_min : sequence of float
        Heating rates in K/min.
    """
    if model not in _G:
        raise ValueError(f"Unknown model {model!r}; choose one of {sorted(_G)}.")
    alphas = ALPHA_GRID if alphas is None else np.asarray(alphas, dtype=float)

    e_j = ea_kj * 1000.0
    grid = np.arange(t_start, t_end, d_t)
    integrand = np.exp(-e_j / (_R_TRUE * grid))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * d_t)])

    g = _G[model](alphas)
    data: Dict[str, np.ndarray] = {"alpha": alphas}
    for k, beta in enumerate(betas_k_per_min, start=1):
        target = g * (beta / 60.0) / pre_exp
        if target.max() >= cum[-1]:
            raise ValueError("Reaction does not complete below t_end; increase A or t_end.")
        data[f"T{k}"] = np.interp(target, cum, grid)
    return pd.DataFrame(data)


def simulate_variable_ea(
    ea_of_alpha_kj: Callable[[float], float],
    pre_exp: float,
    model: str = "F1",
    betas_k_per_min: Sequence[float] = DEFAULT_BETAS,
    alphas: Optional[np.ndarray] = None,
    t_start: float = 300.0,
    t_max: float = 1500.0,
    d_t: float = 0.2,
) -> pd.DataFrame:
    """
    Synthetic dataset for ``d alpha/dT = (A/beta) f(alpha) exp(-Ea(alpha)/(R T))``
    with an activation energy that varies with conversion, integrated with RK4.
    """
    if model not in _F:
        raise ValueError(f"Variable-Ea simulation supports {sorted(_F)}.")
    f = _F[model]
    alphas = ALPHA_GRID if alphas is None else np.asarray(alphas, dtype=float)
    beta_s = np.asarray(betas_k_per_min, dtype=float) / 60.0

    def rhs(T: float, a: np.ndarray) -> np.ndarray:
        ea = np.array([ea_of_alpha_kj(float(x)) for x in np.clip(a, 0.0, 1.0)]) * 1000.0
        return (pre_exp / beta_s) * f(np.clip(a, 0.0, 1.0)) * np.exp(-ea / (_R_TRUE * T))

    a = np.full(len(beta_s), 1e-9)
    T = t_start
    temps, convs = [T], [a.copy()]
    while np.min(a) < 0.995 and T < t_max:
        k1 = rhs(T, a)
        k2 = rhs(T + d_t / 2, a + d_t / 2 * k1)
        k3 = rhs(T + d_t / 2, a + d_t / 2 * k2)
        k4 = rhs(T + d_t, a + d_t * k3)
        a = a + d_t / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        T += d_t
        temps.append(T)
        convs.append(a.copy())

    temps_arr = np.asarray(temps)
    conv_arr = np.vstack(convs)
    data: Dict[str, np.ndarray] = {"alpha": alphas}
    for k in range(len(beta_s)):
        data[f"T{k + 1}"] = np.interp(alphas, conv_arr[:, k], temps_arr)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# Scenarios and tolerances
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    """One synthetic experiment and the acceptance tolerance for each method."""

    name: str
    description: str
    build: Callable[[], pd.DataFrame]
    #: Callable returning the true Ea (kJ/mol) at a given alpha.
    true_ea: Callable[[float], float]
    #: Relative tolerance (percent) per method.
    tolerance_pct: Dict[str, float] = field(default_factory=dict)
    variable: bool = False


#: Tolerances (percent of the true Ea) when Ea is constant.
#: (KAS carries a small inherent error, up to ~0.5 %, from its 1/T^2
#: approximation when Ea/RT is low, hence 1 %.)
_TOL_CONST = {
    "Friedman": 0.5, "FWO": 5.0, "KAS": 1.0, "Starink": 0.5,
    "Vyazovkin": 0.5, "KTH": 0.5,
}
#: ... and when Ea varies strongly with alpha (integral methods are
#: biased by the variation; windowed Vyazovkin, Friedman and KTH are not).
_TOL_VARIABLE = {
    "Friedman": 0.5, "FWO": 5.0, "KAS": 5.0, "Starink": 5.0,
    "Vyazovkin": 1.5, "KTH": 1.5,
}


def default_scenarios() -> List[Scenario]:
    """The five built-in ground-truth experiments."""
    return [
        Scenario(
            "F1, Ea = 150",
            "First-order reaction, 150 kJ/mol, A = 1e12 s\u207b\u00b9 (typical organic decomposition)",
            lambda: simulate_constant_ea(150.0, 1e12, "F1"),
            lambda a: 150.0, dict(_TOL_CONST),
        ),
        Scenario(
            "D3, Ea = 80",
            "3-D diffusion (Jander), 80 kJ/mol, A = 1e8 s\u207b\u00b9 (low Ea, Ea/RT \u2248 19)",
            lambda: simulate_constant_ea(80.0, 1e8, "D3"),
            lambda a: 80.0, dict(_TOL_CONST),
        ),
        Scenario(
            "R3, Ea = 250",
            "Contracting volume, 250 kJ/mol, A = 1e20 s\u207b\u00b9 (high Ea, Ea/RT \u2248 50)",
            lambda: simulate_constant_ea(250.0, 1e20, "R3"),
            lambda a: 250.0, dict(_TOL_CONST),
        ),
        Scenario(
            "F2, Ea = 55 (near-ambient start)",
            "Second order, 55 kJ/mol, A = 1e6 s\u207b\u00b9, reaction starting near 285 K "
            "(regression test for the old fixed 300 K integral limit)",
            lambda: simulate_constant_ea(55.0, 1e6, "F2"),
            lambda a: 55.0,
            {**_TOL_CONST, "FWO": 6.0},
        ),
        Scenario(
            "F1, Ea(\u03b1) = 100 \u2192 200",
            "Ea increasing linearly with conversion (multi-step behaviour)",
            lambda: simulate_variable_ea(lambda a: 100.0 + 100.0 * a, 1e10, "F1"),
            lambda a: 100.0 + 100.0 * a, dict(_TOL_VARIABLE), variable=True,
        ),
    ]


# --------------------------------------------------------------------------- #
# Running the validation
# --------------------------------------------------------------------------- #
@dataclass
class ValidationRecord:
    scenario: str
    method: str
    true_ea: float
    found_ea: float
    error_pct: float
    tolerance_pct: float
    passed: bool
    note: str = ""


@dataclass
class ValidationReport:
    records: List[ValidationRecord]
    betas: Sequence[float]
    seconds: float

    @property
    def all_passed(self) -> bool:
        return bool(self.records) and all(r.passed for r in self.records)

    @property
    def n_passed(self) -> int:
        return sum(r.passed for r in self.records)

    def for_method(self, method: str) -> List[ValidationRecord]:
        return [r for r in self.records if r.method == method]

    @property
    def methods(self) -> List[str]:
        seen: List[str] = []
        for r in self.records:
            if r.method not in seen:
                seen.append(r.method)
        return seen

    @property
    def scenarios(self) -> List[str]:
        seen: List[str] = []
        for r in self.records:
            if r.scenario not in seen:
                seen.append(r.scenario)
        return seen


def _score(
    result: Dict[float, float], scenario: Scenario, lo: float = 0.10, hi: float = 0.90
) -> "tuple[float, float, float]":
    """Return (true_ea, found_ea, error_pct) for one method's Ea(alpha)."""
    pts = [(a, e) for a, e in result.items() if lo <= a <= hi and np.isfinite(e)]
    if not pts:
        return float("nan"), float("nan"), float("nan")
    alphas = np.array([p[0] for p in pts])
    found = np.array([p[1] for p in pts])
    truth = np.array([scenario.true_ea(a) for a in alphas])
    if scenario.variable:
        err = float(np.mean(np.abs(found - truth)) / np.mean(truth) * 100.0)
        return float(np.mean(truth)), float(np.mean(found)), err
    err = float((np.mean(found) / np.mean(truth) - 1.0) * 100.0)
    return float(np.mean(truth)), float(np.mean(found)), err


def run_validation(
    scenarios: Optional[Sequence[Scenario]] = None,
    betas: Sequence[float] = DEFAULT_BETAS,
    include_coats_redfern: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
) -> ValidationReport:
    """
    Run every kinetic method on every synthetic scenario.

    Parameters
    ----------
    scenarios : sequence of Scenario, optional
        Defaults to :func:`default_scenarios`.
    betas : sequence of float
        Heating rates (K/min) used to generate *and* analyse the data.
    include_coats_redfern : bool
        Also check that Coats-Redfern identifies the true model and
        recovers its Ea on the constant-Ea F1/D3/R3/F2 scenarios.
    progress : callable, optional
        ``progress(fraction, label)`` called after every scenario.
    """
    t0 = time.perf_counter()
    scenarios = list(default_scenarios() if scenarios is None else scenarios)
    cr_truth = {"F1, Ea = 150": "F1", "D3, Ea = 80": "D3", "R3, Ea = 250": "R3"}
    records: List[ValidationRecord] = []

    for idx, sc in enumerate(scenarios):
        df = sc.build()
        for method, func in kinetics.METHODS.items():
            tol = sc.tolerance_pct.get(method, 5.0)
            try:
                true_ea, found, err = _score(func(df, list(betas)), sc)
                ok = bool(np.isfinite(err) and abs(err) <= tol)
                note = "" if np.isfinite(err) else "no finite result"
            except Exception as exc:  # noqa: BLE001
                true_ea, found, err, ok, note = float("nan"), float("nan"), float("nan"), False, str(exc)
            records.append(ValidationRecord(sc.name, method, true_ea, found, err, tol, ok, note))

        if include_coats_redfern and sc.name in cr_truth:
            model = cr_truth[sc.name]
            try:
                cr = kinetics.coats_redfern(df, list(betas))
                fit = cr["results"][model]
                true_ea = sc.true_ea(0.5)
                err = (fit["Ea"] / true_ea - 1.0) * 100.0
                ok = cr["recommended_model"] == model and abs(err) <= 1.0
                note = f"recommended: {cr['recommended_model']}"
                records.append(
                    ValidationRecord(sc.name, "Coats-Redfern", true_ea, fit["Ea"], err, 1.0, bool(ok), note)
                )
            except Exception as exc:  # noqa: BLE001
                records.append(
                    ValidationRecord(sc.name, "Coats-Redfern", sc.true_ea(0.5), float("nan"),
                                     float("nan"), 1.0, False, str(exc))
                )

        if progress is not None:
            progress((idx + 1) / len(scenarios), sc.name)

    return ValidationReport(records, betas, time.perf_counter() - t0)


def format_report(report: ValidationReport) -> str:
    """Plain-text table of a :class:`ValidationReport` (used by the CLI)."""
    lines = [
        "Isoconversional kinetics -- validation against synthetic ground truth",
        f"Heating rates: {', '.join(f'{b:g}' for b in report.betas)} K/min   "
        f"({report.seconds:.1f} s)",
        "",
        f"{'Scenario':34s}{'Method':15s}{'True Ea':>10s}{'Found':>10s}{'Error':>10s}{'Tol.':>8s}  Result",
        "-" * 96,
    ]
    for r in report.records:
        lines.append(
            f"{r.scenario:34s}{r.method:15s}{fmt_ea(r.true_ea):>10s}{fmt_ea(r.found_ea):>10s}"
            f"{fmt_pct(r.error_pct, signed=True):>10s}{('\u00b1' + format(r.tolerance_pct, 'g') + ' %'):>8s}  "
            f"{'PASS' if r.passed else 'FAIL'}{('  ' + r.note) if r.note else ''}"
        )
    lines += ["", f"{report.n_passed}/{len(report.records)} checks passed."]
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - thin CLI wrapper
    report = run_validation()
    print(format_report(report))
    return 0 if report.all_passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
