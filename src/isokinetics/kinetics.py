"""
kinetics.py
===========
Isoconversional kinetic analysis of non-isothermal data measured at
several heating rates: activation energy Ea as a function of
conversion alpha.

Implemented methods
--------------------
* Friedman (differential)
* Flynn-Wall-Ozawa (FWO)
* Kissinger-Akahira-Sunose (KAS)
* Starink
* Vyazovkin (advanced, non-linear; Vyazovkin 2001 windowed form)
* KTH (original method; discretised-Arrhenius differential approach)
* Coats-Redfern model fitting (+ reference-Ea screening of models)
* KTH full triplet (Ea, A, f(alpha)) via Malek's method

Units (consistent throughout the module)
-----------------------------------------
* Temperatures: kelvin.
* Heating rates are **given** in K/min (as typed in the GUI) and are
  converted internally to K/s before use, so that every pre-exponential
  factor A returned by this module is in s^-1.
* Activation energies are **returned** in kJ/mol; every internal
  formula works in J/mol with R = 8.314462618 J/(mol K).

Validation
----------
Every method is validated against synthetic data with a known Ea (see
``tests/test_kinetics_ground_truth.py`` and
:mod:`isokinetics.validation`).  Recovered values agree with the true
Ea to < 0.3 % for Friedman, KAS, Starink, Vyazovkin and KTH, and to
within about +/-3 % for FWO, whose Doyle approximation (the constant
1.052) is only exact near Ea/RT ~ 30-40; this bias is inherent to the
FWO method itself, not to this implementation.

Only NumPy and pandas are used (no SciPy).  The temperature integral
is evaluated with Gauss-Laguerre / Gauss-Legendre quadrature written
with NumPy, and the 1-D minimiser is a golden-section search.

All public per-alpha functions return a ``dict`` mapping
``alpha -> Ea_kJ_per_mol`` (NaN where the calculation is not possible).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Universal gas constant, J / (mol K) (CODATA 2018 exact value).
R = 8.314462618

#: Column names (temperature) in the order matching the beta list.
_T_COLUMNS = ["T1", "T2", "T3", "T4"]


def _beta_to_kelvin_per_second(beta_list: Sequence[float]) -> List[float]:
    """Convert a list of heating rates from K/min to K/s."""
    return [b / 60.0 for b in beta_list]


def _linear_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Ordinary least-squares slope of y vs x using ``numpy.polyfit``.

    Returns ``numpy.nan`` if fewer than two valid (finite) points are
    available.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return float("nan")
    slope, _intercept = np.polyfit(x_arr[mask], y_arr[mask], 1)
    return float(slope)


# --------------------------------------------------------------------------- #
# 1. Friedman method
# --------------------------------------------------------------------------- #
def friedman(df: pd.DataFrame, beta_list: Sequence[float]) -> Dict[float, float]:
    """
    Friedman differential isoconversional method.

    ``ln(beta * dalpha/dT) = ln(A * f(alpha)) - Ea / (R * T)``

    For each alpha, the four (1/T, ln(beta * dalpha/dT)) pairs (one per
    heating rate) are fit with a straight line whose slope is -Ea/R.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns alpha, T1, T2, T3, T4 (same alpha grid for all
        heating rates -- see :func:`isokinetics.data_loader.interpolate_alpha`).
    beta_list : Sequence[float]
        The four heating rates, in K/min, in the same order as columns
        T1..T4.

    Returns
    -------
    dict
        Mapping alpha -> Ea (kJ/mol).
    """
    if len(beta_list) != 4:
        raise ValueError("Friedman method requires exactly 4 heating rates.")

    betas_ks = _beta_to_kelvin_per_second(beta_list)
    alpha = df["alpha"].to_numpy(dtype=float)

    # Pre-compute dalpha/dT for every heating rate over the whole series.
    dadT = {}
    for col in _T_COLUMNS:
        T = df[col].to_numpy(dtype=float)
        # np.gradient handles non-uniform spacing correctly.
        dadT[col] = np.gradient(alpha, T)

    results: Dict[float, float] = {}
    for i, a in enumerate(alpha):
        x_vals, y_vals = [], []
        for beta, col in zip(betas_ks, _T_COLUMNS):
            T = df[col].iloc[i]
            rate = beta * dadT[col][i]
            if T > 0 and rate > 0:
                x_vals.append(1.0 / T)
                y_vals.append(np.log(rate))
        slope = _linear_slope(x_vals, y_vals)
        results[float(a)] = -slope * R / 1000.0 if np.isfinite(slope) else float("nan")

    return results


# --------------------------------------------------------------------------- #
# 2. Flynn-Wall-Ozawa (FWO)
# --------------------------------------------------------------------------- #
def fwo(df: pd.DataFrame, beta_list: Sequence[float]) -> Dict[float, float]:
    """
    Flynn-Wall-Ozawa (FWO) integral isoconversional method.

    ``ln(beta) = const - 1.052 * Ea / (R * T)``

    The constant 1.052 comes from Doyle's approximation of the
    temperature integral, which is accurate for ``Ea/(R T)`` of about
    30-40.  Outside that range FWO has a small *method* bias (about
    +3 % at ``Ea/(RT) ~ 15`` and -1 % at ``~ 60``); KAS, Starink,
    Friedman and Vyazovkin do not share it.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns alpha, T1, T2, T3, T4.
    beta_list : Sequence[float]
        The four heating rates, in K/min.

    Returns
    -------
    dict
        Mapping alpha -> Ea (kJ/mol).
    """
    if len(beta_list) != 4:
        raise ValueError("FWO method requires exactly 4 heating rates.")

    betas_ks = _beta_to_kelvin_per_second(beta_list)
    alpha = df["alpha"].to_numpy(dtype=float)

    results: Dict[float, float] = {}
    for i, a in enumerate(alpha):
        x_vals, y_vals = [], []
        for beta, col in zip(betas_ks, _T_COLUMNS):
            T = df[col].iloc[i]
            if T > 0 and beta > 0:
                x_vals.append(1.0 / T)
                y_vals.append(np.log(beta))
        slope = _linear_slope(x_vals, y_vals)
        results[float(a)] = (
            -slope * R / 1.052 / 1000.0 if np.isfinite(slope) else float("nan")
        )

    return results


# --------------------------------------------------------------------------- #
# 3. Kissinger-Akahira-Sunose (KAS)
# --------------------------------------------------------------------------- #
def kas(df: pd.DataFrame, beta_list: Sequence[float]) -> Dict[float, float]:
    """
    Kissinger-Akahira-Sunose (KAS) integral isoconversional method.

    ``ln(beta / T^2) = const - Ea / (R * T)``

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns alpha, T1, T2, T3, T4.
    beta_list : Sequence[float]
        The four heating rates, in K/min.

    Returns
    -------
    dict
        Mapping alpha -> Ea (kJ/mol).
    """
    if len(beta_list) != 4:
        raise ValueError("KAS method requires exactly 4 heating rates.")

    betas_ks = _beta_to_kelvin_per_second(beta_list)
    alpha = df["alpha"].to_numpy(dtype=float)

    results: Dict[float, float] = {}
    for i, a in enumerate(alpha):
        x_vals, y_vals = [], []
        for beta, col in zip(betas_ks, _T_COLUMNS):
            T = df[col].iloc[i]
            if T > 0 and beta > 0:
                x_vals.append(1.0 / T)
                y_vals.append(np.log(beta / T ** 2))
        slope = _linear_slope(x_vals, y_vals)
        results[float(a)] = -slope * R / 1000.0 if np.isfinite(slope) else float("nan")

    return results


# --------------------------------------------------------------------------- #
# 4. Starink
# --------------------------------------------------------------------------- #
def starink(df: pd.DataFrame, beta_list: Sequence[float]) -> Dict[float, float]:
    """
    Starink integral isoconversional method.

    ``ln(beta / T^1.92) = const - 1.0008 * Ea / (R * T)``

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns alpha, T1, T2, T3, T4.
    beta_list : Sequence[float]
        The four heating rates, in K/min.

    Returns
    -------
    dict
        Mapping alpha -> Ea (kJ/mol).
    """
    if len(beta_list) != 4:
        raise ValueError("Starink method requires exactly 4 heating rates.")

    betas_ks = _beta_to_kelvin_per_second(beta_list)
    alpha = df["alpha"].to_numpy(dtype=float)

    results: Dict[float, float] = {}
    for i, a in enumerate(alpha):
        x_vals, y_vals = [], []
        for beta, col in zip(betas_ks, _T_COLUMNS):
            T = df[col].iloc[i]
            if T > 0 and beta > 0:
                x_vals.append(1.0 / T)
                y_vals.append(np.log(beta / T ** 1.92))
        slope = _linear_slope(x_vals, y_vals)
        results[float(a)] = (
            -slope * R / 1.0008 / 1000.0 if np.isfinite(slope) else float("nan")
        )

    return results


# --------------------------------------------------------------------------- #
# Temperature-integral helpers (shared by Vyazovkin and Malek's method)
# --------------------------------------------------------------------------- #
#: Gauss-Laguerre rule used for  p(x) = int_x^inf e^-u/u^2 du
#: =  e^-x * int_0^inf e^-t / (x + t)^2 dt.  48 nodes give ~1e-14
#: relative accuracy for every x >= 2 (verified against the exact
#: closed form  p(x) = e^-x/x - E1(x)  during development).
_GL_NODES, _GL_WEIGHTS = np.polynomial.laguerre.laggauss(48)

#: Gauss-Legendre rule used for integrals over a *finite* temperature
#: window [T_lo, T_hi] of exp(-E/(R T)) (the integrand is smooth there).
_GLEG_NODES, _GLEG_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _log_logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable ``log(sum(exp(values)))`` along ``axis``."""
    values = np.asarray(values, dtype=float)
    m = np.max(values, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(values - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _log_p_exact(x: np.ndarray) -> np.ndarray:
    """
    ``log p(x)`` with ``p(x) = int_x^inf e^-u / u^2 du`` (the classical
    "temperature integral"), computed in the log domain so that very
    large x (large Ea / low T) cannot underflow.

    x >= 2 : Gauss-Laguerre quadrature (machine precision).
    0 < x < 2 : ``p(2)`` plus a log-spaced Gauss-Legendre integral of
    ``e^-u / u^2`` over [x, 2] (rare in practice: it needs Ea < 2RT).
    Non-positive / non-finite x give NaN.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.full(x.shape, np.nan)

    big = np.isfinite(x) & (x >= 2.0)
    if np.any(big):
        xb = x[big]
        terms = np.log(_GL_WEIGHTS)[None, :] - 2.0 * np.log(xb[:, None] + _GL_NODES[None, :])
        out[big] = -xb + _log_logsumexp(terms, axis=1)

    small = np.isfinite(x) & (x > 0.0) & (x < 2.0)
    if np.any(small):
        p2 = float(np.exp(_log_p_exact(np.array([2.0]))[0]))
        for idx in np.flatnonzero(small):
            xi = x[idx]
            # u = e^s  ->  integral of e^-u / u ds over s in [ln xi, ln 2]
            lo, hi = np.log(xi), np.log(2.0)
            s_nodes = 0.5 * (hi - lo) * _GLEG_NODES + 0.5 * (hi + lo)
            u = np.exp(s_nodes)
            part = 0.5 * (hi - lo) * np.sum(_GLEG_WEIGHTS * np.exp(-u) / u)
            out[idx] = np.log(p2 + part)
    return out


def _log_temperature_integral(
    e_j_per_mol: float, T_hi: np.ndarray, T_lo: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    ``log I`` for ``I(E, T_lo, T_hi) = int_{T_lo}^{T_hi} exp(-E/(R T')) dT'``,
    evaluated for each element of ``T_hi`` (kelvin).

    * ``T_lo is None`` (or <= 0): integral from absolute zero,
      ``I = (E/R) p(E/(R T_hi))`` -- exact, no arbitrary reference
      temperature (the previous implementation used a fixed 300 K lower
      limit, which gave wrong results whenever the data were measured
      near or below 300 K, or for low activation energies).
    * otherwise: integral over the finite window ``[T_lo, T_hi]`` with
      16-point Gauss-Legendre quadrature.

    Returns NaN for elements where the window is empty/reversed.
    """
    T_hi = np.atleast_1d(np.asarray(T_hi, dtype=float))
    if T_lo is None:
        x = e_j_per_mol / (R * T_hi)
        return np.log(e_j_per_mol / R) + _log_p_exact(x)

    T_lo = np.atleast_1d(np.asarray(T_lo, dtype=float))
    out = np.full(T_hi.shape, np.nan)
    valid = np.isfinite(T_hi) & np.isfinite(T_lo) & (T_hi > T_lo) & (T_lo > 0)
    if np.any(valid):
        half = 0.5 * (T_hi[valid] - T_lo[valid])
        mid = 0.5 * (T_hi[valid] + T_lo[valid])
        nodes = mid[:, None] + half[:, None] * _GLEG_NODES[None, :]
        terms = np.log(_GLEG_WEIGHTS)[None, :] - e_j_per_mol / (R * nodes)
        out[valid] = np.log(half) + _log_logsumexp(terms, axis=1)
    return out


def _temperature_integral(
    e_j_per_mol: float, T: float, T_ref: float = 0.0, n: int = 16
) -> float:
    """
    ``I(E, T) = int_{T_ref}^{T} exp(-E / (R T')) dT'``  (J/mol, K).

    ``T_ref = 0`` (default) integrates from absolute zero (exact via the
    closed form in terms of p(x)); a positive ``T_ref`` integrates over
    the finite window ``[T_ref, T]`` by Gauss-Legendre quadrature.
    ``n`` is accepted for backwards compatibility and ignored.
    Returns 0.0 if ``T <= T_ref``.
    """
    if T <= T_ref:
        return 0.0
    lo = None if T_ref <= 0 else np.array([T_ref], dtype=float)
    val = _log_temperature_integral(e_j_per_mol, np.array([T], dtype=float), lo)[0]
    return float(np.exp(val)) if np.isfinite(val) else 0.0


def _vyazovkin_log_objective(
    e_kj_per_mol: float,
    T_hi: Sequence[float],
    betas_ks: Sequence[float],
    T_lo: Optional[Sequence[float]] = None,
) -> float:
    """
    ``log( Phi(Ea) + n )`` where Vyazovkin's objective is

    ``Phi(Ea) = sum_i sum_{j != i} [I_i * beta_j] / [I_j * beta_i]
              = (sum_i I_i/beta_i) * (sum_j beta_j/I_j) - n``

    (the closed form on the right is exact and costs O(n) instead of
    O(n^2)).  Minimising ``log(Phi + n)`` is equivalent to minimising
    Phi, but stays finite for extreme Ea where Phi itself would
    overflow.  Returns ``inf`` for a degenerate (empty/invalid) window.
    """
    T_hi = np.asarray(T_hi, dtype=float)
    beta = np.asarray(betas_ks, dtype=float)
    lo = None if T_lo is None else np.asarray(T_lo, dtype=float)
    log_i = _log_temperature_integral(e_kj_per_mol * 1000.0, T_hi, lo)
    if not np.all(np.isfinite(log_i)):
        return float("inf")
    a = _log_logsumexp(log_i - np.log(beta))
    b = _log_logsumexp(np.log(beta) - log_i)
    return float(a + b)


def _vyazovkin_objective(
    e_kj_per_mol: float,
    temperatures: Sequence[float],
    betas_ks: Sequence[float],
    T_lo: Optional[Sequence[float]] = None,
) -> float:
    """
    Vyazovkin's objective ``Phi(Ea)`` for one alpha (see
    :func:`_vyazovkin_log_objective`).  Its minimum value is
    ``n (n - 1)`` and is reached at the activation energy that makes the
    temperature integrals of all heating rates consistent.

    The ``beta_j / beta_i`` weighting is essential (Vyazovkin, J.
    Comput. Chem. 18 (1997) 393): without it the ratios are dominated
    by whichever heating rate has the largest T and there is no
    interior minimum.

    Returns ``inf`` for a degenerate input.
    """
    log_phi_plus_n = _vyazovkin_log_objective(e_kj_per_mol, temperatures, betas_ks, T_lo)
    if not np.isfinite(log_phi_plus_n) or log_phi_plus_n > 700:
        return float("inf")
    return float(np.exp(log_phi_plus_n) - len(temperatures))


def _golden_section_search(
    func, low: float, high: float, tol: float = 0.1, max_iter: int = 200
) -> float:
    """
    Minimise a unimodal scalar function on ``[low, high]`` using a
    golden-section search (SciPy-free replacement for
    ``scipy.optimize.minimize_scalar``).

    Parameters
    ----------
    func : Callable[[float], float]
        The (assumed unimodal) function to minimise.
    low, high : float
        Bracket in which to search.
    tol : float, optional
        Stop once the bracket width is below this tolerance.
    max_iter : int, optional
        Safety cap on the number of iterations.

    Returns
    -------
    float
        The value of x that (approximately) minimises ``func``.
    """
    invphi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618

    a, b = float(low), float(high)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = func(c), func(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = func(d)

    return (a + b) / 2.0


#: Hard limits (kJ/mol) between which the Vyazovkin search bracket may
#: be automatically widened.
_VYAZ_HARD_MIN = 1.0
_VYAZ_HARD_MAX = 2000.0


def _minimise_with_expansion(func, low: float, high: float, tol: float) -> float:
    """
    Golden-section minimisation that **widens its bracket** when the
    minimum lands on an edge, instead of silently returning the edge
    value (the previous behaviour clipped any Ea above 500 kJ/mol to
    ~500).  Returns NaN if the minimum is still on a hard limit, i.e.
    there is no interior minimum (degenerate / non-physical data).
    """
    lo, hi = float(low), float(high)
    x = float("nan")
    for _ in range(8):
        x = _golden_section_search(func, lo, hi, tol=tol)
        margin = max(5.0 * tol, 0.01 * (hi - lo))
        if x > hi - margin:
            if hi >= _VYAZ_HARD_MAX:
                return float("nan")
            hi = min(_VYAZ_HARD_MAX, hi * 2.0)
        elif x < lo + margin:
            if lo <= _VYAZ_HARD_MIN:
                return float("nan")
            lo = max(_VYAZ_HARD_MIN, lo / 2.0)
        else:
            return float(x)
    return float("nan")


def vyazovkin(
    df: pd.DataFrame,
    beta_list: Sequence[float],
    e_min: float = 30.0,
    e_max: float = 500.0,
    tol: float = 0.01,
    advanced: bool = True,
) -> Dict[float, float]:
    """
    Vyazovkin's non-linear isoconversional method.

    For every alpha the activation energy minimises

    ``Phi(Ea) = sum_i sum_{j != i} [J(Ea, T_i) / beta_i] / [J(Ea, T_j) / beta_j]``

    where ``J`` is the temperature integral of ``exp(-Ea / (R T))``.

    * ``advanced=True`` (default): the **modified method of Vyazovkin
      (2001)** -- ``J`` is integrated only over the conversion window
      ``[alpha - d_alpha, alpha]`` (i.e. between the temperatures at
      the previous and the current grid point).  This removes the
      systematic error that the whole-range integral produces when
      Ea varies with alpha.  For the first grid point the window starts
      at absolute zero (equivalent to the onset of the reaction).
    * ``advanced=False``: the original 1997 form, integrating from
      absolute zero up to ``T_alpha``.

    Both integrals are evaluated by quadrature (no fixed reference
    temperature), the objective is minimised by golden-section search,
    and the search bracket is widened automatically if the minimum is
    not inside ``[e_min, e_max]``.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns alpha, T1, T2, T3, T4 (uniform alpha grid, see
        :func:`isokinetics.data_loader.interpolate_alpha`).
    beta_list : Sequence[float]
        The four heating rates, in K/min.
    e_min, e_max : float, optional
        Initial search bracket for Ea, in kJ/mol.
    tol : float, optional
        Convergence tolerance of the search, in kJ/mol.
    advanced : bool, optional
        Use the windowed (2001) formulation (default) or the original.

    Returns
    -------
    dict
        Mapping alpha -> Ea (kJ/mol); NaN where the window is not
        physically valid (non-increasing temperatures) or no interior
        minimum exists.
    """
    if len(beta_list) != 4:
        raise ValueError("Vyazovkin method requires exactly 4 heating rates.")

    betas_ks = np.asarray(_beta_to_kelvin_per_second(beta_list), dtype=float)
    alpha = df["alpha"].to_numpy(dtype=float)
    temps = df[_T_COLUMNS].to_numpy(dtype=float)
    results: Dict[float, float] = {}

    for i, a in enumerate(alpha):
        T_hi = temps[i]
        T_lo = temps[i - 1] if (advanced and i > 0) else None
        try:
            if T_lo is not None and not np.all(T_hi > T_lo):
                results[float(a)] = float("nan")
                continue
            ea = _minimise_with_expansion(
                lambda e: _vyazovkin_log_objective(e, T_hi, betas_ks, T_lo),
                e_min,
                e_max,
                tol,
            )
            results[float(a)] = float(ea)
        except Exception:  # noqa: BLE001 - keep other alphas usable on failure
            logger.debug("Vyazovkin failed at alpha=%s", a, exc_info=True)
            results[float(a)] = float("nan")

    return results


# --------------------------------------------------------------------------- #
# 6. KTH (Kadri-Trache-Hassam) -- discretised-Arrhenius differential method
# --------------------------------------------------------------------------- #
def _kth_regression(
    df: pd.DataFrame, beta_list: Sequence[float]
) -> "tuple[Dict[float, float], Dict[float, float]]":
    """
    Shared regression core for :func:`kth` and :func:`kth_full`.

    For each alpha step, regresses ``ln(rate)`` against ``1/T_mean``
    across heating rates (see :func:`kth` for the full derivation) and
    returns *both* results of that regression:

    * the slope, converted to an activation energy Ea(alpha) (kJ/mol);
    * the intercept, converted to ``Phi(alpha) = exp(intercept) =
      A(alpha) * f(alpha)`` (s^-1) -- the product of the pre-exponential
      factor and the (still unknown, at this stage) reaction model.

    :func:`kth` only exposes the first of these (to stay drop-in
    compatible with the other methods in this module); :func:`kth_full`
    uses both, plus a Malek-method model identification step, to also
    recover A(alpha) and f(alpha).

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns alpha, T1, T2, T3, T4.
    beta_list : Sequence[float]
        The four heating rates, in K/min.

    Returns
    -------
    tuple of dict
        ``(ea_kj_per_mol, phi_per_second)``, both mapping alpha -> value
        (NaN where fewer than 3 valid heating-rate pairs were available).
    """
    if len(beta_list) != 4:
        raise ValueError("KTH method requires exactly 4 heating rates.")

    betas_ks = _beta_to_kelvin_per_second(beta_list)
    alpha = df["alpha"].to_numpy(dtype=float)
    n = len(alpha)

    logger.debug("KTH: starting computation for %d alpha points, betas=%s", n, beta_list)

    ea_results: Dict[float, float] = {}
    phi_results: Dict[float, float] = {}
    n_skipped = 0

    for i in range(n):
        # Step to the right for every point except the last, which has
        # no "next" point -- step to the left instead.
        if i < n - 1:
            i0, i1 = i, i + 1
        else:
            i0, i1 = i - 1, i

        d_alpha = alpha[i1] - alpha[i0]

        x_vals, y_vals = [], []
        for beta, col in zip(betas_ks, _T_COLUMNS):
            T0 = df[col].iloc[i0]
            T1 = df[col].iloc[i1]
            d_t = (T1 - T0) / beta
            T_mean = (T0 + T1) / 2.0

            if d_t > 0 and T_mean > 0 and d_alpha > 0:
                rate = d_alpha / d_t
                if rate > 0:
                    x_vals.append(1.0 / T_mean)
                    y_vals.append(np.log(rate))

        # Per the method's validation requirements: at least 3 valid
        # (heating rate, temperature) pairs are required for a
        # meaningful regression; fewer than that yields NaN rather than
        # an unreliable fit.
        if len(x_vals) < 3:
            n_skipped += 1
            logger.debug(
                "KTH: alpha=%.4f has only %d valid heating-rate pairs "
                "(need >= 3); returning NaN.",
                alpha[i],
                len(x_vals),
            )
            ea_results[float(alpha[i])] = float("nan")
            phi_results[float(alpha[i])] = float("nan")
            continue

        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        slope, intercept = np.polyfit(x_arr, y_arr, 1)

        ea_results[float(alpha[i])] = (
            -slope * R / 1000.0 if np.isfinite(slope) else float("nan")
        )
        phi_results[float(alpha[i])] = (
            float(np.exp(intercept)) if np.isfinite(intercept) else float("nan")
        )

    if n_skipped:
        logger.debug("KTH: %d/%d alpha points returned NaN.", n_skipped, n)
    logger.debug("KTH: finished, average Ea = %.2f kJ/mol", average_ea(ea_results))

    return ea_results, phi_results


def kth(df: pd.DataFrame, beta_list: Sequence[float]) -> Dict[float, float]:
    """
    KTH method: an original isoconversional approach based on a
    discretised form of the Arrhenius equation.

    ``alpha_{i+1} = alpha_i + Phi(alpha_i) * exp(-Ea_i / (R*T_mean,i)) * dt_i``

    Taking the natural logarithm linearises this to::

        ln((alpha_{i+1} - alpha_i) / dt_i) = ln(Phi(alpha_i)) - Ea_i / (R * T_mean,i)

    i.e. ``Y = a*X + b`` with ``Y = ln(rate)``, ``X = 1/T_mean``, and
    slope ``a = -Ea/R``. Where this differs from :func:`friedman`:

    * The reaction rate is estimated as a **discrete step** rate,
      ``(alpha_{i+1} - alpha_i) / dt_i``, over the interval between two
      consecutive grid points on the alpha axis, rather than an
      instantaneous derivative ``dalpha/dT`` at a single point.
    * It is paired with the **mean temperature over that same interval**,
      ``T_mean,i = (T_i + T_{i+1}) / 2``, rather than the temperature at
      a single endpoint.
    * The time step is recovered from the temperature step under the
      linear-heating assumption ``T = T0 + beta*t``, i.e.
      ``dt_i = (T_{i+1} - T_i) / beta`` -- no explicit time column is
      required in the input data.

    As with the other methods here, every heating rate contributes one
    ``(1/T_mean, ln(rate))`` pair for a given alpha step, and a linear
    regression across those points (one per heating rate) gives the
    slope ``-Ea/R``.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns alpha, T1, T2, T3, T4 (same alpha grid for all
        heating rates -- see
        :func:`isokinetics.data_loader.interpolate_alpha`).
    beta_list : Sequence[float]
        The four heating rates, in K/min, in the same order as columns
        T1..T4.

    Returns
    -------
    dict
        Mapping alpha -> Ea (kJ/mol). Every alpha in ``df`` is present:
        each point uses the discrete step to its right (alpha_i to
        alpha_{i+1}) to estimate its rate, except the last point, which
        uses the step to its left (alpha_{i-1} to alpha_i) since there
        is no "next" point to step to.

    Notes
    -----
    This returns only Ea(alpha), so it is drop-in compatible with the
    other methods in this module (all consumed the same way by the
    GUI/plotting/export pipeline, which expects a single ``{alpha: Ea}``
    mapping per method). For the full kinetic triplet -- also recovering
    the pre-exponential factor A(alpha) and the experimental reaction
    model f(alpha) via Malek's method -- see :func:`kth_full`.
    """
    ea_results, _phi_results = _kth_regression(df, beta_list)
    return ea_results


# --------------------------------------------------------------------------- #
# 6b. KTH full triplet: Ea(alpha), A(alpha), f(alpha) via Malek's method
# --------------------------------------------------------------------------- #

#: Candidate theoretical solid-state reaction models used for Malek-method
#: model identification (a standard subset of Table IV.1 in the thesis,
#: covering the four classical mechanistic families). Each entry maps a
#: model code to (f(alpha), g(alpha)) -- the differential and integral
#: forms of the reaction model, respectively.
KTH_CANDIDATE_MODELS: List[str] = [
    "F0", "F1", "F2", "F3",   # reaction order
    "R2", "R3",               # geometric contraction (contracting area/volume)
    "D1", "D2", "D3", "D4",   # diffusion
    "A2", "A3", "A4",         # Avrami-Erofeev (nucleation and growth)
]

#: Human-readable descriptions, purely for display in the GUI/exports.
KTH_MODEL_DESCRIPTIONS: Dict[str, str] = {
    "F0": "Zero-order reaction",
    "F1": "First-order reaction",
    "F2": "Second-order reaction",
    "F3": "Third-order reaction",
    "R2": "Contracting area (geometric contraction)",
    "R3": "Contracting volume (geometric contraction)",
    "D1": "One-dimensional diffusion",
    "D2": "Two-dimensional diffusion (Valensi)",
    "D3": "Three-dimensional diffusion (Jander)",
    "D4": "Three-dimensional diffusion (Ginstling-Brounshtein)",
    "A2": "Avrami-Erofeev, n=2 (nucleation and growth)",
    "A3": "Avrami-Erofeev, n=3 (nucleation and growth)",
    "A4": "Avrami-Erofeev, n=4 (nucleation and growth)",
}


def _model_f_g(name: str, alpha: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """
    Evaluate a named theoretical solid-state reaction model's
    differential and integral forms, f(alpha) and g(alpha), over an
    array of alpha values.

    This is a standard subset of the classical solid-state kinetic
    models (reaction-order Fn, geometric-contraction Rn, diffusion Dn,
    and Avrami-Erofeev nucleation-and-growth An), corresponding to the
    same four mechanistic families as Table IV.1 in the thesis.

    Parameters
    ----------
    name : str
        One of :data:`KTH_CANDIDATE_MODELS`.
    alpha : numpy.ndarray
        Conversion values, clipped away from the 0/1 boundaries
        internally to avoid singularities (several models have f(alpha)
        or g(alpha) diverge at alpha=0 or alpha=1).

    Returns
    -------
    tuple of numpy.ndarray
        ``(f_alpha, g_alpha)``, each the same shape as ``alpha``.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised model code.
    """
    a = np.clip(np.asarray(alpha, dtype=float), 1e-6, 1.0 - 1e-6)

    if name == "F0":
        f = np.ones_like(a)
        g = a
    elif name == "F1":
        f = 1.0 - a
        g = -np.log(1.0 - a)
    elif name == "F2":
        f = (1.0 - a) ** 2
        g = (1.0 / (1.0 - a)) - 1.0
    elif name == "F3":
        f = (1.0 - a) ** 3
        g = 0.5 * ((1.0 / (1.0 - a)) ** 2 - 1.0)
    elif name == "R2":
        f = 2.0 * (1.0 - a) ** 0.5
        g = 1.0 - (1.0 - a) ** 0.5
    elif name == "R3":
        f = 3.0 * (1.0 - a) ** (2.0 / 3.0)
        g = 1.0 - (1.0 - a) ** (1.0 / 3.0)
    elif name == "D1":
        f = 1.0 / (2.0 * a)
        g = a ** 2
    elif name == "D2":
        f = 1.0 / (-np.log(1.0 - a))
        g = (1.0 - a) * np.log(1.0 - a) + a
    elif name == "D3":
        f = 1.5 * (1.0 - a) ** (2.0 / 3.0) / (1.0 - (1.0 - a) ** (1.0 / 3.0))
        g = (1.0 - (1.0 - a) ** (1.0 / 3.0)) ** 2
    elif name == "D4":
        f = 1.5 / ((1.0 - a) ** (-1.0 / 3.0) - 1.0)
        g = 1.0 - (2.0 * a / 3.0) - (1.0 - a) ** (2.0 / 3.0)
    elif name == "A2":
        f = 2.0 * (1.0 - a) * (-np.log(1.0 - a)) ** 0.5
        g = (-np.log(1.0 - a)) ** 0.5
    elif name == "A3":
        f = 3.0 * (1.0 - a) * (-np.log(1.0 - a)) ** (2.0 / 3.0)
        g = (-np.log(1.0 - a)) ** (1.0 / 3.0)
    elif name == "A4":
        f = 4.0 * (1.0 - a) * (-np.log(1.0 - a)) ** (3.0 / 4.0)
        g = (-np.log(1.0 - a)) ** (1.0 / 4.0)
    elif name == "F4":
        f = (1.0 - a) ** 4
        g = ((1.0 / (1.0 - a)) ** 3 - 1.0) / 3.0
    elif name == "D5":
        # "Anti-Jander" 3D diffusion variant (thesis Table IV.1, row 26).
        f = 1.5 * (1.0 + a) ** (2.0 / 3.0) / ((1.0 + a) ** (1.0 / 3.0) - 1.0)
        g = ((1.0 + a) ** (1.0 / 3.0) - 1.0) ** 2
    elif name in ("P1", "P2", "P3", "P4"):
        n = {"P1": 1.5, "P2": 2.0, "P3": 3.0, "P4": 4.0}[name]
        f = n * a ** (1.0 - 1.0 / n)
        g = a ** (1.0 / n)
    else:
        raise ValueError(f"Unknown KTH candidate model: {name!r}")

    return f, g


def _normalize_01(values: np.ndarray) -> np.ndarray:
    """
    Min-max normalise an array to the range [0, 1] (matching the
    thesis's instruction that y(alpha) and z(alpha) be "normalisees
    entre 0 et 1"), ignoring NaNs when computing the min/max.

    Returns an all-zero array (same shape) if the input is constant or
    has no finite values, rather than dividing by zero.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan)
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax - vmin < 1e-300:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _temperature_integral_p(x: np.ndarray, span: float = 60.0, n: int = 2000) -> np.ndarray:
    """
    The temperature integral ``p(x) = int_x^inf e^-u / u^2 du`` for each
    value in ``x`` (``x = Ea / (R T)``, strictly positive).

    Evaluated to ~1e-14 relative accuracy with Gauss-Laguerre quadrature
    (see :func:`_log_p_exact`); the exact closed form is
    ``p(x) = e^-x / x - E1(x)``.  ``span`` and ``n`` are accepted only for
    backwards compatibility with the previous trapezoidal implementation
    (which was accurate to ~1e-4) and are ignored.

    Returns
    -------
    numpy.ndarray
        p(x), same shape as the input (NaN for non-positive x).
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(_log_p_exact(x_arr))


def _malek_y_z_curves(
    df: pd.DataFrame, beta_list: Sequence[float], ea_kj_per_mol: float
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Compute Malek's y(alpha) and z(alpha) diagnostic functions from the
    experimental data, averaged across all heating rates (each
    heating rate's curve is normalised to [0, 1] individually before
    averaging, so no single heating rate dominates the result).

    ``y(alpha) = (dalpha/dt) * exp(Ea / (R*T))``
    ``z(alpha) = (dalpha/dt) * (T/beta) * x * exp(x) * p(x)``, ``x = Ea/(R*T)``

    The z(alpha) form used here is derived from first principles (see
    the module-level notes in :func:`kth_full`) to be *exactly*
    proportional to the theoretical ``f(alpha)*g(alpha)`` combined
    master-plot function -- this was verified numerically against
    synthetic ground-truth data during development (a version without
    the ``x*exp(x)`` factor, closer to a literal reading of the
    thesis's stated equation, was found to misidentify the reaction
    model on synthetic F1 data and was rejected).

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns alpha, T1, T2, T3, T4.
    beta_list : Sequence[float]
        The four heating rates, in K/min.
    ea_kj_per_mol : float
        A single representative activation energy (typically the
        average of :func:`kth`'s Ea(alpha) over the identification
        window) used to evaluate x = Ea/(RT) -- Malek's method assumes
        Ea is approximately constant over that window.

    Returns
    -------
    tuple of numpy.ndarray
        ``(y_alpha, z_alpha)``, each normalised to [0, 1] and the same
        length as ``df``.
    """
    betas_ks = _beta_to_kelvin_per_second(beta_list)
    alpha = df["alpha"].to_numpy(dtype=float)
    ea_j = ea_kj_per_mol * 1000.0

    y_curves = []
    z_curves = []
    for beta, col in zip(betas_ks, _T_COLUMNS):
        T = df[col].to_numpy(dtype=float)
        dadT = np.gradient(alpha, T)
        dadt = beta * dadT

        x = ea_j / (R * T)
        y = dadt * np.exp(x)

        p = _temperature_integral_p(x)
        z = dadT * T * x * np.exp(x) * p

        y_curves.append(_normalize_01(y))
        z_curves.append(_normalize_01(z))

    y_exp = _normalize_01(np.nanmean(np.vstack(y_curves), axis=0))
    z_exp = _normalize_01(np.nanmean(np.vstack(z_curves), axis=0))
    return y_exp, z_exp


def _identify_model(
    alpha: np.ndarray,
    y_exp: np.ndarray,
    z_exp: np.ndarray,
    identification_window: "tuple[float, float]" = (0.2, 0.8),
    ambiguity_score_threshold: float = 0.30,
    ambiguity_margin_ratio: float = 1.15,
) -> "tuple[str, Dict[str, float]]":
    """
    Identify the best-matching theoretical reaction model by comparing
    the experimental y(alpha)/z(alpha) curves against every candidate
    model's theoretical f(alpha) and f(alpha)*g(alpha) curves (each
    normalised to [0, 1] the same way), restricted to the alpha window
    where Malek's method is valid (Ea approximately constant -- the
    thesis states this should hold for a variation under ~10% within
    alpha in [0.2, 0.8]).

    The match score for each candidate is the sum of the RMSE between
    the experimental and theoretical curves for both y and z; the model
    with the lowest score wins.

    Parameters
    ----------
    alpha : numpy.ndarray
        Conversion grid.
    y_exp, z_exp : numpy.ndarray
        Experimental, normalised Malek diagnostic curves (see
        :func:`_malek_y_z_curves`).
    identification_window : tuple of float, optional
        Restrict scoring to this alpha range (default (0.2, 0.8), per
        the thesis's stated validity range for Malek's method).
    ambiguity_score_threshold : float, optional
        If the best score exceeds this, identification is considered
        too poor to trust and ``'unknown'`` is returned.
    ambiguity_margin_ratio : float, optional
        If the second-best score is less than this ratio times the
        best score (i.e. the two best candidates are nearly tied),
        identification is considered ambiguous and ``'unknown'`` is
        returned instead of an arbitrary tie-break.

    Returns
    -------
    tuple
        ``(model_name, scores)`` where ``model_name`` is a code from
        :data:`KTH_CANDIDATE_MODELS` or ``'unknown'``, and ``scores`` is
        a dict mapping every candidate model to its match score (lower
        is better; useful for diagnostics/plotting).
    """
    lo, hi = identification_window
    mask = (alpha >= lo) & (alpha <= hi)
    if mask.sum() < 3:
        # Identification window too narrow for this data -- fall back
        # to using every available point rather than failing outright.
        mask = np.ones_like(alpha, dtype=bool)

    scores: Dict[str, float] = {}
    for name in KTH_CANDIDATE_MODELS:
        f_th, g_th = _model_f_g(name, alpha)
        f_th_n = _normalize_01(f_th)
        fg_th_n = _normalize_01(f_th * g_th)

        valid = (
            mask
            & np.isfinite(y_exp)
            & np.isfinite(z_exp)
            & np.isfinite(f_th_n)
            & np.isfinite(fg_th_n)
        )
        if valid.sum() < 3:
            continue

        rmse_y = float(np.sqrt(np.mean((y_exp[valid] - f_th_n[valid]) ** 2)))
        rmse_z = float(np.sqrt(np.mean((z_exp[valid] - fg_th_n[valid]) ** 2)))
        score = rmse_y + rmse_z
        if np.isfinite(score):
            scores[name] = score
            logger.debug("KTH full: candidate model=%s score=%.4f", name, score)

    finite_scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite_scores:
        logger.debug("KTH full: no candidate model could be scored; model=unknown.")
        return "unknown", scores

    ranked = sorted(finite_scores.items(), key=lambda kv: kv[1])
    best_name, best_score = ranked[0]

    if best_score > ambiguity_score_threshold:
        logger.debug(
            "KTH full: best model=%s score=%.4f exceeds ambiguity threshold "
            "%.2f; model=unknown.",
            best_name,
            best_score,
            ambiguity_score_threshold,
        )
        return "unknown", scores

    if len(ranked) > 1:
        second_score = ranked[1][1]
        if second_score < ambiguity_margin_ratio * best_score:
            logger.debug(
                "KTH full: best model=%s (score=%.4f) and second-best %s "
                "(score=%.4f) are too close to distinguish; model=unknown.",
                best_name,
                best_score,
                ranked[1][0],
                second_score,
            )
            return "unknown", scores

    logger.debug("KTH full: identified model=%s (score=%.4f).", best_name, best_score)
    return best_name, scores


def kth_full(
    df: pd.DataFrame,
    beta_list: Sequence[float],
    identification_window: "tuple[float, float]" = (0.2, 0.8),
) -> Dict[str, object]:
    """
    Full KTH kinetic triplet: activation energy Ea(alpha), identified
    reaction model f(alpha), and pre-exponential factor A(alpha), using
    Malek's method for the model-identification step (thesis, Chapter
    VII, section VII.4.1.3).

    Procedure
    ---------
    1. Run the same regression as :func:`kth` to get Ea(alpha) and
       ``Phi(alpha) = A(alpha)*f(alpha)``.
    2. Compute Malek's y(alpha) and z(alpha) diagnostic curves from the
       experimental data (:func:`_malek_y_z_curves`), using the average
       Ea over the identification window as a single representative
       activation energy (Malek's method assumes Ea is approximately
       constant there).
    3. Compare the experimental curves against every candidate
       theoretical model's f(alpha) and f(alpha)*g(alpha) curves
       (:func:`_identify_model`); the best match (if any is good enough
       and unambiguous) identifies the reaction model.
    4. If a model was identified, compute its theoretical f(alpha) over
       the full alpha range and recover ``A(alpha) = Phi(alpha) /
       f(alpha)``.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns alpha, T1, T2, T3, T4.
    beta_list : Sequence[float]
        The four heating rates, in K/min.
    identification_window : tuple of float, optional
        Alpha range used for the model-identification step (default
        (0.2, 0.8), matching the thesis's stated validity range).

    Returns
    -------
    dict
        ``{"Ea": {alpha: kJ/mol}, "A": {alpha: s^-1}, "f_alpha":
        {alpha: value}, "model": model_code_or_'unknown', "model_scores":
        {model_code: score}, "y_alpha": {alpha: value}, "z_alpha":
        {alpha: value}, "y_theory": {alpha: value}, "z_theory":
        {alpha: value}}``. ``y_theory``/``z_theory`` are the identified
        model's own normalised curves (for overlaying against
        ``y_alpha``/``z_alpha`` on a diagnostic plot). When no model
        could be confidently identified, ``"model"`` is ``"unknown"``
        and ``"A"``, ``"f_alpha"``, ``"y_theory"``, and ``"z_theory"``
        are all filled with NaN (Ea, y_alpha, and z_alpha are always
        populated regardless).
    """
    ea_results, phi_results = _kth_regression(df, beta_list)
    alpha = df["alpha"].to_numpy(dtype=float)

    lo, hi = identification_window
    window_eas = [
        e for a, e in ea_results.items() if np.isfinite(e) and lo <= a <= hi
    ]
    ea_avg = float(np.mean(window_eas)) if window_eas else average_ea(ea_results)

    if not np.isfinite(ea_avg):
        logger.debug("KTH full: no finite Ea available; cannot run Malek identification.")
        nan_map = {float(a): float("nan") for a in alpha}
        return {
            "Ea": ea_results,
            "A": dict(nan_map),
            "f_alpha": dict(nan_map),
            "model": "unknown",
            "model_scores": {},
            "y_alpha": dict(nan_map),
            "z_alpha": dict(nan_map),
            "y_theory": dict(nan_map),
            "z_theory": dict(nan_map),
        }

    y_exp, z_exp = _malek_y_z_curves(df, beta_list, ea_avg)
    model_name, scores = _identify_model(alpha, y_exp, z_exp, identification_window)

    if model_name != "unknown":
        f_theory, g_theory = _model_f_g(model_name, alpha)
        f_alpha_dict = {float(a): float(fv) for a, fv in zip(alpha, f_theory)}
        A_dict: Dict[float, float] = {}
        for a, fv in zip(alpha, f_theory):
            a_key = float(a)
            phi_v = phi_results.get(a_key, float("nan"))
            A_dict[a_key] = (
                phi_v / fv if (np.isfinite(phi_v) and fv > 0) else float("nan")
            )
        y_theory_n = _normalize_01(f_theory)
        z_theory_n = _normalize_01(f_theory * g_theory)
        y_theory_dict = {float(a): float(v) for a, v in zip(alpha, y_theory_n)}
        z_theory_dict = {float(a): float(v) for a, v in zip(alpha, z_theory_n)}
    else:
        f_alpha_dict = {float(a): float("nan") for a in alpha}
        A_dict = {float(a): float("nan") for a in alpha}
        y_theory_dict = {float(a): float("nan") for a in alpha}
        z_theory_dict = {float(a): float("nan") for a in alpha}

    return {
        "Ea": ea_results,
        "A": A_dict,
        "f_alpha": f_alpha_dict,
        "model": model_name,
        "model_scores": scores,
        "y_alpha": {float(a): float(v) for a, v in zip(alpha, y_exp)},
        "z_alpha": {float(a): float(v) for a, v in zip(alpha, z_exp)},
        "y_theory": y_theory_dict,
        "z_theory": z_theory_dict,
    }


#: Registry mapping a human-readable method name to its implementation.
#: Used by the GUI to iterate over the methods the user has selected.
METHODS = {
    "Friedman": friedman,
    "FWO": fwo,
    "KAS": kas,
    "Starink": starink,
    "Vyazovkin": vyazovkin,
    "KTH": kth,
}


def average_ea(results: Dict[float, float]) -> float:
    """
    Compute the mean activation energy over all alpha values, ignoring
    NaNs.

    Parameters
    ----------
    results : dict
        Mapping alpha -> Ea (kJ/mol), as returned by any method above.

    Returns
    -------
    float
        Mean Ea in kJ/mol, or ``numpy.nan`` if no finite values exist.
    """
    values = np.array(list(results.values()), dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


# --------------------------------------------------------------------------- #
# 7. Coats-Redfern (model-fitting integral method) + model ranking
# --------------------------------------------------------------------------- #

#: Candidate models offered to Coats-Redfern (a superset of
#: KTH_CANDIDATE_MODELS: also includes F4, D5, and the P1-P4 power-law
#: family). Kept separate from KTH_CANDIDATE_MODELS so that KTH's
#: already-validated default behaviour is unaffected by this addition.
COATS_REDFERN_MODELS: List[str] = KTH_CANDIDATE_MODELS + ["F4", "D5", "P1", "P2", "P3", "P4"]


def _coats_redfern_single_model(
    df: pd.DataFrame, betas_ks: Sequence[float], model_name: str
) -> Dict[str, float]:
    """
    Fit a single candidate model with the Coats-Redfern method,
    separately for each heating rate, then average the results.

    ``ln(g(alpha)/T^2) = ln[(A*R/(beta*Ea)) * (1 - 2RT/Ea)] - Ea/(R*T)``

    Coats-Redfern is classically a *single*-heating-rate method; with
    multi-rate data (as used throughout this module) it is applied once
    per heating rate and the four results are averaged, which is the
    standard way to make use of the extra data while staying faithful
    to the original method.

    Returns
    -------
    dict
        ``{"Ea": kJ/mol, "A": s^-1, "r2": float, "aic": float, "bic": float}``,
        each averaged across heating rates (NaN if the model could not
        be fit for any heating rate, e.g. a non-positive or non-finite
        g(alpha)).
    """
    alpha = df["alpha"].to_numpy(dtype=float)
    f_theory, g_theory = _model_f_g(model_name, alpha)

    ea_list, a_list, r2_list, aic_list, bic_list = [], [], [], [], []

    for beta, col in zip(betas_ks, _T_COLUMNS):
        T = df[col].to_numpy(dtype=float)
        mask = np.isfinite(T) & (T > 0) & np.isfinite(g_theory) & (g_theory > 0)
        if mask.sum() < 3:
            continue

        Tm = T[mask]
        x = 1.0 / Tm
        y_base = np.log(g_theory[mask] / Tm ** 2)
        if not (np.isfinite(x).all() and np.isfinite(y_base).all()):
            continue

        # Coats-Redfern:  ln(g/T^2) = ln[(A R / (beta E)) (1 - 2RT/E)] - E/(RT).
        # The (1 - 2RT/E) factor is usually dropped (it is ~0.93-0.98 for
        # typical data) but doing so biases A low by that factor and Ea
        # slightly; it is restored here by a short fixed-point iteration
        # that starts from the plain (factor = 1) fit.
        y = y_base
        slope, intercept = np.polyfit(x, y, 1)
        for _ in range(5):
            ea_guess = -slope * R
            if not (ea_guess > 0 and np.isfinite(ea_guess)):
                break
            corr = 1.0 - 2.0 * R * Tm / ea_guess
            if np.any(corr <= 0):
                break
            y = y_base - np.log(corr)
            new_slope, new_intercept = np.polyfit(x, y, 1)
            converged = abs(new_slope - slope) < 1e-9 * abs(slope)
            slope, intercept = new_slope, new_intercept
            if converged:
                break

        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        ea_j = -slope * R
        if ea_j <= 0 or not np.isfinite(ea_j):
            continue
        # intercept = ln(A R / (beta E)) with beta in K/s  ->  A in s^-1
        a_val = float(np.exp(intercept) * beta * ea_j / R)

        n = mask.sum()
        k = 2  # slope + intercept
        mean_sq_err = ss_res / n if n > 0 else float("nan")
        aic = n * np.log(mean_sq_err) + 2 * k if mean_sq_err > 0 else float("nan")
        bic = n * np.log(mean_sq_err) + k * np.log(n) if mean_sq_err > 0 else float("nan")

        ea_list.append(ea_j / 1000.0)
        a_list.append(a_val)
        r2_list.append(r2)
        aic_list.append(aic)
        bic_list.append(bic)

    if not ea_list:
        return {"Ea": float("nan"), "A": float("nan"), "r2": float("nan"),
                "aic": float("nan"), "bic": float("nan")}

    return {
        "Ea": float(np.mean(ea_list)),
        "A": float(np.mean(a_list)),
        "r2": float(np.mean(r2_list)),
        "aic": float(np.mean(aic_list)),
        "bic": float(np.mean(bic_list)),
    }


def coats_redfern(
    df: pd.DataFrame,
    beta_list: Sequence[float],
    model_names: Optional[Sequence[str]] = None,
    reference_ea_kj: Optional[float] = None,
) -> Dict[str, object]:
    """
    Coats-Redfern integral (model-fitting) method: fit every candidate
    reaction model to the data and rank them to recommend a best fit.

    Important limitation (why this needs a reference Ea)
    -------------------------------------------------------
    R^2 alone is well known to be an unreliable way to pick a kinetic
    model with fitting methods like Coats-Redfern: several *wrong*
    models routinely achieve R^2 extremely close to 1.0 with an
    unphysical Ea (the "kinetic compensation effect" -- verified
    numerically here during development: on synthetic data generated
    from a pure F1 model, F1 correctly scores R^2=1.000 with the right
    Ea, but A2 *also* scores R^2=1.000 while recovering an Ea more than
    2x too low). Relying on R^2 alone would silently recommend the
    wrong model.

    To guard against this, every candidate's Coats-Redfern Ea is also
    compared against a model-free reference Ea (by default, this
    module's own :func:`vyazovkin`, averaged over the same data --
    Vyazovkin's method does not assume any reaction model, so it is not
    subject to the same degeneracy). The recommended model is the one
    whose Coats-Redfern Ea is *closest* to that reference, among models
    with a reasonably good fit (R^2 above a minimum threshold) -- not
    simply whichever has the single highest R^2.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns alpha, T1, T2, T3, T4.
    beta_list : Sequence[float]
        The four heating rates, in K/min.
    model_names : Sequence[str], optional
        Candidate models to try (default: :data:`COATS_REDFERN_MODELS`,
        17 models spanning reaction-order, contracting-geometry,
        diffusion, nucleation-and-growth, and power-law families).
    reference_ea_kj : float, optional
        A model-free Ea (kJ/mol) to screen candidates against. If not
        given, it is computed internally via :func:`vyazovkin`.

    Returns
    -------
    dict
        ``{"results": {model_name: {"Ea", "A", "r2", "aic", "bic",
        "ea_gap_kj": |Ea_model - Ea_reference|}}, "reference_ea_kj":
        float, "recommended_model": model_name_or_None}``.
        ``recommended_model`` is ``None`` if no candidate both fits
        reasonably well (R^2 > 0.99) and is within 20% of the reference
        Ea -- i.e. this function also explicitly declines to guess
        rather than force a recommendation onto poor data.
    """
    if len(beta_list) != 4:
        raise ValueError("Coats-Redfern method requires exactly 4 heating rates.")

    if model_names is None:
        model_names = COATS_REDFERN_MODELS

    betas_ks = _beta_to_kelvin_per_second(beta_list)

    if reference_ea_kj is None:
        reference_ea_kj = average_ea(vyazovkin(df, beta_list))

    results: Dict[str, Dict[str, float]] = {}
    for name in model_names:
        fit = _coats_redfern_single_model(df, betas_ks, name)
        if np.isfinite(fit["Ea"]) and np.isfinite(reference_ea_kj) and reference_ea_kj > 0:
            fit["ea_gap_kj"] = abs(fit["Ea"] - reference_ea_kj)
            fit["ea_gap_pct"] = fit["ea_gap_kj"] / reference_ea_kj
        else:
            fit["ea_gap_kj"] = float("nan")
            fit["ea_gap_pct"] = float("nan")
        results[name] = fit
        logger.debug(
            "Coats-Redfern: model=%s Ea=%.2f r2=%.5f gap_from_reference=%.1f%%",
            name, fit["Ea"], fit["r2"], fit["ea_gap_pct"] * 100 if np.isfinite(fit["ea_gap_pct"]) else float("nan"),
        )

    candidates = [
        (name, fit) for name, fit in results.items()
        if np.isfinite(fit["r2"]) and fit["r2"] > 0.99 and np.isfinite(fit["ea_gap_pct"])
    ]
    if candidates:
        best_name, best_fit = min(candidates, key=lambda kv: kv[1]["ea_gap_pct"])
        recommended = best_name if best_fit["ea_gap_pct"] < 0.20 else None
    else:
        recommended = None

    logger.debug("Coats-Redfern: reference Ea=%.2f kJ/mol, recommended model=%s", reference_ea_kj, recommended)

    return {
        "results": results,
        "reference_ea_kj": reference_ea_kj,
        "recommended_model": recommended,
    }


# --------------------------------------------------------------------------- #
# 8. Confidence intervals for Ea(alpha)
# --------------------------------------------------------------------------- #

#: Two-tailed Student's t critical values, keyed by degrees of freedom,
#: for 95% and 99% confidence. Hardcoded (rather than depending on
#: scipy.stats.t.ppf) to keep this module's core dependency footprint
#: at NumPy/pandas only; the degrees of freedom encountered here are
#: always small (typically 1-3, since there are only 4 heating rates
#: and a 2-parameter linear fit), well within this table's range, and
#: values beyond it fall back to the normal-distribution approximation.
_T_CRITICAL_TABLE = {
    1: (12.706, 63.657), 2: (4.303, 9.925), 3: (3.182, 5.841),
    4: (2.776, 4.604), 5: (2.571, 4.032), 6: (2.447, 3.707),
    7: (2.365, 3.499), 8: (2.306, 3.355), 9: (2.262, 3.250),
    10: (2.228, 3.169), 15: (2.131, 2.947), 20: (2.086, 2.845),
    25: (2.060, 2.787), 30: (2.042, 2.750),
}

#: Methods for which confidence_interval() is supported. Restricted to
#: the integral methods (FWO, KAS, Starink) plus Friedman -- see the
#: docstring of confidence_interval() for why Friedman/KTH-style
#: differential methods, while supported, need to be interpreted with
#: real caution.
_CI_SUPPORTED_METHODS = ("Friedman", "FWO", "KAS", "Starink")


def _t_critical(dof: int, confidence: float = 0.95) -> float:
    """
    Two-tailed Student's t critical value for the given degrees of
    freedom and confidence level (0.95 or 0.99), from a small hardcoded
    table (falls back to the normal-distribution z value for dof
    outside the table, and to the nearest tabulated dof below it
    otherwise -- conservative, since t-critical values decrease
    monotonically towards the normal approximation as dof grows).
    """
    z = 1.960 if confidence >= 0.95 and confidence < 0.99 else 2.576
    if confidence not in (0.95, 0.99):
        # Only these two levels are tabulated; anything else falls back
        # to the closest normal approximation rather than guessing.
        return z
    idx = 0 if confidence == 0.95 else 1
    if dof in _T_CRITICAL_TABLE:
        return _T_CRITICAL_TABLE[dof][idx]
    if dof < 1:
        return float("nan")
    if dof > 30:
        return z
    # dof between tabulated entries (e.g. 11-14, 16-19, ...): use the
    # next-lowest tabulated dof, which is the conservative (wider) choice.
    lower_dofs = [d for d in _T_CRITICAL_TABLE if d <= dof]
    return _T_CRITICAL_TABLE[max(lower_dofs)][idx]


def confidence_interval(
    df: pd.DataFrame,
    beta_list: Sequence[float],
    method_name: str = "KAS",
    confidence: float = 0.95,
) -> Dict[float, "tuple[float, float, float]"]:
    """
    Compute a confidence interval for Ea(alpha) for one of the linear
    (OLS-regression-based) isoconversional methods, using the standard
    error of the regression slope and a Student's t critical value.

    Recommended methods: FWO, KAS, Starink (the default is KAS)
    ------------------------------------------------------------------
    These integral methods regress directly against the raw
    temperatures at each heating rate -- no numerical differentiation
    is involved. Validated here with a Monte Carlo coverage check
    (repeatedly adding realistic instrument noise, +-0.5 K, to a
    synthetic dataset with a known Ea and checking how often the
    resulting 95% CI actually contains the true value): KAS achieved
    ~95.4% empirical coverage with a mean CI width of ~25 kJ/mol --
    both statistically correct and practically usable.

    Use Friedman's confidence interval with caution
    ------------------------------------------------------------------
    Friedman's method needs dalpha/dT, estimated via numerical
    differentiation (``numpy.gradient``). The same Monte Carlo check
    showed this is *statistically valid* (~94.5% empirical coverage,
    matching the nominal 95%) but numerical differentiation of noisy
    data is inherently unstable, and the resulting intervals can become
    enormous (mean width ~1300 kJ/mol in that same test) -- correct,
    but often not practically useful to display. KTH shares this
    limitation (it also relies on a discrete rate estimate) and is not
    currently supported by this function; Vyazovkin's non-linear
    minimisation does not have a simple closed-form standard error and
    is also not currently supported.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns alpha, T1, T2, T3, T4.
    beta_list : Sequence[float]
        The four heating rates, in K/min.
    method_name : str, optional
        One of ``"Friedman"``, ``"FWO"``, ``"KAS"``, ``"Starink"``
        (default ``"KAS"``).
    confidence : float, optional
        ``0.95`` or ``0.99`` (default ``0.95``).

    Returns
    -------
    dict
        Mapping alpha -> ``(Ea, lower_bound, upper_bound)`` (kJ/mol).
        Points with fewer than 3 valid heating-rate pairs, or where the
        regression is otherwise degenerate, map to
        ``(Ea_or_nan, nan, nan)``.

    Raises
    ------
    ValueError
        If ``method_name`` is not one of the supported methods, or if
        ``beta_list`` does not have exactly 4 entries.
    """
    if method_name not in _CI_SUPPORTED_METHODS:
        raise ValueError(
            f"confidence_interval() supports {_CI_SUPPORTED_METHODS}, got {method_name!r}."
        )
    if len(beta_list) != 4:
        raise ValueError("confidence_interval() requires exactly 4 heating rates.")

    betas_ks = _beta_to_kelvin_per_second(beta_list)
    alpha = df["alpha"].to_numpy(dtype=float)
    n_points = len(alpha)

    dadT: Dict[str, np.ndarray] = {}
    if method_name == "Friedman":
        for col in _T_COLUMNS:
            dadT[col] = np.gradient(alpha, df[col].to_numpy(dtype=float))

    results: Dict[float, "tuple[float, float, float]"] = {}

    for i in range(n_points):
        x_vals, y_vals = [], []
        for beta, col in zip(betas_ks, _T_COLUMNS):
            T = df[col].iloc[i]
            if T <= 0:
                continue
            if method_name == "Friedman":
                rate = beta * dadT[col][i]
                if rate > 0:
                    x_vals.append(1.0 / T)
                    y_vals.append(np.log(rate))
            elif method_name == "FWO":
                x_vals.append(1.0 / T)
                y_vals.append(np.log(beta))
            elif method_name == "KAS":
                x_vals.append(1.0 / T)
                y_vals.append(np.log(beta / T ** 2))
            else:  # Starink
                x_vals.append(1.0 / T)
                y_vals.append(np.log(beta / T ** 1.92))

        x = np.asarray(x_vals, dtype=float)
        y = np.asarray(y_vals, dtype=float)
        n = len(x)
        a_val = float(alpha[i])

        if n < 3:
            results[a_val] = (float("nan"), float("nan"), float("nan"))
            continue

        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        dof = n - 2
        sum_sq_x = np.sum((x - x.mean()) ** 2)

        if dof < 1 or sum_sq_x <= 0:
            ea_only = -slope * R / 1000.0
            results[a_val] = (ea_only, float("nan"), float("nan"))
            continue

        s_err = float(np.sqrt(np.sum(resid ** 2) / dof))
        se_slope = s_err / float(np.sqrt(sum_sq_x))

        if method_name == "FWO":
            ea = -slope * R / 1.052 / 1000.0
            ea_se = se_slope * R / 1.052 / 1000.0
        elif method_name == "Starink":
            ea = -slope * R / 1.0008 / 1000.0
            ea_se = se_slope * R / 1.0008 / 1000.0
        else:  # Friedman, KAS
            ea = -slope * R / 1000.0
            ea_se = se_slope * R / 1000.0

        tc = _t_critical(dof, confidence)
        if not np.isfinite(tc):
            results[a_val] = (ea, float("nan"), float("nan"))
        else:
            results[a_val] = (ea, ea - tc * ea_se, ea + tc * ea_se)

    return results
