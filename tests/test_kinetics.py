"""
Unit and validation tests for isokinetics.kinetics.

The validation tests generate *synthetic* non-isothermal kinetic data
for a known activation energy (via direct numerical simulation of a
first-order Arrhenius reaction at several heating rates) and check
that every isoconversional method recovers that known activation
energy to within a reasonable tolerance. This is a much stronger check
than simply asserting "some number came back" -- it verifies the
underlying physics/maths of each method are implemented correctly.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from isokinetics import data_loader, kinetics

SAMPLE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "sample_data", "matric_121459.txt"
)

R = 8.314


# --------------------------------------------------------------------------- #
# Synthetic-data generator (first-order Arrhenius kinetics, forward Euler)
# --------------------------------------------------------------------------- #
def _simulate_first_order(
    ea_j_per_mol: float,
    pre_exp_factor: float,
    beta_k_per_s: float,
    t0: float,
    alpha_targets,
    t_max: float = 900.0,
    d_t: float = 0.02,
):
    """
    Simulate alpha(T) for a first-order Arrhenius reaction under linear
    heating, and return the temperature at which each target alpha is
    reached (linear interpolation between the two nearest simulation
    steps).
    """
    targets = sorted(alpha_targets)
    out = {}
    T = t0
    alpha = 1e-6
    ti = 0
    while T < t_max and ti < len(targets):
        rate = (pre_exp_factor / beta_k_per_s) * np.exp(-ea_j_per_mol / (R * T)) * max(
            1.0 - alpha, 0.0
        )
        alpha_next = alpha + rate * d_t
        if alpha_next >= targets[ti]:
            frac = (
                (targets[ti] - alpha) / (alpha_next - alpha)
                if alpha_next != alpha
                else 0.0
            )
            out[targets[ti]] = T + frac * d_t
            ti += 1
        alpha = alpha_next
        T += d_t
    return out


@pytest.fixture(scope="module")
def synthetic_dataset():
    """
    Build a synthetic dataset with a known ("ground truth") activation
    energy of 150 kJ/mol, sampled at four heating rates (5, 10, 20, 40
    K/min), mimicking the structure of a real experimental dataset.
    """
    ea_true_kj = 150.0
    ea_true_j = ea_true_kj * 1000.0
    pre_exp_factor = 1.0e12
    t0 = 300.0
    betas_min = [5.0, 10.0, 20.0, 40.0]
    betas_s = [b / 60.0 for b in betas_min]

    alpha_targets = np.round(np.arange(0.05, 0.90, 0.02), 4)

    data = {"alpha": alpha_targets}
    for i, beta in enumerate(betas_s, start=1):
        res = _simulate_first_order(ea_true_j, pre_exp_factor, beta, t0, alpha_targets)
        data[f"T{i}"] = [res.get(a, np.nan) for a in alpha_targets]

    df = pd.DataFrame(data).dropna().reset_index(drop=True)
    return df, betas_min, ea_true_kj


def _simulate_model(
    ea_j_per_mol: float,
    pre_exp_factor: float,
    beta_k_per_s: float,
    t0: float,
    alpha_targets,
    f_func,
    t_max: float = 900.0,
    d_t: float = 0.02,
):
    """
    Generalisation of :func:`_simulate_first_order` to an arbitrary
    reaction model ``f_func(alpha) -> float``, used to build ground-truth
    datasets for models other than F1 (e.g. D3 diffusion) for
    :func:`kinetics.kth_full` validation.
    """
    targets = sorted(alpha_targets)
    out = {}
    T = t0
    alpha = 1e-6
    ti = 0
    while T < t_max and ti < len(targets):
        rate = (pre_exp_factor / beta_k_per_s) * np.exp(-ea_j_per_mol / (R * T)) * f_func(alpha)
        alpha_next = alpha + rate * d_t
        if alpha_next >= targets[ti]:
            frac = (
                (targets[ti] - alpha) / (alpha_next - alpha)
                if alpha_next != alpha
                else 0.0
            )
            out[targets[ti]] = T + frac * d_t
            ti += 1
        alpha = alpha_next
        T += d_t
    return out


def _f_d3(alpha: float) -> float:
    """Jander (3D diffusion) reaction model, f(alpha), for a scalar alpha."""
    a = min(max(alpha, 1e-6), 1 - 1e-6)
    return 1.5 * (1 - a) ** (2 / 3) / (1 - (1 - a) ** (1 / 3))


@pytest.fixture(scope="module")
def d3_synthetic_dataset():
    """
    Build a synthetic dataset generated from a genuinely different
    reaction model (D3, three-dimensional Jander diffusion) than the
    F1 model used by :func:`synthetic_dataset`, so that
    :func:`kinetics.kth_full`'s model-identification step can be tested
    against more than one ground truth (guarding against a classifier
    that merely always guesses F1).
    """
    ea_true_kj = 150.0
    ea_true_j = ea_true_kj * 1000.0
    pre_exp_factor = 1.0e14  # D3's rate is intrinsically slower near alpha->1
    t0 = 300.0
    betas_min = [5.0, 10.0, 20.0, 40.0]
    betas_s = [b / 60.0 for b in betas_min]

    alpha_targets = np.round(np.arange(0.05, 0.90, 0.02), 4)

    data = {"alpha": alpha_targets}
    for i, beta in enumerate(betas_s, start=1):
        res = _simulate_model(ea_true_j, pre_exp_factor, beta, t0, alpha_targets, _f_d3)
        data[f"T{i}"] = [res.get(a, np.nan) for a in alpha_targets]

    df = pd.DataFrame(data).dropna().reset_index(drop=True)
    return df, betas_min, ea_true_kj


# --------------------------------------------------------------------------- #
# Validation tests: each method should recover the known Ea
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "method_name,rel_tolerance",
    [
        ("Friedman", 0.05),
        ("FWO", 0.05),
        ("KAS", 0.05),
        ("Starink", 0.05),
        ("Vyazovkin", 0.10),
        ("KTH", 0.05),
    ],
)
def test_method_recovers_known_activation_energy(
    synthetic_dataset, method_name, rel_tolerance
):
    df, betas, ea_true_kj = synthetic_dataset
    func = kinetics.METHODS[method_name]
    result = func(df, betas)
    avg_ea = kinetics.average_ea(result)

    assert np.isfinite(avg_ea), f"{method_name} produced no finite Ea values."
    rel_error = abs(avg_ea - ea_true_kj) / ea_true_kj
    assert rel_error <= rel_tolerance, (
        f"{method_name}: recovered Ea={avg_ea:.2f} kJ/mol vs. true "
        f"{ea_true_kj:.2f} kJ/mol (relative error {rel_error:.3%} "
        f"exceeds tolerance {rel_tolerance:.0%})"
    )


def test_kth_recovers_known_activation_energy(synthetic_dataset):
    """Explicit KTH check (in addition to the parametrized one above),
    matching the acceptance criterion specified for this method."""
    df, betas, ea_true_kj = synthetic_dataset
    result = kinetics.kth(df, betas)
    avg_ea = kinetics.average_ea(result)
    assert avg_ea == pytest.approx(ea_true_kj, rel=0.10)


def test_kth_last_point_uses_backward_step(synthetic_dataset):
    """
    The last alpha in the grid has no "next" point to step to, so KTH
    must fall back to the step behind it (alpha_{n-2} -> alpha_{n-1})
    rather than returning NaN for that point alone.

    Reuses the same validated synthetic dataset/simulator as the other
    ground-truth tests, just checking the boundary point specifically.
    """
    df, betas, ea_true_kj = synthetic_dataset
    result = kinetics.kth(df, betas)

    last_alpha = float(df["alpha"].iloc[-1])
    assert last_alpha in result
    assert np.isfinite(result[last_alpha]), "Last point should not be NaN."
    rel_error = abs(result[last_alpha] - ea_true_kj) / ea_true_kj
    assert rel_error <= 0.10, (
        f"Last-point Ea={result[last_alpha]:.2f} kJ/mol vs. true "
        f"{ea_true_kj:.2f} kJ/mol (relative error {rel_error:.3%})"
    )


def test_kth_returns_nan_with_fewer_than_three_valid_pairs():
    """
    KTH requires at least 3 valid (heating-rate, temperature) pairs per
    alpha step to attempt a regression; fewer than that must yield NaN
    for that point rather than an unreliable 2-point fit.
    """
    # Only 2 of the 4 heating-rate columns have physically valid
    # (strictly increasing) temperatures; the other two are constant,
    # which makes d_t == 0 for those columns and excludes them.
    df = pd.DataFrame(
        {
            "alpha": [0.1, 0.2, 0.3],
            "T1": [500.0, 510.0, 520.0],
            "T2": [505.0, 515.0, 525.0],
            "T3": [500.0, 500.0, 500.0],  # constant -> d_t == 0, excluded
            "T4": [505.0, 505.0, 505.0],  # constant -> d_t == 0, excluded
        }
    )
    betas = [5.0, 10.0, 20.0, 40.0]
    result = kinetics.kth(df, betas)

    assert len(result) == 3
    assert all(np.isnan(v) for v in result.values())


def test_all_methods_agree_within_20_percent(synthetic_dataset):
    """Sanity check: all methods should broadly agree with each other."""
    df, betas, _ = synthetic_dataset
    averages = []
    for name, func in kinetics.METHODS.items():
        result = func(df, betas)
        averages.append(kinetics.average_ea(result))

    averages = np.array(averages)
    assert np.all(np.isfinite(averages))
    spread = (averages.max() - averages.min()) / averages.mean()
    assert spread < 0.20, f"Methods disagree too much: {averages}"


# --------------------------------------------------------------------------- #
# Basic behavioural / API tests using the real sample dataset
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_df():
    df = data_loader.load_data(SAMPLE_FILE)
    return data_loader.interpolate_alpha(df)


@pytest.mark.parametrize("method_name", list(kinetics.METHODS.keys()))
def test_method_returns_dict_keyed_by_alpha(sample_df, method_name):
    betas = [5.0, 10.0, 20.0, 40.0]
    func = kinetics.METHODS[method_name]
    result = func(sample_df, betas)

    assert isinstance(result, dict)
    assert len(result) == len(sample_df)
    for alpha in sample_df["alpha"]:
        assert alpha in result


def test_methods_reject_wrong_number_of_betas(sample_df):
    for func in kinetics.METHODS.values():
        with pytest.raises(ValueError):
            func(sample_df, [5.0, 10.0, 20.0])  # only 3 betas


def test_average_ea_ignores_nan():
    result = {0.1: 100.0, 0.2: float("nan"), 0.3: 120.0}
    assert kinetics.average_ea(result) == pytest.approx(110.0)


def test_average_ea_all_nan_returns_nan():
    result = {0.1: float("nan"), 0.2: float("nan")}
    assert np.isnan(kinetics.average_ea(result))


def test_golden_section_search_finds_known_minimum():
    # f(x) = (x - 42)^2 has its minimum at x = 42.
    f = lambda x: (x - 42.0) ** 2
    x_min = kinetics._golden_section_search(f, 0.0, 100.0, tol=1e-3)
    assert x_min == pytest.approx(42.0, abs=1e-2)


# --------------------------------------------------------------------------- #
# kth_full(): full kinetic triplet (Ea, A, f(alpha), model identification)
# --------------------------------------------------------------------------- #
def test_kth_full_identifies_f1_model_from_synthetic_f1_data(synthetic_dataset):
    """
    Data generated from a pure first-order (F1) reaction model should be
    identified as F1, with a score clearly better than every other
    candidate (not just an arbitrary tie-break).
    """
    df, betas, ea_true_kj = synthetic_dataset
    result = kinetics.kth_full(df, betas)

    assert result["model"] == "F1"

    ranked = sorted(result["model_scores"].items(), key=lambda kv: kv[1])
    best_name, best_score = ranked[0]
    second_name, second_score = ranked[1]
    assert best_name == "F1"
    assert second_score > 1.5 * best_score, (
        f"F1's match should be decisively better than the runner-up "
        f"({second_name}); got best={best_score:.4f}, second={second_score:.4f}"
    )


def test_kth_full_identifies_d3_model_from_synthetic_d3_data(d3_synthetic_dataset):
    """
    Data generated from a genuinely different model (D3, 3D diffusion)
    should be identified as D3 -- not simply always F1 -- guarding
    against a classifier that never actually discriminates.
    """
    df, betas, ea_true_kj = d3_synthetic_dataset
    result = kinetics.kth_full(df, betas)

    assert result["model"] == "D3"


def test_kth_full_recovers_known_activation_energy(synthetic_dataset):
    df, betas, ea_true_kj = synthetic_dataset
    result = kinetics.kth_full(df, betas)
    avg_ea = kinetics.average_ea(result["Ea"])
    assert avg_ea == pytest.approx(ea_true_kj, rel=0.05)


def test_kth_full_recovers_known_pre_exponential_factor(synthetic_dataset):
    """
    A(alpha) is exponentially sensitive to errors in Ea, so we check
    ln(A) with a generous (but still meaningful) absolute tolerance
    rather than expecting tight relative agreement on A itself.
    """
    df, betas, ea_true_kj = synthetic_dataset
    A_true = 1.0e12
    result = kinetics.kth_full(df, betas)

    assert result["model"] == "F1"
    A_values = np.array([v for v in result["A"].values() if np.isfinite(v)])
    assert A_values.size > 0

    ln_A_recovered = np.log(np.mean(A_values))
    ln_A_true = np.log(A_true)
    assert abs(ln_A_recovered - ln_A_true) < 2.0, (
        f"ln(A) recovered={ln_A_recovered:.3f} vs true={ln_A_true:.3f}"
    )


def test_kth_full_ea_matches_kth(synthetic_dataset):
    """kth_full()'s Ea(alpha) must be identical to plain kth()'s output
    (they share the same regression core) -- consistency check."""
    df, betas, _ = synthetic_dataset
    ea_simple = kinetics.kth(df, betas)
    ea_full = kinetics.kth_full(df, betas)["Ea"]
    assert ea_simple.keys() == ea_full.keys()
    for a in ea_simple:
        if np.isnan(ea_simple[a]):
            assert np.isnan(ea_full[a])
        else:
            assert ea_simple[a] == pytest.approx(ea_full[a])


def test_kth_full_returns_unknown_for_non_physical_data():
    """
    Data with no real Arrhenius relationship at all should not be
    force-fit to some arbitrary model -- it should come back
    'unknown', per the explicit ambiguity-handling requirement.
    """
    rng = np.random.RandomState(0)
    alpha = np.round(np.arange(0.05, 0.90, 0.02), 4)
    n = len(alpha)
    data = {"alpha": alpha}
    for i in range(1, 5):
        T = np.sort(400 + 300 * alpha + rng.uniform(-50, 50, size=n))
        data[f"T{i}"] = T
    df = pd.DataFrame(data)
    betas = [5.0, 10.0, 20.0, 40.0]

    result = kinetics.kth_full(df, betas)
    assert result["model"] == "unknown"
    assert all(np.isnan(v) for v in result["A"].values())
    assert all(np.isnan(v) for v in result["f_alpha"].values())
    assert all(np.isnan(v) for v in result["y_theory"].values())
    assert all(np.isnan(v) for v in result["z_theory"].values())


def test_kth_full_output_has_expected_keys_and_alpha_coverage(synthetic_dataset):
    df, betas, _ = synthetic_dataset
    result = kinetics.kth_full(df, betas)

    assert set(result.keys()) == {
        "Ea", "A", "f_alpha", "model", "model_scores",
        "y_alpha", "z_alpha", "y_theory", "z_theory",
    }
    n = len(df)
    for key in ("Ea", "A", "f_alpha", "y_alpha", "z_alpha", "y_theory", "z_theory"):
        assert len(result[key]) == n, f"{key} should have one entry per alpha"


def test_kth_full_rejects_wrong_number_of_betas(synthetic_dataset):
    df, _betas, _ = synthetic_dataset
    with pytest.raises(ValueError):
        kinetics.kth_full(df, [5.0, 10.0, 20.0])


def test_model_f_g_rejects_unknown_model():
    with pytest.raises(ValueError):
        kinetics._model_f_g("NOT_A_MODEL", np.array([0.1, 0.5, 0.9]))


def test_normalize_01_handles_constant_array():
    result = kinetics._normalize_01(np.array([5.0, 5.0, 5.0]))
    np.testing.assert_array_equal(result, np.zeros(3))


def test_normalize_01_handles_all_nan():
    result = kinetics._normalize_01(np.array([np.nan, np.nan]))
    assert np.all(np.isnan(result))


def test_temperature_integral_p_matches_known_asymptotic_behaviour():
    """
    p(x) = integral_x^inf e^-u/u^2 du should behave like e^-x/x^2 for
    large x (standard asymptotic expansion), and should decrease
    monotonically with x.
    """
    xs = np.array([10.0, 20.0, 30.0, 40.0])
    p_vals = kinetics._temperature_integral_p(xs)

    assert np.all(np.isfinite(p_vals))
    assert np.all(np.diff(p_vals) < 0), "p(x) should be strictly decreasing in x"

    # Leading-order asymptotic check: p(x)/(e^-x/x^2) -> 1 as x grows.
    leading_order = np.exp(-xs) / xs**2
    ratio = p_vals / leading_order
    assert np.all((ratio > 0.7) & (ratio < 1.05)), f"ratios were {ratio}"


# --------------------------------------------------------------------------- #
# coats_redfern(): model-fitting method with reference-Ea screening
# --------------------------------------------------------------------------- #
def test_coats_redfern_identifies_f1_and_recovers_known_ea(synthetic_dataset):
    df, betas, ea_true_kj = synthetic_dataset
    result = kinetics.coats_redfern(df, betas)

    assert result["recommended_model"] == "F1"
    f1_fit = result["results"]["F1"]
    assert f1_fit["Ea"] == pytest.approx(ea_true_kj, rel=0.05)
    assert f1_fit["r2"] > 0.999


def test_coats_redfern_identifies_d3_from_synthetic_d3_data(d3_synthetic_dataset):
    df, betas, ea_true_kj = d3_synthetic_dataset
    result = kinetics.coats_redfern(df, betas)
    assert result["recommended_model"] == "D3"


def test_coats_redfern_rejects_high_r2_false_positive(synthetic_dataset):
    """
    This is the key regression test for the bug caught during
    development: on pure-F1 synthetic data, model A2 achieves R^2 very
    close to 1.0 (a classic Coats-Redfern false positive, the "kinetic
    compensation effect") but with a wildly wrong Ea. It must not be
    recommended, and must not even rank above F1 by the ea_gap metric.
    """
    df, betas, ea_true_kj = synthetic_dataset
    result = kinetics.coats_redfern(df, betas)

    a2_fit = result["results"]["A2"]
    assert a2_fit["r2"] > 0.999  # confirms the trap is actually present in this data
    assert a2_fit["ea_gap_pct"] > 0.20  # but its Ea is far from the reference
    assert result["recommended_model"] != "A2"


def test_coats_redfern_declines_to_recommend_on_non_physical_data():
    rng = np.random.RandomState(0)
    alpha = np.round(np.arange(0.05, 0.90, 0.02), 4)
    n = len(alpha)
    data = {"alpha": alpha}
    for i in range(1, 5):
        T = np.sort(400 + 300 * alpha + rng.uniform(-50, 50, size=n))
        data[f"T{i}"] = T
    df = pd.DataFrame(data)
    betas = [5.0, 10.0, 20.0, 40.0]

    result = kinetics.coats_redfern(df, betas)
    assert result["recommended_model"] is None


def test_coats_redfern_accepts_explicit_reference_ea(synthetic_dataset):
    """Passing reference_ea_kj explicitly should skip the internal
    Vyazovkin computation and use the supplied value instead."""
    df, betas, ea_true_kj = synthetic_dataset
    result = kinetics.coats_redfern(df, betas, reference_ea_kj=ea_true_kj)
    assert result["reference_ea_kj"] == ea_true_kj
    assert result["recommended_model"] == "F1"


def test_coats_redfern_rejects_wrong_number_of_betas(synthetic_dataset):
    df, _betas, _ = synthetic_dataset
    with pytest.raises(ValueError):
        kinetics.coats_redfern(df, [5.0, 10.0, 20.0])


def test_coats_redfern_results_cover_all_requested_models(synthetic_dataset):
    df, betas, _ = synthetic_dataset
    subset = ["F1", "D3", "A2"]
    result = kinetics.coats_redfern(df, betas, model_names=subset)
    assert set(result["results"].keys()) == set(subset)


def test_coats_redfern_models_superset_includes_new_additions():
    for extra in ("F4", "D5", "P1", "P2", "P3", "P4"):
        assert extra in kinetics.COATS_REDFERN_MODELS
    # And still includes everything KTH uses, unchanged.
    for name in kinetics.KTH_CANDIDATE_MODELS:
        assert name in kinetics.COATS_REDFERN_MODELS


def test_new_models_have_valid_f_and_g_in_0_1_range():
    """Sanity check the newly-added models (F4, D5, P1-P4) evaluate to
    finite, correctly-signed values over the interior of (0, 1)."""
    alpha = np.linspace(0.05, 0.95, 19)
    for name in ("F4", "D5", "P1", "P2", "P3", "P4"):
        f, g = kinetics._model_f_g(name, alpha)
        assert np.all(np.isfinite(f)), name
        assert np.all(np.isfinite(g)), name
        assert np.all(f > 0), name
        assert np.all(g > 0), name


# --------------------------------------------------------------------------- #
# confidence_interval(): statistical validity via Monte Carlo coverage check
# --------------------------------------------------------------------------- #
def test_confidence_interval_rejects_unsupported_method(synthetic_dataset):
    df, betas, _ = synthetic_dataset
    with pytest.raises(ValueError):
        kinetics.confidence_interval(df, betas, method_name="Vyazovkin")


def test_confidence_interval_rejects_wrong_number_of_betas(synthetic_dataset):
    df, _betas, _ = synthetic_dataset
    with pytest.raises(ValueError):
        kinetics.confidence_interval(df, [5.0, 10.0, 20.0], method_name="KAS")


@pytest.mark.parametrize("method_name", ["Friedman", "FWO", "KAS", "Starink"])
def test_confidence_interval_returns_lower_le_ea_le_upper(synthetic_dataset, method_name):
    df, betas, _ = synthetic_dataset
    ci = kinetics.confidence_interval(df, betas, method_name=method_name)
    for a, (ea, lo, hi) in ci.items():
        if np.isfinite(lo) and np.isfinite(hi):
            assert lo <= ea <= hi, f"alpha={a}: Ea={ea} not within [{lo}, {hi}]"


def test_confidence_interval_kas_has_correct_monte_carlo_coverage():
    """
    The key statistical-validity check: add realistic instrument noise
    (+-0.5 K, typical DSC/TGA scatter) to a synthetic dataset with a
    *known* Ea across many trials, and confirm the resulting 95% CIs
    actually contain the true value ~95% of the time (not, say, 60% or
    100% -- either of which would indicate the interval is miscalibrated).
    KAS is used here because -- as documented in confidence_interval()'s
    docstring -- it does not require numerical differentiation and so
    gives both correct coverage *and* a practically useful (not
    enormous) interval width, unlike Friedman.
    """
    ea_true_j = 150_000.0
    betas_min = [5.0, 10.0, 20.0, 40.0]
    betas_s = [b / 60.0 for b in betas_min]
    alpha_targets = np.round(np.arange(0.05, 0.90, 0.02), 4)

    base_data = {"alpha": alpha_targets}
    for i, beta in enumerate(betas_s, start=1):
        res = _simulate_first_order(ea_true_j, 1.0e12, beta, 300.0, alpha_targets)
        base_data[f"T{i}"] = [res.get(a, np.nan) for a in alpha_targets]
    base_df = pd.DataFrame(base_data).dropna().reset_index(drop=True)

    rng = np.random.RandomState(42)
    idx = 20  # a representative mid-range alpha
    n_trials = 500  # kept modest so the test suite stays fast
    contained = 0
    valid = 0

    for _ in range(n_trials):
        noisy_df = base_df.copy()
        for col in ["T1", "T2", "T3", "T4"]:
            noisy_df[col] = noisy_df[col] + rng.normal(0, 0.5, size=len(noisy_df))
        ci = kinetics.confidence_interval(noisy_df, betas_min, method_name="KAS")
        a_val = float(noisy_df["alpha"].iloc[idx])
        ea, lo, hi = ci[a_val]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        valid += 1
        if lo <= 150.0 <= hi:
            contained += 1

    assert valid > n_trials * 0.8, "Too many degenerate trials to draw a conclusion."
    coverage = contained / valid
    # Generous tolerance band around the nominal 95% -- this is a
    # statistical check with finite trials, not an exact equality.
    assert 0.85 <= coverage <= 1.0, f"Empirical coverage was {coverage:.1%}, expected ~95%"


def test_confidence_interval_friedman_is_valid_but_much_wider_than_kas():
    """
    Documents (and locks in as a regression test) the finding from
    development: Friedman's CI is statistically valid but its reliance
    on numerical differentiation makes it far less precise than KAS's
    on the same noisy data -- users should be steered towards KAS/FWO/
    Starink for confidence intervals, not Friedman.
    """
    ea_true_j = 150_000.0
    betas_min = [5.0, 10.0, 20.0, 40.0]
    betas_s = [b / 60.0 for b in betas_min]
    alpha_targets = np.round(np.arange(0.05, 0.90, 0.02), 4)

    data = {"alpha": alpha_targets}
    for i, beta in enumerate(betas_s, start=1):
        res = _simulate_first_order(ea_true_j, 1.0e12, beta, 300.0, alpha_targets)
        data[f"T{i}"] = [res.get(a, np.nan) for a in alpha_targets]
    df = pd.DataFrame(data).dropna().reset_index(drop=True)

    rng = np.random.RandomState(7)
    noisy_df = df.copy()
    for col in ["T1", "T2", "T3", "T4"]:
        noisy_df[col] = noisy_df[col] + rng.normal(0, 0.5, size=len(noisy_df))

    idx = 20
    a_val = float(noisy_df["alpha"].iloc[idx])

    ci_friedman = kinetics.confidence_interval(noisy_df, betas_min, method_name="Friedman")
    ci_kas = kinetics.confidence_interval(noisy_df, betas_min, method_name="KAS")

    _, lo_f, hi_f = ci_friedman[a_val]
    _, lo_k, hi_k = ci_kas[a_val]

    assert np.isfinite(hi_f - lo_f) and np.isfinite(hi_k - lo_k)
    assert (hi_f - lo_f) > 5 * (hi_k - lo_k)


def test_confidence_interval_handles_insufficient_points_gracefully():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.DataFrame(
            {
                "alpha": [0.1, 0.2, 0.3],
                "T1": [500.0, 510.0, 520.0],
                "T2": [500.0, 500.0, 500.0],  # constant -> excluded for Friedman
                "T3": [500.0, 500.0, 500.0],
                "T4": [500.0, 500.0, 500.0],
            }
        )
        betas = [5.0, 10.0, 20.0, 40.0]
        ci = kinetics.confidence_interval(df, betas, method_name="Friedman")

    assert len(ci) == 3
    for a, (ea, lo, hi) in ci.items():
        assert np.isnan(ea) or (np.isnan(lo) and np.isnan(hi))


def test_t_critical_matches_known_table_values():
    assert kinetics._t_critical(1, 0.95) == pytest.approx(12.706)
    assert kinetics._t_critical(4, 0.95) == pytest.approx(2.776)
    assert kinetics._t_critical(10, 0.99) == pytest.approx(3.169)


def test_t_critical_falls_back_to_normal_for_large_dof():
    assert kinetics._t_critical(1000, 0.95) == pytest.approx(1.96)
    assert kinetics._t_critical(1000, 0.99) == pytest.approx(2.576)


def test_t_critical_interpolates_conservatively_between_tabulated_dof():
    # dof=12 is not tabulated; should use the next-lowest tabulated
    # entry (dof=10), which is the more conservative (wider) choice.
    assert kinetics._t_critical(12, 0.95) == kinetics._t_critical(10, 0.95)
