"""Arrhenius lifetime prediction: physics, units, Ea sources and the GUI tab."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from isokinetics import kinetics, lifetime
from isokinetics import formatting as fmt

R = 8.314462618
SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "..", "sample_data", "matric_121459.txt")


# --------------------------------------------------------------------------- #
# Physics (checked against a hand-written formula, not the module's own)
# --------------------------------------------------------------------------- #
def _af(ea_kj, t_storage, t_service):
    return math.exp(ea_kj * 1000.0 / R * (1.0 / t_service - 1.0 / t_storage))


def test_known_value_100kj_80c_to_25c():
    r = lifetime.predict_lifetime(100.0, 353.15, 30.0, 298.15)
    assert r.acceleration_factor == pytest.approx(535.14, rel=1e-4)  # worked by hand
    assert r.lifetime_days == pytest.approx(30.0 * 535.14, rel=1e-4)


def test_years_days_minutes_are_consistent():
    r = lifetime.predict_lifetime(95.0, 343.15, 12.0, 293.15)
    assert r.lifetime_years * 365.25 == pytest.approx(r.lifetime_days, rel=1e-12)
    assert r.lifetime_minutes == pytest.approx(r.lifetime_days * 1440.0, rel=1e-12)


@pytest.mark.parametrize("ea,ts,tu", [(60, 400, 300), (150, 523.15, 313.15), (250, 700, 350)])
def test_matches_independent_formula(ea, ts, tu):
    r = lifetime.predict_lifetime(ea, ts, 7.0, tu)
    assert r.lifetime_days == pytest.approx(7.0 * _af(ea, ts, tu), rel=1e-10)


def test_same_temperature_gives_storage_time():
    r = lifetime.predict_lifetime(100.0, 330.0, 42.0, 330.0)
    assert r.acceleration_factor == pytest.approx(1.0)
    assert r.lifetime_days == pytest.approx(42.0)


def test_cooler_service_lasts_longer_hotter_service_shorter():
    cool = lifetime.predict_lifetime(100.0, 350.0, 10.0, 300.0)
    hot = lifetime.predict_lifetime(100.0, 350.0, 10.0, 400.0)
    assert cool.lifetime_days > 10.0 > hot.lifetime_days
    assert any("higher than the storage" in w for w in hot.warnings)


def test_doubling_ea_squares_the_acceleration_factor():
    a = lifetime.acceleration_factor(60.0, 360.0, 300.0)
    b = lifetime.acceleration_factor(120.0, 360.0, 300.0)
    assert b == pytest.approx(a * a, rel=1e-10)


def test_sensitivity_bracket_is_ordered_and_matches_formula():
    r = lifetime.predict_lifetime(100.0, 353.15, 30.0, 298.15)
    assert r.lifetime_days_low_ea < r.lifetime_days < r.lifetime_days_high_ea
    assert r.lifetime_days_high_ea == pytest.approx(30.0 * _af(110.0, 353.15, 298.15), rel=1e-10)


def test_overflow_is_reported_as_infinite_not_an_exception():
    r = lifetime.predict_lifetime(3000.0, 600.0, 1.0, 200.0)
    assert math.isinf(r.lifetime_days) and not r.is_finite
    assert any("numerical range" in w for w in r.warnings)


# --------------------------------------------------------------------------- #
# Units and validation
# --------------------------------------------------------------------------- #
def test_unit_conversions():
    assert lifetime.to_kelvin(25.0, "C") == pytest.approx(298.15)
    assert lifetime.to_kelvin(25.0, "\u00b0C") == pytest.approx(298.15)
    assert lifetime.to_kelvin(298.15, "K") == pytest.approx(298.15)
    assert lifetime.time_to_days(2, "years") == pytest.approx(730.5)
    assert lifetime.time_to_days(48, "hours") == pytest.approx(2.0)
    assert lifetime.time_to_days(2880, "minutes") == pytest.approx(2.0)
    with pytest.raises(ValueError):
        lifetime.time_to_days(1, "fortnights")
    with pytest.raises(ValueError):
        lifetime.to_kelvin(1, "F")


@pytest.mark.parametrize("args", [
    (0.0, 350, 10, 300), (-5.0, 350, 10, 300), (100, 0.0, 10, 300), (100, 350, 10, -1.0),
    (100, 350, 0.0, 300), (100, 350, -3.0, 300), (float("nan"), 350, 10, 300), (100, 350, float("inf"), 300),
])
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        lifetime.predict_lifetime(*args)


def test_extrapolation_and_variable_ea_warnings():
    r = lifetime.predict_lifetime(100.0, 353.15, 30.0, 298.15,
                                  measured_range_k=(500.0, 700.0), ea_variation_pct=44.0)
    text = " ".join(r.warnings)
    assert "Extrapolation" in text and "multi-step" in text
    quiet = lifetime.predict_lifetime(100.0, 353.15, 30.0, 298.15,
                                      measured_range_k=(300.0, 400.0), ea_variation_pct=5.0)
    assert not quiet.warnings


def test_implausible_lifetime_and_high_ea_warnings():
    """Screenshot case: Ea = 200 kJ/mol, 90 d at 60 degC -> ~1,184 years at 25 degC."""
    r = lifetime.predict_lifetime(200.03, 333.15, 90.0, 298.15)
    assert r.lifetime_years == pytest.approx(1183.8, rel=1e-3)
    text = " ".join(r.warnings)
    assert "Plausibility" in text and "high for slow ageing" in text
    ordinary = lifetime.predict_lifetime(80.0, 333.15, 90.0, 298.15)  # ~7 years
    assert not any("Plausibility" in w or "high for slow ageing" in w for w in ordinary.warnings)


# --------------------------------------------------------------------------- #
# Ea sources: integration with model-fitting results
# --------------------------------------------------------------------------- #
CR = {"recommended_model": "F1", "reference_ea_kj": 150.0, "results": {
    "F1": {"Ea": 149.8, "r2": 0.999}, "R3": {"Ea": 120.0, "r2": 0.98}, "D3": {"Ea": 300.0, "r2": 0.97}}}


def test_recommended_model_is_first_source_with_its_ea():
    sources = lifetime.available_ea_sources({"Vyazovkin": 148.0}, CR)
    assert sources[0].label == "Coats-Redfern recommended (F1)"
    assert sources[0].ea_kj == pytest.approx(149.8)
    assert sources[-1].label == lifetime.MANUAL_SOURCE and sources[-1].ea_kj is None
    labels = [s.label for s in sources]
    assert "Vyazovkin mean" in labels and "Coats-Redfern D3 (not recommended)" in labels


def test_without_recommendation_model_free_mean_comes_first_and_says_why():
    cr = dict(CR, recommended_model=None)
    sources = lifetime.available_ea_sources({"Friedman": 97.0}, cr)
    assert sources[0].label == "Friedman mean"
    assert "No Coats-Redfern model was recommended" in sources[0].note
    assert not any("recommended (" in s.label for s in sources)


def test_no_results_only_manual_entry():
    assert [s.label for s in lifetime.available_ea_sources({}, None)] == [lifetime.MANUAL_SOURCE]


def test_ea_variation_percent():
    res = {"Friedman": {a / 100: 100 + a for a in range(5, 96)}}
    assert lifetime.ea_variation_percent(res) == pytest.approx(80 / 150 * 100, rel=1e-9)  # 0.10-0.90 -> 110..190, mean 150
    assert lifetime.ea_variation_percent({}) is None


def test_end_to_end_with_coats_redfern_on_synthetic_data():
    """Fit a known F1 dataset (Ea = 150 kJ/mol), take the recommended model's Ea
    and predict a lifetime: it must agree with the exact value from the true Ea."""
    from isokinetics import validation
    df = validation.simulate_constant_ea(150.0, 1e12, "F1",
                                         alphas=np.round(np.arange(0.05, 0.95, 0.01), 2))
    cr = kinetics.coats_redfern(df, [5.0, 10.0, 20.0, 40.0])
    src = lifetime.available_ea_sources({}, cr)[0]
    assert src.label == "Coats-Redfern recommended (F1)"
    got = lifetime.predict_lifetime(src.ea_kj, 373.15, 1.0, 298.15).lifetime_days
    true = lifetime.predict_lifetime(150.0, 373.15, 1.0, 298.15).lifetime_days
    assert got == pytest.approx(true, rel=0.05)  # Ea within 0.1 % -> lifetime within a few %


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def test_time_formats():
    assert fmt.fmt_years(43.9482) == "43.95"
    assert fmt.fmt_days(16054.2) == "16,054.20"
    assert fmt.fmt_minutes(23118055.4) == "23,118,055"
    assert fmt.fmt_years(math.inf) == "\u221e"
    assert fmt.fmt_days(float("nan")) == fmt.NA
    assert fmt.fmt_minutes(1.5e15) == "1.500e+15"


# --------------------------------------------------------------------------- #
# GUI tab (needs a display)
# --------------------------------------------------------------------------- #
def _display_available() -> bool:
    try:
        import tkinter as tk
        tk.Tk().destroy()
        return True
    except Exception:
        return False


gui_only = pytest.mark.skipif(not _display_available(), reason="No display (run under Xvfb).")


@pytest.fixture
def app():
    pytest.importorskip("customtkinter")
    from isokinetics.gui import KineticsApp
    application = KineticsApp()
    yield application
    application.destroy()


def _set(entry, text):
    entry.delete(0, "end")
    entry.insert(0, text)


def _feed_analysis(app):
    betas = app._read_betas()
    results = {n: kinetics.METHODS[n](app.working_df, betas) for n in ("Friedman", "Vyazovkin")}
    averages = {n: kinetics.average_ea(r) for n, r in results.items()}
    app._on_analysis_done(results, averages, {}, cr_result=CR)


@gui_only
def test_tab_manual_entry_shows_years_days_minutes(app):
    _set(app.lifetime_storage_temp_entry, "80")
    _set(app.lifetime_service_temp_entry, "25")
    _set(app.lifetime_storage_time_entry, "30")
    _set(app.lifetime_ea_entry, "100")
    res = app._on_calculate_lifetime()
    assert res is not None
    assert app.lifetime_years_label.cget("text") == "43.95"
    assert app.lifetime_days_label.cget("text") == "16,054.20"
    assert app.lifetime_minutes_label.cget("text") == "23,118,055"
    assert "acceleration factor" in app.lifetime_detail_label.cget("text")


@gui_only
def test_tab_uses_recommended_model_after_analysis(app):
    assert app.load_file(SAMPLE_FILE)
    _feed_analysis(app)
    assert app.lifetime_source_var.get() == "Coats-Redfern recommended (F1)"
    assert "149.80" in app.lifetime_ea_label.cget("text")
    assert not app.lifetime_ea_entry.winfo_manager()  # manual box hidden for fitted values
    res = app._on_calculate_lifetime()
    assert res.ea_kj == pytest.approx(149.8)
    # sample data are measured far above 25 C and Ea varies with alpha -> both warnings
    notes = app.lifetime_notes_label.cget("text")
    assert "Extrapolation" in notes and "multi-step" in notes


@gui_only
def test_switching_source_recalculates_and_manual_reappears(app):
    app.load_file(SAMPLE_FILE)
    _feed_analysis(app)
    first = app._on_calculate_lifetime().lifetime_days
    app.lifetime_source_var.set("Coats-Redfern R3 (not recommended)")
    app._on_lifetime_source_changed("Coats-Redfern R3 (not recommended)")
    assert app.lifetime_result.lifetime_days != pytest.approx(first)
    app.lifetime_source_var.set(lifetime.MANUAL_SOURCE)
    app._on_lifetime_source_changed(lifetime.MANUAL_SOURCE)
    assert app.lifetime_ea_entry.winfo_manager()


@gui_only
def test_invalid_input_shows_inline_error_not_dialog(app, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no modal dialog expected")
    monkeypatch.setattr("isokinetics.gui.messagebox.showerror", boom)
    monkeypatch.setattr("isokinetics.gui.messagebox.showwarning", boom)
    _set(app.lifetime_storage_temp_entry, "abc")
    assert app._on_calculate_lifetime() is None
    assert "not a valid number" in app.lifetime_error_label.cget("text")
    _set(app.lifetime_storage_temp_entry, "-300")  # below absolute zero in degC
    assert app._on_calculate_lifetime() is None
    assert "absolute zero" in app.lifetime_error_label.cget("text")
    _set(app.lifetime_storage_temp_entry, "60")
    assert app._on_calculate_lifetime() is not None
    assert app.lifetime_error_label.cget("text") == ""


@gui_only
def test_temperature_unit_toggle_converts_fields_and_keeps_result(app):
    _set(app.lifetime_storage_temp_entry, "80")
    _set(app.lifetime_service_temp_entry, "25")
    before = app._on_calculate_lifetime().lifetime_days
    app.lifetime_temp_unit_var.set("K")
    app._on_lifetime_temp_unit_changed("K")
    assert float(app.lifetime_storage_temp_entry.get()) == pytest.approx(353.15)
    assert app._on_calculate_lifetime().lifetime_days == pytest.approx(before)


@gui_only
def test_time_units_and_stale_result_cleared_on_new_data(app):
    _set(app.lifetime_storage_time_entry, "2")
    app.lifetime_time_unit_var.set("years")
    r = app._on_calculate_lifetime()
    assert r.storage_time_days == pytest.approx(730.5)
    app.load_file(SAMPLE_FILE)
    assert app.lifetime_result is None
    assert app.lifetime_years_label.cget("text") == "-"
