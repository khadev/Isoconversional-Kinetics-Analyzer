"""v2.0: Lifetime-tab synchronisation with the stabilization interval, the manual
'Calculate KTH Full Triplet' button (threaded, stale-result guard) and the version."""

from __future__ import annotations

import os
import re
import time

import numpy as np
import pytest

pytest.importorskip("tkinter")
pytest.importorskip("customtkinter")


def _display_available() -> bool:
    try:
        import tkinter as tk
        tk.Tk().destroy()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _display_available(), reason="No display (run under Xvfb).")

import isokinetics  # noqa: E402
from isokinetics import interval as iv, kinetics  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SAMPLE = os.path.join(ROOT, "sample_data", "matric_121459.txt")
BETAS = [5.0, 10.0, 20.0, 40.0]
FIVE = ("Friedman", "FWO", "KAS", "Starink", "Vyazovkin")      # everything except KTH


@pytest.fixture
def app():
    from isokinetics.gui import KineticsApp
    application = KineticsApp()
    yield application
    application.destroy()


@pytest.fixture(scope="module")
def real_triplet():
    from isokinetics import data_loader
    df = data_loader.interpolate_alpha(data_loader.load_data(SAMPLE))
    return kinetics.kth_full(df, BETAS)


def _analyse(app, methods=FIVE, cr=None):
    assert app.load_file(SAMPLE)
    results = {n: kinetics.METHODS[n](app.working_df, BETAS) for n in methods}
    averages = {n: kinetics.average_ea(r) for n, r in results.items()}
    app._on_analysis_done(results, averages, {}, cr_result=cr)
    app.update()
    return app


CR = {"recommended_model": "F1", "reference_ea_kj": 97.0, "results": {
    "F1": {"Ea": 96.0, "r2": 0.999, "A": 1e10, "ea_gap_pct": 0.01}}}


def pump_until(app, condition, timeout=10.0):
    deadline = time.time() + timeout

    def tick():
        if condition() or time.time() > deadline:
            app.quit()
        else:
            app.after(20, tick)

    app.after(20, tick)
    app.mainloop()
    return condition()


def pump_for(app, seconds):
    """Run the real main loop for a fixed time; returns how many 20 ms timer ticks fired."""
    ticks = [0]
    end = time.time() + seconds

    def tick():
        ticks[0] += 1
        if time.time() > end:
            app.quit()
        else:
            app.after(20, tick)

    app.after(20, tick)
    app.mainloop()
    return ticks[0]


def labels(app):
    return list(app._lifetime_sources)


def select_interval(app, lo=0.2, hi=0.8):
    app._on_alpha_selected(lo, hi)
    assert pump_until(app, lambda: app.alpha_interval is not None)


# =========================================================================== #
# Lifetime tab synchronisation
# =========================================================================== #
def test_selecting_an_interval_relabels_the_lifetime_sources(app):
    _analyse(app, cr=CR)
    assert "Friedman mean" in labels(app)
    select_interval(app)
    got = labels(app)
    assert "Friedman mean (\u03b1 0.20\u20130.80)" in got and "Friedman mean" not in got
    assert all("(\u03b1 0.20\u20130.80)" in l for l in got if l.endswith("mean (\u03b1 0.20\u20130.80)") or " mean" in l)
    assert "Coats-Redfern recommended (F1)" in got and "Manual entry" in got      # unaffected entries


def test_the_chosen_source_stays_selected_and_carries_the_interval_mean(app):
    _analyse(app, cr=CR)
    app.lifetime_source_var.set("Friedman mean")
    app._on_lifetime_source_changed("Friedman mean")
    select_interval(app)
    assert app.lifetime_source_var.get() == "Friedman mean (\u03b1 0.20\u20130.80)"
    expected = iv.mean_ea_in_interval(app.results["Friedman"], app.alpha_interval)
    assert app._selected_lifetime_source().ea_kj == pytest.approx(expected)
    assert f"{expected:.2f}" in app.lifetime_ea_label.cget("text")
    assert "0.20\u20130.80" in app.lifetime_source_note.cget("text")


def test_a_calculated_lifetime_is_recalculated_from_the_interval_then_reverts_on_reset(app):
    _analyse(app, cr=CR)
    app.lifetime_source_var.set("Friedman mean")
    app._on_lifetime_source_changed("Friedman mean")
    full = app._on_calculate_lifetime()
    assert full.ea_kj == pytest.approx(app.averages["Friedman"])

    select_interval(app)
    sub = iv.mean_ea_in_interval(app.results["Friedman"], app.alpha_interval)
    assert app.lifetime_result.ea_kj == pytest.approx(sub)               # recalculated automatically
    assert app.lifetime_result.lifetime_days != pytest.approx(full.lifetime_days)
    assert "Friedman mean (\u03b1 0.20\u20130.80)" in app.lifetime_detail_label.cget("text")

    app._on_reset_interval()
    assert app.lifetime_result.ea_kj == pytest.approx(app.averages["Friedman"])   # back to full range
    assert app.lifetime_result.lifetime_days == pytest.approx(full.lifetime_days)
    assert "Friedman mean" in labels(app) and not any("\u03b1 0.20" in l for l in labels(app))
    assert app.lifetime_source_var.get() == "Friedman mean"


def test_nothing_is_calculated_automatically_if_the_user_never_calculated(app):
    _analyse(app, cr=CR)
    select_interval(app)
    assert app.lifetime_result is None
    assert app.lifetime_years_label.cget("text") == "-"


def test_a_cr_source_and_manual_entry_are_not_changed_by_the_interval(app):
    _analyse(app, cr=CR)
    first = app.lifetime_source_var.get()
    assert first == "Coats-Redfern recommended (F1)"
    before = app._on_calculate_lifetime().ea_kj
    select_interval(app)
    assert app.lifetime_source_var.get() == first and app.lifetime_result.ea_kj == pytest.approx(before)
    app.lifetime_source_var.set("Manual entry")
    app._on_lifetime_source_changed("Manual entry")
    app.lifetime_ea_entry.delete(0, "end")
    app.lifetime_ea_entry.insert(0, "88")
    app._on_reset_interval()
    assert app.lifetime_source_var.get() == "Manual entry"
    assert app._on_calculate_lifetime().ea_kj == pytest.approx(88.0)


def test_a_method_without_enough_points_in_the_interval_is_dropped_from_the_sources(app):
    _analyse(app, cr=CR)
    keep = (0.5, 0.51)                                                  # exactly 2 usable points
    app.results["Friedman"] = {a: (e if any(abs(a - k) < 1e-9 for k in keep) else float("nan"))
                               for a, e in app.results["Friedman"].items()}
    select_interval(app, 0.4, 0.6)
    assert not any(l.startswith("Friedman") for l in labels(app))
    assert any(l.startswith("KAS mean (") for l in labels(app))


def test_a_new_analysis_keeps_the_interval_and_the_lifetime_follows_it(app):
    _analyse(app, cr=CR)
    select_interval(app)
    app.lifetime_source_var.set("KAS mean (\u03b1 0.20\u20130.80)")
    app._on_lifetime_source_changed("KAS mean (\u03b1 0.20\u20130.80)")
    app._on_calculate_lifetime()
    shifted = {n: {a: e + 20.0 for a, e in r.items()} for n, r in app.results.items()}
    app._on_analysis_done(shifted, {n: kinetics.average_ea(r) for n, r in shifted.items()}, {}, cr_result=CR)
    assert app.alpha_interval is not None
    assert app.lifetime_source_var.get() == "Coats-Redfern recommended (F1)"     # fresh analysis: first source
    assert any("(\u03b1 0.20\u20130.80)" in l for l in labels(app))               # ...but still interval-based


def test_sync_is_safe_with_no_results_and_never_raises(app):
    app._sync_lifetime_tab()
    assert labels(app) == ["Manual entry"]
    app.lifetime_source_menu = None            # even a broken widget must not escape
    app._sync_lifetime_tab()


# =========================================================================== #
# SpanSelector re-attachment
# =========================================================================== #
def test_arming_the_selector_never_raises_even_if_attach_fails(app, monkeypatch):
    _analyse(app)
    monkeypatch.setattr(app.span_controller, "attach", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    app._arm_span_selector()          # must be swallowed (and logged)
    app._refresh_main_plot()


def test_selector_is_active_after_redraw_theme_change_style_change_and_new_analysis(app):
    _analyse(app)
    assert app.span_controller.active
    app._refresh_main_plot()
    assert app.span_controller.active
    app.dark_mode_var.set(True)
    app._on_theme_toggled()
    assert app.span_controller.active
    app.plot_style_var.set(list(app.plot_style_menu.cget("values"))[1])
    app._on_plot_style_changed(app.plot_style_var.get())
    assert app.span_controller.active
    results = dict(app.results)
    app._on_analysis_done(results, dict(app.averages), {})
    app.update()
    assert app.span_controller.active


# =========================================================================== #
# KTH full triplet button
# =========================================================================== #
def test_button_is_hidden_without_data_and_visible_once_data_is_loaded(app):
    assert not app.kth_button_visible
    assert app.load_file(SAMPLE)
    assert app.kth_button_visible
    assert app.kth_calc_button.cget("text") == "Calculate KTH Full Triplet"
    assert app.kth_calc_button.place_info()["relx"] in ("0.5", 0.5)


def test_button_hides_when_a_triplet_exists_and_returns_for_a_new_dataset(app, real_triplet):
    _analyse(app)
    assert app.kth_button_visible
    app.kth_full_result = real_triplet
    app._update_kth_button()
    assert not app.kth_button_visible
    app.load_file(SAMPLE)
    assert app.kth_full_result is None and app.kth_button_visible


def test_button_is_hidden_while_an_analysis_runs(app):
    app.load_file(SAMPLE)
    app._set_running_state(True)
    assert not app.kth_button_visible
    app._set_running_state(False)
    assert app.kth_button_visible


def test_analysis_in_full_triplet_mode_leaves_no_button(app, real_triplet):
    assert app.load_file(SAMPLE)
    results = {n: kinetics.METHODS[n](app.working_df, BETAS) for n in (*FIVE, "KTH")}
    app._on_analysis_done(results, {n: kinetics.average_ea(r) for n, r in results.items()}, {},
                          kth_full_result=real_triplet)
    app.update()
    assert not app.kth_button_visible


def test_click_runs_in_a_thread_without_freezing_the_ui_and_updates_every_view(app, monkeypatch):
    _analyse(app, methods=FIVE, cr=CR)
    assert "KTH" not in app.results
    real = kinetics.kth_full

    def slow(df, betas):
        time.sleep(0.8)
        return real(df, betas)

    monkeypatch.setattr(kinetics, "kth_full", slow)
    app.kth_calc_button.invoke()
    assert not app.kth_button_visible and app._kth_busy            # hidden at once
    assert "Calculating KTH" in app.status_label.cget("text")
    ticks = pump_for(app, 0.6)                                      # worker is still sleeping
    assert ticks >= 8, f"UI loop only ran {ticks} times in 0.6 s - main thread was blocked"
    assert app.kth_full_result is None                              # not finished yet
    assert pump_until(app, lambda: app.kth_full_result is not None)

    full = app.kth_full_result
    assert not app._kth_busy and not app.kth_button_visible
    # results table: A and f(alpha) columns now filled, average row carries the model
    rows = app.results_tree.get_children()
    assert app.results_tree.set(rows[0], "KTH_f(\u03b1)").startswith("model=")
    # (on this sample KTH cannot identify a reaction model, so A and f(alpha) are undefined -> "-")
    assert app.results_tree.set(rows[0], "KTH_f(\u03b1)") == f"model={full['model']}"
    # KTH was not part of the analysis: it is added everywhere
    assert "KTH" in app.results and "KTH" in app.averages
    assert any("KTH" in t for t in [x.get_text() for x in app.figure.axes[0].get_legend().get_texts()])
    # KTH model figure no longer shows its placeholder
    assert not [t for ax in app.kth_figure.axes for t in ax.texts if t.get_gid() == "placeholder"]
    # Lifetime tab got the new source
    assert "KTH mean" in labels(app)
    assert str(app.export_button.cget("state")) == "normal"
    assert "KTH full triplet calculated" in app.status_label.cget("text")
    assert full["model"] == app.kth_full_result["model"]


def _synthetic_f1_file(tmp_path):
    from isokinetics import validation
    df = validation.simulate_constant_ea(150.0, 1e12, "F1")
    path = tmp_path / "synthetic_F1_processed.txt"
    with open(path, "w", newline="") as fh:
        for _, r in df.iterrows():
            fh.write("\t".join([f"{r['alpha']:.2f}"] + [f"{r[c]:.5f}" for c in ("T1", "T2", "T3", "T4")]) + "\n")
    return str(path)


def test_an_identified_model_fills_the_A_and_f_columns_of_the_results_table(app, tmp_path):
    assert app.load_file(_synthetic_f1_file(tmp_path))
    assert app.kth_button_visible
    results = {n: kinetics.METHODS[n](app.working_df, BETAS) for n in FIVE}
    app._on_analysis_done(results, {n: kinetics.average_ea(r) for n, r in results.items()}, {})
    app.update()
    app.kth_calc_button.invoke()
    assert pump_until(app, lambda: app.kth_full_result is not None, timeout=30)
    assert app.kth_full_result["model"] == "F1"
    rows = app.results_tree.get_children()
    assert app.results_tree.set(rows[0], "KTH_f(\u03b1)") == "model=F1"
    avg_A = float(app.results_tree.set(rows[0], "KTH_A"))
    assert avg_A == pytest.approx(1e12, rel=0.05)                       # A recovered, in s^-1
    body = rows[1:]
    assert all(app.results_tree.set(r, "KTH_A") != "-" for r in body)   # every alpha has A and f(alpha)
    assert all(app.results_tree.set(r, "KTH_f(\u03b1)") != "-" for r in body)
    assert float(app.results_tree.set(body[0], "KTH_f(\u03b1)")) == pytest.approx(1.0 - 0.01, abs=0.05)  # f(alpha) = 1 - alpha


def test_the_real_unpatched_calculation_works_through_the_button(app):
    _analyse(app, methods=FIVE)
    app._on_calculate_kth_full()
    assert pump_until(app, lambda: app.kth_full_result is not None, timeout=30)
    assert set(app.kth_full_result) >= {"Ea", "A", "f_alpha", "model"}
    assert app.results["KTH"] == app.kth_full_result["Ea"]


def test_when_kth_was_already_in_the_results_they_are_left_alone(app, real_triplet):
    _analyse(app, methods=(*FIVE, "KTH"))
    ea_only = dict(app.results["KTH"])
    app._on_calculate_kth_full()
    assert pump_until(app, lambda: app.kth_full_result is not None, timeout=30)
    assert app.results["KTH"] == ea_only


def test_a_second_click_while_running_is_ignored(app, monkeypatch):
    _analyse(app)
    calls = []
    real = kinetics.kth_full
    monkeypatch.setattr(kinetics, "kth_full", lambda df, b: (calls.append(1), time.sleep(0.4), real(df, b))[2])
    app._on_calculate_kth_full()
    app._on_calculate_kth_full()
    app._on_calculate_kth_full()
    assert pump_until(app, lambda: app.kth_full_result is not None, timeout=30)
    assert len(calls) == 1


# ------------------------------- stale-result guard ------------------------- #
def test_loading_new_data_while_calculating_discards_the_late_result(app, monkeypatch):
    _analyse(app)
    real = kinetics.kth_full
    monkeypatch.setattr(kinetics, "kth_full", lambda df, b: (time.sleep(0.5), real(df, b))[1])
    app._on_calculate_kth_full()
    app.load_file(SAMPLE)                  # new data -> the running calculation is stale
    pump_for(app, 1.2)                     # let the worker finish and post its result
    assert app.kth_full_result is None and "KTH" not in app.results
    assert app.kth_button_visible and not app._kth_busy


def test_a_new_analysis_while_calculating_discards_the_late_result(app, monkeypatch):
    _analyse(app)
    real = kinetics.kth_full
    monkeypatch.setattr(kinetics, "kth_full", lambda df, b: (time.sleep(0.5), real(df, b))[1])
    app._on_calculate_kth_full()
    app._invalidate_kth()                  # what _on_run_analysis does when a run starts
    pump_for(app, 1.2)
    assert app.kth_full_result is None


def test_only_the_newest_request_is_applied(app, real_triplet):
    _analyse(app)
    app._on_calculate_kth_full()
    old_token, version = app._kth_token, app._data_version
    app._invalidate_kth()                                  # superseded...
    app._on_calculate_kth_full()                           # ...by a newer click
    new_token = app._kth_token
    old = dict(real_triplet, model="OLD")
    new = dict(real_triplet, model="NEW")
    app._apply_kth_full(old_token, version, old)           # late result of the first click
    assert app.kth_full_result is None
    app._apply_kth_full(new_token, version, new)
    assert app.kth_full_result["model"] == "NEW"
    app._apply_kth_full(old_token, version, old)           # and a straggler afterwards
    assert app.kth_full_result["model"] == "NEW"


def test_a_stale_failure_is_ignored_and_a_current_failure_offers_the_button_again(app, monkeypatch):
    _analyse(app)
    warnings = []
    monkeypatch.setattr("isokinetics.gui.messagebox.showwarning", lambda *a, **k: warnings.append(a))
    app._on_calculate_kth_full()
    token, version = app._kth_token, app._data_version
    app._on_kth_failed(token - 1, version, "stale")
    assert warnings == [] and app._kth_busy
    app._on_kth_failed(token, version, "no good fit")
    assert len(warnings) == 1 and "no good fit" in warnings[0][1]
    assert not app._kth_busy and app.kth_button_visible and app.kth_full_result is None


def test_an_exception_inside_the_worker_is_reported_not_swallowed(app, monkeypatch):
    _analyse(app)
    warnings = []
    monkeypatch.setattr("isokinetics.gui.messagebox.showwarning", lambda *a, **k: warnings.append(a))

    def boom(df, betas):
        raise ValueError("singular matrix")

    monkeypatch.setattr(kinetics, "kth_full", boom)
    app._on_calculate_kth_full()
    assert pump_until(app, lambda: bool(warnings))
    assert "singular matrix" in warnings[0][1] and app.kth_button_visible


def test_invalid_heating_rates_show_an_error_and_keep_the_button(app, monkeypatch):
    _analyse(app)
    errors = []
    monkeypatch.setattr("isokinetics.gui.messagebox.showerror", lambda *a, **k: errors.append(a))
    app.beta_entries[0].delete(0, "end")
    app.beta_entries[0].insert(0, "abc")
    app._on_calculate_kth_full()
    assert errors and app.kth_button_visible and not app._kth_busy


def test_closing_the_window_while_calculating_is_clean(app, monkeypatch):
    _analyse(app)
    real = kinetics.kth_full
    monkeypatch.setattr(kinetics, "kth_full", lambda df, b: (time.sleep(0.4), real(df, b))[1])
    app._on_calculate_kth_full()
    app.destroy()                          # the worker's late post must be a harmless no-op
    time.sleep(0.6)


def test_the_triplet_joins_an_active_interval(app):
    _analyse(app)
    select_interval(app)
    assert "KTH" not in app.interval_means
    app._on_calculate_kth_full()
    assert pump_until(app, lambda: app.kth_full_result is not None, timeout=30)
    assert app.interval_means["KTH"] == pytest.approx(
        iv.mean_ea_in_interval(app.results["KTH"], app.alpha_interval))
    assert "KTH mean (\u03b1 0.20\u20130.80)" in labels(app)


# =========================================================================== #
# Version
# =========================================================================== #
def test_version_is_2_1_0_everywhere():
    assert isokinetics.__version__ == "2.1.0"
    text = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    assert re.search(r'^version\s*=\s*"2\.1\.0"', text, re.M)
