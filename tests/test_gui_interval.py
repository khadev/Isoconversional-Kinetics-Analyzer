"""Stabilization phase in the GUI: drag-select on the plot, legend, reset, edge cases, export."""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import pandas as pd
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

from matplotlib.backend_bases import MouseEvent  # noqa: E402

from isokinetics import interval as iv, kinetics  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "sample_data", "matric_121459.txt")
METHODS = ("Friedman", "FWO", "KAS", "Starink", "Vyazovkin", "KTH")


@pytest.fixture
def app():
    from isokinetics.gui import KineticsApp
    application = KineticsApp()
    yield application
    application.destroy()


@pytest.fixture
def analysed(app):
    assert app.load_file(SAMPLE)
    betas = app._read_betas()
    results = {n: kinetics.METHODS[n](app.working_df, betas) for n in METHODS}
    averages = {n: kinetics.average_ea(r) for n, r in results.items()}
    app._on_analysis_done(results, averages, {})
    app.update()
    return app


def pump_until(app, condition, timeout=10.0):
    """Run the real Tk main loop until ``condition()`` (worker threads post results
    through ``after``, which needs a running loop)."""
    deadline = time.time() + timeout

    def tick():
        if condition() or time.time() > deadline:
            app.quit()
        else:
            app.after(20, tick)

    app.after(20, tick)
    app.mainloop()
    return condition()


def legend(app):
    return [t.get_text() for t in app.figure.axes[0].get_legend().get_texts()]


def select(app, lo, hi):
    app._on_alpha_selected(lo, hi)


# --------------------------------------------------------------------------- #
def test_initial_state_has_no_selection_and_reset_is_disabled(app):
    assert app.alpha_interval is None
    assert str(app.reset_interval_button.cget("state")) == "disabled"
    assert not app.span_controller.enabled          # nothing to select before an analysis


def test_selector_is_armed_after_an_analysis(analysed):
    assert analysed.span_controller.enabled and analysed.span_controller.active


def test_selecting_a_range_updates_legend_span_label_and_button(analysed):
    app = analysed
    full_legend = legend(app)
    full_means = dict(app.averages)
    select(app, 0.2, 0.8)
    assert pump_until(app, lambda: app.alpha_interval is not None)

    assert (app.alpha_interval.lo, app.alpha_interval.hi) == pytest.approx((0.2, 0.8))
    for name, text in zip(app.results, legend(app)):
        expected = iv.mean_ea_in_interval(app.results[name], app.alpha_interval)
        assert text == f"{name} (mean E\u2090 [0.20\u20130.80] = {expected:.2f} kJ/mol)"
    spans = [p for p in app.figure.axes[0].patches if p.get_gid() == "alpha_interval"]
    assert len(spans) == 1
    assert "0.20\u20130.80" in app.interval_label.cget("text")
    assert str(app.reset_interval_button.cget("state")) == "normal"
    assert app.averages == full_means and legend(app) != full_legend       # full-range numbers untouched
    assert app.span_controller.active                                      # re-armed after the redraw


def test_full_range_results_are_never_modified(analysed):
    app = analysed
    before_results, before_avg = copy.deepcopy(app.results), dict(app.averages)
    select(app, 0.3, 0.6)
    pump_until(app, lambda: app.alpha_interval is not None)
    app._on_reset_interval()
    assert app.results == before_results and app.averages == before_avg


def test_reset_button_restores_legend_and_removes_span(analysed):
    app = analysed
    full_legend = legend(app)
    select(app, 0.2, 0.8)
    pump_until(app, lambda: app.alpha_interval is not None)
    app.reset_interval_button.invoke()
    app.update()
    assert app.alpha_interval is None and app.interval_means == {}
    assert legend(app) == full_legend
    assert not [p for p in app.figure.axes[0].patches if p.get_gid() == "alpha_interval"]
    assert str(app.reset_interval_button.cget("state")) == "disabled"
    assert app.span_controller.active


def test_summary_cards_show_the_interval_mean(analysed):
    app = analysed
    select(app, 0.2, 0.8)
    pump_until(app, lambda: app.alpha_interval is not None)
    texts = []

    def walk(w):
        try:
            texts.append(str(w.cget("text")))
        except Exception:
            pass
        for c in w.winfo_children():
            walk(c)

    walk(app.summary_body)
    expected = iv.mean_ea_in_interval(app.results["Friedman"], app.alpha_interval)
    assert f"mean [0.20\u20130.80] = {expected:.2f}" in texts


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lo,hi", [(1.2, 1.6), (-0.6, -0.1)])
def test_selection_outside_the_data_is_rejected_with_a_message(analysed, lo, hi):
    select(analysed, lo, hi)
    analysed.update()
    assert analysed.alpha_interval is None
    assert "outside the data" in analysed.status_label.cget("text")
    assert analysed.span_controller.active


def test_selection_with_fewer_than_three_points_is_rejected(analysed):
    select(analysed, 0.201, 0.213)           # only the grid points 0.21 (and maybe none) fall inside
    analysed.update()
    assert analysed.alpha_interval is None
    assert "at least 3" in analysed.status_label.cget("text")


def test_rejected_selection_keeps_the_previous_valid_one(analysed):
    app = analysed
    select(app, 0.2, 0.8)
    pump_until(app, lambda: app.alpha_interval is not None)
    first = app.alpha_interval
    select(app, 0.5, 0.505)
    app.update()
    assert app.alpha_interval == first


def test_selection_overhanging_the_data_is_clipped_to_the_data(analysed):
    app = analysed
    select(app, 0.9, 1.4)
    assert pump_until(app, lambda: app.alpha_interval is not None)
    assert app.alpha_interval.hi == pytest.approx(float(app.working_df["alpha"].max()))
    assert app.alpha_interval.lo == pytest.approx(0.9)


def test_newest_selection_wins_when_two_are_made_quickly(analysed):
    app = analysed
    select(app, 0.1, 0.4)
    select(app, 0.5, 0.9)
    pump_until(app, lambda: app.alpha_interval is not None and app.alpha_interval.lo > 0.4)
    time.sleep(0.2)
    app.update()
    assert (app.alpha_interval.lo, app.alpha_interval.hi) == pytest.approx((0.5, 0.9))


def test_reset_while_a_selection_is_being_computed_is_not_overridden(analysed):
    app = analysed
    select(app, 0.2, 0.8)          # worker started...
    app._on_reset_interval()       # ...but the user resets first
    pump_until(app, lambda: False, timeout=0.5)
    assert app.alpha_interval is None


def test_selector_is_disabled_while_an_analysis_runs(analysed):
    analysed._set_running_state(True)
    assert not analysed.span_controller.enabled and not analysed.span_controller.active
    analysed._set_running_state(False)
    assert analysed.span_controller.enabled and analysed.span_controller.active


def test_loading_new_data_clears_the_selection(analysed):
    app = analysed
    select(app, 0.2, 0.8)
    pump_until(app, lambda: app.alpha_interval is not None)
    app.load_file(SAMPLE)
    assert app.alpha_interval is None and app.interval_means == {}
    assert not app.span_controller.enabled


def test_a_new_analysis_keeps_the_selected_range_and_recomputes_its_means(analysed):
    app = analysed
    select(app, 0.2, 0.8)
    pump_until(app, lambda: app.alpha_interval is not None)
    shifted = {n: {a: e + 10.0 for a, e in r.items()} for n, r in app.results.items()}
    app._on_analysis_done(shifted, {n: kinetics.average_ea(r) for n, r in shifted.items()}, {})
    assert app.alpha_interval is not None
    assert app.interval_means["Friedman"] == pytest.approx(
        iv.mean_ea_in_interval(shifted["Friedman"], app.alpha_interval))


def test_theme_toggle_keeps_the_selection_visible(analysed):
    app = analysed
    select(app, 0.2, 0.8)
    pump_until(app, lambda: app.alpha_interval is not None)
    app.dark_mode_var.set(True)
    app._on_theme_toggled()
    assert [p for p in app.figure.axes[0].patches if p.get_gid() == "alpha_interval"]
    assert "[0.20\u20130.80]" in legend(app)[0]
    assert app.span_controller.active


# --------------------------------------------------------------------------- #
# The real mouse path: press / move / release on the Tk canvas, double-click
# --------------------------------------------------------------------------- #
def _mouse(app, name, x, **kw):
    ax = app.figure.axes[0]
    px, py = ax.transData.transform((x, float(np.mean(ax.get_ylim()))))
    app.canvas.callbacks.process(name, MouseEvent(name, app.canvas, px, py, button=1, **kw))


def test_dragging_on_the_plot_selects_and_double_click_resets(analysed):
    app = analysed
    app.tabview.set("Kinetic Plot")
    app.update()
    for name, x in (("button_press_event", 0.25), ("motion_notify_event", 0.5),
                    ("motion_notify_event", 0.75), ("button_release_event", 0.75)):
        _mouse(app, name, x)
    assert pump_until(app, lambda: app.alpha_interval is not None)
    assert (app.alpha_interval.lo, app.alpha_interval.hi) == pytest.approx((0.25, 0.75), abs=0.02)
    assert "Selecting" not in app.status_label.cget("text")      # live message replaced by the result

    _mouse(app, "button_press_event", 0.5, dblclick=True)
    app.update()
    assert app.alpha_interval is None
    assert not [p for p in app.figure.axes[0].patches if p.get_gid() == "alpha_interval"]


def test_live_feedback_while_dragging(analysed):
    analysed._on_alpha_dragging(0.2, 0.55)
    assert "0.20\u20130.55" in analysed.status_label.cget("text")


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def _export(app, monkeypatch, tmp_path):
    monkeypatch.setattr("isokinetics.gui.filedialog.askdirectory", lambda **k: str(tmp_path))
    monkeypatch.setattr("isokinetics.gui.messagebox.showinfo", lambda *a, **k: None)
    monkeypatch.setattr("isokinetics.gui.messagebox.showwarning", lambda *a, **k: None)
    app._on_export()
    return next(p for p in tmp_path.iterdir() if p.name.startswith("Kinetics_Results_"))


def test_export_after_a_selection_reports_the_stabilization_phase(analysed, monkeypatch, tmp_path):
    app = analysed
    select(app, 0.2, 0.8)
    pump_until(app, lambda: app.alpha_interval is not None)
    folder = _export(app, monkeypatch, tmp_path)
    summary = (folder / "Summary.txt").read_text(encoding="utf-8")
    comment = "# Selected stabilization interval: alpha 0.20 to 0.80"
    assert summary.splitlines()[0] == comment
    assert "Stabilization phase: alpha = 0.20\u20130.80" in summary          # metadata line (also in image tags)
    expected = iv.mean_ea_in_interval(app.results["KAS"], app.alpha_interval)
    assert f"{expected:.2f}" in summary.split("Selected stabilization interval: alpha 0.20 to 0.80 (")[1]
    data = (folder / "Ea_data.txt").read_text(encoding="utf-8").splitlines()
    assert data[0] == comment
    avg = next(l for l in data if l.startswith("(average)")).split("\t")
    header = next(l for l in data if l.startswith("alpha")).split("\t")
    assert float(avg[header.index("KAS")]) == pytest.approx(expected, abs=5e-5)   # (average) = interval mean
    sheet = pd.read_excel(folder / "Ea_results.xlsx", sheet_name="Averages", header=1)
    assert sheet.loc[sheet["method"] == "KAS", "average_Ea_kJ_per_mol"].iloc[0] == pytest.approx(expected)
    assert (folder / "Ea_vs_alpha.png").exists()


def test_export_without_a_selection_is_the_plain_full_range(analysed, monkeypatch, tmp_path):
    folder = _export(analysed, monkeypatch, tmp_path)
    assert "tabilization" not in (folder / "Summary.txt").read_text(encoding="utf-8")


def test_exported_png_of_a_selection_is_white_with_the_span_and_screen_is_restored(analysed, monkeypatch, tmp_path):
    from PIL import Image
    app = analysed
    app.dark_mode_var.set(True)
    app._on_theme_toggled()
    select(app, 0.2, 0.8)
    pump_until(app, lambda: app.alpha_interval is not None)
    out = str(tmp_path / "shot.png")
    monkeypatch.setattr("isokinetics.gui.filedialog.asksaveasfilename", lambda **k: out)
    monkeypatch.setattr("isokinetics.gui.messagebox.showinfo", lambda *a, **k: None)
    app.tabview.set("Kinetic Plot")
    app._on_save_image()
    assert Image.open(out).convert("RGB").getpixel((2, 2)) == (255, 255, 255)
    assert [p for p in app.figure.axes[0].patches if p.get_gid() == "alpha_interval"]
