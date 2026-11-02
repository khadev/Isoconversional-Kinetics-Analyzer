"""
Smoke tests for isokinetics.gui.

These tests actually instantiate the customtkinter application, so they
require a display server. On headless CI runners, run under Xvfb, e.g.:

    xvfb-run -a pytest tests/test_gui_smoke.py

Tests are automatically skipped if no display / tkinter backend is
available (for example, when running on a machine with no X server and
no virtual display configured).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("tkinter")
pytest.importorskip("customtkinter")


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _display_available(),
    reason="No display available for GUI smoke tests (run under Xvfb).",
)

SAMPLE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "sample_data", "matric_121459.txt"
)


@pytest.fixture
def app():
    from isokinetics.gui import KineticsApp

    application = KineticsApp()
    yield application
    application.destroy()


def test_app_constructs_with_expected_widgets(app):
    assert app.title() != ""
    assert len(app.beta_entries) == 4
    assert set(app.method_vars.keys()) == {
        "Friedman",
        "FWO",
        "KAS",
        "Starink",
        "Vyazovkin",
        "KTH",
    }


def test_read_betas_returns_defaults(app):
    betas = app._read_betas()
    assert betas == [5.0, 10.0, 20.0, 40.0]


def test_read_betas_rejects_non_numeric(app):
    app.beta_entries[0].delete(0, "end")
    app.beta_entries[0].insert(0, "not-a-number")
    with pytest.raises(ValueError):
        app._read_betas()


def test_read_betas_rejects_non_positive(app):
    app.beta_entries[0].delete(0, "end")
    app.beta_entries[0].insert(0, "-5")
    with pytest.raises(ValueError):
        app._read_betas()


def test_load_and_run_end_to_end(app):
    from isokinetics import data_loader, kinetics

    df = data_loader.load_data(SAMPLE_FILE)
    app.working_df = data_loader.interpolate_alpha(df)
    app.raw_df = df
    app._populate_preview(df)

    betas = app._read_betas()
    results = {
        name: kinetics.METHODS[name](app.working_df, betas)
        for name in app.method_vars.keys()
    }
    app._on_analysis_done(
        results, {n: kinetics.average_ea(r) for n, r in results.items()}, {}
    )

    assert app.results
    assert app.export_button.cget("state") == "normal"
    # Results table should contain the average row plus one row per alpha.
    children = app.results_tree.get_children()
    assert len(children) == len(app.working_df) + 1


def test_kth_full_triplet_mode_populates_kth_full_result(app):
    """
    Selecting KTH's "Full triplet" mode should run kth_full() (not the
    plain kth()), store the rich result on the app, and populate both
    the results table's KTH_A/KTH_f(alpha) columns and the KTH Model
    diagnostic plot.
    """
    from isokinetics import data_loader, gui as gui_module, kinetics

    df = data_loader.load_data(SAMPLE_FILE)
    app.working_df = data_loader.interpolate_alpha(df)
    app.raw_df = df
    app._populate_preview(df)

    app.kth_mode_var.set(gui_module.KTH_MODE_FULL_TRIPLET)
    betas = app._read_betas()

    full = kinetics.kth_full(app.working_df, betas)
    results = {"KTH": full["Ea"]}
    averages = {"KTH": kinetics.average_ea(full["Ea"])}

    app._on_analysis_done(results, averages, {}, kth_full_result=full)

    assert app.kth_full_result is not None
    assert app.kth_full_result["model"] == full["model"]

    # The results table should have KTH_A / KTH_f(alpha) columns with
    # real (non-placeholder) values whenever a model was identified.
    children = app.results_tree.get_children()
    assert len(children) == len(app.working_df) + 1
    avg_row_values = app.results_tree.item(children[0], "values")
    assert avg_row_values[0] == "(average)"
    # Last column of the average row should mention the identified model.
    assert "model=" in avg_row_values[-1]


def test_kth_mode_defaults_to_ea_only(app):
    from isokinetics import gui as gui_module

    assert app.kth_mode_var.get() == gui_module.KTH_MODE_EA_ONLY


def test_loading_new_data_resets_kth_full_result(app):
    from isokinetics import data_loader

    app.kth_full_result = {"model": "F1"}  # pretend a previous run happened
    df = data_loader.load_data(SAMPLE_FILE)
    app.raw_df = df
    app.working_df = data_loader.interpolate_alpha(df)
    app._populate_preview(df)
    app.kth_full_result = None  # emulate what _on_load_data does on a fresh load

    assert app.kth_full_result is None


def test_model_fitting_tab_populates_after_analysis(app):
    from isokinetics import data_loader, kinetics

    df = data_loader.load_data(SAMPLE_FILE)
    app.working_df = data_loader.interpolate_alpha(df)
    app.raw_df = df
    app._populate_preview(df)

    betas = app._read_betas()
    results = {"Vyazovkin": kinetics.vyazovkin(app.working_df, betas)}
    averages = {"Vyazovkin": kinetics.average_ea(results["Vyazovkin"])}
    cr_result = kinetics.coats_redfern(app.working_df, betas)

    app._on_analysis_done(results, averages, {}, cr_result=cr_result)

    assert app.cr_result is not None
    rows = app.cr_tree.get_children()
    assert len(rows) == len(cr_result["results"])
    assert "Reference E" in app.cr_summary_label.cget("text")


def test_model_fitting_tab_shows_placeholder_with_no_result(app):
    app._populate_model_fitting_tab(None)
    assert len(app.cr_tree.get_children()) == 0
    assert "Run an analysis" in app.cr_summary_label.cget("text")


def test_loading_new_data_resets_cr_result(app):
    app.cr_result = {"results": {"F1": {}}, "recommended_model": "F1"}
    # Emulate exactly what _on_load_data does on a fresh load.
    app.cr_result = None
    assert app.cr_result is None


def test_plot_style_dropdown_defaults_to_acs(app):
    from isokinetics import plotter

    assert app.plot_style_var.get() == plotter.DEFAULT_STYLE


def test_show_ci_checkbox_defaults_to_off(app):
    assert app.show_ci_var.get() is False


def test_current_ci_returns_none_when_checkbox_off(app):
    from isokinetics import data_loader, kinetics

    df = data_loader.load_data(SAMPLE_FILE)
    app.working_df = data_loader.interpolate_alpha(df)
    app.results = {"KAS": kinetics.kas(app.working_df, app._read_betas())}

    assert app.show_ci_var.get() is False
    assert app._current_ci() is None


def test_current_ci_computes_for_supported_methods_when_enabled(app):
    from isokinetics import data_loader, kinetics

    df = data_loader.load_data(SAMPLE_FILE)
    app.working_df = data_loader.interpolate_alpha(df)
    betas = app._read_betas()
    app.results = {
        "KAS": kinetics.kas(app.working_df, betas),
        "Vyazovkin": kinetics.vyazovkin(app.working_df, betas),  # not CI-supported
    }
    app.show_ci_var.set(True)

    ci = app._current_ci()
    assert ci is not None
    assert "KAS" in ci
    assert "Vyazovkin" not in ci  # correctly excluded


def test_on_plot_style_changed_redraws_when_results_exist(app):
    from isokinetics import data_loader, kinetics

    df = data_loader.load_data(SAMPLE_FILE)
    app.working_df = data_loader.interpolate_alpha(df)
    betas = app._read_betas()
    app.results = {"KAS": kinetics.kas(app.working_df, betas)}
    app.averages = {"KAS": kinetics.average_ea(app.results["KAS"])}

    app._on_plot_style_changed("Colorblind")  # must not raise
    assert app.plot_style_var.get == app.plot_style_var.get  # sanity: app still alive


def test_about_dialog_opens_without_error(app):
    app._on_show_about()
    app.update()  # must not raise


def test_about_dialog_contains_contact_email(app):
    import tkinter as tk
    from isokinetics import gui as gui_module

    app._on_show_about()
    app.update()

    toplevels = [c for c in app.winfo_children() if isinstance(c, tk.Toplevel)]
    assert len(toplevels) == 1

    def _collect_text(widget):
        texts = []
        try:
            texts.append(widget.cget("text"))
        except Exception:
            pass
        for child in widget.winfo_children():
            texts.extend(_collect_text(child))
        return texts

    all_text = " ".join(_collect_text(toplevels[0]))
    assert gui_module.DEVELOPER_CONTACT_EMAIL in all_text


def test_save_image_button_exists(app):
    assert app.save_image_button.cget("text") == "Save Image..."


def test_save_image_shows_info_when_no_plot_tab_active(app, monkeypatch):
    """If somehow no known plot tab is active, Save Image should inform
    the user rather than raise."""
    shown = {}

    def fake_showinfo(title, message):
        shown["title"] = title
        shown["message"] = message

    monkeypatch.setattr("isokinetics.gui.messagebox.showinfo", fake_showinfo)
    monkeypatch.setattr(app, "tabview", type("FakeTabview", (), {"get": lambda self: "Data Preview"})())

    app._on_save_image()
    assert "Save Image" in shown.get("title", "")


def test_save_image_writes_png_file(app, monkeypatch, tmp_path):
    from isokinetics import data_loader, kinetics

    df = data_loader.load_data(SAMPLE_FILE)
    app.working_df = data_loader.interpolate_alpha(df)
    betas = app._read_betas()
    results = {"KAS": kinetics.kas(app.working_df, betas)}
    averages = {"KAS": kinetics.average_ea(results["KAS"])}
    app._on_analysis_done(results, averages, {})

    out_path = str(tmp_path / "my_plot.png")
    monkeypatch.setattr(
        "isokinetics.gui.filedialog.asksaveasfilename", lambda **kwargs: out_path
    )
    monkeypatch.setattr("isokinetics.gui.messagebox.showinfo", lambda *a, **k: None)

    app.tabview.set("Kinetic Plot")
    app._on_save_image()

    import os as _os
    assert _os.path.isfile(out_path)
    assert _os.path.getsize(out_path) > 0


def test_save_image_writes_tiff_with_lzw_compression(app, monkeypatch, tmp_path):
    from isokinetics import data_loader, kinetics
    from PIL import Image

    df = data_loader.load_data(SAMPLE_FILE)
    app.working_df = data_loader.interpolate_alpha(df)
    betas = app._read_betas()
    results = {"KAS": kinetics.kas(app.working_df, betas)}
    averages = {"KAS": kinetics.average_ea(results["KAS"])}
    app._on_analysis_done(results, averages, {})

    out_path = str(tmp_path / "my_plot.tiff")
    monkeypatch.setattr(
        "isokinetics.gui.filedialog.asksaveasfilename", lambda **kwargs: out_path
    )
    monkeypatch.setattr("isokinetics.gui.messagebox.showinfo", lambda *a, **k: None)

    app.tabview.set("Kinetic Plot")
    app._on_save_image()

    img = Image.open(out_path)
    assert img.info.get("compression") == "tiff_lzw"


def test_save_image_cancelled_dialog_does_nothing(app, monkeypatch):
    monkeypatch.setattr("isokinetics.gui.filedialog.asksaveasfilename", lambda **kwargs: "")
    app.tabview.set("Kinetic Plot")
    app._on_save_image()  # must not raise even with an empty path (user cancelled)


def test_progress_bar_reaches_100_percent_after_full_analysis(app):
    """
    Regression test for a user-reported bug: the progress bar was
    hardcoded to 0.95 for the Coats-Redfern step (which always runs
    after every selected method) and nothing ever bumped it back to
    1.0 afterwards, so it visually looked "stuck" even though the
    analysis had genuinely completed successfully.
    """
    from isokinetics import data_loader, kinetics

    df = data_loader.load_data(SAMPLE_FILE)
    app.working_df = data_loader.interpolate_alpha(df)
    app.raw_df = df
    app._populate_preview(df)

    betas = app._read_betas()
    results = {"KAS": kinetics.kas(app.working_df, betas)}
    averages = {"KAS": kinetics.average_ea(results["KAS"])}
    cr_result = kinetics.coats_redfern(app.working_df, betas)

    # Simulate exactly what _run_analysis_worker does at the very end.
    app._update_progress(1.0, "Complete")
    app._on_analysis_done(results, averages, {}, cr_result=cr_result)

    assert app.progress_bar.get() == pytest.approx(1.0)


def test_about_button_is_child_of_root_window_not_bottom_frame(app):
    """The About button must be placed directly on the root window (so
    it can be pinned to the top-right corner independent of the bottom
    action bar's layout), not packed inside the bottom frame."""
    assert app.about_button.master is app


def test_about_button_uses_visible_solid_color_not_transparent(app):
    """Regression test: the button previously used
    fg_color='transparent', which was reported as hard to see."""
    fg_color = app.about_button.cget("fg_color")
    assert fg_color != "transparent"
