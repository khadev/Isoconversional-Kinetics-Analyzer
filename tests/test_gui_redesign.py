"""Tests for the redesigned GUI: theme toggle, progress feedback, new tabs,
number formatting, and light-theme exports.  Run under Xvfb on headless CI."""

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


pytestmark = pytest.mark.skipif(not _display_available(), reason="No display (run under Xvfb).")

SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "..", "sample_data", "matric_121459.txt")


@pytest.fixture
def app():
    from isokinetics.gui import KineticsApp
    application = KineticsApp()
    yield application
    application.destroy()


@pytest.fixture
def loaded(app):
    assert app.load_file(SAMPLE_FILE)
    return app


def _run_analysis_sync(app, selected=("Friedman", "KAS", "Vyazovkin")):
    from isokinetics import kinetics
    betas = app._read_betas()
    results = {n: kinetics.METHODS[n](app.working_df, betas) for n in selected}
    averages = {n: kinetics.average_ea(r) for n, r in results.items()}
    cr = kinetics.coats_redfern(app.working_df, betas)
    app._on_analysis_done(results, averages, {}, cr_result=cr)
    return results


def test_all_expected_tabs_exist_in_logical_order(app):
    from isokinetics import gui
    names = list(app.tabview._tab_dict.keys())
    assert names == [gui.TAB_PREVIEW, gui.TAB_SUMMARY, gui.TAB_PLOT, gui.TAB_RESULTS,
                     gui.TAB_MODEL_FITTING, gui.TAB_LIFETIME, gui.TAB_KTH, gui.TAB_VALIDATION]


def test_theme_toggle_switches_figures_and_back(loaded):
    from isokinetics import plotter
    _run_analysis_sync(loaded)
    loaded.dark_mode_var.set(True)
    loaded._on_theme_toggled()
    assert plotter.get_theme() == "dark"
    assert loaded.figure.get_facecolor()[:3] != (1.0, 1.0, 1.0)
    loaded.dark_mode_var.set(False)
    loaded._on_theme_toggled()
    assert plotter.get_theme() == "light"
    assert loaded.figure.get_facecolor()[:3] == (1.0, 1.0, 1.0)


def test_exported_image_is_white_even_in_dark_theme(loaded, monkeypatch, tmp_path):
    from PIL import Image
    _run_analysis_sync(loaded)
    loaded.dark_mode_var.set(True)
    loaded._on_theme_toggled()
    out = str(tmp_path / "dark_screen.png")
    monkeypatch.setattr("isokinetics.gui.filedialog.asksaveasfilename", lambda **k: out)
    monkeypatch.setattr("isokinetics.gui.messagebox.showinfo", lambda *a, **k: None)
    loaded.tabview.set("Kinetic Plot")
    loaded._on_save_image()
    img = Image.open(out).convert("RGB")
    assert img.getpixel((2, 2)) == (255, 255, 255)
    # ...and the on-screen figure is back to dark afterwards
    assert loaded.figure.get_facecolor()[:3] != (1.0, 1.0, 1.0)


def test_tables_use_consistent_number_format(loaded):
    _run_analysis_sync(loaded)
    rows = loaded.results_tree.get_children()
    # Tree.set() returns the stored text (item(..., "values") would coerce to numbers)
    assert loaded.results_tree.set(rows[1], "alpha") == "0.010"  # alpha: 3 decimals
    for col in ("Friedman", "KAS", "Vyazovkin"):
        text = loaded.results_tree.set(rows[1], col)
        assert len(text.split(".")[1]) == 2, (col, text)  # Ea: 2 decimals
    pre_row = loaded.preview_tree.get_children()[0]
    assert len(loaded.preview_tree.set(pre_row, "T1").split(".")[1]) == 2  # T: 2 decimals


def test_summary_tab_shows_a_card_per_method_and_insights(loaded):
    _run_analysis_sync(loaded)
    texts = []

    def walk(w):
        try:
            texts.append(str(w.cget("text")))
        except Exception:
            pass
        for c in w.winfo_children():
            walk(c)

    walk(loaded.summary_body)
    joined = " ".join(texts)
    for name in ("Friedman", "KAS", "Vyazovkin"):
        assert name in joined
    assert "What the results say" in joined
    assert loaded.tabview.get() == "Summary"


def test_progress_stage_text_and_eta_fields(app):
    app._begin_stage("Vyazovkin", 3, 7, 0.40)
    assert "Vyazovkin" in app.status_label.cget("text")
    assert "step 3 of 7" in app.status_label.cget("text")
    assert app.progress_bar.get() == pytest.approx(0.40)
    assert app.progress_label.cget("text") == "40 %"
    import time
    app._run_started = time.perf_counter() - 2.0
    app._update_progress(0.5, "KAS")
    assert "left" in app.eta_label.cget("text")
    app._update_progress(1.0, "Complete")
    assert app.progress_label.cget("text") == "100 %"
    assert "Done in" in app.eta_label.cget("text")


def test_progress_is_monotonic_during_a_real_run(app):
    """Run the real worker in-process and collect the fractions it reports."""
    from isokinetics import data_loader
    df = data_loader.load_data(SAMPLE_FILE)
    app.raw_df = df
    app.working_df = data_loader.interpolate_alpha(df)
    seen = []
    app._post = lambda fn, *a: seen.append((fn.__name__, a))
    app._run_analysis_worker(["Friedman", "KAS", "Vyazovkin", "KTH"], [5., 10., 20., 40.], "Ea only")
    fractions = [a[0] for name, a in seen if name == "_update_progress"]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    stage_names = [a[0] for name, a in seen if name == "_begin_stage"]
    assert stage_names == ["Friedman", "KAS", "Vyazovkin", "KTH", "Coats-Redfern"]


def test_validation_tab_populates_from_report(app):
    from isokinetics import validation
    report = validation.run_validation(scenarios=validation.default_scenarios()[:1])
    app._on_validation_done(report, "")
    rows = app.validation_tree.get_children()
    assert len(rows) == len(report.records)
    assert "checks passed" in app.validation_status_label.cget("text")


def test_duplicate_heating_rates_rejected(app):
    app.beta_entries[1].delete(0, "end")
    app.beta_entries[1].insert(0, "5")
    with pytest.raises(ValueError):
        app._read_betas()


def test_beta_order_warning(loaded):
    assert loaded._check_beta_order([5., 10., 20., 40.]) is None
    assert loaded._check_beta_order([40., 20., 10., 5.]) is not None


def test_methods_all_none_buttons(app):
    app._set_all_methods(False)
    assert not any(v.get() for v in app.method_vars.values())
    app._set_all_methods(True)
    assert all(v.get() for v in app.method_vars.values())


def test_copy_table_puts_tsv_on_clipboard(loaded):
    _run_analysis_sync(loaded)
    loaded._on_copy_table()
    text = loaded.clipboard_get()
    lines = text.splitlines()
    assert lines[0].split("\t")[0] == "\u03b1"
    assert len(lines) == len(loaded.results_tree.get_children()) + 1


def test_export_writes_summary_and_model_fitting(loaded, monkeypatch, tmp_path):
    _run_analysis_sync(loaded)
    monkeypatch.setattr("isokinetics.gui.filedialog.askdirectory", lambda **k: str(tmp_path))
    monkeypatch.setattr("isokinetics.gui.messagebox.showinfo", lambda *a, **k: None)
    monkeypatch.setattr("isokinetics.gui.messagebox.showwarning", lambda *a, **k: None)
    loaded._on_export()
    folder = next(p for p in tmp_path.iterdir() if p.name.startswith("Kinetics_Results_"))
    names = {p.name for p in folder.iterdir()}
    assert "Summary.txt" in names
    assert "Coats_Redfern_plot.png" in names
    import pandas as pd
    assert "Model_Fitting" in pd.ExcelFile(folder / "Ea_results.xlsx").sheet_names


# --------------------------------------------------------------------------- #
# Raw (4-column) vs processed (5-column) loading
# --------------------------------------------------------------------------- #
RAW_FILE = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_raw_4columns_F1_endpoints.txt")
INTERIOR_FILE = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_raw_4columns_interior.txt")


def test_processed_file_skips_the_dialog(app, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("dialog must not open for a processed file")
    monkeypatch.setattr(app, "_ask_alpha_step", boom)
    assert app.load_file(SAMPLE_FILE)
    assert app.source_kind == "processed" and app.alpha_step is None
    assert not app.raw_actions_frame.winfo_manager()  # raw-only buttons hidden


def test_raw_file_asks_for_step_and_generates_alpha(app, monkeypatch):
    asked = []
    monkeypatch.setattr(app, "_ask_alpha_step", lambda table, name, step, mapping: asked.append((name, step)) or (0.02, "endpoints"))
    assert app.load_file(RAW_FILE)
    assert asked == [("sample_raw_4columns_F1_endpoints.txt", 0.01)]  # default step proposed
    assert app.source_kind == "raw" and app.alpha_step == 0.02
    assert len(app.working_df) == 49
    assert app.working_df["alpha"].iloc[0] == pytest.approx(0.02) and app.working_df["alpha"].iloc[-1] == pytest.approx(0.98)
    assert len(app.preview_tree.get_children()) == 49
    assert "Raw file" in app.data_info_label.cget("text") and "49 points" in app.data_info_label.cget("text")
    assert app.raw_actions_frame.winfo_manager()  # "Change step" / "Save processed" visible


def test_raw_analysis_runs_on_generated_grid_and_recovers_ea(app, monkeypatch):
    monkeypatch.setattr(app, "_ask_alpha_step", lambda *a: (0.01, "endpoints"))
    app.load_file(RAW_FILE)
    results = _run_analysis_sync(app, ("Friedman", "KAS", "Vyazovkin"))
    for name in ("Friedman", "Vyazovkin"):
        vals = [e for a, e in results[name].items() if 0.1 <= a <= 0.9]
        assert abs(sum(vals) / len(vals) / 150.0 - 1) < 0.005
    assert sorted(results["Friedman"])[0] == pytest.approx(0.01)  # analysed on the chosen grid (not re-gridded)


def test_cancelling_the_dialog_keeps_current_data(app, monkeypatch):
    app.load_file(SAMPLE_FILE)
    before = app.working_df.copy()
    monkeypatch.setattr(app, "_ask_alpha_step", lambda *a: None)
    assert app.load_file(RAW_FILE) is False
    assert app.file_path == SAMPLE_FILE
    assert app.working_df.equals(before)


def test_change_alpha_step_regenerates_and_clears_results(app, monkeypatch):
    answers = iter([(0.02, "endpoints"), (0.05, "endpoints")])
    monkeypatch.setattr(app, "_ask_alpha_step", lambda *a: next(answers))
    app.load_file(RAW_FILE)
    _run_analysis_sync(app, ("Friedman",))
    assert app.results
    app._on_change_alpha_step()
    assert app.alpha_step == 0.05 and len(app.working_df) == 19
    assert app.results == {}  # stale results from the old grid are discarded
    assert app._last_alpha_step == 0.05  # remembered as the next default


def test_interior_mapping_through_the_gui_reproduces_processed_sample(app, monkeypatch):
    monkeypatch.setattr(app, "_ask_alpha_step", lambda *a: (0.01, "interior"))
    app.load_file(INTERIOR_FILE)
    proc = app.raw_df
    from isokinetics import data_loader
    ref = data_loader.load_data(SAMPLE_FILE)
    assert len(proc) == 99
    assert (proc[["T1", "T2", "T3", "T4"]].to_numpy() == pytest.approx(ref[["T1", "T2", "T3", "T4"]].to_numpy()))


def test_save_processed_table_from_gui_reloads_as_processed(app, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_ask_alpha_step", lambda *a: (0.02, "endpoints"))
    app.load_file(RAW_FILE)
    out = str(tmp_path / "gen.txt")
    monkeypatch.setattr("isokinetics.gui.filedialog.asksaveasfilename", lambda **k: out)
    monkeypatch.setattr("isokinetics.gui.messagebox.showinfo", lambda *a, **k: None)
    app._on_save_processed()
    from isokinetics import data_loader
    assert data_loader.load_table(out).kind == "processed"
    monkeypatch.setattr(app, "_ask_alpha_step", lambda *a: (_ for _ in ()).throw(AssertionError("no dialog expected")))
    assert app.load_file(out) and app.source_kind == "processed"


def test_bad_raw_file_shows_error_dialog(app, monkeypatch, tmp_path):
    shown = []
    monkeypatch.setattr("isokinetics.gui.messagebox.showerror", lambda title, msg: shown.append(msg))
    bad = tmp_path / "three.txt"
    bad.write_text("\n".join("300\t310\t320" for _ in range(10)))
    assert app.load_file(str(bad)) is False
    assert shown and "column" in shown[0].lower()


def test_export_metadata_mentions_generated_alpha(app, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_ask_alpha_step", lambda *a: (0.02, "endpoints"))
    app.load_file(RAW_FILE)
    _run_analysis_sync(app, ("Friedman", "KAS"))
    monkeypatch.setattr("isokinetics.gui.filedialog.askdirectory", lambda **k: str(tmp_path))
    monkeypatch.setattr("isokinetics.gui.messagebox.showinfo", lambda *a, **k: None)
    monkeypatch.setattr("isokinetics.gui.messagebox.showwarning", lambda *a, **k: None)
    app._on_export()
    folder = next(p for p in tmp_path.iterdir() if p.name.startswith("Kinetics_Results_"))
    assert "raw 4-column file, alpha generated with step 0.02" in (folder / "Summary.txt").read_text(encoding="utf-8")


# ---- the dialog itself ------------------------------------------------------
@pytest.fixture
def dialog(app):
    from isokinetics import data_loader, gui
    table = data_loader.load_table(RAW_FILE)
    dlg = gui.AlphaStepDialog(app, table, "raw.txt", "0.01")
    yield dlg
    try:
        dlg.destroy()
    except Exception:
        pass


@pytest.mark.parametrize("text,step,points", [("0.02", 0.02, 49), ("1/50", 0.02, 49), ("50", 0.02, 49),
                                              ("100", 0.01, 99), ("1/100", 0.01, 99)])
def test_dialog_accepts_step_fraction_or_point_count(dialog, text, step, points):
    dialog.step_entry.delete(0, "end")
    dialog.step_entry.insert(0, text)
    dialog._update_preview()
    assert f"{points} points" in dialog.preview_label.cget("text")
    dialog._on_ok()
    assert dialog.result == (pytest.approx(step), "endpoints")


def test_dialog_invalid_input_shows_error_and_blocks_ok(dialog):
    dialog.step_entry.delete(0, "end")
    dialog.step_entry.insert(0, "abc")
    dialog._update_preview()
    assert "not a valid step" in dialog.error_label.cget("text")
    assert str(dialog.ok_button.cget("state")) == "disabled"
    dialog._on_ok()
    assert dialog.result is None and dialog.winfo_exists()  # stays open


def test_dialog_cancel_and_mapping_choice(dialog):
    from isokinetics import data_loader
    dialog.mapping_var.set(data_loader.ROW_MAPPINGS["interior"])
    dialog._update_preview()
    assert "points" in dialog.preview_label.cget("text")
    dialog._on_ok()
    assert dialog.result[1] == "interior"


def test_dialog_cancel_returns_none(app):
    from isokinetics import data_loader, gui
    dlg = gui.AlphaStepDialog(app, data_loader.load_table(RAW_FILE), "raw.txt")
    dlg._on_cancel()
    assert dlg.result is None


def test_dialog_warns_when_step_finer_than_data(dialog):
    dialog.step_entry.delete(0, "end")
    dialog.step_entry.insert(0, "0.001")
    dialog._update_preview()
    assert "finer than the data" in dialog.note_label.cget("text")


def test_dialog_preselects_interior_layout_for_a_99_row_raw_file(app, monkeypatch):
    seen = []
    monkeypatch.setattr(app, "_ask_alpha_step", lambda table, name, step, mapping: seen.append(mapping) or None)
    app.load_file(INTERIOR_FILE)   # 99 rows
    app.load_file(RAW_FILE)        # 101 rows: nothing special -> default layout
    assert seen == ["interior", "endpoints"]
