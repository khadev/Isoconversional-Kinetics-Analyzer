"""Raw instrument import: loader (encodings, headers, units), conversion physics against a
KNOWN activation energy, the wizard dialog and its hand-off to the main application."""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd
import pytest

import dsc_fixtures as F
from isokinetics import data_loader as D
from isokinetics import kinetics

ROOT = os.path.join(os.path.dirname(__file__), "..")
REAL_DIR = os.path.join(ROOT, "sample_data", "dsc_instrument")
RATES = list(F.RATES)
METHODS = ("Friedman", "FWO", "KAS", "Starink", "Vyazovkin", "KTH")
MIXED = [F.write_utf16_with_header, F.write_utf16_no_header, F.write_utf8_decimal_comma, F.write_xlsx]


def mean_ea(table, method, lo=0.1, hi=0.9):
    res = kinetics.METHODS[method](table, RATES)
    return float(np.mean([e for a, e in res.items() if lo <= a <= hi]))


def ranges_for(spans, margin=2.0):
    """Per-file intervals (kelvin) covering the whole reaction of the synthetic curves."""
    return [(s[0] - margin + 273.15, s[1] + margin + 273.15) for s in spans]


@pytest.fixture(scope="module")
def mixed_files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("mixed")
    return F.make_four_files(folder, writers=MIXED)


# =========================================================================== #
# Loader: encodings, headers, units
# =========================================================================== #
def test_utf16_file_with_metadata_and_header_is_mapped_automatically(mixed_files):
    paths, _ = mixed_files
    table = D.load_raw_table(paths[0])
    assert not table.generic_names and table.header_row is not None
    assert [("temp" in n.lower() or "mp" in n.lower()) for n in table.names][0]
    x, y = D.detect_columns(table.names)
    assert (x, y) == (0, 1) and "HeatFlow" in table.names[y]
    assert table.heating_rate == 3.0


def test_file_without_header_gets_generic_column_names_and_does_not_crash(mixed_files):
    paths, _ = mixed_files
    table = D.load_raw_table(paths[1])                         # UTF-16, no header at all
    assert table.generic_names and table.names == ["Column 1", "Column 2"]
    assert D.detect_columns(table.names) == (0, 1)
    assert table.header_row is None and any("generic" in n for n in table.notes)
    curve = D.curve_from_table(table, 0, 1)                    # still usable once mapped
    assert len(curve) > 100 and curve.attrs["heating_rate"] == 5.0


def test_utf8_semicolon_decimal_comma_file(mixed_files):
    paths, _ = mixed_files
    table = D.load_raw_table(paths[2])
    assert table.names[0].startswith("Temp") and table.n_rows > 100
    assert table.data.iloc[:, 0].between(-100, 1000).all()    # '123,45' was read as 123.45


def test_xlsx_with_the_header_on_row_4_and_kelvin_header(tmp_path):
    t_c, hf, _ = F.synthetic_curve(5.0)
    path = str(tmp_path / "x_5_C.xlsx")
    F.write_xlsx(path, t_c, hf, header_row=3, kelvin=True)
    table = D.load_raw_table(path)
    assert table.header_row == 3 and not table.generic_names
    curve = D.load_raw_instrument_excel(path)                  # "(K)" in the header -> no +273.15
    assert curve[D.TEMP_COL].iloc[0] == pytest.approx(t_c[0] + 273.15, abs=1e-6)


def test_legacy_xls_file(tmp_path):
    pytest.importorskip("xlrd")
    pytest.importorskip("xlwt")
    t_c, hf, _ = F.synthetic_curve(7.0)
    path = str(tmp_path / "Ech_7_C.xls")
    F.write_xls(path, t_c, hf)
    curve = D.load_raw_instrument_excel(path)
    assert curve.attrs["heating_rate"] == 7.0 and len(curve) > 100


def test_celsius_is_converted_to_kelvin_exactly(mixed_files):
    paths, _ = mixed_files
    t_c, _, _ = F.synthetic_curve(3.0)
    curve = D.load_raw_instrument_excel(paths[0])
    assert curve[D.TEMP_COL].iloc[0] == pytest.approx(t_c[0] + 273.15, abs=1e-4)
    assert curve[D.TEMP_COL].iloc[-1] == pytest.approx(t_c[-1] + 273.15, abs=1e-4)
    kelvin = D.curve_from_table(D.load_raw_table(paths[0]), 0, 1, unit="K")      # unit override
    assert kelvin[D.TEMP_COL].iloc[0] == pytest.approx(t_c[0], abs=1e-4)


def test_pre_run_zeros_and_cooling_segment_are_removed(mixed_files):
    paths, _ = mixed_files
    table = D.load_raw_table(paths[0])
    assert (table.data.iloc[:, 1] == 0).sum() > 400            # the raw file has the flat blocks
    curve = D.curve_from_table(table, 0, 1)
    assert (curve[D.HEATFLOW_COL] == 0).sum() == 0
    assert (np.diff(curve[D.TEMP_COL]) > 0).all()              # strictly increasing: one ramp
    full = D.curve_from_table(table, 0, 1, heating_only=False)
    assert len(full) == table.n_rows and len(curve) < len(full)
    notes = " ".join(curve.attrs["notes"])
    assert "cooling segment removed" in notes and "heat flow exactly 0" in notes


def test_nan_rows_are_dropped(tmp_path):
    t_c, hf, _ = F.synthetic_curve(5.0)
    path = str(tmp_path / "gaps_5_C.xlsx")
    F.write_xlsx(path, t_c, hf)
    df = pd.read_excel(path, header=None)
    df.iloc[[10, 50, 80], 1] = np.nan
    df.to_excel(path, header=False, index=False)
    table = D.load_raw_table(path)
    assert table.data.notna().all().all()
    assert table.n_rows == len(df) - 4 - 3                     # 3 metadata rows + header row, and 3 blank cells


def test_text_with_metadata_numbers_is_not_mistaken_for_data(tmp_path):
    path = tmp_path / "m_5_C.txt"
    path.write_text("Masse initiale : 0.895 mg\nDate 16/02/2026\nTemp (\u00b0C)\tHeatFlow (mW)\n"
                    "30.0\t0\n31.0\t-1.5\n32.0\t-1.4\n33.0\t-1.2\n", encoding="utf-8")
    table = D.load_raw_table(str(path))
    assert table.n_rows == 4 and table.names == ["Temp (\u00b0C)", "HeatFlow (mW)"]


@pytest.mark.parametrize("names,expected", [
    (["Temp\u00e9rature de l'\u00e9chantillon (\u00b0C)", "HeatFlow |c (mW)"], (0, 1)),
    (["HeatFlow (mW)", "Temperature (K)"], (1, 0)),
    (["Time (s)", "Temperature (\u00b0C)", "Heat Flow (mW)"], (1, 2)),
    (["Column 1", "Column 2"], (0, 1)),
    (["Time", "Signal", "Flux de chaleur"], (0, 2)),
])
def test_detect_columns(names, expected):
    assert D.detect_columns(names) == expected


def test_x_and_y_must_differ(mixed_files):
    table = D.load_raw_table(mixed_files[0][0])
    with pytest.raises(ValueError, match="different columns"):
        D.curve_from_table(table, 0, 0)


@pytest.mark.parametrize("name,rate", [
    ("Ech-DRD-E01-1_5_C_100711.txt", 5.0), ("Ech-DRD-E01-1_10_C_100559.txt", 10.0),
    ("Ech-DRD-E01-1_3_C_100639.txt", 3.0), ("sample 7\u00b0C.xlsx", 7.0), ("run_2.5_C_x.txt", 2.5),
    ("no_rate_here.txt", None), ("E01-1_100711.txt", None),
])
def test_heating_rate_from_name(name, rate):
    assert D.heating_rate_from_name(name) == rate


def test_files_are_ordered_by_heating_rate_not_alphabetically():
    names = ["Ech_1_10_C_1.txt", "Ech_1_3_C_2.txt", "Ech_1_5_C_3.txt", "Ech_1_7_C_4.txt"]
    assert sorted(names)[0].endswith("_10_C_1.txt")                       # the trap
    assert [D.heating_rate_from_name(p) for p in D.order_by_heating_rate(names)] == [3, 5, 7, 10]
    unnamed = ["a_10.txt", "a_3.txt", "a_5.txt", "a_7.txt"]               # no rates: natural order
    assert [os.path.basename(p) for p in D.order_by_heating_rate(unnamed)] == ["a_3.txt", "a_5.txt", "a_7.txt", "a_10.txt"]


def test_the_bundled_real_sample_files_load():
    paths = D.order_by_heating_rate(glob.glob(os.path.join(REAL_DIR, "*.txt")))
    assert [D.heating_rate_from_name(p) for p in paths] == [3, 5, 7, 10]
    generic = [D.load_raw_table(p).generic_names for p in paths]
    assert generic == [True, False, False, False]                          # only the _3 export has no header
    for p in paths:
        curve = D.load_raw_instrument_excel(p)
        assert (np.diff(curve[D.TEMP_COL]) > 0).all() and (curve[D.HEATFLOW_COL] != 0).all()
        assert 300 < curve[D.TEMP_COL].min() < 320 and 660 < curve[D.TEMP_COL].max() < 680


def test_unreadable_files_raise_a_clear_error(tmp_path):
    empty = tmp_path / "e_5_C.txt"
    empty.write_text("nothing numeric here\nat all\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no table of numbers"):
        D.load_raw_table(str(empty))
    assert not D.is_instrument_file(str(empty))
    assert not D.is_instrument_file(str(tmp_path / "x.png"))


# =========================================================================== #
# Conversion: alpha is the cumulative peak area (validated against a known Ea)
# =========================================================================== #
def test_alpha_rises_monotonically_from_0_to_1_whatever_the_peak_sign():
    t_c, hf, span = F.synthetic_curve(5.0)
    t = t_c + 273.15
    sel = (t_c >= span[0] - 2) & (t_c <= span[1] + 2)
    a_neg, _, _ = D.conversion_from_heat_flow(t[sel], hf[sel])
    a_pos, _, _ = D.conversion_from_heat_flow(t[sel], -hf[sel])            # exothermic version
    assert a_neg[0] == 0 and a_neg[-1] == pytest.approx(1.0)
    assert (np.diff(a_neg) >= 0).all()
    assert a_neg == pytest.approx(a_pos, abs=1e-9)


def test_alpha_ignores_a_constant_offset_of_the_heat_flow():
    t_c, hf, span = F.synthetic_curve(5.0)
    t = t_c + 273.15
    sel = (t_c >= span[0] - 2) & (t_c <= span[1] + 2)
    a, _, _ = D.conversion_from_heat_flow(t[sel], hf[sel])
    shifted, _, _ = D.conversion_from_heat_flow(t[sel], hf[sel] + 7.5)
    assert shifted == pytest.approx(a, abs=1e-9)


def test_normalised_heat_flow_is_not_a_conversion():
    """Why the engine integrates: min-max scaling of a peak rises AND falls."""
    t_c, hf, span = F.synthetic_curve(5.0)
    sel = (t_c >= span[0] - 2) & (t_c <= span[1] + 2)
    y = hf[sel]
    naive = (y - y.min()) / (y.max() - y.min())
    assert not (np.diff(naive) >= -1e-9).all()


def test_all_methods_recover_the_known_ea_from_files_of_four_formats(mixed_files):
    """UTF-16 with header, UTF-16 without, UTF-8 with decimal commas, xlsx -> the same Ea."""
    paths, spans = mixed_files
    ordered = D.order_by_heating_rate(paths)
    curves = [D.load_raw_instrument_excel(p) for p in ordered]
    res = D.build_instrument_table(curves, RATES, ranges_for(spans))
    assert res.warnings == []
    for method in METHODS:
        got = mean_ea(res.table, method)
        tol = 0.03 if method == "FWO" else 0.015
        assert got == pytest.approx(F.EA_TRUE, rel=tol), method


def test_sigmoidal_baseline_handles_a_heat_capacity_step_that_linear_cannot(tmp_path):
    paths, spans = F.make_four_files(tmp_path, baseline=(-3.0, 0.0), step_baseline=1.5)
    curves = [D.load_raw_instrument_excel(p) for p in D.order_by_heating_rate(paths)]
    rng = ranges_for(spans)
    sig = D.build_instrument_table(curves, RATES, rng, baseline="sigmoidal")
    lin = D.build_instrument_table(curves, RATES, rng, baseline="linear")
    assert mean_ea(sig.table, "Friedman") == pytest.approx(F.EA_TRUE, rel=0.015)
    assert abs(mean_ea(lin.table, "Friedman") - F.EA_TRUE) > 10          # linear is clearly off here


def test_per_file_intervals_may_differ_and_are_cropped_strictly(mixed_files):
    paths, spans = mixed_files
    curves = [D.load_raw_instrument_excel(p) for p in D.order_by_heating_rate(paths)]
    wide = [(s[0] - 2 + 273.15, s[1] + 2 + 273.15) for s in spans]
    uneven = [(lo - 4.0 * i, hi + 6.0 * i) for i, (lo, hi) in enumerate(wide)]
    a = D.build_instrument_table(curves, RATES, wide)
    b = D.build_instrument_table(curves, RATES, uneven)
    assert b.counts[3] > a.counts[3] and b.counts[0] == a.counts[0]
    assert mean_ea(b.table, "Friedman") == pytest.approx(mean_ea(a.table, "Friedman"), rel=0.01)
    narrow = D.build_instrument_table(curves, RATES, [(r[0] + 1, r[1] - 1) for r in wide])
    assert all(n <= m for n, m in zip(narrow.counts, a.counts))


def test_table_has_the_standard_layout_sorted_by_heating_rate(mixed_files):
    paths, spans = mixed_files
    ordered = D.order_by_heating_rate(paths)
    curves = [D.load_raw_instrument_excel(p) for p in ordered]
    rng = ranges_for(spans)
    shuffled = [2, 0, 3, 1]                                   # hand them over in a scrambled order
    res = D.build_instrument_table([curves[i] for i in shuffled], [RATES[i] for i in shuffled],
                                   [rng[i] for i in shuffled])
    assert list(res.table.columns) == ["alpha", "T1", "T2", "T3", "T4"]
    assert res.betas == RATES and [os.path.basename(f) for f in res.files] == [os.path.basename(p) for p in ordered]
    assert res.table["alpha"].tolist() == pytest.approx(D.alpha_grid(0.01).tolist())
    assert len(res.table) == 99 and res.table.notna().all().all()
    assert (res.table[["T1", "T2", "T3", "T4"]].diff().dropna() >= 0).all().all()
    D.validate_data(res.table)                                # the engine's own validator accepts it
    assert len(D.build_instrument_table(curves, RATES, rng, step=0.02).table) == 49
    assert "4 instrument files" in res.describe()


def test_a_selection_with_fewer_than_ten_points_is_refused(mixed_files):
    paths, spans = mixed_files
    curves = [D.load_raw_instrument_excel(p) for p in D.order_by_heating_rate(paths)]
    tiny = [(s[0] + 273.15, s[0] + 273.15 + 0.2) for s in spans]
    with pytest.raises(D.SelectionTooSmall, match="at least 10"):
        D.build_instrument_table(curves, RATES, tiny)
    with pytest.raises(D.SelectionTooSmall):
        D.conversion_from_heat_flow(np.arange(5.0) + 300, np.arange(5.0))
    # exactly 10 points is accepted by the check
    t0 = float(curves[0][D.TEMP_COL].iloc[100])
    t9 = float(curves[0][D.TEMP_COL].iloc[109])
    assert D.check_selection(curves[0], t0, t9) == 10
    with pytest.raises(D.SelectionTooSmall):
        D.check_selection(curves[0], t0, float(curves[0][D.TEMP_COL].iloc[108]))


def test_other_input_errors(mixed_files):
    paths, spans = mixed_files
    curves = [D.load_raw_instrument_excel(p) for p in D.order_by_heating_rate(paths)]
    rng = ranges_for(spans)
    with pytest.raises(ValueError, match="Exactly 4"):
        D.build_instrument_table(curves[:3], RATES[:3], rng[:3])
    with pytest.raises(ValueError, match="all be different"):
        D.build_instrument_table(curves, [5, 5, 7, 10], rng)
    with pytest.raises(ValueError, match="positive"):
        D.build_instrument_table(curves, [0, 5, 7, 10], rng)
    with pytest.raises(ValueError, match="empty"):
        D.build_instrument_table(curves, RATES, [(rng[0][0], rng[0][0])] + rng[1:])
    flat = curves[0].copy()
    flat[D.HEATFLOW_COL] = -3.0 + 0.001 * np.arange(len(flat))
    with pytest.raises(ValueError, match="no measurable area"):
        D.build_instrument_table([flat] + curves[1:], RATES, rng)
    with pytest.raises(ValueError, match="Unknown baseline"):
        D.build_instrument_table(curves, RATES, rng, baseline="cubic")


def test_swapped_rates_trigger_the_ordering_warning(mixed_files):
    paths, spans = mixed_files
    curves = [D.load_raw_instrument_excel(p) for p in D.order_by_heating_rate(paths)]
    res = D.build_instrument_table(curves, [3, 10, 7, 5], ranges_for(spans))   # 5 and 10 mislabeled
    assert any("does not rise with the heating rate" in w for w in res.warnings)


def test_real_sample_files_run_through_the_pipeline_and_the_ordering_is_checked():
    """Real data: the cleaned curves convert to monotone alpha(T) and the tool reports honestly
    when the four runs do not behave like one kinetic process."""
    paths = D.order_by_heating_rate(glob.glob(os.path.join(REAL_DIR, "*.txt")))
    curves = [D.load_raw_instrument_excel(p) for p in paths]
    rng = [(320 + 273.15, 368 + 273.15)] * 4
    res = D.build_instrument_table(curves, RATES, rng, baseline="sigmoidal")
    assert len(res.table) == 99 and (res.table[["T1", "T2", "T3", "T4"]].diff().dropna() >= 0).all().all()
    assert all(c > 50 for c in res.counts)
    D.validate_data(res.table)


# =========================================================================== #
# GUI: the wizard and the hand-off to the main window
# =========================================================================== #
def _display_available() -> bool:
    try:
        import tkinter as tk
        tk.Tk().destroy()
        return True
    except Exception:
        return False


gui = pytest.mark.skipif(not _display_available(), reason="No display (run under Xvfb).")


@pytest.fixture
def app():
    pytest.importorskip("customtkinter")
    from isokinetics.gui import KineticsApp
    application = KineticsApp()
    yield application
    application.destroy()


@pytest.fixture
def errors(monkeypatch):
    shown = []
    monkeypatch.setattr("isokinetics.gui.messagebox.showerror", lambda *a, **k: shown.append(a))
    monkeypatch.setattr("isokinetics.gui.messagebox.showinfo", lambda *a, **k: shown.append(("info",) + a))
    return shown


def pump(widget, seconds=0.25):
    end = time.time() + seconds
    while time.time() < end:
        widget.update()
        time.sleep(0.01)


def drag(dlg, x0, x1):
    """A real mouse drag on the wizard's plot (synthetic matplotlib events)."""
    from matplotlib.backend_bases import MouseEvent
    ax = dlg.fig.axes[0]
    y = float(np.mean(ax.get_ylim()))

    def fire(name, x):
        px, py = ax.transData.transform((x, y))
        dlg.canvas.callbacks.process(name, MouseEvent(name, dlg.canvas, px, py, button=1))

    fire("button_press_event", x0)
    fire("motion_notify_event", (x0 + x1) / 2)
    fire("motion_notify_event", x1)
    fire("button_release_event", x1)
    pump(dlg, 0.15)


def select_reaction(dlg, spans, margin=2.0):
    s = spans[dlg.index]
    drag(dlg, s[0] - margin, s[1] + margin)


def make_wizard(app, paths, **kw):
    from isokinetics.gui import ImportWizardDialog
    dlg = ImportWizardDialog(app, paths, **kw)
    pump(dlg, 0.2)
    return dlg


@gui
def test_wizard_shows_progress_preview_mapping_and_plot_for_the_first_file(app, mixed_files):
    paths, _ = mixed_files
    dlg = make_wizard(app, list(reversed(paths)))                 # handed over in the wrong order
    assert dlg.header_label.cget("text") == f"File 1 of 4: {os.path.basename(paths[0])}"   # sorted by rate
    assert len(dlg.preview_tree.get_children()) == 10
    assert list(dlg.preview_tree["columns"]) == D.load_raw_table(paths[0]).names
    assert dlg.x_menu.get().startswith("Temp") and "HeatFlow" in dlg.y_menu.get()
    assert dlg.rate_entry.get() == "3" and str(dlg.back_button.cget("state")) == "disabled"
    assert dlg.next_button.cget("text") == "Next File"
    assert len(dlg.fig.axes[0].lines) == 1                        # the data are plotted
    assert dlg.span.active                                        # the drag selector is armed
    dlg.destroy()


@gui
def test_headerless_file_shows_generic_names(app, mixed_files):
    paths, _ = mixed_files
    dlg = make_wizard(app, paths)
    dlg.index = 1                                                 # the _5 file has no header
    dlg._show_file()
    pump(dlg)
    assert [dlg.x_menu.get(), dlg.y_menu.get()] == ["Column 1", "Column 2"]
    assert "No header found" in dlg.notes_label.cget("text")
    assert dlg.fig.axes[0].get_ylabel() == "Column 2" and dlg.current.curve is not None
    dlg.destroy()


@gui
def test_columns_of_a_headerless_three_column_file_can_be_mapped_by_hand(app, tmp_path):
    t_c, hf, span = F.synthetic_curve(5.0)
    path = str(tmp_path / "three_5_C.txt")
    F.write_three_columns_no_header(path, t_c, hf)
    other = []
    for i, w in enumerate((F.write_utf16_with_header,) * 3):
        t2, h2, _ = F.synthetic_curve(F.RATES[(0, 2, 3)[i]])
        q = str(tmp_path / f"o_{F.RATES[(0, 2, 3)[i]]:g}_C.txt")
        w(q, t2, h2)
        other.append(q)
    dlg = make_wizard(app, [path] + other)
    dlg.index = 1                                                 # (sorted by rate: 3, 5, 7, 10)
    dlg._show_file()
    pump(dlg)
    cur = dlg.current
    assert cur.table.names == ["Column 1", "Column 2", "Column 3"]
    assert [dlg.x_menu.get(), dlg.y_menu.get()] == ["Column 1", "Column 2"]       # auto guess = wrong here
    assert list(dlg.x_menu.cget("values")) == ["Column 1", "Column 2", "Column 3"]
    dlg.x_menu.set("Column 2")                                    # the user maps temperature and heat flow
    dlg._on_mapping_changed("Column 2")
    dlg.y_menu.set("Column 3")
    dlg._on_mapping_changed("Column 3")
    pump(dlg)
    assert dlg.fig.axes[0].get_ylabel() == "Column 3" and "Column 2" in dlg.fig.axes[0].get_xlabel()
    curve = dlg.current.curve
    assert curve[D.TEMP_COL].iloc[0] == pytest.approx(t_c[0] + 273.15, abs=1e-3)
    drag(dlg, span[0] - 2, span[1] + 2)
    assert dlg.current.selection_k is not None
    dlg.destroy()


@gui
def test_an_impossible_mapping_does_not_crash_and_changing_columns_clears_the_selection(app, mixed_files):
    paths, spans = mixed_files
    dlg = make_wizard(app, paths)
    dlg.index = 1
    dlg._show_file()
    pump(dlg)
    drag(dlg, spans[1][0] - 2, spans[1][1] + 2)
    assert dlg.current.selection_k is not None
    dlg.x_menu.set("Column 2")                                    # heat flow as "temperature": nonsense
    dlg.y_menu.set("Column 1")
    dlg._on_mapping_changed("Column 2")
    pump(dlg)
    assert dlg.current.selection_k is None                        # columns changed -> interval dropped
    assert dlg.current.curve is None or len(dlg.current.curve) < 50
    assert dlg.next_button.cget("state") == "normal"              # no crash; Next explains what is missing
    dlg.destroy()


@gui
def test_unit_toggle_reinterprets_the_raw_values_and_keeps_the_selection(app, mixed_files):
    paths, spans = mixed_files
    dlg = make_wizard(app, paths)
    select_reaction(dlg, spans)
    lo_k = dlg.current.selection_k[0]
    assert lo_k == pytest.approx(spans[0][0] - 2 + 273.15, abs=0.6)
    shown_before = dlg.current.selection_k[0] - 273.15            # what the user sees on the axis (degC)
    dlg.unit_button.set("K")
    dlg._on_unit_changed("K")
    pump(dlg)
    assert "[K]" in dlg.fig.axes[0].get_xlabel()
    assert dlg.current.selection_k is not None                    # not silently dropped
    assert dlg.current.selection_k[0] == pytest.approx(shown_before)      # same raw numbers, now kelvin
    assert [p for p in dlg.fig.axes[0].patches if p.get_gid() == "alpha_interval"]
    dlg.unit_button.set("\u00b0C")
    dlg._on_unit_changed("\u00b0C")
    assert dlg.current.selection_k[0] == pytest.approx(lo_k)              # and back again
    dlg.destroy()


@gui
def test_dragging_selects_an_interval_shades_it_and_previews_the_conversion(app, mixed_files):
    paths, spans = mixed_files
    dlg = make_wizard(app, paths)
    assert "No interval selected" in dlg.readout_label.cget("text")
    select_reaction(dlg, spans)
    sel = dlg.current.selection_k
    assert sel is not None and sel[0] < sel[1]
    assert [p for p in dlg.fig.axes[0].patches if p.get_gid() == "alpha_interval"]     # shaded span
    assert len(dlg.fig.axes[1].lines) == 1                                              # alpha(T) preview
    y = dlg.fig.axes[1].lines[0].get_ydata()
    assert y[0] == 0 and y[-1] == pytest.approx(1.0) and (np.diff(y) >= 0).all()
    assert "data points" in dlg.readout_label.cget("text")
    dlg.destroy()


@gui
def test_next_without_a_selection_shows_an_error_and_stays(app, mixed_files, errors):
    paths, _ = mixed_files
    dlg = make_wizard(app, paths)
    dlg._on_next()
    assert errors and "drag over the decomposition peak" in errors[0][1]
    assert dlg.index == 0 and "drag over" in dlg.error_label.cget("text")
    dlg.destroy()


@gui
def test_a_selection_of_fewer_than_ten_points_is_refused_with_an_error(app, mixed_files, errors):
    paths, spans = mixed_files
    dlg = make_wizard(app, paths)
    start = spans[0][0] + 20                                      # inside the ramp, ~0.06 K per point
    dlg._on_span(start, start + 0.3)                              # about 5 points
    pump(dlg)
    assert "too few" in dlg.readout_label.cget("text")
    dlg._on_next()
    assert errors and "at least 10" in errors[0][1]
    assert dlg.index == 0 and dlg.result is None                  # aborted: still on file 1
    dlg.destroy()


@gui
def test_missing_heating_rate_must_be_typed_before_continuing(app, tmp_path, errors):
    t_c, hf, span = F.synthetic_curve(5.0)
    paths = []
    for i in range(4):                                            # names without any rate
        p = str(tmp_path / f"unnamed_{i}.txt")
        t, h, _ = F.synthetic_curve(F.RATES[i])
        F.write_utf16_with_header(p, t, h)
        paths.append(p)
    dlg = make_wizard(app, paths)
    assert dlg.rate_entry.get() == ""
    drag(dlg, span[0] - 2, span[1] + 2)
    dlg._on_next()
    assert errors and "heating rate" in errors[0][1] and dlg.index == 0
    dlg.rate_entry.insert(0, "3")
    dlg._on_next()
    assert dlg.index == 1
    dlg.destroy()


@gui
def test_back_keeps_the_work_and_next_file_advances(app, mixed_files):
    paths, spans = mixed_files
    dlg = make_wizard(app, paths)
    select_reaction(dlg, spans)
    first = dlg.current.selection_k
    dlg._on_next()
    pump(dlg)
    assert dlg.index == 1 and dlg.header_label.cget("text").startswith("File 2 of 4")
    assert str(dlg.back_button.cget("state")) == "normal"
    dlg._on_back()
    pump(dlg)
    assert dlg.index == 0 and dlg.current.selection_k == first
    assert [p for p in dlg.fig.axes[0].patches if p.get_gid() == "alpha_interval"]     # span redrawn
    dlg.destroy()


@gui
def test_last_page_offers_finish_and_cancel_returns_nothing(app, mixed_files):
    paths, _ = mixed_files
    dlg = make_wizard(app, paths)
    dlg.index = 3
    dlg._show_file()
    assert dlg.next_button.cget("text") == "Finish"
    dlg._on_cancel()
    assert dlg.result is None and not dlg.winfo_exists()


@gui
def test_a_file_that_cannot_be_read_blocks_next_but_not_cancel(app, mixed_files, tmp_path):
    paths, _ = mixed_files
    bad = tmp_path / "bad_3_C.txt"
    bad.write_text("nothing numeric\n", encoding="utf-8")
    dlg = make_wizard(app, [str(bad)] + paths[1:])
    assert "no table of numbers" in dlg.error_label.cget("text")
    assert str(dlg.next_button.cget("state")) == "disabled"
    dlg._on_cancel()
    assert dlg.result is None


@gui
def test_complete_wizard_run_returns_a_table_that_recovers_the_known_ea(app, mixed_files):
    paths, spans = mixed_files
    dlg = make_wizard(app, paths)
    for _ in range(4):
        select_reaction(dlg, spans)
        dlg._on_next()
        pump(dlg, 0.1)
    assert dlg.result is not None and not dlg.winfo_exists()
    res = dlg.result
    assert res.betas == RATES and list(res.table.columns) == ["alpha", "T1", "T2", "T3", "T4"]
    assert len(res.table) == 99 and res.table.notna().all().all()
    for method in ("Friedman", "KAS", "Vyazovkin"):
        assert mean_ea(res.table, method) == pytest.approx(F.EA_TRUE, rel=0.02), method


@gui
def test_baseline_and_step_settings_reach_the_result(app, mixed_files):
    paths, spans = mixed_files
    dlg = make_wizard(app, paths, step_text="0.02", baseline="sigmoidal")
    assert dlg.baseline_key == "sigmoidal" and "49 points" in dlg.step_label.cget("text")
    for _ in range(4):
        select_reaction(dlg, spans)
        dlg._on_next()
        pump(dlg, 0.1)
    assert dlg.result.baseline == "sigmoidal" and len(dlg.result.table) == 49


@gui
def test_invalid_step_is_reported_instead_of_crashing(app, mixed_files, errors):
    paths, spans = mixed_files
    dlg = make_wizard(app, paths, step_text="abc")
    for _ in range(3):
        select_reaction(dlg, spans)
        dlg._on_next()
        pump(dlg, 0.1)
    select_reaction(dlg, spans)
    dlg._on_next()
    assert dlg.result is None and errors and dlg.winfo_exists()
    dlg.destroy()


# ------------------------------ main window ---------------------------------- #
def result_for(paths, spans):
    curves = [D.load_raw_instrument_excel(p) for p in D.order_by_heating_rate(paths)]
    return D.build_instrument_table(curves, RATES, ranges_for(spans))


@gui
def test_four_files_open_the_wizard_and_the_result_becomes_the_dataset(app, mixed_files, errors, monkeypatch):
    paths, spans = mixed_files
    asked = []
    result = result_for(paths, spans)
    monkeypatch.setattr(app, "_ask_import_wizard", lambda p, initial=None: asked.append(list(p)) or result)
    monkeypatch.setattr("isokinetics.gui.filedialog.askopenfilenames", lambda **k: tuple(paths))
    app._on_load_data()
    assert asked == [list(paths)]
    assert app.working_df.equals(result.table) and app.source_kind == "instrument"
    assert [e.get() for e in app.beta_entries] == ["3", "5", "7", "10"]
    assert str(app.run_button.cget("state")) == "normal"
    assert len(app.preview_tree.get_children()) == 99                          # Data Preview populated
    assert "4 instrument files" in app.file_label.cget("text") or "4 files" in app.file_label.cget("text")
    assert "Imported 4 instrument files" in app.status_label.cget("text")
    assert errors and errors[-1][0] == "info" and "Import complete" in errors[-1][1]
    assert app.reprocess_button.cget("text").startswith("Re-open import wizard")
    assert app.kth_button_visible                                              # ready for the next step


@gui
def test_the_imported_dataset_runs_through_the_kinetics_engine_from_the_gui(app, mixed_files, errors, monkeypatch):
    paths, spans = mixed_files
    app.load_instrument_files(paths, wizard_result=result_for(paths, spans))
    betas = app._read_betas()
    assert betas == RATES
    res = kinetics.METHODS["Friedman"](app.working_df, betas)
    assert np.mean([e for a, e in res.items() if 0.1 <= a <= 0.9]) == pytest.approx(F.EA_TRUE, rel=0.02)


@gui
def test_cancelling_the_wizard_loads_nothing(app, mixed_files, monkeypatch):
    paths, _ = mixed_files
    monkeypatch.setattr(app, "_ask_import_wizard", lambda p, initial=None: None)
    assert app.load_instrument_files(paths) is False
    assert app.working_df is None and "cancelled" in app.status_label.cget("text")


@gui
@pytest.mark.parametrize("count", [2, 3, 5])
def test_wrong_number_of_files_is_explained_and_no_wizard_opens(app, mixed_files, errors, monkeypatch, count):
    paths, _ = mixed_files
    chosen = (paths * 2)[:count]
    monkeypatch.setattr(app, "_ask_import_wizard", lambda *a, **k: pytest.fail("wizard must not open"))
    monkeypatch.setattr("isokinetics.gui.filedialog.askopenfilenames", lambda **k: tuple(chosen))
    app._on_load_data()
    assert errors and "exactly 4" in errors[0][1] and app.working_df is None


@gui
def test_unsupported_extensions_are_rejected(app, mixed_files, errors):
    paths, _ = mixed_files
    assert app.load_instrument_files(paths[:3] + [paths[3] + ".png"]) is False
    assert errors and "not .xls" in errors[0][1]


@gui
def test_a_single_file_still_uses_the_normal_loader(app, monkeypatch):
    sample = os.path.join(ROOT, "sample_data", "matric_121459.txt")
    loaded = []
    monkeypatch.setattr(app, "load_file", lambda p: loaded.append(p) or True)
    monkeypatch.setattr(app, "_ask_import_wizard", lambda *a, **k: pytest.fail("no wizard for one file"))
    monkeypatch.setattr("isokinetics.gui.filedialog.askopenfilenames", lambda **k: (sample,))
    app._on_load_data()
    assert loaded == [sample]
    monkeypatch.setattr("isokinetics.gui.filedialog.askopenfilenames", lambda **k: ())
    app._on_load_data()                                           # cancelling the file dialog does nothing
    assert loaded == [sample]


@gui
def test_the_main_window_really_waits_for_the_wizard_to_finish(app, mixed_files):
    """Open the modal wizard through the app and drive it from inside its own event loop."""
    from isokinetics.gui import ImportWizardDialog
    paths, spans = mixed_files
    order = []

    def driver():
        dlg = next(w for w in app.winfo_children() if isinstance(w, ImportWizardDialog))
        order.append("wizard open")
        for _ in range(4):
            s = spans[dlg.index]
            dlg._on_span(s[0] - 2, s[1] + 2)
            dlg._on_next()

    app.after(300, driver)
    result = app._ask_import_wizard(paths)                        # blocks until the dialog is closed
    order.append("app resumed")
    assert order == ["wizard open", "app resumed"]
    assert result is not None and result.betas == RATES
    assert not [w for w in app.winfo_children() if isinstance(w, ImportWizardDialog)]


@gui
def test_closing_the_application_while_the_wizard_is_open_is_clean(app, mixed_files):
    from isokinetics.gui import ImportWizardDialog
    paths, _ = mixed_files
    app.after(300, app.destroy)
    result = app._ask_import_wizard(paths)
    assert result is None


@gui
def test_reopening_the_wizard_prefills_rates_and_intervals(app, mixed_files, errors):
    paths, spans = mixed_files
    result = result_for(paths, spans)
    app.load_instrument_files(paths, wizard_result=result)
    dlg = make_wizard(app, result.files, initial=result)
    assert dlg.current.rate_text == "3" and dlg.current.selection_k == pytest.approx(result.ranges[0])
    assert [p for p in dlg.fig.axes[0].patches if p.get_gid() == "alpha_interval"]
    dlg.destroy()


@gui
def test_reopen_button_replaces_the_dataset(app, mixed_files, errors, monkeypatch):
    paths, spans = mixed_files
    first = result_for(paths, spans)
    app.load_instrument_files(paths, wizard_result=first)
    curves = [D.load_raw_instrument_excel(p) for p in first.files]
    second = D.build_instrument_table(curves, RATES, ranges_for(spans, margin=4.0), step=0.02)
    monkeypatch.setattr(app, "_ask_import_wizard", lambda p, initial=None: second)
    app._on_reopen_wizard()
    assert len(app.working_df) == 49 and app.instrument_result is second


@gui
def test_export_records_where_the_data_came_from(app, mixed_files, errors, monkeypatch, tmp_path):
    paths, spans = mixed_files
    app.load_instrument_files(paths, wizard_result=result_for(paths, spans))
    betas = app._read_betas()
    results = {n: kinetics.METHODS[n](app.working_df, betas) for n in ("Friedman", "KAS")}
    app._on_analysis_done(results, {n: kinetics.average_ea(r) for n, r in results.items()}, {})
    app.update()
    monkeypatch.setattr("isokinetics.gui.filedialog.askdirectory", lambda **k: str(tmp_path))
    monkeypatch.setattr("isokinetics.gui.messagebox.showwarning", lambda *a, **k: None)
    app._on_export()
    folder = next(p for p in tmp_path.iterdir() if p.name.startswith("Kinetics_Results_"))
    summary = (folder / "Summary.txt").read_text(encoding="utf-8")
    assert "4 instrument files" in summary and "3, 5, 7, 10" in summary
