"""Stabilization phase: interval maths, plot output, mouse controller and exports.
None of these tests needs a display."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from isokinetics import exporter, interaction, interval as iv, kinetics, plotter, validation  # noqa: E402
from isokinetics.formatting import fmt_interval  # noqa: E402

BETAS = [5.0, 10.0, 20.0, 40.0]
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def _linear(slope=100.0, base=50.0):
    """Ea(alpha) = base + slope*alpha on the 0.01 grid -> means are easy to know exactly."""
    return {float(a): base + slope * float(a) for a in GRID}


# --------------------------------------------------------------------------- #
# interval.py
# --------------------------------------------------------------------------- #
def test_mean_over_interval_is_exact_and_inclusive():
    res = _linear()
    m = iv.mean_ea_in_interval(res, iv.AlphaInterval(0.2, 0.8))
    assert m == pytest.approx(50 + 100 * 0.5)  # mean of a linear function = value at the midpoint
    assert iv.count_in_interval(res, iv.AlphaInterval(0.2, 0.8)) == 61  # 0.20 ... 0.80 inclusive


def test_edges_survive_floating_point_noise():
    res = {0.1 + 0.1: 1.0, 0.3: 2.0, 0.4: 3.0}  # 0.1+0.1 == 0.2 exactly, 0.30000000000000004 trap below
    res[0.1 + 0.2] = 9.0  # 0.30000000000000004
    assert iv.count_in_interval(res, iv.AlphaInterval(0.2, 0.3)) == 3


def test_right_to_left_drag_is_normalised():
    a = iv.AlphaInterval(0.8, 0.2)
    assert (a.lo, a.hi) == (0.2, 0.8) and a.label == "0.20\u20130.80"


def test_fewer_than_three_points_gives_nan_not_a_misleading_mean():
    res = _linear()
    assert math.isnan(iv.mean_ea_in_interval(res, iv.AlphaInterval(0.2, 0.21)))  # 2 points
    assert not math.isnan(iv.mean_ea_in_interval(res, iv.AlphaInterval(0.2, 0.22)))  # 3 points
    assert iv.mean_ea_in_interval(res, iv.AlphaInterval(0.2, 0.21), min_points=2) == pytest.approx(
        np.mean([res[0.2], res[0.21]]))


def test_nan_and_missing_values_are_ignored():
    res = {0.1: 10.0, 0.2: float("nan"), 0.3: 30.0, 0.4: 40.0, 0.5: None}
    assert iv.count_in_interval(res, iv.AlphaInterval(0.0, 1.0)) == 3
    assert iv.mean_ea_in_interval(res, iv.AlphaInterval(0.0, 1.0)) == pytest.approx(80 / 3)


@pytest.mark.parametrize("lo,hi,expected", [
    (0.2, 0.8, (0.2, 0.8)),
    (0.8, 0.2, (0.2, 0.8)),       # swapped
    (-0.5, 0.4, (0.01, 0.4)),     # overhangs the low end
    (0.9, 1.5, (0.9, 0.99)),      # overhangs the high end
    (-1.0, 2.0, (0.01, 0.99)),    # covers everything
])
def test_clip_to_data_bounds(lo, hi, expected):
    got = iv.clip_interval(lo, hi, 0.01, 0.99)
    assert (got.lo, got.hi) == pytest.approx(expected)


@pytest.mark.parametrize("lo,hi", [(1.2, 1.6), (-0.5, -0.1), (float("nan"), 0.5), (0.2, float("inf"))])
def test_selection_outside_data_or_invalid_is_rejected(lo, hi):
    assert iv.clip_interval(lo, hi, 0.01, 0.99) is None


def test_statistics_and_max_point_count():
    results = {"A": _linear(), "B": {0.5: 1.0, 0.6: 2.0}}
    stats = iv.interval_statistics(results, iv.AlphaInterval(0.4, 0.7))
    assert stats["A"]["n"] == 31 and stats["A"]["mean"] == pytest.approx(50 + 55)
    assert stats["B"]["n"] == 2 and math.isnan(stats["B"]["mean"])
    assert iv.max_point_count(results, iv.AlphaInterval(0.4, 0.7)) == 31
    assert iv.max_point_count({}, iv.AlphaInterval(0.4, 0.7)) == 0


def test_fmt_interval():
    assert fmt_interval(0.2, 0.8) == "0.20\u20130.80"
    assert fmt_interval(0.25, 0.753) == "0.25\u20130.753"
    assert fmt_interval(float("nan"), 0.5) == "N/A"


def test_helpers_never_modify_the_full_range_results():
    res = {"A": _linear()}
    before = {k: dict(v) for k, v in res.items()}
    iv.interval_means(res, iv.AlphaInterval(0.3, 0.6))
    iv.interval_statistics(res, iv.AlphaInterval(0.3, 0.6))
    assert res == before


# --------------------------------------------------------------------------- #
# plotter: legend, shaded span
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def results():
    df = validation.simulate_variable_ea(lambda a: 100 + 100 * a, 1e10, "F1")
    return {m: kinetics.METHODS[m](df, BETAS) for m in ("Friedman", "KAS", "Vyazovkin")}


def _legend(fig):
    return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]


def test_legend_shows_interval_mean_and_full_range_mean_without_selection(results):
    fig = plotter.build_figure()
    averages = {n: kinetics.average_ea(r) for n, r in results.items()}
    plotter.update_plot(fig, results, averages)
    full = _legend(fig)
    assert all("[" not in t and "mean E\u2090 = " in t for t in full)

    sel = iv.AlphaInterval(0.2, 0.8)
    plotter.update_plot(fig, results, averages, interval=sel)
    labels = _legend(fig)
    for name, text in zip(results, labels):
        expected = iv.mean_ea_in_interval(results[name], sel)
        assert f"{name} (mean E\u2090 [0.20\u20130.80] = {expected:.2f} kJ/mol)" == text

    plotter.update_plot(fig, results, averages)          # back to full range: identical to before
    assert _legend(fig) == full


def test_interval_changes_the_label_not_the_curves(results):
    fig = plotter.build_figure()
    plotter.update_plot(fig, results)
    before = [(l.get_xdata().tolist(), l.get_ydata().tolist()) for l in fig.axes[0].lines]
    plotter.update_plot(fig, results, interval=iv.AlphaInterval(0.3, 0.6))
    after = [(l.get_xdata().tolist(), l.get_ydata().tolist()) for l in fig.axes[0].lines]
    assert before == after and len(after) == len(results)  # still exactly one line per method


def _span_x(patch):
    """x-extent of an axvspan patch (a Rectangle in recent matplotlib, a Polygon in older ones)."""
    if hasattr(patch, "get_width"):
        return patch.get_x(), patch.get_x() + patch.get_width()
    xs = np.asarray(patch.get_xy())[:, 0]
    return xs.min(), xs.max()


def test_axvspan_is_drawn_only_when_an_interval_is_selected(results):
    fig = plotter.build_figure()
    plotter.update_plot(fig, results)
    assert not [p for p in fig.axes[0].patches if p.get_gid() == "alpha_interval"]
    plotter.update_plot(fig, results, interval=iv.AlphaInterval(0.2, 0.8))
    spans = [p for p in fig.axes[0].patches if p.get_gid() == "alpha_interval"]
    assert len(spans) == 1
    lo, hi = _span_x(spans[0])
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.8)


def test_method_with_too_few_points_is_labelled_na(results):
    fig = plotter.build_figure()
    sparse = dict(results, Sparse={0.5: 100.0, 0.9: 110.0})
    plotter.update_plot(fig, sparse, interval=iv.AlphaInterval(0.4, 0.6))
    label = [t for t in _legend(fig) if t.startswith("Sparse")][0]
    assert "N/A" in label and "1 of 3 points needed" in label


def test_span_survives_the_light_export_theme_and_returns_to_dark(results):
    fig = plotter.build_figure()
    plotter.set_theme("dark")
    try:
        plotter.update_plot(fig, results, interval=iv.AlphaInterval(0.2, 0.8))
        span = [p for p in fig.axes[0].patches if p.get_gid() == "alpha_interval"][0]
        dark_edge = tuple(span.get_edgecolor())
        with plotter.export_theme(fig, "light"):
            assert tuple(span.get_edgecolor()) != dark_edge
            assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        assert tuple(span.get_edgecolor()) == dark_edge
    finally:
        plotter.set_theme("light")


# --------------------------------------------------------------------------- #
# Mouse controller (real SpanSelector, synthetic mouse events)
# --------------------------------------------------------------------------- #
class Harness:
    def __init__(self, results):
        self.fig = plotter.build_figure()
        self.canvas = FigureCanvasAgg(self.fig)
        self.results = results
        self.selected, self.moved, self.resets = [], [], 0
        self.ctl = interaction.AlphaSpanController(
            self.canvas, lambda a, b: self.selected.append((a, b)), self._reset,
            lambda a, b: self.moved.append((a, b)))
        self.redraw()

    def _reset(self):
        self.resets += 1

    def redraw(self, interval=None):
        plotter.update_plot(self.fig, self.results, interval=interval)
        self.canvas.draw()
        self.ctl.attach()

    def fire(self, name, x, **kw):
        ax = self.fig.axes[0]
        px, py = ax.transData.transform((x, float(np.mean(ax.get_ylim()))))
        self.canvas.callbacks.process(name, MouseEvent(name, self.canvas, px, py, button=1, **kw))

    def drag(self, x0, x1):
        self.fire("button_press_event", x0)
        self.fire("motion_notify_event", (x0 + x1) / 2)
        self.fire("motion_notify_event", x1)
        self.fire("button_release_event", x1)


@pytest.fixture
def harness(results):
    h = Harness(results)
    h.ctl.enabled = True
    h.ctl.attach()
    yield h
    h.ctl.close()


def test_drag_reports_the_range_on_release_and_live_while_moving(harness):
    harness.drag(0.2, 0.8)
    assert len(harness.selected) == 1
    assert harness.selected[0] == pytest.approx((0.2, 0.8), abs=0.01)
    assert len(harness.moved) >= 2


def test_right_to_left_drag_is_reported_in_order(harness):
    harness.drag(0.8, 0.3)
    lo, hi = harness.selected[-1]
    assert lo < hi and (lo, hi) == pytest.approx((0.3, 0.8), abs=0.01)


def test_a_click_after_a_selection_is_not_a_selection(harness):
    harness.drag(0.2, 0.8)
    n = len(harness.selected)
    harness.drag(0.5, 0.5005)  # a click: matplotlib would call onselect with a ~0-width span
    assert len(harness.selected) == n


def test_double_click_resets_and_single_click_does_not(harness):
    harness.fire("button_press_event", 0.5)
    assert harness.resets == 0
    harness.fire("button_press_event", 0.5, dblclick=True)
    assert harness.resets == 1


def test_selection_still_works_after_the_plot_was_redrawn(harness):
    harness.redraw(interval=iv.AlphaInterval(0.2, 0.8))  # fig.clear() destroys the old selector
    harness.drag(0.1, 0.4)
    assert harness.selected[-1] == pytest.approx((0.1, 0.4), abs=0.01)


def test_disabled_controller_ignores_the_mouse(results):
    h = Harness(results)          # enabled=False by default (no results yet in the GUI)
    h.drag(0.2, 0.8)
    h.fire("button_press_event", 0.5, dblclick=True)
    assert h.selected == [] and h.resets == 0 and not h.ctl.active
    h.ctl.close()


def test_drag_past_the_axes_edge_is_clipped_by_the_axes(harness):
    harness.fire("button_press_event", 0.5)
    ax = harness.fig.axes[0]
    px, py = ax.transData.transform((1.3, float(np.mean(ax.get_ylim()))))  # far right of the axes
    for name in ("motion_notify_event", "button_release_event"):
        harness.canvas.callbacks.process(name, MouseEvent(name, harness.canvas, px, py, button=1))
    lo, hi = harness.selected[-1]
    assert hi <= 1.0 + 1e-9 and lo == pytest.approx(0.5, abs=0.01)


# --------------------------------------------------------------------------- #
# kinetics.calculate_interval_mean  (the math engine behind everything above)
# --------------------------------------------------------------------------- #
def test_calculate_interval_mean_basic_inclusive_and_order_independent():
    res = _linear()
    assert kinetics.calculate_interval_mean(res, 0.2, 0.8) == pytest.approx(100.0)
    assert kinetics.calculate_interval_mean(res, 0.8, 0.2) == pytest.approx(100.0)   # reversed ends
    assert kinetics.calculate_interval_mean(res, 0.2, 0.2) == pytest.approx(res[0.2])  # one point


def test_calculate_interval_mean_ignores_nan_and_handles_empty_or_bad_input():
    res = {0.1: 10.0, 0.2: float("nan"), 0.3: 30.0, 0.4: None}
    assert kinetics.calculate_interval_mean(res, 0.0, 1.0) == pytest.approx(20.0)
    assert math.isnan(kinetics.calculate_interval_mean({}, 0.0, 1.0))
    assert math.isnan(kinetics.calculate_interval_mean(res, 0.5, 0.9))               # nothing inside
    assert math.isnan(kinetics.calculate_interval_mean(res, float("nan"), 0.9))
    assert math.isnan(kinetics.calculate_interval_mean(res, 0.0, float("inf")))
    assert math.isnan(kinetics.calculate_interval_mean(res, "a", 0.9))


def test_calculate_interval_mean_min_points_and_float_noise_at_the_ends():
    res = _linear()
    assert math.isnan(kinetics.calculate_interval_mean(res, 0.2, 0.21, min_points=3))
    assert not math.isnan(kinetics.calculate_interval_mean(res, 0.2, 0.21, min_points=2))
    noisy = {0.1 + 0.2: 5.0, 0.4: 7.0, 0.5: 9.0}          # first key is 0.30000000000000004
    assert kinetics.calculate_interval_mean(noisy, 0.3, 0.5) == pytest.approx(7.0)


def test_calculate_interval_mean_does_not_change_the_input_or_the_full_range_average():
    res = _linear()
    before, full = dict(res), kinetics.average_ea(res)
    kinetics.calculate_interval_mean(res, 0.3, 0.6)
    assert res == before and kinetics.average_ea(res) == full


def test_interval_module_and_kinetics_agree():
    res = _linear()
    sel = iv.AlphaInterval(0.25, 0.7)
    assert iv.mean_ea_in_interval(res, sel) == kinetics.calculate_interval_mean(res, 0.25, 0.7, min_points=3)


# --------------------------------------------------------------------------- #
# Export integration:  interval = (alpha_min, alpha_max)
# --------------------------------------------------------------------------- #
COMMENT = "# Selected stabilization interval: alpha 0.20 to 0.80"


def _avg_row(path):
    for line in open(path, encoding="utf-8"):
        if line.startswith("(average)"):
            return line.rstrip("\n").split("\t")
    raise AssertionError("no (average) row")


def test_txt_average_row_is_the_interval_mean_and_header_comment_is_written(tmp_path, results):
    path = exporter.save_data_txt(results, str(tmp_path), interval=(0.2, 0.8))
    lines = open(path, encoding="utf-8").read().splitlines()
    assert lines[0] == COMMENT
    header = lines[1].split("\t")
    row = _avg_row(path)
    for name in results:
        expected = kinetics.calculate_interval_mean(results[name], 0.2, 0.8, min_points=3)
        assert float(row[header.index(name)]) == pytest.approx(expected, abs=5e-5)
    full = kinetics.average_ea(results["Friedman"])
    assert float(row[header.index("Friedman")]) != pytest.approx(full, abs=1e-3)   # not the full-range mean
    # the data rows still cover the whole alpha range
    assert len([l for l in lines if l[:1].isdigit()]) == len(next(iter(results.values())))


def test_txt_without_interval_is_unchanged_full_range(tmp_path, results):
    path = exporter.save_data_txt(results, str(tmp_path))
    lines = open(path, encoding="utf-8").read().splitlines()
    assert lines[0].startswith("alpha") and "stabilization" not in "".join(lines).lower()
    row = _avg_row(path)
    assert float(row[1]) == pytest.approx(kinetics.average_ea(results["Friedman"]), abs=5e-5)


def test_txt_interval_accepts_alphainterval_lists_and_reversed_tuples(tmp_path, results):
    a = _avg_row(exporter.save_data_txt(results, str(tmp_path), filename="a.txt", interval=(0.2, 0.8)))
    b = _avg_row(exporter.save_data_txt(results, str(tmp_path), filename="b.txt", interval=(0.8, 0.2)))
    c = _avg_row(exporter.save_data_txt(results, str(tmp_path), filename="c.txt", interval=iv.AlphaInterval(0.2, 0.8)))
    d = _avg_row(exporter.save_data_txt(results, str(tmp_path), filename="d.txt", interval=[0.2, 0.8]))
    assert a == b == c == d


def test_txt_interval_with_too_few_points_gives_nan_in_the_average_row(tmp_path, results):
    row = _avg_row(exporter.save_data_txt(results, str(tmp_path), interval=(0.2, 0.21)))   # 2 grid points
    assert row[1:] == ["NaN"] * len(results)


def test_txt_kth_A_average_also_follows_the_interval(tmp_path, results):
    A = {float(a): 1e10 * (1 + float(a)) for a in GRID}
    full = {"Ea": results["Friedman"], "A": A, "f_alpha": {a: 1.0 for a in A}, "model": "F1"}
    path = exporter.save_data_txt(results, str(tmp_path), kth_full_result=full, interval=(0.2, 0.8))
    header = [l for l in open(path, encoding="utf-8").read().splitlines() if l.startswith("alpha")][0].split("\t")
    expected = np.mean([A[a] for a in A if 0.2 - 1e-9 <= a <= 0.8 + 1e-9])
    assert float(_avg_row(path)[header.index("KTH_A")]) == pytest.approx(expected, rel=1e-5)


def test_summary_txt_starts_with_the_comment_and_adds_the_interval_table(tmp_path, results):
    path = exporter.save_summary_txt(results, str(tmp_path), interval=(0.2, 0.8))
    text = open(path, encoding="utf-8").read()
    assert text.splitlines()[0] == COMMENT
    assert "Activation energy Ea (kJ/mol) over the alpha grid" in text            # full range kept
    sub = text.split("Selected stabilization interval: alpha 0.20 to 0.80 (")[1]
    expected = kinetics.calculate_interval_mean(results["Friedman"], 0.2, 0.8, min_points=3)
    assert f"{expected:.2f}" in sub and " 61" in sub                                # mean and N = 61


def test_summary_txt_without_interval_is_unchanged(tmp_path, results):
    text = open(exporter.save_summary_txt(results, str(tmp_path)), encoding="utf-8").read()
    assert "stabilization" not in text.lower() and not text.startswith("#")


def test_xlsx_averages_sheet_holds_the_interval_means_and_a_comment_header(tmp_path, results):
    averages = {n: kinetics.average_ea(r) for n, r in results.items()}
    path = exporter.save_data_xlsx(results, str(tmp_path), averages=averages, interval=(0.2, 0.8))
    raw = pd.read_excel(path, sheet_name="Averages", header=None)
    assert raw.iloc[0, 0] == COMMENT                                               # comment header, row 1
    sheet = pd.read_excel(path, sheet_name="Averages", header=1)                  # table header, row 2
    assert list(sheet.columns[:3]) == ["method", "average_Ea_kJ_per_mol", "n_points_in_interval"]
    for name, mean, full, n in zip(sheet["method"], sheet["average_Ea_kJ_per_mol"],
                                   sheet["full_range_average_Ea_kJ_per_mol"], sheet["n_points_in_interval"]):
        assert mean == pytest.approx(kinetics.calculate_interval_mean(results[name], 0.2, 0.8, 3))
        assert full == pytest.approx(averages[name]) and n == 61
    from openpyxl import load_workbook
    # also recorded in the workbook's own properties (File > Info), visible from any sheet
    assert load_workbook(path).properties.description == "Selected stabilization interval: alpha 0.20 to 0.80"


def test_xlsx_without_interval_keeps_the_original_layout(tmp_path, results):
    averages = {n: kinetics.average_ea(r) for n, r in results.items()}
    path = exporter.save_data_xlsx(results, str(tmp_path), averages=averages)
    sheet = pd.read_excel(path, sheet_name="Averages")                             # header on row 1
    assert list(sheet.columns) == ["method", "average_Ea_kJ_per_mol"]
    assert sheet["average_Ea_kJ_per_mol"].iloc[0] == pytest.approx(averages["Friedman"])


def test_export_all_forwards_the_interval_to_every_file(tmp_path, results):
    import os
    fig = plotter.build_figure()
    plotter.update_plot(fig, results, interval=iv.AlphaInterval(0.2, 0.8))
    folder = exporter.export_all(str(tmp_path), results, fig, interval=(0.2, 0.8))
    assert open(os.path.join(folder, "Ea_data.txt"), encoding="utf-8").readline().strip() == COMMENT
    assert open(os.path.join(folder, "Summary.txt"), encoding="utf-8").readline().strip() == COMMENT
    assert os.path.exists(os.path.join(folder, "Ea_vs_alpha.png"))
    sheet = pd.read_excel(os.path.join(folder, "Ea_results.xlsx"), sheet_name="Averages", header=1)
    assert "n_points_in_interval" in sheet.columns


def test_export_all_without_interval_writes_no_interval_text(tmp_path, results):
    import os
    folder = exporter.export_all(str(tmp_path), results, plotter.build_figure())
    for name in ("Ea_data.txt", "Summary.txt"):
        assert "stabilization" not in open(os.path.join(folder, name), encoding="utf-8").read().lower()
