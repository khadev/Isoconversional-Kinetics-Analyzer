"""Unit tests for isokinetics.exporter and isokinetics.plotter."""

from __future__ import annotations

import os

import pytest

from isokinetics import exporter, kinetics, plotter


@pytest.fixture
def dummy_results():
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    return {
        "Friedman": {a: 100.0 + 10 * a for a in alphas},
        "FWO": {a: 95.0 + 8 * a for a in alphas},
    }


def test_build_figure_has_axes():
    fig = plotter.build_figure()
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel()
    assert ax.get_ylabel()


def test_update_plot_adds_one_line_per_method(dummy_results):
    fig = plotter.build_figure()
    plotter.update_plot(fig, dummy_results)
    ax = fig.axes[0]
    assert len(ax.lines) == len(dummy_results)
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == len(dummy_results)


def test_update_plot_handles_empty_results():
    fig = plotter.build_figure()
    # Should not raise, and should show a placeholder message instead of a legend.
    plotter.update_plot(fig, {})
    ax = fig.axes[0]
    assert ax.get_legend() is None


def test_make_output_folder_creates_timestamped_directory(tmp_path):
    out = exporter.make_output_folder(str(tmp_path))
    assert os.path.isdir(out)
    assert os.path.basename(out).startswith("Kinetics_Results_")


def test_make_output_folder_missing_base_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        exporter.make_output_folder(str(missing))


def test_save_graph_creates_four_formats_including_tiff_by_default(tmp_path, dummy_results):
    fig = plotter.build_figure()
    plotter.update_plot(fig, dummy_results)
    paths = exporter.save_graph(fig, str(tmp_path))
    assert set(paths.keys()) == {"png", "svg", "pdf", "tiff"}
    for p in paths.values():
        assert os.path.isfile(p)
        assert os.path.getsize(p) > 0


def test_save_graph_can_skip_tiff(tmp_path, dummy_results):
    fig = plotter.build_figure()
    plotter.update_plot(fig, dummy_results)
    paths = exporter.save_graph(fig, str(tmp_path), include_tiff=False)
    assert set(paths.keys()) == {"png", "svg", "pdf"}


def test_save_graph_tiff_uses_lzw_compression(tmp_path, dummy_results):
    from PIL import Image

    fig = plotter.build_figure()
    plotter.update_plot(fig, dummy_results)
    paths = exporter.save_graph(fig, str(tmp_path))

    img = Image.open(paths["tiff"])
    assert img.info.get("compression") == "tiff_lzw"


def test_save_graph_tiff_embeds_metadata(tmp_path, dummy_results):
    from PIL import Image

    fig = plotter.build_figure()
    plotter.update_plot(fig, dummy_results)
    metadata = {"Author": "Test Author", "Software": "IsoKinetics v1.6.0"}
    paths = exporter.save_graph(fig, str(tmp_path), metadata=metadata)

    img = Image.open(paths["tiff"])
    description = img.tag_v2.get(270)  # 270 = ImageDescription TIFF tag
    assert "Test Author" in description
    assert "IsoKinetics v1.6.0" in description


def test_export_all_metadata_flows_through_to_tiff(tmp_path, dummy_results):
    from PIL import Image

    fig = plotter.build_figure()
    plotter.update_plot(fig, dummy_results)
    out_folder = exporter.export_all(
        str(tmp_path), dummy_results, fig, metadata={"Sample": "test_sample.txt"},
    )
    img = Image.open(os.path.join(out_folder, "Ea_vs_alpha.tiff"))
    assert "test_sample.txt" in img.tag_v2.get(270)


def test_save_data_txt_contains_header_and_average_row(tmp_path, dummy_results):
    path = exporter.save_data_txt(dummy_results, str(tmp_path))
    assert os.path.isfile(path)
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    header = lines[0].strip().split("\t")
    assert header[0] == "alpha"
    assert "Friedman" in header
    assert "FWO" in header
    avg_row = lines[1].strip().split("\t")
    assert avg_row[0] == "(average)"
    # 2 methods + alpha column
    assert len(header) == 3
    # 1 header + 1 average + 5 alpha rows
    assert len(lines) == 1 + 1 + 5


def test_save_data_xlsx_creates_file(tmp_path, dummy_results):
    path = exporter.save_data_xlsx(dummy_results, str(tmp_path))
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0


def test_export_all_creates_expected_files(tmp_path, dummy_results):
    fig = plotter.build_figure()
    plotter.update_plot(fig, dummy_results)
    out_folder = exporter.export_all(str(tmp_path), dummy_results, fig)

    expected = {
        "Ea_vs_alpha.png",
        "Ea_vs_alpha.svg",
        "Ea_vs_alpha.pdf",
        "Ea_data.txt",
        "Ea_results.xlsx",
    }
    actual = set(os.listdir(out_folder))
    assert expected.issubset(actual)


@pytest.fixture
def dummy_kth_full_result(dummy_results):
    alphas = list(dummy_results["Friedman"].keys())
    return {
        "Ea": dummy_results["Friedman"],
        "A": {a: 1.0e12 for a in alphas},
        "f_alpha": {a: 1.0 - a for a in alphas},
        "model": "F1",
        "model_scores": {"F1": 0.05, "D1": 0.6},
        "y_alpha": {a: 1.0 - a for a in alphas},
        "z_alpha": {a: 1.0 - a for a in alphas},
        "y_theory": {a: 1.0 - a for a in alphas},
        "z_theory": {a: 1.0 - a for a in alphas},
    }


def test_save_data_txt_includes_kth_columns_when_provided(
    tmp_path, dummy_results, dummy_kth_full_result
):
    path = exporter.save_data_txt(
        dummy_results, str(tmp_path), kth_full_result=dummy_kth_full_result
    )
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    assert lines[0].startswith("# KTH identified reaction model: F1")
    header = lines[1].strip().split("\t")
    assert "KTH_A" in header
    assert "KTH_f_alpha" in header


def test_save_data_txt_omits_kth_columns_when_not_provided(tmp_path, dummy_results):
    path = exporter.save_data_txt(dummy_results, str(tmp_path))
    with open(path, "r", encoding="utf-8") as fh:
        first_line = fh.readline()
    # No KTH comment line, and the header (first line) starts with alpha.
    assert first_line.startswith("alpha")
    assert "KTH_A" not in first_line


def test_save_data_xlsx_adds_kth_model_sheet_when_provided(
    tmp_path, dummy_results, dummy_kth_full_result
):
    import pandas as pd

    path = exporter.save_data_xlsx(
        dummy_results, str(tmp_path), kth_full_result=dummy_kth_full_result
    )
    xl = pd.ExcelFile(path)
    assert "KTH_Model" in xl.sheet_names

    main_sheet = xl.parse("Ea_vs_alpha")
    assert "KTH_A" in main_sheet.columns
    assert "KTH_f_alpha" in main_sheet.columns


def test_export_all_saves_kth_model_plot_when_both_provided(
    tmp_path, dummy_results, dummy_kth_full_result
):
    fig = plotter.build_figure()
    plotter.update_plot(fig, dummy_results)
    kth_fig = plotter.build_kth_model_figure()
    plotter.update_kth_model_plot(kth_fig, dummy_kth_full_result)

    out_folder = exporter.export_all(
        str(tmp_path),
        dummy_results,
        fig,
        kth_full_result=dummy_kth_full_result,
        kth_figure=kth_fig,
    )

    actual = set(os.listdir(out_folder))
    assert "KTH_model_plot.png" in actual
    assert "KTH_model_plot.svg" in actual
    assert "KTH_model_plot.pdf" in actual


def test_export_all_skips_kth_model_plot_when_kth_full_result_missing(
    tmp_path, dummy_results
):
    fig = plotter.build_figure()
    plotter.update_plot(fig, dummy_results)
    kth_fig = plotter.build_kth_model_figure()

    out_folder = exporter.export_all(
        str(tmp_path), dummy_results, fig, kth_figure=kth_fig
    )
    actual = set(os.listdir(out_folder))
    assert "KTH_model_plot.png" not in actual


def test_kth_model_figure_builds_and_updates(dummy_kth_full_result):
    fig = plotter.build_kth_model_figure()
    assert len(fig.axes) == 2

    updated = plotter.update_kth_model_plot(fig, dummy_kth_full_result)
    assert len(updated.axes) == 2
    # Both experimental and theoretical curves should have been drawn.
    total_lines = sum(len(ax.lines) for ax in updated.axes)
    assert total_lines >= 4  # 2 experimental + 2 theoretical, one per axis


def test_kth_model_figure_shows_placeholder_when_no_result():
    fig = plotter.build_kth_model_figure()
    updated = plotter.update_kth_model_plot(fig, None)
    for ax in updated.axes:
        assert len(ax.lines) == 0


# --------------------------------------------------------------------------- #
# Journal style presets and confidence bands
# --------------------------------------------------------------------------- #
def test_apply_plot_style_returns_the_applied_dict():
    style = plotter.apply_plot_style("Nature")
    assert style == plotter.PLOT_STYLES["Nature"]
    plotter.apply_plot_style("ACS")  # reset for other tests


def test_apply_plot_style_falls_back_to_default_for_unknown_name():
    style = plotter.apply_plot_style("NotARealJournal")
    assert style == plotter.PLOT_STYLES[plotter.DEFAULT_STYLE]
    plotter.apply_plot_style("ACS")  # reset for other tests


def test_colorblind_style_actually_changes_rendered_colors(dummy_results):
    """
    Regression test for the bug caught during development: setting the
    Colorblind/Monochrome rcParams color cycle alone has zero effect,
    because update_plot() always passes an explicit color= per method.
    """
    plotter.apply_plot_style("ACS")
    fig_default = plotter.build_figure()
    plotter.update_plot(fig_default, dummy_results)
    default_color = fig_default.axes[0].lines[0].get_color()

    plotter.apply_plot_style("Colorblind")
    fig_cb = plotter.build_figure()
    plotter.update_plot(fig_cb, dummy_results)
    colorblind_color = fig_cb.axes[0].lines[0].get_color()

    plotter.apply_plot_style("ACS")  # reset for other tests
    assert default_color != colorblind_color


def test_update_plot_draws_confidence_band_when_provided():
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = {"KAS": {a: 100.0 for a in alphas}}
    ci = {"KAS": {a: (100.0, 90.0, 110.0) for a in alphas}}

    fig = plotter.build_figure()
    plotter.update_plot(fig, results, confidence_intervals=ci)

    ax = fig.axes[0]
    # fill_between creates a PolyCollection; there should be exactly one.
    assert len(ax.collections) == 1


def test_update_plot_skips_confidence_band_when_absent():
    alphas = [0.1, 0.2, 0.3]
    results = {"KAS": {a: 100.0 for a in alphas}}

    fig = plotter.build_figure()
    plotter.update_plot(fig, results)

    ax = fig.axes[0]
    assert len(ax.collections) == 0


def test_update_plot_skips_confidence_band_entries_with_nan():
    alphas = [0.1, 0.2, 0.3]
    results = {"KAS": {a: 100.0 for a in alphas}}
    ci = {"KAS": {a: (100.0, float("nan"), float("nan")) for a in alphas}}

    fig = plotter.build_figure()
    # Must not raise despite every CI entry being NaN.
    plotter.update_plot(fig, results, confidence_intervals=ci)
    ax = fig.axes[0]
    assert len(ax.collections) == 0


# --------------------------------------------------------------------------- #
# xlsxwriter-missing resilience (regression test for a production bug)
# --------------------------------------------------------------------------- #
def test_export_all_never_crashes_when_xlsxwriter_is_missing(tmp_path, dummy_results, monkeypatch):
    """
    Regression test for a production bug: with xlsxwriter not
    installed, export_all() used to crash with an uncaught ImportError,
    even though the image files and the tab-delimited text file had
    already been written successfully at that point. It must now skip
    the Excel file gracefully and still return normally.
    """
    monkeypatch.setattr(exporter, "EXCEL_EXPORT_AVAILABLE", False)

    fig = plotter.build_figure()
    plotter.update_plot(fig, dummy_results)
    out_folder = exporter.export_all(str(tmp_path), dummy_results, fig)  # must not raise

    actual = set(os.listdir(out_folder))
    assert "Ea_data.txt" in actual
    assert "Ea_vs_alpha.png" in actual
    assert "Ea_vs_alpha.tiff" in actual
    assert "Ea_results.xlsx" not in actual


def test_save_data_xlsx_raises_clear_importerror_when_unavailable(tmp_path, dummy_results, monkeypatch):
    monkeypatch.setattr(exporter, "EXCEL_EXPORT_AVAILABLE", False)
    with pytest.raises(ImportError, match="xlsxwriter"):
        exporter.save_data_xlsx(dummy_results, str(tmp_path))


def test_excel_export_available_reflects_real_import():
    # In this test environment xlsxwriter is an actual dependency, so
    # this should be True; mainly guards against the flag never being
    # computed at all (e.g. accidentally hardcoded).
    import importlib

    try:
        importlib.import_module("xlsxwriter")
        expected = True
    except ImportError:
        expected = False
    assert exporter.EXCEL_EXPORT_AVAILABLE == expected
