"""
exporter.py
===========
Functions to persist analysis results (graphs and numeric data) to a
timestamped output folder, in formats convenient for further use:

* Ea_vs_alpha.png / .svg / .pdf / .tiff -- the plot, in common formats.
* Ea_data.txt                    -- tab-delimited text, directly
                                     importable into Origin, Excel, etc.
* Ea_results.xlsx                -- optional multi-sheet Excel workbook
                                     (skipped gracefully, with a logged
                                     warning, if the optional
                                     ``xlsxwriter`` package is missing --
                                     see EXCEL_EXPORT_AVAILABLE).
* Summary.txt                    -- human-readable run summary (mean,
                                     spread and range of Ea per method).

Figures are always written with the **light** theme (white background)
even when the application is showing the dark theme, and every number
follows the conventions in :mod:`isokinetics.formatting` (data files keep
a few more digits than the on-screen tables).
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Dict, Optional

import numpy as np
from matplotlib.figure import Figure

from . import plotter
from .formatting import (
    EXPORT_ALPHA_DECIMALS,
    EXPORT_EA_DECIMALS,
    EXPORT_SCI_DECIMALS,
    EXPORT_SCORE_DECIMALS,
    fmt_ea,
    fmt_fixed,
    fmt_sci,
)

logger = logging.getLogger(__name__)


def _xlsxwriter_available() -> bool:
    """Whether the optional ``xlsxwriter`` package can be imported."""
    try:
        import xlsxwriter  # noqa: F401
        return True
    except ImportError:
        return False


#: Computed once at import time. The GUI checks this to warn the user
#: upfront (rather than only after a failed export) if Excel export
#: will be unavailable; ``export_all``/``save_data_xlsx`` also check it
#: themselves so a missing ``xlsxwriter`` can never crash an export --
#: it just skips that one file with a logged warning.
EXCEL_EXPORT_AVAILABLE = _xlsxwriter_available()

from .kinetics import average_ea


def make_output_folder(base_dir: str) -> str:
    """
    Create (and return the path to) a new timestamped subfolder inside
    ``base_dir``, named ``Kinetics_Results_YYYYMMDD_HHMMSS``.

    Parameters
    ----------
    base_dir : str
        Directory chosen by the user (must already exist).

    Returns
    -------
    str
        Full path to the newly created subfolder.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"Kinetics_Results_{timestamp}"
    out_path = os.path.join(base_dir, folder_name)
    os.makedirs(out_path, exist_ok=False)
    return out_path


def save_graph(
    fig: Figure,
    out_folder: str,
    basename: str = "Ea_vs_alpha",
    include_tiff: bool = True,
    tiff_dpi: int = 300,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Save the given matplotlib figure to PNG (300 dpi), SVG, PDF, and
    (by default) TIFF with LZW compression and embedded metadata.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to export (as currently drawn).
    out_folder : str
        Destination folder (must exist).
    basename : str, optional
        File name (without extension) used for all formats.
    include_tiff : bool, optional
        Whether to also save a ``.tiff`` file (default True). TIFF is
        widely required by journals for print-quality raster figures;
        LZW compression keeps the file size reasonable without any
        quality loss (it's a lossless compression scheme).
    tiff_dpi : int, optional
        Resolution for the TIFF export specifically (default 300;
        300-600 is typical for print, matching common journal
        requirements).
    metadata : dict, optional
        String key/value pairs (e.g. ``{"Author": "...", "Software":
        "Isoconversional Kinetics Analyzer v1.6.0", "Sample": "...",
        "Methods": "..."}``) embedded in the TIFF's ImageDescription
        tag as ``"key: value; key: value; ..."``. Ignored if
        ``include_tiff`` is False.

    Returns
    -------
    dict
        Mapping of format -> full file path written (keys: ``"png"``,
        ``"svg"``, ``"pdf"``, and ``"tiff"`` if ``include_tiff``).
    """
    paths = {}
    # Always export on a white background, whatever theme is on screen.
    with plotter.export_theme(fig, "light"):
        for ext, kwargs in (
            ("png", {"dpi": 300}),
            ("svg", {}),
            ("pdf", {}),
        ):
            path = os.path.join(out_folder, f"{basename}.{ext}")
            fig.savefig(path, bbox_inches="tight", **kwargs)
            paths[ext] = path

        if include_tiff:
            tiff_path = os.path.join(out_folder, f"{basename}.tiff")
            pil_kwargs = {"compression": "tiff_lzw"}
            if metadata:
                pil_kwargs["description"] = "; ".join(
                    f"{k}: {v}" for k, v in metadata.items()
                )
            fig.savefig(tiff_path, dpi=tiff_dpi, bbox_inches="tight", pil_kwargs=pil_kwargs)
            paths["tiff"] = tiff_path

    return paths


def save_data_txt(
    results: Dict[str, Dict[float, float]],
    out_folder: str,
    filename: str = "Ea_data.txt",
    averages: Optional[Dict[str, float]] = None,
    kth_full_result: Optional[Dict[str, object]] = None,
) -> str:
    """
    Write a tab-delimited text file with the Ea(alpha) results for every
    method, directly importable into Origin (or any spreadsheet/plotting
    tool that reads tab-delimited text).

    File layout::

        alpha    Friedman   FWO   KAS   Starink   Vyazovkin
        (avg)    98.3       83.8  78.1  78.4      78.6
        0.01     ...        ...   ...   ...       ...
        0.02     ...        ...   ...   ...       ...
        ...

    If ``kth_full_result`` is provided (i.e. KTH was run in "Full
    triplet" mode), two extra columns -- ``KTH_A`` (pre-exponential
    factor, s^-1) and ``KTH_f_alpha`` (identified reaction model's
    value) -- are appended, along with a comment line naming the
    identified model.

    Parameters
    ----------
    results : dict
        Mapping ``method_name -> {alpha: Ea}``.
    out_folder : str
        Destination folder (must exist).
    filename : str, optional
        Output file name (default ``Ea_data.txt``).
    averages : dict, optional
        Mapping ``method_name -> average Ea``. Computed automatically if
        omitted.
    kth_full_result : dict, optional
        The dict returned by :func:`isokinetics.kinetics.kth_full`, if
        KTH was run in "Full triplet" mode. Adds ``KTH_A`` and
        ``KTH_f_alpha`` columns when present.

    Returns
    -------
    str
        Full path to the written file.
    """
    method_names = list(results.keys())
    if averages is None:
        averages = {name: average_ea(results[name]) for name in method_names}

    # Union of every alpha value seen across all methods, sorted ascending.
    all_alphas = sorted(
        {a for method_results in results.values() for a in method_results.keys()}
    )

    kth_A = kth_full_result["A"] if kth_full_result else {}
    kth_f = kth_full_result["f_alpha"] if kth_full_result else {}
    extra_cols = ["KTH_A", "KTH_f_alpha"] if kth_full_result else []

    path = os.path.join(out_folder, filename)
    with open(path, "w", encoding="utf-8") as fh:
        if kth_full_result is not None:
            fh.write(f"# KTH identified reaction model: {kth_full_result.get('model', 'unknown')}\n")

        header = ["alpha"] + method_names + extra_cols
        fh.write("\t".join(header) + "\n")

        avg_row = ["(average)"] + [
            fmt_fixed(averages.get(name, float("nan")), EXPORT_EA_DECIMALS, na="NaN")
            for name in method_names
        ]
        if kth_full_result is not None:
            finite_A = [v for v in kth_A.values() if v == v]
            avg_A = (
                fmt_sci(sum(finite_A) / len(finite_A), EXPORT_SCI_DECIMALS, na="NaN")
                if finite_A else "NaN"
            )
            avg_row += [avg_A, "-"]
        fh.write("\t".join(avg_row) + "\n")

        for a in all_alphas:
            row = [fmt_fixed(a, EXPORT_ALPHA_DECIMALS, na="NaN")]
            for name in method_names:
                ea = results[name].get(a, float("nan"))
                row.append(fmt_fixed(ea, EXPORT_EA_DECIMALS, na="NaN"))
            if kth_full_result is not None:
                a_val = kth_A.get(a, float("nan"))
                f_val = kth_f.get(a, float("nan"))
                row.append(fmt_sci(a_val, EXPORT_SCI_DECIMALS, na="NaN"))
                row.append(fmt_fixed(f_val, EXPORT_SCORE_DECIMALS, na="NaN"))
            fh.write("\t".join(row) + "\n")

    return path


def method_statistics(results: Dict[str, Dict[float, float]]) -> Dict[str, Dict[str, float]]:
    """Mean, standard deviation, minimum and maximum of Ea(alpha) per
    method (NaNs ignored); used by the summary file and the GUI."""
    stats: Dict[str, Dict[str, float]] = {}
    for name, res in results.items():
        vals = np.array([v for v in res.values() if v == v], dtype=float)
        if vals.size == 0:
            stats[name] = {"mean": float("nan"), "std": float("nan"),
                           "min": float("nan"), "max": float("nan"), "n": 0}
        else:
            stats[name] = {"mean": float(vals.mean()), "std": float(vals.std()),
                           "min": float(vals.min()), "max": float(vals.max()),
                           "n": int(vals.size)}
    return stats


def save_summary_txt(
    results: Dict[str, Dict[float, float]],
    out_folder: str,
    filename: str = "Summary.txt",
    metadata: Optional[Dict[str, str]] = None,
    cr_result: Optional[Dict[str, object]] = None,
    kth_full_result: Optional[Dict[str, object]] = None,
) -> str:
    """Write a short human-readable summary of the run (metadata, per-method
    mean/std/range of Ea, KTH model, Coats-Redfern recommendation)."""
    stats = method_statistics(results)
    path = os.path.join(out_folder, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Isoconversional kinetics - run summary\n")
        fh.write("=" * 40 + "\n")
        for k, v in (metadata or {}).items():
            fh.write(f"{k}: {v}\n")
        fh.write("\nActivation energy Ea (kJ/mol) over the alpha grid\n")
        fh.write(f"{'Method':<12}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}{'N':>6}\n")
        for name, st in stats.items():
            fh.write(
                f"{name:<12}{fmt_ea(st['mean']):>10}{fmt_ea(st['std']):>10}"
                f"{fmt_ea(st['min']):>10}{fmt_ea(st['max']):>10}{st['n']:>6d}\n"
            )
        if kth_full_result is not None:
            fh.write(f"\nKTH identified reaction model: {kth_full_result.get('model', 'unknown')}\n")
        if cr_result:
            rec = cr_result.get("recommended_model")
            fh.write(
                f"\nCoats-Redfern: reference Ea = {fmt_ea(cr_result.get('reference_ea_kj'), unit=True)}; "
                f"recommended model = {rec or 'none (inconclusive)'}\n"
            )
    return path


def _cr_rows(cr_result: Dict[str, object]):
    """Rows (model, Ea, A, R2, AIC, BIC, gap %) sorted by R^2, best first."""
    rows = []
    for name, fit in (cr_result.get("results") or {}).items():
        rows.append({
            "model": name, "Ea_kJ_per_mol": fit.get("Ea", float("nan")),
            "A_per_s": fit.get("A", float("nan")), "R2": fit.get("r2", float("nan")),
            "AIC": fit.get("aic", float("nan")), "BIC": fit.get("bic", float("nan")),
            "Ea_gap_vs_reference_pct": 100.0 * fit.get("ea_gap_pct", float("nan")),
        })
    rows.sort(key=lambda r: (r["R2"] != r["R2"], -(r["R2"] if r["R2"] == r["R2"] else 0)))
    return rows


def save_data_xlsx(
    results: Dict[str, Dict[float, float]],
    out_folder: str,
    filename: str = "Ea_results.xlsx",
    averages: Optional[Dict[str, float]] = None,
    kth_full_result: Optional[Dict[str, object]] = None,
    cr_result: Optional[Dict[str, object]] = None,
) -> str:
    """
    Write an Excel workbook with two sheets: "Ea_vs_alpha" (full table)
    and "Averages" (one row per method). If ``kth_full_result`` is
    provided, the "Ea_vs_alpha" sheet also gets ``KTH_A`` and
    ``KTH_f_alpha`` columns, and a third sheet, "KTH_Model", is added
    with the identified model, its match scores against every candidate,
    and the y(alpha)/z(alpha) diagnostic curves (experimental and
    theoretical).

    Parameters
    ----------
    results : dict
        Mapping ``method_name -> {alpha: Ea}``.
    out_folder : str
        Destination folder (must exist).
    filename : str, optional
        Output file name (default ``Ea_results.xlsx``).
    averages : dict, optional
        Mapping ``method_name -> average Ea``. Computed automatically if
        omitted.
    kth_full_result : dict, optional
        The dict returned by :func:`isokinetics.kinetics.kth_full`, if
        KTH was run in "Full triplet" mode.

    Returns
    -------
    str
        Full path to the written file.

    Raises
    ------
    ImportError
        If the optional ``xlsxwriter`` package is not installed (check
        :data:`EXCEL_EXPORT_AVAILABLE` beforehand to avoid this, or call
        this via :func:`export_all`, which catches it and skips the
        Excel file gracefully instead of raising).
    """
    if not EXCEL_EXPORT_AVAILABLE:
        raise ImportError(
            "Excel export requires the 'xlsxwriter' package, which is not "
            "installed. Install it with: pip install xlsxwriter"
        )

    import pandas as pd  # local import keeps module import light for GUI startup

    method_names = list(results.keys())
    if averages is None:
        averages = {name: average_ea(results[name]) for name in method_names}

    all_alphas = sorted(
        {a for method_results in results.values() for a in method_results.keys()}
    )

    table = {"alpha": all_alphas}
    for name in method_names:
        table[name] = [results[name].get(a, float("nan")) for a in all_alphas]

    if kth_full_result is not None:
        kth_A = kth_full_result["A"]
        kth_f = kth_full_result["f_alpha"]
        table["KTH_A"] = [kth_A.get(a, float("nan")) for a in all_alphas]
        table["KTH_f_alpha"] = [kth_f.get(a, float("nan")) for a in all_alphas]

    df_full = pd.DataFrame(table)

    df_avg = pd.DataFrame(
        {"method": method_names, "average_Ea_kJ_per_mol": [averages[n] for n in method_names]}
    )

    path = os.path.join(out_folder, filename)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        book = writer.book
        fmt_a = book.add_format({"num_format": "0.000"})
        fmt_ea_x = book.add_format({"num_format": "0.00"})
        fmt_sci_x = book.add_format({"num_format": "0.000E+00"})
        fmt_head = book.add_format({"bold": True, "bg_color": "#DBE7FF", "border": 1})

        df_full.to_excel(writer, sheet_name="Ea_vs_alpha", index=False)
        ws = writer.sheets["Ea_vs_alpha"]
        for col_idx, col in enumerate(df_full.columns):
            if col == "alpha":
                ws.set_column(col_idx, col_idx, 10, fmt_a)
            elif col == "KTH_A":
                ws.set_column(col_idx, col_idx, 14, fmt_sci_x)
            elif col == "KTH_f_alpha":
                ws.set_column(col_idx, col_idx, 12, book.add_format({"num_format": "0.0000"}))
            else:
                ws.set_column(col_idx, col_idx, 12, fmt_ea_x)
            ws.write(0, col_idx, col, fmt_head)
        ws.freeze_panes(1, 1)

        df_avg.to_excel(writer, sheet_name="Averages", index=False)
        writer.sheets["Averages"].set_column(0, 0, 14)
        writer.sheets["Averages"].set_column(1, 1, 22, fmt_ea_x)

        if cr_result and cr_result.get("results"):
            df_cr = pd.DataFrame(_cr_rows(cr_result))
            df_cr.to_excel(writer, sheet_name="Model_Fitting", index=False)
            wsm = writer.sheets["Model_Fitting"]
            wsm.set_column(0, 0, 10)
            wsm.set_column(1, 1, 14, fmt_ea_x)
            wsm.set_column(2, 2, 14, fmt_sci_x)
            wsm.set_column(3, 6, 14, book.add_format({"num_format": "0.0000"}))
            wsm.write(len(df_cr) + 2, 0,
                      f"Reference Ea (kJ/mol): {cr_result.get('reference_ea_kj')}")
            wsm.write(len(df_cr) + 3, 0,
                      f"Recommended model: {cr_result.get('recommended_model') or 'none (inconclusive)'}")

        if kth_full_result is not None:
            model_scores = kth_full_result.get("model_scores", {}) or {}
            df_model_summary = pd.DataFrame(
                {
                    "identified_model": [kth_full_result.get("model", "unknown")],
                }
            )
            df_scores = pd.DataFrame(
                sorted(model_scores.items(), key=lambda kv: kv[1]),
                columns=["candidate_model", "match_score_lower_is_better"],
            )
            df_curves = pd.DataFrame(
                {
                    "alpha": all_alphas,
                    "y_alpha_experimental": [
                        kth_full_result.get("y_alpha", {}).get(a, float("nan"))
                        for a in all_alphas
                    ],
                    "z_alpha_experimental": [
                        kth_full_result.get("z_alpha", {}).get(a, float("nan"))
                        for a in all_alphas
                    ],
                    "y_alpha_theoretical": [
                        kth_full_result.get("y_theory", {}).get(a, float("nan"))
                        for a in all_alphas
                    ],
                    "z_alpha_theoretical": [
                        kth_full_result.get("z_theory", {}).get(a, float("nan"))
                        for a in all_alphas
                    ],
                }
            )
            df_model_summary.to_excel(
                writer, sheet_name="KTH_Model", index=False, startrow=0
            )
            df_scores.to_excel(writer, sheet_name="KTH_Model", index=False, startrow=3)
            df_curves.to_excel(
                writer,
                sheet_name="KTH_Model",
                index=False,
                startrow=3 + len(df_scores) + 3,
            )

    return path


def export_all(
    base_dir: str,
    results: Dict[str, Dict[float, float]],
    fig: Figure,
    include_excel: bool = True,
    kth_full_result: Optional[Dict[str, object]] = None,
    kth_figure: Optional[Figure] = None,
    include_tiff: bool = True,
    metadata: Optional[Dict[str, str]] = None,
    cr_result: Optional[Dict[str, object]] = None,
    cr_figure: Optional[Figure] = None,
) -> str:
    """
    Convenience wrapper that creates the timestamped output folder and
    writes graphs, the tab-delimited text file, and (optionally) the
    Excel workbook, in a single call.

    Parameters
    ----------
    base_dir : str
        Directory chosen by the user via the folder-selection dialog.
    results : dict
        Mapping ``method_name -> {alpha: Ea}`` for every method that was
        run.
    fig : matplotlib.figure.Figure
        The currently displayed Ea(alpha) plot.
    include_excel : bool, optional
        Whether to also write the ``.xlsx`` workbook (default True).
    kth_full_result : dict, optional
        The dict returned by :func:`isokinetics.kinetics.kth_full`, if
        KTH was run in "Full triplet" mode. When provided, adds
        ``KTH_A``/``KTH_f_alpha`` columns to the text/Excel exports and
        (if ``kth_figure`` is also given) saves the KTH model diagnostic
        plot alongside the main graph.
    kth_figure : matplotlib.figure.Figure, optional
        The currently displayed KTH model diagnostic plot (y(alpha)/
        z(alpha) vs. the identified model), saved as
        ``KTH_model_plot.png/.svg/.pdf`` when both this and
        ``kth_full_result`` are provided.
    include_tiff : bool, optional
        Whether the graph exports also include a 300 dpi, LZW-compressed
        ``.tiff`` (default True) -- see :func:`save_graph`.
    metadata : dict, optional
        String key/value pairs embedded in every exported TIFF's
        ImageDescription tag (e.g. author, sample name, methods used).
        Ignored if ``include_tiff`` is False.

    cr_result : dict, optional
        The dict returned by :func:`isokinetics.kinetics.coats_redfern`.
        Adds a ``Model_Fitting`` sheet to the workbook and a line to
        ``Summary.txt``.
    cr_figure : matplotlib.figure.Figure, optional
        The Coats-Redfern R^2 chart, saved as ``Coats_Redfern_plot.*``
        when ``cr_result`` is also given.

    Returns
    -------
    str
        Full path to the created output folder.
    """
    out_folder = make_output_folder(base_dir)
    averages = {name: average_ea(res) for name, res in results.items()}

    save_graph(fig, out_folder, include_tiff=include_tiff, metadata=metadata)
    if kth_full_result is not None and kth_figure is not None:
        save_graph(
            kth_figure,
            out_folder,
            basename="KTH_model_plot",
            include_tiff=include_tiff,
            metadata=metadata,
        )

    if cr_result and cr_result.get("results") and cr_figure is not None:
        save_graph(
            cr_figure, out_folder, basename="Coats_Redfern_plot",
            include_tiff=include_tiff, metadata=metadata,
        )

    save_data_txt(results, out_folder, averages=averages, kth_full_result=kth_full_result)
    save_summary_txt(
        results, out_folder, metadata=metadata, cr_result=cr_result,
        kth_full_result=kth_full_result,
    )
    if include_excel:
        try:
            save_data_xlsx(
                results, out_folder, averages=averages,
                kth_full_result=kth_full_result, cr_result=cr_result,
            )
        except ImportError as exc:
            # A missing optional dependency (xlsxwriter) must never
            # take down an otherwise-successful export -- the graphs
            # and the tab-delimited text file above have already been
            # written successfully at this point.
            logger.warning(
                "Skipping Excel export (%s). PNG/TIFF/SVG/PDF and the "
                "tab-delimited text file were still saved successfully.",
                exc,
            )

    return out_folder
