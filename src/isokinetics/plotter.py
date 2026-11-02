"""
plotter.py
==========
Matplotlib helpers for rendering the Ea(alpha) isoconversional plots.

The plots are designed to be publication-ready: clean grid, Unicode
Greek-letter axis labels, one legend entry per method carrying its mean
activation energy, and number formatting shared with the rest of the
application (see :mod:`isokinetics.formatting`).

Light / dark theme
------------------
The on-screen figures follow the application's theme.  The theme only
changes *colours* (backgrounds, text, grid, spines), never the data, and
is applied in place by :func:`apply_figure_theme`.  Anything written to
disk goes through :func:`export_theme`, which temporarily switches a
figure to the light theme so exported figures always have a white
background suitable for publication, whatever the screen theme is.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # safe default; the GUI switches to TkAgg explicitly.
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

from .formatting import UNIT_EA, fmt_ea  # noqa: E402

#: Colour palettes for the two application themes.  The GUI imports the
#: same table so widgets and plots always agree.
THEMES: Dict[str, Dict[str, str]] = {
    "light": {
        "window": "#f3f5f9", "card": "#ffffff", "card_alt": "#f7f8fb",
        "figure": "#ffffff", "axes": "#ffffff",
        "text": "#1f2933", "muted": "#6b7785", "grid": "#d9dee6",
        "spine": "#9aa5b1", "legend": "#ffffff",
        "accent": "#2563eb", "accent_hover": "#1d4ed8",
        "ok": "#15803d", "bad": "#dc2626", "warn": "#b45309",
        "row_alt": "#f4f6fa", "select": "#dbe7ff", "border": "#e2e6ee",
    },
    "dark": {
        "window": "#14171c", "card": "#1d2128", "card_alt": "#232831",
        "figure": "#1d2128", "axes": "#1d2128",
        "text": "#e6e9ee", "muted": "#9aa5b4", "grid": "#323946",
        "spine": "#5b6574", "legend": "#262c35",
        "accent": "#3b82f6", "accent_hover": "#60a5fa",
        "ok": "#4ade80", "bad": "#f87171", "warn": "#fbbf24",
        "row_alt": "#222730", "select": "#27406b", "border": "#2d333d",
    },
}

#: Theme applied to newly built / redrawn figures (module-level state,
#: like matplotlib's own rcParams).
_active_theme: str = "light"

#: Colours that would be invisible on a dark background are lightened
#: while the dark theme is shown (and restored for export).
_DARK_UNSAFE_COLORS = {"#000000", "#252525", "#404040"}

#: Consistent colour per method so repeated redraws look stable.
METHOD_COLORS = {
    "Friedman": "#1f77b4",
    "FWO": "#ff7f0e",
    "KAS": "#2ca02c",
    "Starink": "#d62728",
    "Vyazovkin": "#9467bd",
    "KTH": "#e377c2",
}

METHOD_MARKERS = {
    "Friedman": "o",
    "FWO": "s",
    "KAS": "^",
    "Starink": "D",
    "Vyazovkin": "v",
    "KTH": "x",
}

#: Journal-style presets. Each is a dict of matplotlib rcParams applied
#: on top of the base clean style (see _apply_clean_style). These are
#: deliberately modest, "safe" approximations of each journal's typical
#: look (font choice, line weight, grid style) -- not a guarantee of
#: exact compliance with any specific journal's author guidelines,
#: which should always be checked directly for a real submission.
PLOT_STYLES: Dict[str, Dict[str, object]] = {
    "ACS": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "figure.figsize": (7.5, 5.0),
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "legend.framealpha": 0.9,
    },
    "RSC": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "figure.figsize": (6.0, 4.5),
        "lines.linewidth": 1.2,
        "axes.grid": False,
        "legend.framealpha": 1.0,
    },
    "Nature": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "figure.figsize": (6.5, 4.8),
        "lines.linewidth": 1.2,
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.alpha": 0.3,
        "legend.framealpha": 0.8,
    },
    "Elsevier": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "figure.figsize": (7.0, 5.0),
        "lines.linewidth": 1.3,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "legend.framealpha": 0.9,
    },
    "Colorblind": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "figure.figsize": (7.5, 5.0),
        "lines.linewidth": 1.6,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        # Okabe-Ito colorblind-safe palette.
        "axes.prop_cycle": plt.cycler(
            color=["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                   "#0072B2", "#D55E00", "#CC79A7", "#000000"]
        ),
    },
    "Monochrome": {
        "font.family": "sans-serif",
        "figure.figsize": (7.0, 5.0),
        "lines.linewidth": 1.4,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "axes.prop_cycle": plt.cycler(
            color=["#000000", "#404040", "#707070", "#969696",
                   "#bdbdbd", "#252525", "#525252", "#737373"]
        ),
    },
}

#: The style applied by default (matches the original hardcoded look).
DEFAULT_STYLE = "ACS"

#: Per-method color overrides for styles where accessibility/print
#: constraints matter more than the app's usual color identity (see
#: the note in apply_plot_style() below for why this is necessary).
_STYLE_METHOD_COLOR_OVERRIDES: Dict[str, Dict[str, str]] = {
    "Colorblind": {
        "Friedman": "#0072B2", "FWO": "#E69F00", "KAS": "#009E73",
        "Starink": "#D55E00", "Vyazovkin": "#CC79A7", "KTH": "#F0E442",
    },
    "Monochrome": {
        "Friedman": "#000000", "FWO": "#404040", "KAS": "#707070",
        "Starink": "#969696", "Vyazovkin": "#252525", "KTH": "#525252",
    },
}

#: Tracks the currently-applied style name so update_plot() knows
#: whether to use the standard METHOD_COLORS or a style-specific
#: override (module-level state, mirroring how matplotlib's own
#: rcParams are global/module-level too).
_active_style_name: str = DEFAULT_STYLE


def apply_plot_style(style_name: str) -> Dict[str, object]:
    """
    Apply a named journal-style preset to matplotlib's global rcParams
    (on top of the base clean style).

    Note on per-method colors
    ---------------------------
    ``update_plot()`` normally draws each method in a fixed, consistent
    color (:data:`METHOD_COLORS`) so a given method looks the same
    across every plot in the app. Matplotlib's rcParams color cycle
    (``axes.prop_cycle``, used by the "Colorblind" and "Monochrome"
    styles here) has **no effect** when a plot call passes an explicit
    ``color=`` argument, as this module's plotting functions always do
    -- so setting a color-cycle rcParam alone would silently do
    nothing (caught during testing: the "Colorblind" style rendered
    with zero pixels of its intended palette). To actually change
    colors for these two styles, :func:`update_plot` consults
    :data:`_STYLE_METHOD_COLOR_OVERRIDES` for the currently-active
    style instead of (or in addition to) the rcParam cycle.

    Parameters
    ----------
    style_name : str
        A key of :data:`PLOT_STYLES` (e.g. ``"ACS"``, ``"Nature"``,
        ``"Colorblind"``). Falls back to :data:`DEFAULT_STYLE` if the
        name is not recognised.

    Returns
    -------
    dict
        The rcParams dict that was applied, for reference/testing.
    """
    global _active_style_name
    _apply_clean_style()
    if style_name not in PLOT_STYLES:
        style_name = DEFAULT_STYLE
    style = PLOT_STYLES[style_name]
    plt.rcParams.update(style)
    _active_style_name = style_name
    return style


def _method_color(method_name: str) -> Optional[str]:
    """Resolve the color to use for a method, honoring the currently
    active style's override table (Colorblind/Monochrome) if any."""
    override_table = _STYLE_METHOD_COLOR_OVERRIDES.get(_active_style_name)
    if override_table and method_name in override_table:
        return override_table[method_name]
    return METHOD_COLORS.get(method_name, None)


def set_theme(name: str) -> str:
    """Select the theme (``"light"`` or ``"dark"``) used by the
    ``build_*`` / ``update_*`` functions.  Unknown names fall back to
    light.  Returns the theme actually selected."""
    global _active_theme
    _active_theme = name if name in THEMES else "light"
    return _active_theme


def get_theme() -> str:
    """Name of the currently selected theme."""
    return _active_theme


def apply_figure_theme(fig: Figure, theme: Optional[str] = None) -> Figure:
    """
    Recolour an existing figure *in place* for ``theme`` (default: the
    current one): figure/axes backgrounds, spines, ticks, labels, titles,
    grid, legend, placeholder messages, and lines that would be invisible
    on a dark background.  Data and layout are untouched, so this is
    cheap and can be called repeatedly.
    """
    name = theme if theme in THEMES else _active_theme
    pal = THEMES[name]
    fig._iso_theme = name  # remembered so export_theme() can restore it

    fig.set_facecolor(pal["figure"])
    sup = getattr(fig, "_suptitle", None)
    if sup is not None:
        sup.set_color(pal["text"])

    for ax in fig.axes:
        ax.set_facecolor(pal["axes"])
        for spine in ax.spines.values():
            spine.set_edgecolor(pal["spine"])
        ax.tick_params(axis="both", which="both", colors=pal["text"])
        ax.xaxis.label.set_color(pal["text"])
        ax.yaxis.label.set_color(pal["text"])
        ax.title.set_color(pal["text"])
        for gl in ax.get_xgridlines() + ax.get_ygridlines():
            gl.set_color(pal["grid"])

        for text in ax.texts:
            if text.get_gid() == "placeholder":
                text.set_color(pal["muted"])

        legend = ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor(pal["legend"])
            frame.set_edgecolor(pal["spine"])
            for text in legend.get_texts():
                text.set_color(pal["text"])

        for line in ax.lines:
            original = getattr(line, "_iso_orig_color", None)
            if name == "dark":
                try:
                    current = to_hex(line.get_color())
                except ValueError:
                    continue
                if original is None and current in _DARK_UNSAFE_COLORS:
                    line._iso_orig_color = line.get_color()
                    line.set_color("#e5e7eb")
                    line.set_markerfacecolor("#e5e7eb")
                    line.set_markeredgecolor("#e5e7eb")
            elif original is not None:
                line.set_color(original)
                line.set_markerfacecolor(original)
                line.set_markeredgecolor(original)
                del line._iso_orig_color
    return fig


@contextmanager
def export_theme(fig: Figure, theme: str = "light") -> Iterator[Figure]:
    """
    Context manager that shows ``fig`` in ``theme`` (default: light, i.e.
    a white publication background) while it is saved to disk, then
    restores whatever theme it had on screen.
    """
    previous = getattr(fig, "_iso_theme", _active_theme)
    apply_figure_theme(fig, theme)
    try:
        yield fig
    finally:
        apply_figure_theme(fig, previous)


def _apply_clean_style() -> None:
    """Apply a clean, publication-style matplotlib configuration."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#333333",
            "grid.color": "#cccccc",
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )


def build_figure() -> Figure:
    """
    Create a fresh, empty matplotlib :class:`~matplotlib.figure.Figure`
    pre-configured with the axis labels, title, and grid used throughout
    the application.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with a single axes, ready to be populated by
        :func:`update_plot`.
    """
    _apply_clean_style()
    fig = Figure(figsize=(7.5, 5.0), dpi=100)
    ax = fig.add_subplot(111)
    _configure_axes(ax)
    fig.tight_layout()
    apply_figure_theme(fig)
    return fig


def _configure_axes(ax) -> None:
    """Apply consistent labels/title/grid to an Axes instance."""
    ax.set_xlabel("\u03b1 (conversion)")
    ax.set_ylabel(f"E\u2090 ({UNIT_EA})")
    ax.set_title("Isoconversional Activation Energy")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def _placeholder(ax, message: str, fontsize: int = 11) -> None:
    """Centered 'nothing to show yet' message (recoloured by the theme)."""
    ax.text(
        0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
        fontsize=fontsize, color=THEMES[_active_theme]["muted"], gid="placeholder",
    )


def update_plot(
    fig: Figure,
    results: Dict[str, Dict[float, float]],
    averages: Optional[Dict[str, float]] = None,
    confidence_intervals: Optional[Dict[str, Dict[float, "tuple[float, float, float]"]]] = None,
) -> Figure:
    """
    Clear and redraw the given figure with one Ea(alpha) curve per
    method.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure previously created with :func:`build_figure` (or any
        figure with at least one axes -- it will be cleared).
    results : dict
        Mapping ``method_name -> {alpha: Ea}`` for every method the user
        selected to run.
    averages : dict, optional
        Mapping ``method_name -> average Ea``. If omitted, averages are
        computed on the fly (ignoring NaNs).
    confidence_intervals : dict, optional
        Mapping ``method_name -> {alpha: (Ea, lower, upper)}``, as
        returned by :func:`isokinetics.kinetics.confidence_interval`.
        When present for a given method, a shaded 95%/99% confidence
        band is drawn behind its line.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure object, redrawn in place (convenient for reuse
        with FigureCanvasTkAgg.draw()).
    """
    fig.clear()
    ax = fig.add_subplot(111)
    _configure_axes(ax)

    if averages is None:
        averages = {}
    if confidence_intervals is None:
        confidence_intervals = {}

    any_plotted = False
    for method_name, alpha_to_ea in results.items():
        if not alpha_to_ea:
            continue
        alphas = sorted(alpha_to_ea.keys())
        eas = [alpha_to_ea[a] for a in alphas]

        # Filter out NaNs for plotting, but keep the line contiguous.
        clean_alphas = [a for a, e in zip(alphas, eas) if e == e]  # e == e -> not NaN
        clean_eas = [e for e in eas if e == e]
        if not clean_alphas:
            continue

        avg = averages.get(method_name)
        if avg is None or avg != avg:
            finite = [e for e in clean_eas]
            avg = sum(finite) / len(finite) if finite else float("nan")

        label = f"{method_name} (mean E\u2090 = {fmt_ea(avg, unit=True)})"
        color = _method_color(method_name)
        marker = METHOD_MARKERS.get(method_name, "o")

        ci = confidence_intervals.get(method_name)
        if ci:
            ci_alphas, ci_lo, ci_hi = [], [], []
            for a in clean_alphas:
                entry = ci.get(a)
                if entry is None:
                    continue
                _, lo, hi = entry
                if lo == lo and hi == hi:  # both finite
                    ci_alphas.append(a)
                    ci_lo.append(lo)
                    ci_hi.append(hi)
            if ci_alphas:
                ax.fill_between(
                    ci_alphas, ci_lo, ci_hi, color=color, alpha=0.15, linewidth=0,
                )

        ax.plot(
            clean_alphas,
            clean_eas,
            marker=marker,
            markersize=3.5,
            linewidth=1.6,
            label=label,
            color=color,
        )
        any_plotted = True

    if any_plotted:
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
    else:
        _placeholder(ax, "No results to display yet.\nLoad data and click 'Run Analysis'.")

    fig.tight_layout()
    apply_figure_theme(fig)
    return fig


def build_kth_model_figure() -> Figure:
    """
    Create a fresh, empty two-panel figure for the KTH model
    identification diagnostic plot: Malek's y(alpha) on the left, z(alpha)
    on the right, each ready to show the experimental curve overlaid
    with the identified theoretical model's curve.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with two axes, ready to be populated by
        :func:`update_kth_model_plot`.
    """
    _apply_clean_style()
    fig = Figure(figsize=(9.0, 4.2), dpi=100)
    ax_y = fig.add_subplot(121)
    ax_z = fig.add_subplot(122)
    _configure_kth_axes(ax_y, ax_z)
    fig.tight_layout()
    apply_figure_theme(fig)
    return fig


def _configure_kth_axes(ax_y, ax_z) -> None:
    """Apply consistent labels/titles/grids to the two KTH diagnostic axes."""
    ax_y.set_xlabel("\u03b1 (conversion)")
    ax_y.set_ylabel("y(\u03b1)  (normalised)")
    ax_y.set_title("Malek y(\u03b1)")
    ax_y.grid(True, linestyle="--", alpha=0.6)
    ax_y.set_xlim(0.0, 1.0)
    ax_y.set_ylim(-0.05, 1.05)

    ax_z.set_xlabel("\u03b1 (conversion)")
    ax_z.set_ylabel("z(\u03b1)  (normalised)")
    ax_z.set_title("Malek z(\u03b1)")
    ax_z.grid(True, linestyle="--", alpha=0.6)
    ax_z.set_xlim(0.0, 1.0)
    ax_z.set_ylim(-0.05, 1.05)


def update_kth_model_plot(fig: Figure, kth_full_result: Optional[Dict[str, object]]) -> Figure:
    """
    Clear and redraw the KTH model-identification diagnostic figure.

    Plots the experimental Malek y(alpha)/z(alpha) curves (computed by
    :func:`isokinetics.kinetics.kth_full`) against the identified
    theoretical model's own curves, so the quality of the match is
    visible at a glance. If no model could be confidently identified,
    only the experimental curves are shown with a note to that effect.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure previously created with :func:`build_kth_model_figure`
        (or any figure with two axes -- it will be cleared).
    kth_full_result : dict or None
        The dict returned by :func:`isokinetics.kinetics.kth_full`, or
        ``None`` if KTH's full-triplet mode has not been run yet (in
        which case a placeholder message is shown instead).

    Returns
    -------
    matplotlib.figure.Figure
        The same figure object, redrawn in place.
    """
    fig.clear()
    ax_y = fig.add_subplot(121)
    ax_z = fig.add_subplot(122)
    _configure_kth_axes(ax_y, ax_z)

    if not kth_full_result:
        for ax in (ax_y, ax_z):
            _placeholder(ax, "Run KTH in\n'Full triplet' mode\nto see this plot.", fontsize=10)
        fig.tight_layout()
        apply_figure_theme(fig)
        return fig

    model_name = kth_full_result.get("model", "unknown")

    def _sorted_xy(mapping):
        alphas = sorted(mapping.keys())
        values = [mapping[a] for a in alphas]
        clean_a = [a for a, v in zip(alphas, values) if v == v]
        clean_v = [v for v in values if v == v]
        return clean_a, clean_v

    y_a, y_v = _sorted_xy(kth_full_result.get("y_alpha", {}))
    z_a, z_v = _sorted_xy(kth_full_result.get("z_alpha", {}))

    ax_y.plot(y_a, y_v, color="#2563eb", linewidth=1.8, label="Experimental")
    ax_z.plot(z_a, z_v, color="#2563eb", linewidth=1.8, label="Experimental")

    if model_name != "unknown":
        yt_a, yt_v = _sorted_xy(kth_full_result.get("y_theory", {}))
        zt_a, zt_v = _sorted_xy(kth_full_result.get("z_theory", {}))
        ax_y.plot(
            yt_a, yt_v, color="#d62728", linewidth=1.4, linestyle="--",
            label=f"Theoretical ({model_name})",
        )
        ax_z.plot(
            zt_a, zt_v, color="#d62728", linewidth=1.4, linestyle="--",
            label=f"Theoretical ({model_name})",
        )
        fig.suptitle(f"KTH Model Identification: {model_name}", fontsize=12)
    else:
        fig.suptitle("KTH Model Identification: unknown (ambiguous match)", fontsize=12)

    ax_y.legend(loc="best", fontsize=8, framealpha=0.9)
    ax_z.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    apply_figure_theme(fig)
    return fig


def build_coats_redfern_figure() -> Figure:
    """
    Create a fresh, empty figure for the Coats-Redfern R^2 comparison
    bar chart, ready to be populated by
    :func:`update_coats_redfern_plot`.
    """
    _apply_clean_style()
    fig = Figure(figsize=(8.0, 4.5), dpi=100)
    ax = fig.add_subplot(111)
    _configure_coats_redfern_axes(ax)
    fig.tight_layout()
    apply_figure_theme(fig)
    return fig


def _configure_coats_redfern_axes(ax) -> None:
    ax.set_xlabel("Candidate model")
    ax.set_ylabel("R\u00b2")
    ax.set_title("Coats-Redfern: R\u00b2 by candidate model")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(0.0, 1.05)


def update_coats_redfern_plot(fig: Figure, cr_result: Optional[Dict[str, object]]) -> Figure:
    """
    Clear and redraw the Coats-Redfern R^2 comparison bar chart, one
    bar per candidate model, colored green if that model passed the
    reference-Ea screen (see
    :func:`isokinetics.kinetics.coats_redfern`) or red if it did not
    (even if its R^2 alone looks good -- this is the point: a red bar
    with a high R^2 is exactly the "kinetic compensation effect" false
    positive the screening step exists to catch).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure previously created with :func:`build_coats_redfern_figure`.
    cr_result : dict or None
        The dict returned by :func:`isokinetics.kinetics.coats_redfern`,
        or ``None`` if it hasn't been run yet.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure, redrawn in place.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    _configure_coats_redfern_axes(ax)

    if not cr_result or not cr_result.get("results"):
        _placeholder(ax, "Run an analysis to see\nCoats-Redfern model comparison.")
        fig.tight_layout()
        apply_figure_theme(fig)
        return fig

    results = cr_result["results"]
    recommended = cr_result.get("recommended_model")
    reference_ea = cr_result.get("reference_ea_kj", float("nan"))

    # Rank by R^2 descending purely for a readable bar order; color
    # (not position) is what signals pass/fail.
    names = sorted(results.keys(), key=lambda n: results[n].get("r2", 0) or 0, reverse=True)
    r2_values = [results[n].get("r2", float("nan")) for n in names]

    passed_mask = []
    for n, fit in zip(names, [results[n] for n in names]):
        gap_pct = fit.get("ea_gap_pct", float("nan"))
        passed = np.isfinite(fit.get("r2", float("nan"))) and fit["r2"] > 0.99 and \
            np.isfinite(gap_pct) and gap_pct < 0.20
        passed_mask.append(passed)

    colors = ["#2ca02c" if p else "#d62728" for p in passed_mask]  # pass / reject
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, [v if v == v else 0.0 for v in r2_values], color=colors)

    if recommended is not None and recommended in names:
        idx = names.index(recommended)
        bars[idx].set_edgecolor(THEMES[_active_theme]["text"])
        bars[idx].set_linewidth(2.0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)

    title = f"Coats-Redfern: R\u00b2 by candidate model  (reference E\u2090 = {fmt_ea(reference_ea, unit=True)})"
    if recommended:
        title += f"  \u2192 recommended: {recommended}"
    else:
        title += "  \u2192 no model confidently recommended"
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    apply_figure_theme(fig)
    return fig
