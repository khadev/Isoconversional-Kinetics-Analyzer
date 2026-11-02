"""
gui.py
======
Desktop GUI for the isoconversional kinetics toolkit (customtkinter).

Layout
------
* **Header**   -- title, light/dark theme switch, About button.
* **Sidebar**  -- the workflow top to bottom: 1 Data, 2 Heating rates,
  3 Methods, 4 Display options; *Run Analysis*, *Export* and *Save Image*
  stay pinned at the bottom so they are always visible.
* **Tabs**     -- Data Preview | Summary | Kinetic Plot | Results Table |
  Model Fitting | Lifetime | KTH Model | Validation.
* **Status bar** -- what is happening now, a determinate progress bar,
  the percentage, elapsed time and an estimate of the time remaining.

All heavy computation runs in a background thread so the window never
freezes; progress and results are marshalled back to the Tk thread with
``after()``.  Every number shown follows :mod:`isokinetics.formatting`.

Keyboard shortcuts: Ctrl+O load data, Ctrl+R / F5 run, Ctrl+E export,
Ctrl+T toggle theme.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import customtkinter as ctk
import numpy as np
import pandas as pd

try:
    import darkdetect
except Exception:  # pragma: no cover - darkdetect is optional at runtime
    darkdetect = None

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from . import __version__ as isokinetics_version
from . import data_loader, exporter, kinetics, lifetime, plotter, validation
from ._tk_utils import cancel_all_pending_after_callbacks
from .formatting import (
    EMPTY,
    UNIT_EA_PLAIN,
    alpha_range_text,
    fmt_alpha,
    fmt_beta,
    fmt_days,
    fmt_duration,
    fmt_ea,
    fmt_minutes,
    fmt_pct,
    fmt_r2,
    fmt_sci,
    fmt_score,
    fmt_temp,
    fmt_years,
)

logger = logging.getLogger(__name__)

APP_TITLE = "Isoconversional Kinetics Analyzer"
DEFAULT_BETAS = [5.0, 10.0, 20.0, 40.0]
METHOD_NAMES = ["Friedman", "FWO", "KAS", "Starink", "Vyazovkin", "KTH"]

#: One line per method, shown as a tooltip-style hint under the checkboxes.
METHOD_HINTS = {
    "Friedman": "differential \u2013 exact for variable Ea, noise-sensitive",
    "FWO": "integral \u2013 Doyle approximation (\u00b13 % method bias)",
    "KAS": "integral \u2013 robust, mild bias if Ea varies",
    "Starink": "integral \u2013 most accurate of the linear integral methods",
    "Vyazovkin": "non-linear, windowed \u2013 accurate for variable Ea",
    "KTH": "discretised differential \u2013 optional full triplet",
}

#: Relative cost of each stage, used only to make the progress bar advance
#: roughly in proportion to the real time spent.
_STAGE_WEIGHTS = {
    "Friedman": 1.0, "FWO": 1.0, "KAS": 1.0, "Starink": 1.0,
    "Vyazovkin": 5.0, "KTH": 1.0, "Coats-Redfern": 3.0,
}

#: The two KTH computation modes selectable in the GUI.
KTH_MODE_EA_ONLY = "E\u2090 only"
KTH_MODE_FULL_TRIPLET = "Full triplet (A, f(\u03b1), model)"

#: Content shown in the "About" dialog.
APP_NAME_FOR_ABOUT = "Isoconversional Kinetics Analyzer"
ABOUT_OBJECTIVE_TEXT = (
    "\u2022 Isoconversional kinetic analysis of thermal decomposition data\n"
    "\u2022 Methods: Friedman, FWO, KAS, Starink, Vyazovkin, KTH, Coats-Redfern\n"
    "\u2022 Full kinetic triplet (E\u2090, A, f(\u03b1)) via KTH + Malek's method\n"
    "\u2022 Arrhenius shelf-life / service-lifetime prediction\n"
    "\u2022 Built-in validation against synthetic data with a known E\u2090\n"
    "\u2022 Publication-ready plots and multi-format export"
)
DEVELOPER_CONTACT_EMAIL = "Oukil.khaled@gmail.com"

TAB_PREVIEW = "Data Preview"
TAB_SUMMARY = "Summary"
TAB_PLOT = "Kinetic Plot"
TAB_RESULTS = "Results Table"
TAB_MODEL_FITTING = "Model Fitting"
TAB_LIFETIME = "Lifetime"
TAB_KTH = "KTH Model"
TAB_VALIDATION = "Validation"


def _c(key: str):
    """(light, dark) colour pair for a palette key, as customtkinter wants."""
    return (plotter.THEMES["light"][key], plotter.THEMES["dark"][key])


def _initial_theme() -> str:
    """'dark' or 'light' following the operating system when detectable."""
    if darkdetect is not None:
        try:
            return "dark" if darkdetect.isDark() else "light"
        except Exception:  # pragma: no cover
            pass
    return "light"


def data_loader_fmt_step(step: float) -> str:
    """Text for the dialog entry: ``0.02`` (shortest faithful form)."""
    return f"{float(step):g}"


class AlphaStepDialog(ctk.CTkToplevel):
    """
    Modal dialog shown when a **raw** (4-column, temperatures-only) file is
    loaded: the user chooses the conversion step with which alpha is
    generated.  Accepts a step (``0.02``), a fraction (``1/50``) or a number
    of points (``50``) and shows a live preview of the resulting grid.

    After the dialog closes, ``result`` is ``(step, mapping)`` or ``None`` if
    it was cancelled.
    """

    PRESETS = ("0.01", "0.02", "0.05")

    def __init__(
        self,
        master,
        table: "data_loader.LoadedTable",
        file_name: str = "",
        initial: str = "0.01",
        mapping: str = data_loader.DEFAULT_ROW_MAPPING,
    ) -> None:
        super().__init__(master)
        self.table = table
        self.result: Optional["tuple[float, str]"] = None
        self.title("Raw data detected")
        self.resizable(False, False)
        self.transient(master)
        self._labels_to_keys = {label: key for key, label in data_loader.ROW_MAPPINGS.items()}

        rows = table.rows_per_column
        rows_text = (f"{rows[0]} rows" if len(set(rows)) == 1
                     else "rows per column: " + ", ".join(str(r) for r in rows))
        pad = {"padx": 22, "anchor": "w"}

        ctk.CTkLabel(self, text="Raw data file detected", font=ctk.CTkFont(size=17, weight="bold"),
                     text_color=_c("text")).pack(pady=(18, 2), **pad)
        self.summary_label = ctk.CTkLabel(
            self, justify="left", anchor="w", wraplength=480, text_color=_c("text"),
            text=f"{file_name}\n4 columns (T\u2081 \u2026 T\u2084), {rows_text}. "
                 "There is no conversion (\u03b1) column, so \u03b1 will be generated and the "
                 "temperatures interpolated at each \u03b1.")
        self.summary_label.pack(pady=(0, 8), **pad)

        ctk.CTkLabel(self, text="Assumption: each column is a curve whose rows are equally spaced "
                                "in conversion. Where do the rows sit on the \u03b1 axis?",
                     justify="left", anchor="w", wraplength=450, text_color=_c("muted"),
                     font=self.master.font_small if hasattr(self.master, "font_small") else None
                     ).pack(pady=(0, 4), **pad)
        self.mapping_var = ctk.StringVar(value=data_loader.ROW_MAPPINGS[mapping])
        self.mapping_menu = ctk.CTkOptionMenu(
            self, values=list(data_loader.ROW_MAPPINGS.values()), variable=self.mapping_var,
            width=476, height=28, dynamic_resizing=False, command=lambda _v: self._update_preview())
        self.mapping_menu.pack(**pad)

        ctk.CTkLabel(self, text="Conversion step \u0394\u03b1  \u2014  or number of points",
                     font=ctk.CTkFont(size=12, weight="bold"), text_color=_c("text")).pack(pady=(14, 0), **pad)
        ctk.CTkLabel(self, text="Examples:  0.01   0.02   1/50   100   (a whole number N means a step of 1/N)",
                     text_color=_c("muted"), anchor="w").pack(pady=(0, 4), **pad)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", **{"padx": 22})
        self.step_entry = ctk.CTkEntry(row, width=110, height=30)
        self.step_entry.insert(0, initial)
        self.step_entry.pack(side="left")
        self.step_entry.bind("<KeyRelease>", lambda _e: self._update_preview())
        for preset in self.PRESETS:
            ctk.CTkButton(row, text=preset, width=52, height=28, fg_color=_c("card_alt"),
                          text_color=_c("text"), hover_color=_c("select"),
                          command=lambda p=preset: self._use_preset(p)).pack(side="left", padx=(8, 0))

        self.preview_label = ctk.CTkLabel(self, text="", anchor="w", justify="left", wraplength=480,
                                          text_color=_c("accent"), font=ctk.CTkFont(size=13, weight="bold"))
        self.preview_label.pack(pady=(12, 0), **pad)
        self.note_label = ctk.CTkLabel(self, text="", anchor="w", justify="left", wraplength=480,
                                       text_color=_c("warn"))
        self.note_label.pack(pady=(2, 0), **pad)
        self.error_label = ctk.CTkLabel(self, text="", anchor="w", justify="left", wraplength=480,
                                        text_color=_c("bad"))
        self.error_label.pack(pady=(2, 0), **pad)
        ctk.CTkLabel(self, text="\u03b1 = 0 and \u03b1 = 1 are never included (E\u2090 is undefined there).",
                     text_color=_c("muted"), anchor="w").pack(pady=(2, 8), **pad)

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=22, pady=(0, 18))
        self.ok_button = ctk.CTkButton(buttons, text="Generate \u03b1 and continue", width=200, height=34,
                                       command=self._on_ok, fg_color=_c("accent"),
                                       hover_color=_c("accent_hover"))
        self.ok_button.pack(side="right")
        ctk.CTkButton(buttons, text="Cancel", width=90, height=34, command=self._on_cancel,
                      fg_color=_c("card_alt"), text_color=_c("text"), hover_color=_c("select"),
                      border_width=1, border_color=_c("border")).pack(side="right", padx=(0, 8))

        self.bind("<Return>", lambda _e: self._on_ok())
        self.bind("<Escape>", lambda _e: self._on_cancel())
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._update_preview()
        self._place_over(master)
        self.after(80, self._take_focus)

    # -- helpers -------------------------------------------------------------
    def _place_over(self, master) -> None:
        self.update_idletasks()
        w, h = 520, self.winfo_reqheight()
        try:
            x = master.winfo_rootx() + max(0, (master.winfo_width() - w) // 2)
            y = master.winfo_rooty() + max(0, (master.winfo_height() - h) // 3)
        except Exception:  # noqa: BLE001
            x, y = 200, 120
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _take_focus(self) -> None:
        try:
            self.grab_set()
            self.step_entry.focus_set()
        except Exception:  # noqa: BLE001 - window not viewable yet / already closed
            pass

    def _mapping_key(self) -> str:
        return self._labels_to_keys.get(self.mapping_var.get(), data_loader.DEFAULT_ROW_MAPPING)

    def _use_preset(self, text: str) -> None:
        self.step_entry.delete(0, "end")
        self.step_entry.insert(0, text)
        self._update_preview()

    def _update_preview(self) -> None:
        self.error_label.configure(text="")
        self.note_label.configure(text="")
        try:
            step = data_loader.parse_alpha_step(self.step_entry.get())
            table = data_loader.build_alpha_table(self.table.data, step, self._mapping_key())
        except ValueError as exc:
            self.preview_label.configure(text="")
            self.error_label.configure(text=str(exc))
            self.ok_button.configure(state="disabled")
            return
        grid = table["alpha"]
        self.preview_label.configure(
            text=f"\u2192 {len(table)} points,  \u03b1 = {grid.iloc[0]:.3f} \u2026 {grid.iloc[-1]:.3f}  "
                 f"(\u0394\u03b1 = {step:g})")
        note = data_loader.resolution_note(self.table.data, step, self._mapping_key())
        self.note_label.configure(text=("\u26a0 " + note) if note else "")
        self.ok_button.configure(state="normal")

    def _on_ok(self) -> None:
        try:
            step = data_loader.parse_alpha_step(self.step_entry.get())
            data_loader.build_alpha_table(self.table.data, step, self._mapping_key())
        except ValueError as exc:
            self.error_label.configure(text=str(exc))
            return
        self.result = (step, self._mapping_key())
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()


class KineticsApp(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        # --- appearance -----------------------------------------------------
        self._theme_name = _initial_theme()
        ctk.set_appearance_mode(self._theme_name)
        ctk.set_default_color_theme("blue")
        plotter.set_theme(self._theme_name)

        self.title(APP_TITLE)
        self.geometry("1320x850")
        self.minsize(1100, 700)
        self.configure(fg_color=_c("window"))

        self.font_title = ctk.CTkFont(size=20, weight="bold")
        self.font_h = ctk.CTkFont(size=12, weight="bold")
        self.font_small = ctk.CTkFont(size=11)
        self.font_big = ctk.CTkFont(size=26, weight="bold")

        # --- state ------------------------------------------------------------
        self.source_kind = "processed"  # "processed" (5 columns) or "raw" (4 columns -> alpha generated)
        self._raw_loaded: Optional[data_loader.LoadedTable] = None
        self.alpha_step: Optional[float] = None  # step used for the current raw file
        self.row_mapping = data_loader.DEFAULT_ROW_MAPPING
        self._last_alpha_step = 0.01  # remembered between files
        self._last_row_mapping = data_loader.DEFAULT_ROW_MAPPING
        self.raw_df: Optional[pd.DataFrame] = None
        self.working_df: Optional[pd.DataFrame] = None
        self.results: Dict[str, Dict[float, float]] = {}
        self.averages: Dict[str, float] = {}
        self.file_path: Optional[str] = None
        self.is_running = False  # True while an analysis computation is in progress
        self.kth_full_result: Optional[Dict[str, object]] = None  # only in "Full triplet" mode
        self.cr_result: Optional[Dict[str, object]] = None  # Coats-Redfern results
        self.validation_report: Optional[validation.ValidationReport] = None
        self.lifetime_result: Optional[lifetime.LifetimeResult] = None
        self._lifetime_sources: Dict[str, lifetime.EaSource] = {}

        # progress bookkeeping (see _begin_stage / _tick)
        self._run_started: Optional[float] = None
        self._progress_fraction = 0.0
        self._stage_text = ""
        self._tick_job: Optional[str] = None

        # --- shutdown / lifecycle state -----------------------------------
        # `_alive` is False as soon as the window starts closing. Any
        # background-thread callback marshalled in via `self.after(...)`
        # checks this flag before touching a widget. `_stop_event` is the
        # signal *to* the background threads that they should stop.
        self._alive = True
        self._stop_event = threading.Event()
        self._analysis_thread: Optional[threading.Thread] = None
        self._validation_thread: Optional[threading.Thread] = None

        self._trees: List[ttk.Treeview] = []

        self._build_header()
        self._build_body()
        self._build_statusbar()
        self._build_about_button()
        self._style_treeviews()
        self._bind_shortcuts()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    # Small widget factories
    # ------------------------------------------------------------------ #
    def _card(self, parent, title: str, step: Optional[str] = None):
        """A rounded 'card' with a small title; returns the body frame."""
        card = ctk.CTkFrame(
            parent, corner_radius=12, fg_color=_c("card"),
            border_width=1, border_color=_c("border"),
        )
        card.pack(fill="x", padx=2, pady=(0, 10))
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(10, 2))
        if step:
            ctk.CTkLabel(
                head, text=step, width=22, height=22, corner_radius=11,
                fg_color=_c("accent"), text_color="white", font=self.font_small,
            ).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(head, text=title, font=self.font_h, text_color=_c("text")).pack(side="left")
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=14, pady=(2, 12))
        return body

    def _muted(self, parent, text: str = "", **kw) -> ctk.CTkLabel:
        return ctk.CTkLabel(
            parent, text=text, text_color=_c("muted"), font=self.font_small, **kw
        )

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color=_c("card"))
        header.pack(side="top", fill="x")
        header.pack_propagate(False)

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left", padx=20)
        ctk.CTkLabel(title_box, text=APP_TITLE, font=self.font_title,
                     text_color=_c("text")).pack(anchor="w")
        self._muted(title_box, f"Model-free kinetics of thermal decomposition  \u00b7  v{isokinetics_version}").pack(anchor="w")

        # Right padding leaves room for the About button, which is placed
        # directly on the root window (see _build_about_button).
        self.dark_mode_var = ctk.BooleanVar(value=self._theme_name == "dark")
        self.theme_switch = ctk.CTkSwitch(
            header, text="Dark mode", variable=self.dark_mode_var,
            command=self._on_theme_toggled, text_color=_c("text"),
        )
        self.theme_switch.pack(side="right", padx=(0, 110))

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(side="top", fill="both", expand=True, padx=12, pady=(10, 4))

        sidebar = ctk.CTkFrame(body, width=340, fg_color="transparent")
        sidebar.pack(side="left", fill="y", padx=(0, 10))
        sidebar.pack_propagate(False)

        self._sidebar_scroll = ctk.CTkScrollableFrame(sidebar, fg_color="transparent")
        self._sidebar_scroll.pack(side="top", fill="both", expand=True)
        self._build_data_card(self._sidebar_scroll)
        self._build_rates_card(self._sidebar_scroll)
        self._build_methods_card(self._sidebar_scroll)
        self._build_display_card(self._sidebar_scroll)
        self._build_action_bar(sidebar)

        main = ctk.CTkFrame(body, fg_color=_c("card"), corner_radius=12,
                            border_width=1, border_color=_c("border"))
        main.pack(side="left", fill="both", expand=True)
        self._build_tabview(main)

    # -- sidebar cards ------------------------------------------------------
    def _build_data_card(self, parent) -> None:
        body = self._card(parent, "Data", "1")
        self.load_button = ctk.CTkButton(
            body, text="Load Data\u2026", height=34, command=self._on_load_data,
            fg_color=_c("accent"), hover_color=_c("accent_hover"),
        )
        self.load_button.pack(fill="x")

        self.sample_path = self._find_sample_file()
        self.sample_button = None
        if self.sample_path:
            self.sample_button = ctk.CTkButton(
                body, text="Use bundled sample", height=26, fg_color="transparent",
                border_width=1, border_color=_c("border"), text_color=_c("text"),
                hover_color=_c("card_alt"), command=self._on_load_sample,
            )
            self.sample_button.pack(fill="x", pady=(6, 0))

        self.file_label = ctk.CTkLabel(
            body, text="No file loaded.", anchor="w", justify="left",
            wraplength=290, text_color=_c("text"),
        )
        self.file_label.pack(fill="x", pady=(8, 0))
        self.data_info_label = self._muted(
            body, "5 columns (\u03b1, T\u2081\u2026T\u2084) = processed data\n"
                  "4 columns (T\u2081\u2026T\u2084) = raw data: \u03b1 is generated for you",
            anchor="w", justify="left", wraplength=290)
        self.data_info_label.pack(fill="x")

        # Raw-data actions (shown only while a raw 4-column file is loaded).
        self.raw_actions_frame = ctk.CTkFrame(body, fg_color="transparent")
        self.reprocess_button = ctk.CTkButton(
            self.raw_actions_frame, text="Change \u03b1 step\u2026", height=26,
            command=self._on_change_alpha_step, fg_color=_c("card_alt"), text_color=_c("text"),
            hover_color=_c("select"), border_width=1, border_color=_c("border"))
        self.reprocess_button.pack(fill="x")
        self.save_processed_button = ctk.CTkButton(
            self.raw_actions_frame, text="Save processed table\u2026", height=26,
            command=self._on_save_processed, fg_color=_c("card_alt"), text_color=_c("text"),
            hover_color=_c("select"), border_width=1, border_color=_c("border"))
        self.save_processed_button.pack(fill="x", pady=(4, 0))

    def _build_rates_card(self, parent) -> None:
        body = self._card(parent, "Heating rates \u03b2 (K/min)", "2")
        grid = ctk.CTkFrame(body, fg_color="transparent")
        grid.pack(fill="x")
        self.beta_entries: List[ctk.CTkEntry] = []
        subs = "\u2081\u2082\u2083\u2084"
        for i, default in enumerate(DEFAULT_BETAS):
            r, c = divmod(i, 2)
            cell = ctk.CTkFrame(grid, fg_color="transparent")
            cell.grid(row=r, column=c, padx=(0, 8), pady=3, sticky="w")
            ctk.CTkLabel(cell, text=f"\u03b2{subs[i]}", width=24, text_color=_c("text")).pack(side="left")
            entry = ctk.CTkEntry(cell, width=78, height=30, placeholder_text=f"\u03b2{i + 1}")
            entry.insert(0, fmt_beta(default))
            entry.pack(side="left")
            entry.bind("<KeyRelease>", lambda _e, en=entry: self._validate_beta_entry(en))
            self.beta_entries.append(entry)
        self._muted(body, "One per T column, in the same order as the data file.",
                    anchor="w", wraplength=290, justify="left").pack(fill="x", pady=(4, 0))

    def _build_methods_card(self, parent) -> None:
        body = self._card(parent, "Methods", "3")
        quick = ctk.CTkFrame(body, fg_color="transparent")
        quick.pack(fill="x", pady=(0, 4))
        for text, value in (("All", True), ("None", False)):
            ctk.CTkButton(
                quick, text=text, width=52, height=24, fg_color=_c("card_alt"),
                text_color=_c("text"), hover_color=_c("select"),
                command=lambda v=value: self._set_all_methods(v),
            ).pack(side="left", padx=(0, 6))

        self.method_vars: Dict[str, ctk.BooleanVar] = {}
        for name in METHOD_NAMES:
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", pady=1)
            var = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(
                row, text=name, variable=var, text_color=_c("text"),
                checkbox_width=18, checkbox_height=18,
                fg_color=plotter.METHOD_COLORS.get(name, "#2563eb"),
            ).pack(anchor="w")
            self._muted(row, METHOD_HINTS[name], anchor="w", wraplength=280,
                        justify="left").pack(fill="x", padx=(26, 0))
            self.method_vars[name] = var

        ctk.CTkLabel(body, text="KTH output", font=self.font_small,
                     text_color=_c("muted")).pack(anchor="w", pady=(8, 0))
        self.kth_mode_var = ctk.StringVar(value=KTH_MODE_EA_ONLY)
        self.kth_mode_menu = ctk.CTkOptionMenu(
            body, values=[KTH_MODE_EA_ONLY, KTH_MODE_FULL_TRIPLET],
            variable=self.kth_mode_var, height=28, dynamic_resizing=False,
        )
        self.kth_mode_menu.pack(fill="x")

    def _build_display_card(self, parent) -> None:
        body = self._card(parent, "Display", "4")
        ctk.CTkLabel(body, text="Plot style", font=self.font_small,
                     text_color=_c("muted")).pack(anchor="w")
        self.plot_style_var = ctk.StringVar(value=plotter.DEFAULT_STYLE)
        self.plot_style_menu = ctk.CTkOptionMenu(
            body, values=list(plotter.PLOT_STYLES.keys()), variable=self.plot_style_var,
            height=28, dynamic_resizing=False, command=self._on_plot_style_changed,
        )
        self.plot_style_menu.pack(fill="x", pady=(0, 8))
        self.show_ci_var = ctk.BooleanVar(value=False)
        self.show_ci_checkbox = ctk.CTkCheckBox(
            body, text="95 % confidence bands\n(FWO / KAS / Starink / Friedman)",
            variable=self.show_ci_var, command=self._on_show_ci_toggled,
            text_color=_c("text"), checkbox_width=18, checkbox_height=18,
        )
        self.show_ci_checkbox.pack(anchor="w")

    def _build_action_bar(self, parent) -> None:
        bar = ctk.CTkFrame(parent, fg_color=_c("card"), corner_radius=12,
                           border_width=1, border_color=_c("border"))
        bar.pack(side="bottom", fill="x", pady=(8, 0))
        self.run_button = ctk.CTkButton(
            bar, text="\u25b6  Run Analysis", height=42, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=_c("accent"), hover_color=_c("accent_hover"),
            command=self._on_run_analysis,
        )
        self.run_button.pack(fill="x", padx=12, pady=(12, 8))
        row = ctk.CTkFrame(bar, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))
        row.grid_columnconfigure((0, 1), weight=1)
        self.export_button = ctk.CTkButton(
            row, text="Export", height=32, command=self._on_export, state="disabled",
            fg_color=_c("card_alt"), text_color=_c("text"), hover_color=_c("select"),
            border_width=1, border_color=_c("border"),
        )
        self.export_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.save_image_button = ctk.CTkButton(
            row, text="Save Image...", height=32, command=self._on_save_image,
            fg_color=_c("card_alt"), text_color=_c("text"), hover_color=_c("select"),
            border_width=1, border_color=_c("border"),
        )
        self.save_image_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    # -- tabs -----------------------------------------------------------------
    def _build_tabview(self, parent) -> None:
        self.tabview = ctk.CTkTabview(parent, fg_color="transparent")
        self.tabview.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        self.tab_preview = self.tabview.add(TAB_PREVIEW)
        self.tab_summary = self.tabview.add(TAB_SUMMARY)
        self.tab_plot = self.tabview.add(TAB_PLOT)
        self.tab_results = self.tabview.add(TAB_RESULTS)
        self.tab_model_fitting = self.tabview.add(TAB_MODEL_FITTING)
        self.tab_lifetime = self.tabview.add(TAB_LIFETIME)
        self.tab_kth = self.tabview.add(TAB_KTH)
        self.tab_validation = self.tabview.add(TAB_VALIDATION)

        self._build_preview_tab()
        self._build_summary_tab()
        self._build_plot_tab()
        self._build_results_tab()
        self._build_model_fitting_tab()
        self._build_lifetime_tab()
        self._build_kth_tab()
        self._build_validation_tab()

    def _make_tree(self, parent, columns, headings, widths, height=14, anchor="center"):
        """A themed Treeview + vertical scrollbar packed into ``parent``."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height,
                            style="Iso.Treeview")
        for col, head, width in zip(columns, headings, widths):
            tree.heading(col, text=head)
            tree.column(col, width=width, anchor=anchor, stretch=True)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview,
                            style="Iso.Vertical.TScrollbar")
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._trees.append(tree)
        return tree

    def _build_preview_tab(self) -> None:
        self.preview_info_label = ctk.CTkLabel(
            self.tab_preview, text="Load a data file to preview it here.",
            anchor="w", text_color=_c("muted"))
        self.preview_info_label.pack(fill="x", padx=6, pady=(4, 6))
        columns = ["alpha", "T1", "T2", "T3", "T4"]
        headings = ["\u03b1", "T\u2081 (K)", "T\u2082 (K)", "T\u2083 (K)", "T\u2084 (K)"]
        self.preview_tree = self._make_tree(self.tab_preview, columns, headings, [110] * 5, height=20)

    def _build_summary_tab(self) -> None:
        self.summary_body = ctk.CTkScrollableFrame(self.tab_summary, fg_color="transparent")
        self.summary_body.pack(fill="both", expand=True)
        self._populate_summary({}, {})

    def _build_plot_tab(self) -> None:
        self.figure = plotter.build_figure()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.tab_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.draw()

    def _build_results_tab(self) -> None:
        top = ctk.CTkFrame(self.tab_results, fg_color="transparent")
        top.pack(fill="x", padx=4, pady=(2, 6))
        ctk.CTkLabel(
            top, text=f"E\u2090 ({UNIT_EA_PLAIN}) per method \u00b7 KTH A in s\u207b\u00b9",
            anchor="w", text_color=_c("muted"),
        ).pack(side="left")
        self.copy_table_button = ctk.CTkButton(
            top, text="Copy table", width=100, height=26, command=self._on_copy_table,
            fg_color=_c("card_alt"), text_color=_c("text"), hover_color=_c("select"),
            border_width=1, border_color=_c("border"),
        )
        self.copy_table_button.pack(side="right")

        columns = ["alpha"] + METHOD_NAMES + ["KTH_A", "KTH_f(\u03b1)"]
        headings = ["\u03b1"] + METHOD_NAMES + ["KTH A", "KTH f(\u03b1)"]
        self.results_tree = self._make_tree(self.tab_results, columns, headings, [90] + [95] * 6 + [105, 150], height=20)

    def _build_kth_tab(self) -> None:
        self.kth_figure = plotter.build_kth_model_figure()
        self.kth_canvas = FigureCanvasTkAgg(self.kth_figure, master=self.tab_kth)
        self.kth_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self.kth_canvas.draw()

    def _build_model_fitting_tab(self) -> None:
        self.cr_summary_label = ctk.CTkLabel(
            self.tab_model_fitting, text="Run an analysis to see Coats-Redfern model fitting.",
            anchor="w", justify="left", wraplength=820, text_color=_c("text"),
        )
        self.cr_summary_label.pack(side="top", fill="x", padx=6, pady=(4, 6))

        table_holder = ctk.CTkFrame(self.tab_model_fitting, fg_color="transparent", height=210)
        table_holder.pack(side="top", fill="x")
        table_holder.pack_propagate(False)
        columns = ["Model", "Ea (kJ/mol)", "A (s\u207b\u00b9)", "R\u00b2", "Status"]
        self.cr_tree = self._make_tree(table_holder, columns, columns, [90, 120, 130, 100, 230], height=8)

        self.cr_figure = plotter.build_coats_redfern_figure()
        self.cr_canvas = FigureCanvasTkAgg(self.cr_figure, master=self.tab_model_fitting)
        self.cr_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self.cr_canvas.draw()

    def _build_lifetime_tab(self) -> None:
        """Arrhenius shelf-life extrapolation (see :mod:`isokinetics.lifetime`)."""
        body = ctk.CTkScrollableFrame(self.tab_lifetime, fg_color="transparent")
        body.pack(fill="both", expand=True)
        self.lifetime_body = body

        ctk.CTkLabel(
            body, anchor="w", justify="left", wraplength=820, text_color=_c("muted"),
            text="Arrhenius shelf-life extrapolation:\n"
                 "t(service) = t(storage) \u00d7 exp[ E\u2090/R \u00d7 ( 1/T(service) \u2212 1/T(storage) ) ]\n"
                 "t(storage) is the time to end of life at the (hotter) storage/ageing temperature; "
                 "the result is the time to end of life at the service temperature.",
        ).pack(fill="x", padx=6, pady=(2, 8))

        columns = ctk.CTkFrame(body, fg_color="transparent")
        columns.pack(fill="x")
        columns.grid_columnconfigure(0, weight=0)
        columns.grid_columnconfigure(1, weight=1)

        # ---- inputs ------------------------------------------------------
        inputs = ctk.CTkFrame(columns, corner_radius=12, fg_color=_c("card_alt"),
                              border_width=1, border_color=_c("border"), width=330)
        inputs.grid(row=0, column=0, sticky="n", padx=(4, 8), pady=4)
        pad = {"padx": 14, "anchor": "w"}

        ctk.CTkLabel(inputs, text="Activation energy source", font=self.font_h,
                     text_color=_c("text")).pack(pady=(12, 2), **pad)
        self.lifetime_source_var = ctk.StringVar(value=lifetime.MANUAL_SOURCE)
        self.lifetime_source_menu = ctk.CTkOptionMenu(
            inputs, values=[lifetime.MANUAL_SOURCE], variable=self.lifetime_source_var,
            width=300, height=28, dynamic_resizing=False,
            command=self._on_lifetime_source_changed,
        )
        self.lifetime_source_menu.pack(padx=14, anchor="w")
        self.lifetime_ea_label = ctk.CTkLabel(
            inputs, text="", font=self.font_h, text_color=_c("accent"), anchor="w")
        self.lifetime_ea_label.pack(pady=(6, 0), **pad)
        self.lifetime_ea_entry = ctk.CTkEntry(inputs, width=120, height=28,
                                              placeholder_text="kJ/mol")
        self.lifetime_ea_entry.insert(0, "100")
        self.lifetime_ea_entry.bind("<Return>", lambda _e: self._on_calculate_lifetime())
        self.lifetime_source_note = self._muted(inputs, "", anchor="w", justify="left", wraplength=300)
        self.lifetime_source_note.pack(pady=(2, 0), **pad)

        ctk.CTkLabel(inputs, text="Temperatures", font=self.font_h,
                     text_color=_c("text")).pack(pady=(12, 2), **pad)
        self.lifetime_temp_unit_var = ctk.StringVar(value="\u00b0C")
        self.lifetime_temp_unit_button = ctk.CTkSegmentedButton(
            inputs, values=["\u00b0C", "K"], variable=self.lifetime_temp_unit_var, width=110,
            command=self._on_lifetime_temp_unit_changed)
        self.lifetime_temp_unit_button.pack(**pad)

        def labelled_entry(parent, text, default, attr):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=(6, 0))
            ctk.CTkLabel(row, text=text, width=150, anchor="w", text_color=_c("text")).pack(side="left")
            entry = ctk.CTkEntry(row, width=90, height=28)
            entry.insert(0, default)
            entry.pack(side="left")
            entry.bind("<Return>", lambda _e: self._on_calculate_lifetime())
            setattr(self, attr, entry)

        labelled_entry(inputs, "Storage temperature", "60", "lifetime_storage_temp_entry")
        labelled_entry(inputs, "Service temperature", "25", "lifetime_service_temp_entry")

        ctk.CTkLabel(inputs, text="Time to end of life at storage temp.", font=self.font_h,
                     text_color=_c("text")).pack(pady=(12, 0), **pad)
        self._muted(inputs, "How long the material lasts (until your failure criterion) when kept "
                            "at the storage temperature, e.g. from an accelerated-ageing test.",
                    anchor="w", justify="left", wraplength=300).pack(pady=(0, 4), **pad)
        trow = ctk.CTkFrame(inputs, fg_color="transparent")
        trow.pack(fill="x", padx=14)
        self.lifetime_storage_time_entry = ctk.CTkEntry(trow, width=90, height=28)
        self.lifetime_storage_time_entry.insert(0, "90")
        self.lifetime_storage_time_entry.pack(side="left")
        self.lifetime_storage_time_entry.bind("<Return>", lambda _e: self._on_calculate_lifetime())
        self.lifetime_time_unit_var = ctk.StringVar(value="days")
        self.lifetime_time_unit_menu = ctk.CTkOptionMenu(
            trow, values=list(lifetime.TIME_UNITS_IN_DAYS), variable=self.lifetime_time_unit_var,
            width=100, height=28, dynamic_resizing=False)
        self.lifetime_time_unit_menu.pack(side="left", padx=(8, 0))

        self.lifetime_calc_button = ctk.CTkButton(
            inputs, text="Calculate lifetime", height=36, command=self._on_calculate_lifetime,
            fg_color=_c("accent"), hover_color=_c("accent_hover"))
        self.lifetime_calc_button.pack(fill="x", padx=14, pady=(16, 6))
        self.lifetime_error_label = ctk.CTkLabel(
            inputs, text="", text_color=_c("bad"), anchor="w", justify="left", wraplength=300)
        self.lifetime_error_label.pack(padx=14, pady=(0, 12), anchor="w")

        # ---- results -------------------------------------------------------
        results = ctk.CTkFrame(columns, fg_color="transparent")
        results.grid(row=0, column=1, sticky="nsew", pady=4)
        ctk.CTkLabel(results, text="Predicted service lifetime", font=self.font_h,
                     text_color=_c("text")).pack(anchor="w", padx=4)

        cards = ctk.CTkFrame(results, fg_color="transparent")
        cards.pack(fill="x", pady=(4, 6))
        for col in range(3):
            cards.grid_columnconfigure(col, weight=1, uniform="lt")
        self.lifetime_value_labels: Dict[str, ctk.CTkLabel] = {}
        for col, key in enumerate(("years", "days", "minutes")):
            card = ctk.CTkFrame(cards, corner_radius=12, fg_color=_c("card_alt"),
                                border_width=1, border_color=_c("border"))
            card.grid(row=0, column=col, padx=4, sticky="nsew")
            value = ctk.CTkLabel(card, text=EMPTY, font=ctk.CTkFont(size=22, weight="bold"),
                                 text_color=_c("text"))
            value.pack(padx=12, pady=(14, 0))
            self._muted(card, key).pack(pady=(0, 12))
            self.lifetime_value_labels[key] = value
        self.lifetime_years_label = self.lifetime_value_labels["years"]
        self.lifetime_days_label = self.lifetime_value_labels["days"]
        self.lifetime_minutes_label = self.lifetime_value_labels["minutes"]

        self.lifetime_detail_label = ctk.CTkLabel(
            results, text="Enter the temperatures and press Calculate lifetime.",
            anchor="w", justify="left", wraplength=470, text_color=_c("text"))
        self.lifetime_detail_label.pack(fill="x", padx=6, pady=(4, 2))
        self.lifetime_notes_label = ctk.CTkLabel(
            results, text="", anchor="w", justify="left", wraplength=470, text_color=_c("warn"))
        self.lifetime_notes_label.pack(fill="x", padx=6, pady=(6, 0))

        self._refresh_lifetime_sources()

    def _build_validation_tab(self) -> None:
        top = ctk.CTkFrame(self.tab_validation, fg_color="transparent")
        top.pack(fill="x", padx=4, pady=(2, 6))
        self.validate_button = ctk.CTkButton(
            top, text="Run validation", width=130, height=30, command=self._on_run_validation,
            fg_color=_c("accent"), hover_color=_c("accent_hover"),
        )
        self.validate_button.pack(side="left")
        self.validation_status_label = ctk.CTkLabel(
            top, anchor="w", justify="left", wraplength=640, text_color=_c("muted"),
            text="Checks every method against synthetic data with a known E\u2090 "
                 "(constant and variable). Takes a couple of seconds.",
        )
        self.validation_status_label.pack(side="left", padx=12)

        columns = ["scenario", "method", "true", "found", "error", "tol", "result"]
        headings = ["Scenario", "Method", "True E\u2090", "Found", "Error", "Tolerance", "Result"]
        self.validation_tree = self._make_tree(
            self.tab_validation, columns, headings, [230, 110, 80, 80, 80, 90, 80], height=20)

    def _build_statusbar(self) -> None:
        bar = ctk.CTkFrame(self, height=44, corner_radius=0, fg_color=_c("card"))
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)

        self.status_label = ctk.CTkLabel(bar, text="Ready. Load a data file to begin.",
                                         anchor="w", text_color=_c("text"))
        self.status_label.pack(side="left", padx=(16, 8))

        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.pack(side="right", padx=(8, 16))
        self.eta_label = ctk.CTkLabel(right, text="", width=210, anchor="e", text_color=_c("muted"),
                                      font=self.font_small)
        self.eta_label.pack(side="right", padx=(8, 0))
        self.progress_label = ctk.CTkLabel(right, text="0 %", width=44, anchor="e", text_color=_c("text"))
        self.progress_label.pack(side="right", padx=(8, 0))
        self.progress_bar = ctk.CTkProgressBar(right, width=300, height=10,
                                               progress_color=_c("accent"))
        self.progress_bar.pack(side="right")
        self.progress_bar.set(0.0)

    def _build_about_button(self) -> None:
        """
        Place the "About" button in the top-right corner of the whole
        window with ``place()`` (independent of the header's pack layout)
        so it stays pinned there when the window is resized.  It uses a
        solid fill (a transparent button was reported as hard to see).
        """
        self.about_button = ctk.CTkButton(
            self, text="About", width=70, height=30, command=self._on_show_about,
            fg_color=_c("accent"), hover_color=_c("accent_hover"),
        )
        self.about_button.place(relx=1.0, rely=0.0, anchor="ne", x=-16, y=15)
        self.about_button.lift()

    def _bind_shortcuts(self) -> None:
        self.bind("<Control-o>", lambda _e: self._on_load_data())
        self.bind("<Control-r>", lambda _e: self._on_run_analysis())
        self.bind("<F5>", lambda _e: self._on_run_analysis())
        self.bind("<Control-e>", lambda _e: self._on_export())
        self.bind("<Control-t>", lambda _e: self._toggle_theme_via_shortcut())

    # ------------------------------------------------------------------ #
    # Theme handling
    # ------------------------------------------------------------------ #
    def _style_treeviews(self) -> None:
        """(Re)configure the ttk styles/tags so tables follow the theme."""
        pal = plotter.THEMES[self._theme_name]
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:  # pragma: no cover
            pass
        style.configure(
            "Iso.Treeview", background=pal["card"], fieldbackground=pal["card"],
            foreground=pal["text"], rowheight=26, borderwidth=0, relief="flat",
            bordercolor=pal["card"], lightcolor=pal["card"], darkcolor=pal["card"],
            font=("TkDefaultFont", 10),
        )
        style.layout("Iso.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.map("Iso.Treeview", background=[("selected", pal["select"])],
                  foreground=[("selected", pal["text"])])
        style.configure(
            "Iso.Treeview.Heading", background=pal["card_alt"], foreground=pal["text"],
            relief="flat", borderwidth=0, font=("TkDefaultFont", 10, "bold"), padding=(4, 6),
        )
        style.map("Iso.Treeview.Heading", background=[("active", pal["select"])])
        style.configure(
            "Iso.Vertical.TScrollbar", background=pal["card_alt"], troughcolor=pal["card"],
            bordercolor=pal["card"], arrowcolor=pal["muted"], relief="flat",
        )
        for tree in self._trees:
            tree.tag_configure("odd", background=pal["row_alt"])
            tree.tag_configure("even", background=pal["card"])
            tree.tag_configure("avg", background=pal["select"], foreground=pal["text"])
            tree.tag_configure("pass", foreground=pal["ok"])
            tree.tag_configure("fail", foreground=pal["bad"])

    def _on_theme_toggled(self) -> None:
        self._apply_theme("dark" if self.dark_mode_var.get() else "light")

    def _toggle_theme_via_shortcut(self) -> None:
        self.dark_mode_var.set(not self.dark_mode_var.get())
        self._on_theme_toggled()

    def _apply_theme(self, name: str) -> None:
        """Switch widgets, tables and plots to ``name`` ('light'/'dark')."""
        self._theme_name = plotter.set_theme(name)
        ctk.set_appearance_mode(self._theme_name)
        self.configure(fg_color=_c("window"))
        self._style_treeviews()
        self._redraw_all_plots()

    def _redraw_all_plots(self) -> None:
        pal = plotter.THEMES[self._theme_name]
        plotter.update_plot(self.figure, self.results, self.averages, self._current_ci())
        plotter.update_kth_model_plot(self.kth_figure, self.kth_full_result)
        plotter.update_coats_redfern_plot(self.cr_figure, self.cr_result)
        for canvas in (self.canvas, self.kth_canvas, self.cr_canvas):
            canvas.get_tk_widget().configure(bg=pal["figure"], highlightthickness=0)
            canvas.draw()

    # ------------------------------------------------------------------ #
    # Display-option handlers
    # ------------------------------------------------------------------ #
    def _select_tab(self, name: str) -> None:
        """Select a tab, robustly.

        customtkinter's ``CTkTabview.set`` hides every *other* tab 100 ms
        later.  If another selection lands inside that window (e.g. a quick
        click right after the automatic switch to *Summary*), the newly
        selected tab would be highlighted but blank.  Re-asserting the
        selection once the window has passed makes the final choice stick.
        """
        self.tabview.set(name)
        try:
            self.after(150, self._ensure_tab_visible)
        except Exception:  # noqa: BLE001 - window being torn down
            pass

    def _ensure_tab_visible(self) -> None:
        if not self._alive:
            return
        try:
            current = self.tabview.get()
            if not self.tabview.tab(current).winfo_ismapped():
                self.tabview.set(current)
        except Exception:  # noqa: BLE001
            pass

    def _on_plot_style_changed(self, style_name: str) -> None:
        plotter.apply_plot_style(style_name)
        if self.results:
            plotter.update_plot(self.figure, self.results, self.averages, self._current_ci())
            self.canvas.draw()

    def _on_show_ci_toggled(self) -> None:
        if self.results:
            plotter.update_plot(self.figure, self.results, self.averages, self._current_ci())
            self.canvas.draw()

    def _current_ci(self) -> Optional[Dict[str, Dict[float, "tuple[float, float, float]"]]]:
        """Recompute confidence intervals for whichever supported
        methods are currently in self.results, if the checkbox is on."""
        if not self.show_ci_var.get() or self.working_df is None:
            return None
        try:
            betas = self._read_betas()
        except ValueError:
            return None
        ci: Dict[str, Dict[float, "tuple[float, float, float]"]] = {}
        for name in self.results:
            if name in kinetics._CI_SUPPORTED_METHODS:
                try:
                    ci[name] = kinetics.confidence_interval(self.working_df, betas, method_name=name)
                except Exception:  # noqa: BLE001
                    continue
        return ci or None

    def _set_all_methods(self, value: bool) -> None:
        for var in self.method_vars.values():
            var.set(value)

    def _validate_beta_entry(self, entry: ctk.CTkEntry) -> None:
        """Red border while the text is not a positive number."""
        text = entry.get().strip()
        try:
            ok = float(text) > 0
        except ValueError:
            ok = False
        entry.configure(border_color=_c("border") if ok else _c("bad"))

    # ------------------------------------------------------------------ #
    # Loading data
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_sample_file() -> Optional[str]:
        """Path of the bundled sample dataset, if it ships next to the package."""
        here = os.path.dirname(os.path.abspath(__file__))
        for rel in (("..", "..", "sample_data"), ("sample_data",)):
            path = os.path.normpath(os.path.join(here, *rel, "matric_121459.txt"))
            if os.path.isfile(path):
                return path
        return None

    def _on_load_data(self) -> None:
        if self.is_running:
            return
        path = filedialog.askopenfilename(
            title="Select kinetics data file",
            filetypes=[
                ("Data files", "*.txt *.csv *.dat *.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.load_file(path)

    def _on_load_sample(self) -> None:
        if self.sample_path and not self.is_running:
            self.load_file(self.sample_path)

    def load_file(self, path: str) -> bool:
        """Load ``path`` into the app (also usable from scripts/tests).

        The file type is detected from its columns:

        * 5 columns  -> *processed* data ``[alpha, T1..T4]``, used directly;
        * 4 columns  -> *raw* data ``[T1..T4]``: a dialog asks for the
          conversion step, alpha is generated and the temperatures are
          interpolated onto the new alpha grid (the former Origin step).

        Returns True on success; errors are shown in a dialog and a
        cancelled dialog leaves the current data untouched (returns False).
        """
        try:
            table = data_loader.load_table(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Load error", str(exc))
            return False

        if table.is_raw:
            # Pre-select the row layout the row count points to (99 rows ->
            # interior); otherwise keep whatever was used last.
            suggested = data_loader.suggest_mapping(min(table.rows_per_column))
            mapping = suggested if suggested != data_loader.DEFAULT_ROW_MAPPING else self._last_row_mapping
            choice = self._ask_alpha_step(
                table, os.path.basename(path), self._last_alpha_step, mapping)
            if choice is None:
                self._set_status("Loading cancelled.")
                return False
            return self._install_raw(path, table, *choice)

        df = table.data
        self.source_kind = "processed"
        self._raw_loaded = None
        self.alpha_step = None
        working = data_loader.interpolate_alpha(df)
        info = (f"Processed file (5 columns: \u03b1, T\u2081\u2026T\u2084) \u00b7 {len(df)} rows \u00b7 "
                f"\u03b1 {alpha_range_text(df['alpha'])}\nAnalysed on {len(working)} grid points (\u0394\u03b1 = 0.01)")
        self._install_dataset(path, df, working, info, f"Loaded {len(df)} rows from {path}.")
        return True

    def _ask_alpha_step(self, table, file_name: str, initial_step: float, mapping: str):
        """Open the modal dialog; returns ``(step, mapping)`` or ``None``.
        (A separate method so tests/scripts can replace it.)"""
        dialog = AlphaStepDialog(self, table, file_name, data_loader_fmt_step(initial_step), mapping)
        self.wait_window(dialog)
        return dialog.result

    def _install_raw(self, path: str, table, step: float, mapping: str) -> bool:
        """Generate alpha for a raw table and make it the current dataset."""
        try:
            processed = data_loader.build_alpha_table(table.data, step, mapping)
        except ValueError as exc:
            messagebox.showerror("Cannot generate \u03b1", str(exc))
            return False
        self.source_kind = "raw"
        self._raw_loaded = table
        self.alpha_step = step
        self.row_mapping = mapping
        self._last_alpha_step, self._last_row_mapping = step, mapping
        rows = table.rows_per_column
        rows_text = f"{rows[0]} rows" if len(set(rows)) == 1 else "rows " + "/".join(str(r) for r in rows)
        info = (f"Raw file (4 columns: T\u2081\u2026T\u2084) \u00b7 {rows_text}\n"
                f"\u03b1 generated: {data_loader.describe_grid(step)}")
        self._install_dataset(
            path, processed, processed, info,
            f"Raw data: generated {len(processed)} \u03b1 points (\u0394\u03b1 = {step:g}) from {path}.")
        return True

    def _install_dataset(self, path: str, preview_df, working_df, info: str, status: str) -> None:
        """Make a (processed) table the current dataset and reset all results."""
        self.raw_df = preview_df
        self.working_df = working_df
        self.file_path = path
        self.file_label.configure(text=os.path.basename(path))
        self.data_info_label.configure(text=info)
        self._update_raw_controls()
        self._populate_preview(preview_df)
        self._set_status(status)

        # Reset previous results whenever new data is loaded.
        self.results = {}
        self.averages = {}
        self.kth_full_result = None
        self.cr_result = None
        self.export_button.configure(state="disabled")
        self._redraw_all_plots()
        self._populate_model_fitting_tab(None)
        self._populate_summary({}, {})
        self._clear_results_table()
        self._clear_lifetime_result()
        self._refresh_lifetime_sources()
        self._reset_progress()
        self._select_tab(TAB_PREVIEW)

    def _update_raw_controls(self) -> None:
        """Show the raw-data actions only while a raw file is loaded."""
        if self.source_kind == "raw":
            self.raw_actions_frame.pack(fill="x", pady=(8, 0))
        else:
            self.raw_actions_frame.pack_forget()

    def _on_change_alpha_step(self) -> None:
        """Re-open the dialog for the loaded raw file and regenerate alpha."""
        if self.is_running or self.source_kind != "raw" or self._raw_loaded is None:
            return
        choice = self._ask_alpha_step(
            self._raw_loaded, os.path.basename(self.file_path or ""),
            self.alpha_step or self._last_alpha_step, self.row_mapping)
        if choice is not None:
            self._install_raw(self.file_path or "", self._raw_loaded, *choice)

    def _on_save_processed(self) -> None:
        """Save the generated [alpha, T1..T4] table so it can be reloaded
        later (or used elsewhere) without repeating the interpolation."""
        if self.working_df is None:
            return
        base = os.path.splitext(os.path.basename(self.file_path or "data"))[0]
        path = filedialog.asksaveasfilename(
            title="Save processed table", defaultextension=".txt",
            initialfile=f"{base}_processed.txt",
            filetypes=[("Tab-delimited text", "*.txt"), ("CSV", "*.csv")])
        if not path:
            return
        try:
            data_loader.save_processed_table(self.working_df, path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save error", str(exc))
            return
        self._set_status(f"Processed table saved to {path}")
        messagebox.showinfo("Processed table saved", f"Saved to:\n{path}")

    # ------------------------------------------------------------------ #
    # Running the analysis
    # ------------------------------------------------------------------ #
    def _on_run_analysis(self) -> None:
        if self.is_running:
            return

        if self.working_df is None:
            messagebox.showwarning("No data", "Please load a data file first.")
            return

        selected = [name for name, var in self.method_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning(
                "No method selected", "Please select at least one kinetic method."
            )
            return

        try:
            betas = self._read_betas()
        except ValueError as exc:
            messagebox.showerror("Invalid heating rates", str(exc))
            return

        order_warning = self._check_beta_order(betas)

        self._set_running_state(True)
        self._reset_progress()
        self._run_started = time.perf_counter()
        self._set_status("Starting analysis\u2026" if not order_warning else order_warning)
        self._schedule_tick()

        kth_mode = self.kth_mode_var.get()

        self._stop_event.clear()
        thread = threading.Thread(
            target=self._run_analysis_worker,
            args=(selected, betas, kth_mode),
            daemon=True,
        )
        self._analysis_thread = thread
        thread.start()

    def _check_beta_order(self, betas: List[float]) -> Optional[str]:
        """Non-blocking sanity check: at fixed alpha, a faster heating rate
        must give a higher temperature.  Returns a warning text or None."""
        try:
            means = [float(self.working_df[c].mean()) for c in ("T1", "T2", "T3", "T4")]
        except Exception:  # noqa: BLE001
            return None
        if np.argsort(betas).tolist() != np.argsort(means).tolist():
            return ("\u26a0 Heating rates do not follow the temperature columns "
                    "(faster rate should give higher T) \u2013 check the \u03b2 order.")
        return None

    def _post(self, func, *args) -> None:
        """Marshal ``func(*args)`` onto the Tk thread; silently ignored once
        the window is closing (so late worker callbacks never touch widgets)."""
        if self._stop_event.is_set() or not self._alive:
            return
        try:
            self.after(0, func, *args)
        except Exception:  # noqa: BLE001 - window already torn down
            pass

    def _run_analysis_worker(
        self, selected: List[str], betas: List[float], kth_mode: str
    ) -> None:
        """Runs in a background thread; must not touch Tk widgets directly."""
        results: Dict[str, Dict[float, float]] = {}
        errors: Dict[str, str] = {}
        kth_full_result: Optional[Dict[str, object]] = None

        stages = list(selected) + ["Coats-Redfern"]
        weights = [_STAGE_WEIGHTS.get(s, 1.0) for s in stages]
        if "KTH" in selected and kth_mode == KTH_MODE_FULL_TRIPLET:
            weights[stages.index("KTH")] = 3.0
        total = float(sum(weights))
        done = 0.0

        for idx, name in enumerate(selected):
            if self._stop_event.is_set():
                return  # window is closing -- abandon without touching widgets
            self._post(self._begin_stage, name, idx + 1, len(stages), done / total)

            try:
                if name == "KTH" and kth_mode == KTH_MODE_FULL_TRIPLET:
                    full = kinetics.kth_full(self.working_df, betas)
                    kth_full_result = full
                    results[name] = full["Ea"]
                else:
                    func = kinetics.METHODS[name]
                    results[name] = func(self.working_df, betas)
            except Exception as exc:  # noqa: BLE001
                errors[name] = str(exc)
                results[name] = {}

            if self._stop_event.is_set():
                return
            done += weights[idx]
            self._post(self._update_progress, done / total, name)

        if self._stop_event.is_set():
            return

        # Coats-Redfern model fitting is fast and self-contained, so it runs
        # on every analysis; it needs a model-free reference Ea, taken from
        # Vyazovkin when that was selected (else computed internally).
        self._post(self._begin_stage, "Coats-Redfern", len(stages), len(stages), done / total)
        try:
            reference_ea = None
            if "Vyazovkin" in results and results["Vyazovkin"]:
                reference_ea = kinetics.average_ea(results["Vyazovkin"])
            cr_result = kinetics.coats_redfern(
                self.working_df, betas, reference_ea_kj=reference_ea
            )
        except Exception as exc:  # noqa: BLE001
            errors["Coats-Redfern"] = str(exc)
            cr_result = None

        if self._stop_event.is_set():
            return

        averages = {name: kinetics.average_ea(res) for name, res in results.items()}

        # The bar must end at exactly 100 % (a past bug left it at 95 %).
        self._post(self._update_progress, 1.0, "Complete")
        self._post(
            self._on_analysis_done, results, averages, errors, kth_full_result, cr_result
        )

    # -- progress -------------------------------------------------------------
    def _reset_progress(self) -> None:
        self._progress_fraction = 0.0
        self._stage_text = ""
        self.progress_bar.set(0.0)
        self.progress_label.configure(text="0 %")
        self.eta_label.configure(text="")

    def _begin_stage(self, name: str, index: int, total: int, fraction: float) -> None:
        """A stage is starting: say which one, at what position."""
        if not self._alive:
            return
        self._stage_text = f"Running {name}  (step {index} of {total})"
        self._progress_fraction = max(self._progress_fraction, fraction)
        self.progress_bar.set(self._progress_fraction)
        self.progress_label.configure(text=f"{self._progress_fraction * 100:.0f} %")
        self._set_status(self._stage_text + "\u2026")

    def _update_progress(self, fraction: float, method_name: str) -> None:
        if not self._alive:
            return
        fraction = min(max(float(fraction), 0.0), 1.0)
        self._progress_fraction = fraction
        self.progress_bar.set(fraction)
        self.progress_label.configure(text=f"{fraction * 100:.0f} %")
        if method_name == "Complete":
            self._set_status("Analysis complete.")
        else:
            self._set_status(f"Computed {method_name} ({fraction * 100:.0f} %)")
        self._refresh_eta()

    def _refresh_eta(self) -> None:
        if self._run_started is None:
            return
        elapsed = time.perf_counter() - self._run_started
        frac = self._progress_fraction
        if frac >= 1.0:
            text = f"Done in {fmt_duration(elapsed)}"
        elif frac > 0.05:
            text = f"Elapsed {fmt_duration(elapsed)} \u00b7 ~{fmt_duration(elapsed * (1 - frac) / frac)} left"
        else:
            text = f"Elapsed {fmt_duration(elapsed)}"
        self.eta_label.configure(text=text)

    def _schedule_tick(self) -> None:
        """Refresh elapsed/remaining time every 250 ms while running."""
        if not self._alive or not self.is_running:
            self._tick_job = None
            return
        self._refresh_eta()
        try:
            self._tick_job = self.after(250, self._schedule_tick)
        except Exception:  # noqa: BLE001
            self._tick_job = None

    def _on_analysis_done(
        self,
        results: Dict[str, Dict[float, float]],
        averages: Dict[str, float],
        errors: Dict[str, str],
        kth_full_result: Optional[Dict[str, object]] = None,
        cr_result: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self._alive:
            return

        self.results = results
        self.averages = averages
        self.kth_full_result = kth_full_result
        self.cr_result = cr_result

        plotter.update_plot(self.figure, results, averages, self._current_ci())
        self.canvas.draw()
        plotter.update_kth_model_plot(self.kth_figure, kth_full_result)
        self.kth_canvas.draw()
        self._populate_model_fitting_tab(cr_result)
        self._populate_results_table(results)
        self._populate_summary(results, averages)
        had_prediction = self.lifetime_result is not None
        self._refresh_lifetime_sources()
        if had_prediction:  # don't leave a prediction from the previous run on screen
            self._on_calculate_lifetime()

        self._set_running_state(False)
        self.progress_bar.set(1.0)
        self.progress_label.configure(text="100 %")
        self._refresh_eta()
        self.export_button.configure(state="normal" if results else "disabled")
        if results:
            self._select_tab(TAB_SUMMARY)

        if errors:
            detail = "\n".join(f"- {k}: {v}" for k, v in errors.items())
            messagebox.showwarning(
                "Some methods failed",
                f"The following method(s) could not be computed:\n{detail}",
            )
            self._set_status("Analysis finished with warnings.")
        else:
            self._set_status("Analysis complete.")

    # ------------------------------------------------------------------ #
    # Tables
    # ------------------------------------------------------------------ #
    def _populate_preview(self, df: pd.DataFrame) -> None:
        self.preview_tree.delete(*self.preview_tree.get_children())
        for i, (_, row) in enumerate(df.iterrows()):
            values = [fmt_alpha(row["alpha"])] + [fmt_temp(row[c]) for c in ["T1", "T2", "T3", "T4"]]
            self.preview_tree.insert("", "end", values=values, tags=("odd" if i % 2 else "even",))
        tmin = float(df[["T1", "T2", "T3", "T4"]].min().min())
        tmax = float(df[["T1", "T2", "T3", "T4"]].max().max())
        self.preview_info_label.configure(
            text=f"{len(df)} rows \u00b7 \u03b1 {alpha_range_text(df['alpha'])} \u00b7 "
                 f"T {fmt_temp(tmin)} \u2013 {fmt_temp(tmax)} K"
        )

    def _clear_results_table(self) -> None:
        self.results_tree.delete(*self.results_tree.get_children())

    def _populate_results_table(self, results: Dict[str, Dict[float, float]]) -> None:
        self._clear_results_table()

        all_alphas = sorted(
            {a for method_results in results.values() for a in method_results.keys()}
        )

        kth_A = self.kth_full_result["A"] if self.kth_full_result else {}
        kth_f = self.kth_full_result["f_alpha"] if self.kth_full_result else {}

        for i, a in enumerate(all_alphas):
            row = [fmt_alpha(a)]
            for name in METHOD_NAMES:
                if name in results:
                    row.append(fmt_ea(results[name].get(a, float("nan"))))
                else:
                    row.append(EMPTY)
            a_val = kth_A.get(a, float("nan"))
            f_val = kth_f.get(a, float("nan"))
            row.append(fmt_sci(a_val, na=EMPTY))
            row.append(fmt_score(f_val, na=EMPTY))
            self.results_tree.insert("", "end", values=row, tags=("odd" if i % 2 else "even",))

        # Add an "average" summary row at the top for convenience.
        avg_row = ["(average)"]
        for name in METHOD_NAMES:
            avg_row.append(fmt_ea(kinetics.average_ea(results[name])) if name in results else EMPTY)

        if self.kth_full_result:
            finite_A = [v for v in kth_A.values() if v == v]
            avg_row.append(fmt_sci(sum(finite_A) / len(finite_A), na=EMPTY) if finite_A else EMPTY)
            avg_row.append(f"model={self.kth_full_result.get('model', '-')}")
        else:
            avg_row.append(EMPTY)
            avg_row.append(EMPTY)

        self.results_tree.insert("", 0, values=avg_row, tags=("avg",))

    def _on_copy_table(self) -> None:
        """Copy the results table to the clipboard as tab-separated text."""
        cols = self.results_tree["columns"]
        heads = [self.results_tree.heading(c, "text") for c in cols]
        lines = ["\t".join(heads)]
        for item in self.results_tree.get_children():
            lines.append("\t".join(str(v) for v in self.results_tree.item(item, "values")))
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        self._set_status(f"Copied {len(lines) - 1} table rows to the clipboard.")

    def _populate_model_fitting_tab(self, cr_result: Optional[Dict[str, object]]) -> None:
        self.cr_tree.delete(*self.cr_tree.get_children())

        if not cr_result or not cr_result.get("results"):
            self.cr_summary_label.configure(
                text="Run an analysis to see Coats-Redfern model fitting."
            )
            plotter.update_coats_redfern_plot(self.cr_figure, None)
            self.cr_canvas.draw()
            return

        results = cr_result["results"]
        recommended = cr_result.get("recommended_model")
        reference_ea = cr_result.get("reference_ea_kj", float("nan"))

        n_passed = 0
        for name, fit in sorted(results.items(), key=lambda kv: kv[1].get("r2", 0) or 0, reverse=True):
            r2 = fit.get("r2", float("nan"))
            gap_pct = fit.get("ea_gap_pct", float("nan"))
            passed = np.isfinite(r2) and r2 > 0.99 and np.isfinite(gap_pct) and gap_pct < 0.20

            if passed:
                n_passed += 1
                status = "\u2714 Recommended" if name == recommended else "\u2714 Passed"
                tag = "pass"
            elif np.isfinite(gap_pct):
                status = f"\u2718 Rejected (E\u2090 gap {fmt_pct(gap_pct * 100, 0)})"
                tag = "fail"
            else:
                status = "\u2718 Rejected"
                tag = "fail"

            self.cr_tree.insert(
                "", "end", tags=(tag,),
                values=[name, fmt_ea(fit.get("Ea")), fmt_sci(fit.get("A")), fmt_r2(r2), status],
            )

        n_total = len(results)
        summary = (
            f"Reference E\u2090 (Vyazovkin): {fmt_ea(reference_ea, unit=True)}   |   "
            f"Models screened: {n_total}   |   Passed: {n_passed}   |   "
            f"Rejected: {n_total - n_passed}   |   "
            f"Recommended: {recommended or 'none (inconclusive)'}"
        )
        if not recommended:
            summary += (
                "\nNo single reaction model fits well. This is typical when E\u2090 changes "
                "with \u03b1 (multi-step kinetics); rely on the isoconversional results."
            )
        self.cr_summary_label.configure(text=summary)

        plotter.update_coats_redfern_plot(self.cr_figure, cr_result)
        self.cr_canvas.draw()

    # ------------------------------------------------------------------ #
    # Summary tab
    # ------------------------------------------------------------------ #
    def _insights(self, results, averages) -> List[str]:
        """Plain-language observations derived from the results."""
        notes: List[str] = []
        stats = exporter.method_statistics(results)
        usable = {n: s for n, s in stats.items() if s["n"] > 0 and s["mean"] > 0}
        if not usable:
            return notes

        def window_range(name):
            vals = [(a, e) for a, e in results[name].items() if 0.10 <= a <= 0.90 and e == e]
            if len(vals) < 3:
                return None
            es = np.array([e for _, e in vals])
            return float(es.max() - es.min()), float(es.mean()), float(es.min()), float(es.max())

        ref = next((n for n in ("Friedman", "Vyazovkin", "KTH") if n in usable), next(iter(usable)))
        wr = window_range(ref)
        if wr:
            span, mean, lo, hi = wr
            rel = span / mean * 100.0
            if rel > 20.0:
                notes.append(
                    f"E\u2090 changes by {fmt_pct(rel, 0)} between \u03b1 = 0.10 and 0.90 "
                    f"({fmt_ea(lo)} \u2192 {fmt_ea(hi)} {UNIT_EA_PLAIN}, {ref}). This points to "
                    "multi-step kinetics: favour Friedman, KTH and Vyazovkin, which follow "
                    "E\u2090(\u03b1), over FWO/KAS/Starink, which average over the whole range."
                )
            else:
                notes.append(
                    f"E\u2090 is nearly constant ({fmt_pct(rel, 0)} variation over \u03b1 = 0.10\u20130.90): "
                    "single-step kinetics is plausible and all methods should agree closely."
                )

        means = [s["mean"] for s in usable.values()]
        if len(means) > 1:
            spread = (max(means) - min(means)) / (sum(means) / len(means)) * 100.0
            notes.append(
                f"Mean E\u2090 differs by {fmt_pct(spread, 0)} between methods "
                f"({fmt_ea(min(means))} \u2013 {fmt_ea(max(means))} {UNIT_EA_PLAIN})."
                + (" Good agreement." if spread < 5 else
                   " Differences of this size are expected when E\u2090 varies with \u03b1.")
            )
        if self.cr_result is not None:
            rec = self.cr_result.get("recommended_model")
            notes.append(
                f"Coats-Redfern recommends model {rec}." if rec
                else "Coats-Redfern found no single reaction model that fits."
            )
        if self.kth_full_result:
            notes.append(f"KTH + Malek identified the reaction model: {self.kth_full_result.get('model')}.")
        return notes

    def _populate_summary(self, results, averages) -> None:
        for child in self.summary_body.winfo_children():
            child.destroy()

        if not results:
            ctk.CTkLabel(
                self.summary_body, text_color=_c("muted"), justify="left",
                text="Run an analysis to see the activation energy of every method here.\n\n"
                     "1  Load a data file (or use the bundled sample)\n"
                     "2  Check the heating rates\n"
                     "3  Pick methods and press  \u25b6 Run Analysis",
            ).pack(anchor="w", padx=12, pady=16)
            return

        stats = exporter.method_statistics(results)
        grid = ctk.CTkFrame(self.summary_body, fg_color="transparent")
        grid.pack(fill="x")
        for col in range(3):
            grid.grid_columnconfigure(col, weight=1, uniform="cards")

        for i, (name, st) in enumerate(stats.items()):
            card = ctk.CTkFrame(grid, corner_radius=12, fg_color=_c("card_alt"),
                                border_width=1, border_color=_c("border"))
            card.grid(row=i // 3, column=i % 3, padx=5, pady=5, sticky="nsew")
            head = ctk.CTkFrame(card, fg_color="transparent")
            head.pack(fill="x", padx=14, pady=(10, 0))
            ctk.CTkLabel(head, text="\u25cf", text_color=plotter.METHOD_COLORS.get(name, "#888"),
                         width=14).pack(side="left")
            ctk.CTkLabel(head, text=name, font=self.font_h, text_color=_c("text")).pack(side="left")
            ctk.CTkLabel(card, text=fmt_ea(st["mean"]), font=self.font_big,
                         text_color=_c("text")).pack(anchor="w", padx=14)
            self._muted(card, f"{UNIT_EA_PLAIN}  \u00b7  \u00b1 {fmt_ea(st['std'])}  (std over \u03b1)").pack(anchor="w", padx=14)
            self._muted(card, f"range {fmt_ea(st['min'])} \u2013 {fmt_ea(st['max'])}").pack(
                anchor="w", padx=14, pady=(0, 10))

        notes = self._insights(results, averages)
        if notes:
            box = ctk.CTkFrame(self.summary_body, corner_radius=12, fg_color=_c("card_alt"),
                               border_width=1, border_color=_c("border"))
            box.pack(fill="x", padx=5, pady=(10, 5))
            ctk.CTkLabel(box, text="What the results say", font=self.font_h,
                         text_color=_c("text")).pack(anchor="w", padx=14, pady=(10, 4))
            for note in notes:
                ctk.CTkLabel(box, text="\u2022 " + note, justify="left", anchor="w",
                             wraplength=760, text_color=_c("text")).pack(
                    anchor="w", padx=14, pady=(0, 6))

    # ------------------------------------------------------------------ #
    # Lifetime tab
    # ------------------------------------------------------------------ #
    def _refresh_lifetime_sources(self) -> None:
        """Rebuild the Ea-source menu from the current analysis results;
        the recommended Coats-Redfern model (if any) is selected first."""
        sources = lifetime.available_ea_sources(self.averages, self.cr_result)
        self._lifetime_sources = {src.label: src for src in sources}
        labels = list(self._lifetime_sources)
        self.lifetime_source_menu.configure(values=labels)
        self.lifetime_source_var.set(labels[0])
        self._update_lifetime_ea_display()

    def _selected_lifetime_source(self) -> lifetime.EaSource:
        label = self.lifetime_source_var.get()
        return self._lifetime_sources.get(label) or lifetime.EaSource(lifetime.MANUAL_SOURCE, None)

    def _update_lifetime_ea_display(self) -> None:
        source = self._selected_lifetime_source()
        note = source.note
        if source.ea_kj is None:
            self.lifetime_ea_label.configure(text="E\u2090 (kJ/mol):")
            self.lifetime_ea_entry.pack(after=self.lifetime_ea_label, padx=14, pady=(2, 0), anchor="w")
            if len(self._lifetime_sources) == 1:
                note += "  Run an analysis to use fitted values."
        else:
            self.lifetime_ea_entry.pack_forget()
            self.lifetime_ea_label.configure(text=f"E\u2090 = {fmt_ea(source.ea_kj, unit=True)}")
        self.lifetime_source_note.configure(text=note)

    def _on_lifetime_source_changed(self, _label: str) -> None:
        self._update_lifetime_ea_display()
        if self.lifetime_result is not None:  # keep the shown result in step
            self._on_calculate_lifetime()

    def _on_lifetime_temp_unit_changed(self, new_unit: str) -> None:
        """Convert the two temperature fields when switching \u00b0C <-> K."""
        for entry in (self.lifetime_storage_temp_entry, self.lifetime_service_temp_entry):
            try:
                value = float(entry.get().replace(",", "."))
            except ValueError:
                continue
            converted = value + 273.15 if new_unit == "K" else value - 273.15
            text = f"{converted:.2f}"
            entry.delete(0, "end")
            entry.insert(0, text.rstrip("0").rstrip(".") if "." in text else text)

    def _read_lifetime_inputs(self) -> "tuple[float, float, float, float]":
        """(Ea kJ/mol, T_storage K, t_storage days, T_service K) from the form."""
        def number(entry, name: str) -> float:
            text = entry.get().strip().replace(",", ".")
            try:
                return float(text)
            except ValueError:
                raise ValueError(f"{name} ('{text}') is not a valid number.") from None

        source = self._selected_lifetime_source()
        ea = source.ea_kj if source.ea_kj is not None else number(self.lifetime_ea_entry, "Activation energy")
        unit = "K" if self.lifetime_temp_unit_var.get() == "K" else "C"
        t_storage = lifetime.to_kelvin(number(self.lifetime_storage_temp_entry, "Storage temperature"), unit)
        t_service = lifetime.to_kelvin(number(self.lifetime_service_temp_entry, "Service temperature"), unit)
        days = lifetime.time_to_days(
            number(self.lifetime_storage_time_entry, "Storage time"), self.lifetime_time_unit_var.get())
        return float(ea), t_storage, days, t_service

    def _measured_temperature_range(self) -> Optional["tuple[float, float]"]:
        if self.working_df is None:
            return None
        try:
            cols = self.working_df[["T1", "T2", "T3", "T4"]]
            return float(cols.min().min()), float(cols.max().max())
        except Exception:  # noqa: BLE001
            return None

    def _on_calculate_lifetime(self) -> Optional[lifetime.LifetimeResult]:
        """Validate the form, compute the lifetime and show it in years,
        days and minutes.  Errors are shown inline (no modal dialog)."""
        self.lifetime_error_label.configure(text="")
        try:
            ea, t_storage, days, t_service = self._read_lifetime_inputs()
            result = lifetime.predict_lifetime(
                ea, t_storage, days, t_service,
                measured_range_k=self._measured_temperature_range(),
                ea_variation_pct=lifetime.ea_variation_percent(self.results) if self.results else None,
            )
        except ValueError as exc:
            self.lifetime_error_label.configure(text=str(exc))
            return None

        self.lifetime_result = result
        self.lifetime_years_label.configure(text=fmt_years(result.lifetime_years))
        self.lifetime_days_label.configure(text=fmt_days(result.lifetime_days))
        self.lifetime_minutes_label.configure(text=fmt_minutes(result.lifetime_minutes))

        af = result.acceleration_factor
        af_text = fmt_sci(af, na="\u221e") if af >= 1e6 else fmt_score(af, 2, na="\u221e")
        low_y = result.lifetime_days_low_ea / lifetime.DAYS_PER_YEAR
        high_y = result.lifetime_days_high_ea / lifetime.DAYS_PER_YEAR
        unc = f"{result.ea_uncertainty_pct:g}"
        self.lifetime_detail_label.configure(
            text=f"Using E\u2090 = {fmt_ea(result.ea_kj, unit=True)}  \u00b7  acceleration factor = {af_text}\n"
                 f"Sensitivity: E\u2090 \u2212{unc} % \u2192 {fmt_years(low_y)} years;  "
                 f"E\u2090 +{unc} % \u2192 {fmt_years(high_y)} years"
        )
        self.lifetime_notes_label.configure(text="\n".join("\u26a0 " + w for w in result.warnings))
        self._set_status(
            f"Predicted service lifetime: {fmt_years(result.lifetime_years)} years "
            f"({fmt_days(result.lifetime_days)} days).")
        return result

    def _clear_lifetime_result(self) -> None:
        self.lifetime_result = None
        for label in self.lifetime_value_labels.values():
            label.configure(text=EMPTY)
        self.lifetime_detail_label.configure(text="Enter the temperatures and press Calculate lifetime.")
        self.lifetime_notes_label.configure(text="")
        self.lifetime_error_label.configure(text="")

    # ------------------------------------------------------------------ #
    # Validation tab
    # ------------------------------------------------------------------ #
    def _on_run_validation(self) -> None:
        if self.is_running:
            return
        self._set_running_state(True)
        self._reset_progress()
        self._run_started = time.perf_counter()
        self.validation_status_label.configure(text="Running validation\u2026")
        self._set_status("Validating methods against synthetic data\u2026")
        self._schedule_tick()
        self._stop_event.clear()
        thread = threading.Thread(target=self._run_validation_worker, daemon=True)
        self._validation_thread = thread
        thread.start()

    def _run_validation_worker(self) -> None:
        def progress(fraction: float, label: str) -> None:
            self._post(self._update_progress, fraction * 0.98, f"scenario '{label}'")

        try:
            report = validation.run_validation(progress=progress)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Validation failed")
            self._post(self._on_validation_done, None, str(exc))
            return
        self._post(self._on_validation_done, report, "")

    def _on_validation_done(self, report, error: str) -> None:
        if not self._alive:
            return
        self._set_running_state(False)
        self.validation_tree.delete(*self.validation_tree.get_children())
        if report is None:
            self.validation_status_label.configure(text=f"Validation failed: {error}")
            self._set_status("Validation failed.")
            return
        self.validation_report = report
        for r in report.records:
            self.validation_tree.insert(
                "", "end", tags=("pass" if r.passed else "fail",),
                values=[
                    r.scenario, r.method, fmt_ea(r.true_ea), fmt_ea(r.found_ea),
                    fmt_pct(r.error_pct, signed=True), f"\u00b1 {r.tolerance_pct:g} %",
                    "PASS" if r.passed else "FAIL",
                ],
            )
        verdict = "All methods recover the known E\u2090 within tolerance." if report.all_passed \
            else "Some checks failed \u2013 see the red rows."
        self.validation_status_label.configure(
            text=f"{report.n_passed}/{len(report.records)} checks passed in "
                 f"{fmt_duration(report.seconds)}.  {verdict}")
        self._update_progress(1.0, "Complete")
        self._set_status("Validation complete.")

    # ------------------------------------------------------------------ #
    # Export / save image / about
    # ------------------------------------------------------------------ #
    def _on_export(self) -> None:
        if not self.results:
            messagebox.showwarning("Nothing to export", "Run an analysis first.")
            return

        folder = filedialog.askdirectory(title="Select destination folder")
        if not folder:
            return

        metadata = {
            "Software": f"Isoconversional Kinetics Analyzer v{isokinetics_version}",
            "Sample": os.path.basename(self.file_path) if self.file_path else "unknown",
            "Input": ("raw 4-column file, alpha generated with step %g" % self.alpha_step)
                     if self.source_kind == "raw" and self.alpha_step else "processed 5-column file",
            "Methods": ", ".join(self.results.keys()),
            "Date": datetime.date.today().isoformat(),
        }

        try:
            out_path = exporter.export_all(
                folder,
                self.results,
                self.figure,
                kth_full_result=self.kth_full_result,
                kth_figure=self.kth_figure,
                metadata=metadata,
                cr_result=self.cr_result,
                cr_figure=self.cr_figure,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export error", str(exc))
            return

        messagebox.showinfo("Export complete", f"Results saved to:\n{out_path}")
        if not exporter.EXCEL_EXPORT_AVAILABLE:
            messagebox.showwarning(
                "Excel file skipped",
                "The .xlsx workbook was not created because the optional "
                "'xlsxwriter' package is not installed.\n\n"
                "Everything else (PNG, TIFF, SVG, PDF, the tab-delimited "
                ".txt file and Summary.txt) was saved normally.\n\n"
                "To also get the Excel file, close the app and run:\n"
                "pip install xlsxwriter",
            )
        self._set_status(f"Exported to {out_path}")

    def _on_save_image(self) -> None:
        """
        Save the plot shown in the active tab as PNG, TIFF, PDF or SVG.
        Images are always written with the light (white-background) theme,
        even while the dark theme is on screen.
        """
        tab_to_figure = {
            TAB_PLOT: self.figure,
            TAB_KTH: self.kth_figure,
            TAB_MODEL_FITTING: self.cr_figure,
        }
        current_tab = self.tabview.get()
        fig = tab_to_figure.get(current_tab)
        if fig is None:
            messagebox.showinfo(
                "Save Image",
                "Switch to a tab with a plot first (Kinetic Plot, KTH Model, "
                "or Model Fitting), then click Save Image.",
            )
            return

        default_name = current_tab.replace(" ", "_") + ".png"
        path = filedialog.asksaveasfilename(
            title="Save current plot as image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG image", "*.png"),
                ("TIFF image", "*.tif *.tiff"),
                ("PDF document", "*.pdf"),
                ("SVG vector image", "*.svg"),
            ],
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower().lstrip(".")
        try:
            with plotter.export_theme(fig, "light"):
                if ext in ("tif", "tiff"):
                    fig.savefig(
                        path, dpi=300, bbox_inches="tight",
                        pil_kwargs={"compression": "tiff_lzw"},
                    )
                elif ext == "png":
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                else:  # pdf, svg, or anything else matplotlib understands
                    fig.savefig(path, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save Image error", str(exc))
            return

        self._set_status(f"Image saved to {path}")
        messagebox.showinfo("Image saved", f"Saved to:\n{path}")

    def _on_show_about(self) -> None:
        """Small, static About window: what the app does and how to reach
        the developer. A normal modal dialog -- no background work."""
        about_win = ctk.CTkToplevel(self)
        about_win.title("About")
        about_win.geometry("470x340")
        about_win.resizable(False, False)
        about_win.transient(self)
        about_win.grab_set()

        ctk.CTkLabel(
            about_win, text=APP_NAME_FOR_ABOUT,
            font=ctk.CTkFont(size=17, weight="bold"),
        ).pack(pady=(22, 2), padx=20)
        ctk.CTkLabel(
            about_win, text=f"v{isokinetics_version}", text_color=("gray30", "gray70"),
        ).pack()

        ctk.CTkLabel(
            about_win, text=ABOUT_OBJECTIVE_TEXT, justify="left", anchor="w",
        ).pack(pady=(18, 18), padx=28, fill="x")

        ctk.CTkLabel(
            about_win, text="Developer contact",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack()
        ctk.CTkLabel(
            about_win, text=DEVELOPER_CONTACT_EMAIL, text_color=("#1a5fb4", "#62a0ea"),
        ).pack(pady=(0, 18))

        ctk.CTkButton(about_win, text="Close", width=100, command=about_win.destroy).pack(
            pady=(0, 16)
        )

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    def _on_close(self) -> None:
        """Handle the window's close button (WM_DELETE_WINDOW)."""
        self._shutdown()

    def destroy(self) -> None:
        """
        Override so that *any* code path that closes this window goes
        through the same clean shutdown sequence exactly once.  Calling
        this more than once is safe (``tkinter.Tk.destroy`` itself is not
        idempotent and raises TclError on a second call).
        """
        if self._alive:
            self._shutdown()

    def _shutdown(self) -> None:
        """
        Clean shutdown sequence, run exactly once:

        1. Mark the app as no longer alive, so already-queued ``after``
           callbacks from background threads become no-ops.
        2. Signal the worker threads to stop and give them a brief moment
           (they are daemon threads, so this is a courtesy only).
        3. Close child windows (e.g. About) first -- leaving a modal
           Toplevel open while the main window tears down can raise
           ``TclError: can't delete Tcl command`` in some sequences.
        4. Cancel every pending ``after``/``after_idle`` callback,
           including customtkinter's own polling loops, so none fires
           after its widgets are gone (see :mod:`isokinetics._tk_utils`).
        5. Destroy the window.
        """
        self._alive = False
        self._stop_event.set()

        for thread in (self._analysis_thread, self._validation_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.5)

        for child in list(self.winfo_children()):
            if isinstance(child, tk.Toplevel):
                try:
                    child.destroy()
                except Exception:  # noqa: BLE001 - best-effort during teardown
                    pass

        cancel_all_pending_after_callbacks(self)

        try:
            super().destroy()
        except Exception:  # noqa: BLE001
            # The window is being torn down regardless; an isolated failure
            # to delete one internal Tcl command during that cascade (a rare
            # customtkinter/Tcl interaction) must never surface to the user.
            logger.debug("Non-fatal error while destroying the main window.", exc_info=True)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _read_betas(self) -> List[float]:
        betas = []
        for i, entry in enumerate(self.beta_entries):
            text = entry.get().strip()
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"Heating rate #{i + 1} ('{text}') is not a valid number.")
            if value <= 0:
                raise ValueError(f"Heating rate #{i + 1} must be a positive number.")
            betas.append(value)
        if len(set(betas)) < len(betas):
            raise ValueError("The four heating rates must all be different.")
        return betas

    def _set_running_state(self, running: bool) -> None:
        self.is_running = running
        state = "disabled" if running else "normal"
        self.run_button.configure(
            state=state, text="Running\u2026" if running else "\u25b6  Run Analysis")
        self.load_button.configure(state=state)
        if self.sample_button is not None:
            self.sample_button.configure(state=state)
        self.validate_button.configure(state=state)
        self.reprocess_button.configure(state=state)
        self.save_processed_button.configure(state=state)
        for entry in self.beta_entries:
            entry.configure(state=state)

    def _set_status(self, message: str) -> None:
        self.status_label.configure(text=message)


def run() -> None:
    """Instantiate and start the GUI main loop."""
    app = KineticsApp()
    app.mainloop()
