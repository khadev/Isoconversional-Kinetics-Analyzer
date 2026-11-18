"""
splash.py
=========
A branded splash screen shown while the application starts up.

Design goals
------------
* Frameless, centered window with a clean dark theme (independent of
  the user's OS light/dark setting, so branding always looks the same).
* A small vector "activation energy curve" logo drawn on a
  ``tkinter.Canvas`` (no external image assets required).
* A determinate progress bar that advances through real startup stages
  (importing NumPy/pandas, initialising Matplotlib's Tk backend,
  building the main window) rather than a fake/cosmetic animation.
* Author credit and version, always visible at the bottom.

The heavy lifting happens in a background thread (see
:meth:`SplashScreen.run_loading_sequence`) so the progress bar animates
smoothly instead of freezing while each stage's imports run.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import customtkinter as ctk

from . import __version__
from ._tk_utils import cancel_all_pending_after_callbacks

logger = logging.getLogger(__name__)

APP_NAME = "Isoconversional Kinetics Analyzer"
APP_TAGLINE = "Friedman \u2022 FWO \u2022 KAS \u2022 Starink \u2022 Vyazovkin"
AUTHOR_NAME = "Oukil khaled ibn el-walid"

#: (status text, target progress fraction, minimum time to spend on this
#: stage in seconds -- purely for a smooth, readable animation; the real
#: work done at each stage may finish faster or slower than this).
_LOADING_STAGES = [
    ("Starting up...", 0.10, 0.15),
    ("Loading numerical engine (NumPy / pandas)...", 0.40, 0.25),
    ("Preparing plotting backend (Matplotlib)...", 0.70, 0.25),
    ("Building user interface...", 0.92, 0.25),
    ("Ready.", 1.0, 0.15),
]

_BG_COLOR = "#161625"
_ACCENT_COLOR = "#5B8CFF"
_ACCENT_COLOR_2 = "#8B5CF6"
_TEXT_COLOR = "#F2F2F7"
_MUTED_COLOR = "#9A9AB0"


class SplashScreen(ctk.CTk):
    """A frameless, centered splash window with a progress bar."""

    WIDTH = 560
    HEIGHT = 340

    def __init__(self) -> None:
        super().__init__()

        self._destroyed = False
        self.overrideredirect(True)  # no OS title bar / borders
        self.configure(fg_color=_BG_COLOR)
        self.attributes("-topmost", True)
        self._center_on_screen()

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _center_on_screen(self) -> None:
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - self.WIDTH) // 2
        y = (screen_h - self.HEIGHT) // 2
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x}+{y}")

    def _build_ui(self) -> None:
        # Thin accent border effect: an outer colored frame with a small
        # inset holding the actual dark content frame.
        outer = ctk.CTkFrame(self, fg_color=_ACCENT_COLOR, corner_radius=0)
        outer.pack(fill="both", expand=True)

        content = ctk.CTkFrame(outer, fg_color=_BG_COLOR, corner_radius=0)
        content.pack(fill="both", expand=True, padx=2, pady=2)

        # --- Logo (drawn on a Canvas so no image assets are required) ---
        logo_canvas = ctk.CTkCanvas(
            content, width=88, height=88, bg=_BG_COLOR, highlightthickness=0
        )
        logo_canvas.pack(pady=(34, 6))
        self._draw_logo(logo_canvas)

        # --- Title & tagline ---
        ctk.CTkLabel(
            content,
            text=APP_NAME,
            font=ctk.CTkFont(size=21, weight="bold"),
            text_color=_TEXT_COLOR,
        ).pack(pady=(2, 0))

        ctk.CTkLabel(
            content,
            text=APP_TAGLINE,
            font=ctk.CTkFont(size=12),
            text_color=_MUTED_COLOR,
        ).pack(pady=(2, 22))

        # --- Progress bar + percentage ---
        progress_row = ctk.CTkFrame(content, fg_color="transparent")
        progress_row.pack(fill="x", padx=60)

        self.progress = ctk.CTkProgressBar(
            progress_row,
            height=8,
            corner_radius=4,
            progress_color=_ACCENT_COLOR,
            fg_color="#2A2A3D",
        )
        self.progress.pack(side="left", fill="x", expand=True)
        self.progress.set(0.0)

        self.percent_label = ctk.CTkLabel(
            progress_row,
            text="0%",
            width=40,
            font=ctk.CTkFont(size=12),
            text_color=_MUTED_COLOR,
        )
        self.percent_label.pack(side="left", padx=(10, 0))

        self.status_label = ctk.CTkLabel(
            content,
            text="Starting up...",
            font=ctk.CTkFont(size=12),
            text_color=_MUTED_COLOR,
        )
        self.status_label.pack(pady=(10, 0))

        # --- Footer: author + version ---
        footer = ctk.CTkFrame(content, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=20, pady=(0, 16))

        ctk.CTkLabel(
            footer,
            text=f"Author: {AUTHOR_NAME}",
            font=ctk.CTkFont(size=11),
            text_color=_MUTED_COLOR,
        ).pack(side="left")

        ctk.CTkLabel(
            footer,
            text=f"v{__version__}",
            font=ctk.CTkFont(size=11),
            text_color=_MUTED_COLOR,
        ).pack(side="right")

    def _draw_logo(self, canvas) -> None:
        """
        Draw a small stylised "activation-energy curve" logo: a rising
        curve (evoking the Ea(alpha) plots this app produces) inside a
        rounded accent-colored circle.
        """
        size = 88
        pad = 6
        canvas.create_oval(
            pad, pad, size - pad, size - pad, outline=_ACCENT_COLOR, width=2.5
        )
        # A smooth rising polyline reminiscent of an Ea(alpha) curve.
        points = [
            (20, 62),
            (30, 58),
            (38, 50),
            (46, 46),
            (54, 36),
            (62, 30),
            (68, 20),
        ]
        flat_points = [coord for point in points for coord in point]
        canvas.create_line(
            *flat_points,
            fill=_ACCENT_COLOR_2,
            width=3,
            smooth=True,
            capstyle="round",
        )
        for x, y in (points[0], points[-1]):
            canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3, fill=_ACCENT_COLOR, outline=""
            )

    # ------------------------------------------------------------------ #
    # Progress driving
    # ------------------------------------------------------------------ #
    def _set_stage(self, text: str, fraction: float) -> None:
        """Thread-safe update of the progress bar/labels (marshals to the
        Tk main loop via ``after``).

        Guards against a real race reported in production (Windows,
        Python 3.13): the background worker thread calls this once per
        stage, each call scheduling a *new* ``self.after(0, _apply)``.
        If the splash is destroyed (main window ready, or the app
        closing) in between two stages, a freshly-scheduled ``_apply``
        can still fire afterwards -- cancelling *already-pending*
        callbacks in ``destroy()`` (see :mod:`isokinetics._tk_utils`)
        does not protect against a *new* one being scheduled after that
        point. The fix is this explicit ``self._destroyed`` check,
        mirroring the ``_alive`` guard already used by ``KineticsApp``
        for the exact same class of race.
        """

        def _apply() -> None:
            if self._destroyed:
                return
            self.progress.set(fraction)
            self.percent_label.configure(text=f"{int(fraction * 100)}%")
            self.status_label.configure(text=text)

        if not self._destroyed:
            self.after(0, _apply)

    def run_loading_sequence(self, on_complete: Callable[[], None]) -> None:
        """
        Advance the splash screen through its loading stages in a
        background thread, performing the real (lazy) imports that the
        main application needs, then invoke ``on_complete`` back on the
        main thread once finished.

        Robustness notes
        -----------------
        * The worker thread is started via ``self.after(...)`` rather
          than directly, so it only begins once the Tk main loop is
          definitely running. ``Tk.after()`` callbacks scheduled from a
          background thread require the main loop to be active (Tkinter
          raises ``RuntimeError: main thread is not in main loop``
          otherwise); starting the thread from inside an ``after``
          callback guarantees that ordering instead of racing against
          ``mainloop()`` being called.
        * A safety-net timer guarantees ``on_complete`` fires at most
          once, even if a stage raises an unexpected exception or an
          import hangs -- the user is never left stuck on the splash
          screen indefinitely.

        Parameters
        ----------
        on_complete : Callable[[], None]
            Called (on the Tk main thread) once every stage has finished
            (or the safety-net timeout elapses, whichever is first).
        """
        already_called = {"flag": False}
        scheduled_after_ids: list = []

        def call_once() -> None:
            if already_called["flag"]:
                return
            already_called["flag"] = True
            # Whichever of {start_worker, the 20s safety-net} did NOT
            # fire is still pending at this point -- cancel it now
            # rather than leaving it scheduled against a window that
            # may be destroyed moments from now (see _tk_utils for why
            # that matters).
            for after_id in scheduled_after_ids:
                try:
                    self.after_cancel(after_id)
                except Exception:
                    pass
            on_complete()

        def worker() -> None:
            try:
                for text, target_fraction, min_duration in _LOADING_STAGES:
                    start = time.monotonic()
                    self._set_stage(text, target_fraction)
                    self._do_stage_work(text)
                    elapsed = time.monotonic() - start
                    if elapsed < min_duration:
                        time.sleep(min_duration - elapsed)
            except Exception:
                # A hiccup in a startup import must never strand the
                # user on the splash screen -- fall through to
                # call_once() below regardless.
                pass
            self.after(0, call_once)

        def start_worker() -> None:
            threading.Thread(target=worker, daemon=True).start()

        # Defer starting the thread until the main loop is guaranteed to
        # be running (see docstring above), and install an independent
        # safety-net in case anything above goes wrong.
        scheduled_after_ids.append(self.after(20, start_worker))
        scheduled_after_ids.append(self.after(20_000, call_once))

    def destroy(self) -> None:
        """
        Cancel any callbacks this window (or customtkinter internally)
        still has scheduled before tearing down the widget hierarchy,
        to avoid stray "invalid command name ... (\"after\" script)"
        errors on shutdown. See :mod:`isokinetics._tk_utils` for why
        this is necessary.

        Idempotent: a second call is a safe no-op. ``tkinter.Tk.destroy()``
        itself raises ``TclError`` if invoked again after the Tcl
        interpreter has actually been torn down, so we must guard
        against calling it twice.
        """
        if self._destroyed:
            return
        self._destroyed = True
        cancel_all_pending_after_callbacks(self)
        try:
            super().destroy()
        except Exception:  # noqa: BLE001
            # The splash is being torn down regardless; an isolated
            # Tcl-level teardown error here must never surface as a
            # crash (see the matching note in gui.py's _shutdown()).
            logger.debug(
                "Non-fatal error while destroying the splash screen.", exc_info=True
            )

    @staticmethod
    def _do_stage_work(stage_text: str) -> None:
        """
        Perform the real work associated with a loading stage. Imports
        are cheap to repeat (Python caches modules in ``sys.modules``),
        so this both gives an accurate "what's happening" message and
        genuinely front-loads the cost of first-use imports before the
        main window appears.
        """
        if "numerical engine" in stage_text:
            import numpy  # noqa: F401
            import pandas  # noqa: F401
        elif "plotting backend" in stage_text:
            import matplotlib  # noqa: F401
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: F401
        elif "user interface" in stage_text:
            from . import gui  # noqa: F401


def show_splash_then_launch() -> None:
    """
    Show the splash screen, run the loading sequence, then destroy the
    splash and launch the main application window.

    This is the standard entry point used by :func:`isokinetics.main.main`.
    """
    splash = SplashScreen()

    def _launch_main_app() -> None:
        splash.destroy()
        from .gui import KineticsApp

        app = KineticsApp()
        app.mainloop()

    splash.run_loading_sequence(on_complete=_launch_main_app)
    splash.mainloop()
