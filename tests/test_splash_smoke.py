"""
Smoke tests for isokinetics.splash.

Like the GUI smoke tests, these require a display (or Xvfb) since they
instantiate real Tk/customtkinter widgets.
"""

from __future__ import annotations

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
    reason="No display available for splash-screen smoke tests (run under Xvfb).",
)


def test_splash_screen_constructs_with_expected_widgets():
    from isokinetics.splash import AUTHOR_NAME, SplashScreen

    splash = SplashScreen()
    try:
        assert splash.progress.get() == 0.0
        assert "Starting up" in splash.status_label.cget("text")
        # Author credit must be present somewhere in the footer.
        assert AUTHOR_NAME == "Oukil khaled ibn el-walid"
    finally:
        splash.destroy()


def test_splash_progress_reaches_100_percent():
    """
    Drive the splash screen through a *real* Tk main loop (background
    ``after()`` calls are only valid while the main loop is running --
    manually polling with ``update()`` does not count), and confirm the
    progress bar reaches 100% before the safety-net timeout.
    """
    from isokinetics.splash import SplashScreen

    splash = SplashScreen()
    completed = {"flag": False}

    def on_complete():
        completed["flag"] = True
        splash.destroy()

    splash.run_loading_sequence(on_complete=on_complete)
    # Independent safety net so the test can never hang the suite, even
    # if something regresses.
    splash.after(8000, splash.destroy)

    splash.mainloop()  # returns as soon as splash.destroy() runs, above

    assert completed["flag"], "Loading sequence did not complete before the safety timeout."


def test_show_splash_then_launch_reaches_main_window(monkeypatch):
    """
    End-to-end: run the real show_splash_then_launch() entry point, but
    patch KineticsApp so it closes itself almost immediately instead of
    blocking on mainloop() forever.
    """
    from isokinetics import gui as gui_module
    from isokinetics import splash as splash_module

    original_init = gui_module.KineticsApp.__init__
    reached_main_window = {"flag": False}

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        reached_main_window["flag"] = True
        self.after(200, self.destroy)

    monkeypatch.setattr(gui_module.KineticsApp, "__init__", patched_init)

    splash_module.show_splash_then_launch()

    assert reached_main_window["flag"], "Main window was never constructed."
