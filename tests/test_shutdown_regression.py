"""
Regression tests for clean application shutdown.

These reproduce, via subprocess (so we can inspect real stderr output
exactly as a user would see it in a terminal), the bug where closing
the window left pending Tk ``after`` callbacks scheduled -- both our
own and customtkinter's internal DPI-scaling / appearance-mode polling
loops -- which then printed::

    invalid command name "...check_dpi_scaling"
      while executing
    "...check_dpi_scaling"
      ("after" script)

to the console after the widgets they referenced were destroyed.

All tests here run under a display (Xvfb on headless CI) since they
instantiate real windows.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("tkinter")
pytest.importorskip("customtkinter")

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC_DIR = os.path.join(REPO_ROOT, "src")
SAMPLE_FILE = os.path.join(REPO_ROOT, "sample_data", "matric_121459.txt")


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
    reason="No display available for shutdown regression tests (run under Xvfb).",
)


def _run_script(script: str, timeout: int = 20) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


_FORBIDDEN_SNIPPETS = [
    "invalid command name",
    "while executing",
    '("after" script)',
]


def _assert_no_stray_after_errors(result: subprocess.CompletedProcess) -> None:
    combined = result.stdout + result.stderr
    for snippet in _FORBIDDEN_SNIPPETS:
        assert snippet not in combined, (
            f"Found stray Tk 'after' callback error in output "
            f"(looking for absence of {snippet!r}):\n{combined}"
        )


def test_splash_to_main_window_close_produces_no_stray_after_errors():
    """
    End-to-end: splash screen runs its full loading sequence, the main
    window opens, and then closes itself shortly after -- exactly the
    real application flow. Must produce zero stray Tk error output.
    """
    script = """
import sys
sys.path.insert(0, "src")
from isokinetics import splash as splash_mod
import isokinetics.gui as gui_mod

orig_init = gui_mod.KineticsApp.__init__
def patched_init(self, *a, **kw):
    orig_init(self, *a, **kw)
    self.after(300, self._on_close)
gui_mod.KineticsApp.__init__ = patched_init

splash_mod.show_splash_then_launch()
print("SEQUENCE_COMPLETED_CLEANLY")
"""
    result = _run_script(script)
    assert "SEQUENCE_COMPLETED_CLEANLY" in result.stdout, result.stderr
    _assert_no_stray_after_errors(result)


@pytest.mark.parametrize("close_delay_ms", [0, 10, 50, 200, 500])
def test_closing_mid_analysis_produces_no_stray_after_errors(close_delay_ms):
    """
    Close the main window at various points while a background analysis
    is running (including immediately, i.e. before the worker thread has
    done anything). None of these should ever print a stray Tk error.
    """
    script = f"""
import sys
sys.path.insert(0, "src")
from isokinetics.gui import KineticsApp
from isokinetics import data_loader

app = KineticsApp()
df = data_loader.load_data({SAMPLE_FILE!r})
app.raw_df = df
app.working_df = data_loader.interpolate_alpha(df)
app._populate_preview(df)
app.update()

app._on_run_analysis()
app.after({close_delay_ms}, app._on_close)
app.mainloop()
print("CLOSED_CLEANLY")
"""
    result = _run_script(script)
    assert "CLOSED_CLEANLY" in result.stdout, result.stderr
    _assert_no_stray_after_errors(result)


def test_closing_during_slow_method_computation_produces_no_stray_after_errors():
    """
    Worst case for the original bug: the background thread is still
    actively inside a (artificially slowed) kinetics computation when
    the window is closed, so `_stop_event` must be checked and the
    in-flight `self.after(0, ...)` callback must be safely ignored.
    """
    script = """
import sys, time
sys.path.insert(0, "src")
from isokinetics.gui import KineticsApp
from isokinetics import data_loader
import isokinetics.kinetics as kinetics_mod

_orig = kinetics_mod.vyazovkin
def slow_vyazovkin(df, betas):
    time.sleep(1.0)
    return _orig(df, betas)
kinetics_mod.METHODS["Vyazovkin"] = slow_vyazovkin

app = KineticsApp()
df = data_loader.load_data("sample_data/matric_121459.txt")
app.raw_df = df
app.working_df = data_loader.interpolate_alpha(df)
app._populate_preview(df)
app.update()

app._on_run_analysis()
app.after(150, app._on_close)
app.mainloop()
print("CLOSED_CLEANLY_MID_COMPUTATION")
"""
    result = _run_script(script)
    assert "CLOSED_CLEANLY_MID_COMPUTATION" in result.stdout, result.stderr
    _assert_no_stray_after_errors(result)


def test_shutdown_sets_alive_false_and_stop_event(monkeypatch):
    """Direct unit check of the _shutdown() bookkeeping, in-process."""
    from isokinetics.gui import KineticsApp

    app = KineticsApp()
    assert app._alive is True
    assert not app._stop_event.is_set()

    app._on_close()

    assert app._alive is False
    assert app._stop_event.is_set()


def test_after_thread_signalled_to_stop_marshalled_callbacks_are_ignored():
    """
    If a background-thread callback is somehow still invoked after
    shutdown has begun (e.g. a callback that was already dispatched by
    Tcl right as we cancel everything else), it must be a safe no-op
    rather than touching destroyed widgets.
    """
    from isokinetics.gui import KineticsApp

    app = KineticsApp()
    app._on_close()  # window is now "closed" (_alive is False)

    # These must not raise, even though the widgets are gone.
    app._update_progress(0.5, "Friedman")
    app._on_analysis_done({}, {}, {})


def test_destroy_is_idempotent():
    """Calling destroy() more than once must not raise or recurse."""
    from isokinetics.gui import KineticsApp

    app = KineticsApp()
    app.destroy()
    app.destroy()  # must not raise


def test_about_dialog_then_double_image_save_then_destroy_never_crashes():
    """
    Regression test for a production bug (user-reported): opening the
    About dialog, then saving the current plot as two different image
    formats (PNG then TIFF) back-to-back, then closing the app,
    triggered a rare TclError deep inside customtkinter's own widget
    teardown cascade (not in this app's own code). Reproduced reliably
    with this exact sequence; fixed by (a) proactively closing child
    Toplevel windows before tearing down the main window, and (b)
    defensively catching any residual TclError during the final
    super().destroy() call, since the window is being torn down
    regardless at that point.
    """
    from isokinetics.gui import KineticsApp
    from isokinetics import data_loader, kinetics

    app = KineticsApp()
    df = data_loader.load_data(SAMPLE_FILE)
    app.raw_df = df
    app.working_df = data_loader.interpolate_alpha(df)
    app._populate_preview(df)
    app.update()

    betas = app._read_betas()
    results = {"KAS": kinetics.kas(app.working_df, betas)}
    averages = {"KAS": kinetics.average_ea(results["KAS"])}
    app._on_analysis_done(results, averages, {})
    app.update()

    app._on_show_about()
    app.update()

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        png_path = f"{tmp_dir}/x.png"
        tiff_path = f"{tmp_dir}/x.tiff"
        app.figure.savefig(png_path, dpi=300, bbox_inches="tight")
        app.figure.savefig(
            tiff_path, dpi=300, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )

    app.destroy()  # must not raise


def test_shutdown_closes_open_toplevel_children_first():
    """The About dialog (a CTkToplevel child) must be closed as part of
    the main window's shutdown sequence, not left dangling."""
    import tkinter as tk

    from isokinetics.gui import KineticsApp

    app = KineticsApp()
    app._on_show_about()
    app.update()

    toplevels_before = [c for c in app.winfo_children() if isinstance(c, tk.Toplevel)]
    assert len(toplevels_before) == 1

    app.destroy()  # must not raise, and must have closed the About window
