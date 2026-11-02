"""
_tk_utils.py
============
Small internal helpers shared by the splash screen and the main window
for clean Tkinter/customtkinter shutdown.

Background
----------
customtkinter schedules a couple of *recurring* internal ``after()``
loops on every root window it creates -- most notably
``ScalingTracker.check_dpi_scaling`` and ``AppearanceModeTracker.update``
(both reschedule themselves every ~100ms for the lifetime of the
window). If a window is destroyed while one of these (or any of our
own ``self.after(...)`` callbacks) is still pending, Tcl will
eventually try to invoke a Tcl command that no longer exists once that
timer fires, and prints directly to stderr::

    invalid command name "...check_dpi_scaling"
      while executing
    "...check_dpi_scaling"
      ("after" script)

This is not a Python exception we can ``try/except`` around -- it's
Tcl's own error reporting for a stale ``after`` event -- so the correct
fix is to make sure nothing is left pending in the first place, by
cancelling every scheduled ``after``/``after_idle`` callback *before*
the widget hierarchy is destroyed.
"""

from __future__ import annotations


def cancel_all_pending_after_callbacks(widget) -> None:
    """
    Cancel every ``after``/``after_idle`` callback still scheduled on
    ``widget``'s underlying Tcl interpreter.

    This cancels callbacks *we* scheduled (e.g. progress-bar updates,
    splash-screen safety-net timers) as well as ones customtkinter
    schedules internally for itself -- both go through the exact same
    Tk ``after`` mechanism, so a single ``after info`` query finds all
    of them regardless of who registered them.

    Parameters
    ----------
    widget : tkinter.Misc
        Any live widget belonging to the interpreter to clean up --
        typically the root window (``CTk``/``Tk`` instance) that is
        about to be destroyed.

    Notes
    -----
    Safe to call multiple times, and safe to call during teardown when
    parts of the widget tree may already be gone: every step is
    wrapped so a failure to cancel one stale id never prevents the
    rest from being cleaned up.
    """
    try:
        pending_ids = widget.tk.eval("after info").split()
    except Exception:
        return

    for after_id in pending_ids:
        # Use the raw Tcl ``after cancel`` rather than ``widget.after_cancel``.
        # The Python wrapper also deletes the Tcl command that backs the
        # callback -- but only from *this* widget's bookkeeping.  If a nested
        # customtkinter widget registered the callback itself, that widget
        # still lists the command and tries to delete it again when it is
        # destroyed, raising ``TclError: can't delete Tcl command`` and
        # leaving the main window alive.  Cancelling the timer alone stops
        # the callback from ever firing; the owner then cleans up normally.
        try:
            widget.tk.call("after", "cancel", after_id)
        except Exception:
            pass
