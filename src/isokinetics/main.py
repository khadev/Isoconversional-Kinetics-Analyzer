"""
main.py
=======
Entry point for the Isoconversional Kinetics Analyzer desktop
application. Shows a branded splash screen (progress bar + author
credit) while the app initializes, then opens the main window.

Run with one of::

    isokinetics-gui            # after `pip install -e .` / `pip install isoconversional-kinetics`
    python -m isokinetics       # from a source checkout, no install needed (see note below)
    python run.py               # convenience launcher, from the repo root

Do **not** run this file directly with ``python src/isokinetics/main.py``
-- see the note in :func:`main` below for why, and what to do instead.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Launch the GUI application (splash screen, then main window).

    Returns
    -------
    int
        Process exit code (0 on success, 1 on failure).
    """
    if __package__ in (None, ""):
        # This happens when the file is executed directly, e.g.
        # `python src/isokinetics/main.py`, rather than imported as part
        # of the `isokinetics` package. In that mode, the relative
        # imports used throughout this package (e.g. `from .gui import
        # run` / `from . import gui`) have no parent package to resolve
        # against, and Python raises:
        #   ImportError: attempted relative import with no known parent package
        # Rather than let that confusing error surface, explain the fix.
        print(
            "This file must be run as part of the 'isokinetics' package, "
            "not as a standalone script.\n\n"
            "Please use one of the following instead:\n"
            "  1)  python run.py                 (from the repository root)\n"
            "  2)  python -m isokinetics          (from the repository root, "
            "or after `pip install -e .`)\n"
            "  3)  isokinetics-gui                (after `pip install -e .` "
            "or `pip install isoconversional-kinetics`)\n",
            file=sys.stderr,
        )
        return 1

    try:
        from .splash import show_splash_then_launch
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to import GUI dependencies: {exc}", file=sys.stderr)
        print(
            "Make sure all dependencies are installed: "
            "pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    show_splash_then_launch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
