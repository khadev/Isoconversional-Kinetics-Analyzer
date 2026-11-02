#!/usr/bin/env python3
"""
run.py
======
Convenience launcher for people who have NOT installed the package
(i.e. have not run ``pip install -e .``) and just want to run the app
straight from a source checkout.

Why this file exists
---------------------
``src/isokinetics/main.py`` uses a *relative* import
(``from .gui import run``) because it lives inside the ``isokinetics``
package, alongside ``gui.py``, ``kinetics.py``, etc. Relative imports
only work when the file is imported *as part of its package* -- they
fail with::

    ImportError: attempted relative import with no known parent package

if the file is executed directly as a script, e.g.::

    python src/isokinetics/main.py      # <-- fails, no package context

This launcher avoids that problem entirely: it adds ``src/`` to
``sys.path`` and then *imports* ``isokinetics.main`` as a module (rather
than running ``main.py`` as the top-level script), so Python sees
``isokinetics`` as a proper package and every relative import inside it
resolves correctly.

Usage
-----
From the repository root::

    python run.py

This is equivalent to (and exists purely as a convenience alongside):

    pip install -e .
    isokinetics-gui
    # or:
    python -m isokinetics
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from isokinetics.main import main  # noqa: E402  (import after sys.path fix, by design)

if __name__ == "__main__":
    sys.exit(main())
