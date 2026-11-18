"""
Tests for the application entry point: the run.py launcher, the
`python -m isokinetics` path, and the friendly diagnostic message shown
if someone runs main.py directly (which cannot work, by design -- see
main.py's module docstring).

These use subprocess so they exercise exactly what a user would type in
a terminal, without needing a display (they only test the *error path*
/ import wiring, not the GUI itself).
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC_DIR = os.path.join(REPO_ROOT, "src")


def _run(args, **kwargs):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        **kwargs,
    )


def test_running_main_py_directly_gives_helpful_error_not_traceback():
    """
    `python src/isokinetics/main.py` cannot work (relative imports need
    a package context) -- but it must fail with our friendly, actionable
    message, not a raw ImportError traceback.
    """
    result = _run([sys.executable, os.path.join("src", "isokinetics", "main.py")])

    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "run.py" in combined
    assert "python -m isokinetics" in combined
    assert "isokinetics-gui" in combined
    # The old confusing message should NOT be what the user sees.
    assert "attempted relative import" not in combined


def test_module_invocation_imports_cleanly():
    """
    `python -m isokinetics.main` (or `-m isokinetics`) should be able to
    import everything without error up to the point of actually opening
    a GUI window (which needs a display, tested separately under Xvfb).
    We simply verify the import machinery / package wiring is correct.
    """
    result = _run(
        [
            sys.executable,
            "-c",
            "import isokinetics.main; print('IMPORT_OK')",
        ]
    )
    assert "IMPORT_OK" in result.stdout, result.stderr
    assert result.returncode == 0


def test_run_py_launcher_imports_cleanly():
    """
    run.py must be importable (i.e. its sys.path shim + subsequent
    `from isokinetics.main import main` must succeed) without needing
    the package to be pip-installed first.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; "
            "sys.argv = ['run.py']; "
            "mod = runpy.run_path('run.py', run_name='not_main'); "
            "print('MAIN_CALLABLE_OK' if callable(mod['main']) else 'FAIL')",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert "MAIN_CALLABLE_OK" in result.stdout, result.stderr
