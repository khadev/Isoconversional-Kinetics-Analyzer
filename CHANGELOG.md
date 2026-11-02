# Changelog

All notable changes to this project are documented in this file.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.9.0] - 2026-10-12

### Changed
- The raw-file dialog now pre-selects the row layout suggested by the row
  count (`data_loader.suggest_mapping`: 99/49/199 ... rows -> *interior*), because
  the wrong layout shifts every temperature by up to one step.

### Added
- **Raw-data import (replaces the manual Origin step).** *Load Data* now
  detects the file type from its columns:
  - 5 columns `[alpha, T1..T4]` -> *processed*, analysed directly (unchanged);
  - 4 columns `[T1..T4]` -> *raw*: a dialog asks for the conversion step
    (`0.01`, `0.02`, `1/50`, or a number of points such as `100`), alpha is
    generated and the temperatures are linearly interpolated at each alpha.
  The result is a clean processed table that goes straight into the analysis.
- Dialog with live preview of the grid ("49 points, alpha = 0.020 ... 0.980"),
  a choice of how the rows sit on the alpha axis (first row alpha = 0 / last
  alpha = 1, or interior points alpha = i/(n+1)), and a warning when the step
  is finer than the data.
- *Change alpha step...* and *Save processed table...* for raw files; the
  chosen step is remembered and written to the export metadata.
- `data_loader.load_table`, `validate_raw_data`, `parse_alpha_step`,
  `alpha_grid`, `build_alpha_table`, `save_processed_table`.
- Raw files may have columns of different lengths, a header line, tab / space /
  comma / semicolon separators, or be Excel workbooks.
- `sample_data/sample_raw_4columns_*.txt`; `tests/test_raw_data.py`.

### Changed
- A file with too few columns now reports how many columns were found.
- `load_data` / `validate_data` are unchanged (5-column files only).

## [1.8.0] - 2026-10-12

### Added
- **Lifetime tab** (`isokinetics.lifetime`): Arrhenius shelf-life
  extrapolation. Enter storage temperature, time at that temperature and
  service temperature (degC or K; years/days/hours/minutes) to get the
  predicted service lifetime in years, days and minutes, the acceleration
  factor and the lifetime with Ea +/-10 %.
- Ea source defaults to the **recommended Coats-Redfern model**; falls back
  to a model-free mean (with an explanatory note) when no model is
  recommended; other methods, other models and manual entry selectable.
- Warnings for extrapolation far outside the measured temperature range,
  strongly variable Ea(alpha), and extreme Ea sensitivity.
- `formatting.fmt_years / fmt_days / fmt_minutes`.
- `tests/test_lifetime.py` (32 tests).

### Fixed
- customtkinter drops a tab selection made within 100 ms of another one,
  which could leave a tab highlighted but blank (`KineticsApp._select_tab`).

## [1.7.0] - 2026-10-11

### Changed (interface)
- Redesigned GUI: header with **dark/light theme switch**, sidebar workflow
  (Data, Heating rates, Methods, Display) with pinned Run/Export/Save
  buttons, reorganised tabs (new **Summary** and **Validation** tabs),
  themed tables with zebra rows, and a detailed status bar (stage, %,
  elapsed, time remaining).
- One number-format policy for the GUI, plots and exports
  (`isokinetics.formatting`).
- Images and figure exports are always written on a white background,
  whatever theme is on screen (`plotter.export_theme`).
- Export now also writes `Summary.txt`, a `Model_Fitting` Excel sheet,
  Excel number formats, and `Coats_Redfern_plot.*`.

### Fixed (kinetics)
- **Vyazovkin**: fixed 300 K lower integration limit gave wrong Ea for data
  near or below 300 K / low Ea; Ea was silently capped at 500 kJ/mol; the
  "advanced" method was the 1997 whole-range form. Now the windowed (2001)
  form with exact quadrature, automatic bracket widening and NaN for
  degenerate points. `advanced=False` keeps the old form.
- **Coats-Redfern**: restored the `(1 - 2RT/E)` term; A was ~10 % low.
- Temperature integral `p(x)` for Malek's method: ~1e-4 -> ~1e-13.
- Gas constant now CODATA 8.314462618 J/(mol K).
- Shutdown: `_tk_utils.cancel_all_pending_after_callbacks` now uses the raw
  Tcl `after cancel`; the Python wrapper left stale command names in nested
  customtkinter widgets, so `destroy()` could raise `can't delete Tcl
  command` and the window stayed open (also hung the full test run).

### Added
- `isokinetics.validation` (`python -m isokinetics.validation`): independent
  synthetic ground-truth generator and a 33-check validation report.
- `tests/test_kinetics_ground_truth.py` (54 tests), `test_formatting.py`,
  `test_gui_redesign.py`.
- Heating rates must be distinct; non-blocking warning if their order
  contradicts the data.

## [1.6.2] - 2026-07-16

### Fixed (real user-reported bugs)
- **Progress bar got visually stuck at 95% after a full analysis run**,
  even though the analysis (and any export afterwards) had genuinely
  completed successfully. Root cause: the Coats-Redfern step added in
  v1.6.0 sets progress to a hardcoded 0.95 right before it runs, but
  nothing ever bumped it back up to 100% once it (and the rest of the
  analysis) finished. Fixed by explicitly setting progress to 1.0
  ("Complete") right before `_on_analysis_done` runs. Verified against
  the real (non-mocked) background-thread analysis pipeline via a full
  `mainloop()`-driven run, confirming `progress_bar.get() == 1.0` and
  status text "Analysis complete." at the end.
- **"About" button was hard to see** (it used `fg_color="transparent"`
  with a thin border) **and moved to the top-right corner** of the
  window as requested, using `place(relx=1.0, rely=0.0, anchor="ne")`
  on the root window directly -- independent of the bottom action
  bar's own layout, so it stays pinned to that corner regardless of
  window resizing -- with the same solid, visible button color as
  every other button in the app.
- 3 new regression tests for both fixes.

## [1.6.1] - 2026-07-13

### Added
- **"Save Image..." button**: save the plot in the currently active tab
  (Kinetic Plot, KTH Model, or Model Fitting) directly as PNG, TIFF
  (300 dpi, LZW), PDF, or SVG via a normal file-save dialog -- no full
  export folder, no screenshot required.
- **"About" dialog**: a short, bulleted summary of the app's purpose
  and the developer's contact email (for feature requests / help),
  `gui.APP_NAME_FOR_ABOUT` / `ABOUT_OBJECTIVE_TEXT` /
  `DEVELOPER_CONTACT_EMAIL`.
- `exporter.EXCEL_EXPORT_AVAILABLE`: computed once at import time,
  checked by the GUI to warn upfront if the Excel file will be skipped.
- 15 new tests (About dialog, Save Image incl. TIFF compression check,
  xlsxwriter-missing resilience, the shutdown-crash regression below).

### Fixed (real user-reported bugs)
- **Splash screen `TclError` on close, reported from a production
  Windows run.** The background loading thread schedules one
  progress-bar update per stage; if the splash was destroyed in
  between two stages, a *newly*-scheduled update (not one that was
  already pending, which the existing cancellation logic already
  handled) could still fire afterwards against already-destroyed
  widgets. Fixed by adding the same "check `_destroyed`/`_alive` before
  touching any widget" guard `KineticsApp` already used for the exact
  same class of race, to the splash screen's update callback too.
  Verified with a targeted stress reproduction (destroy immediately
  after starting the loading sequence, 5 repeated runs) before and
  after the fix.
- **Export crashed entirely when the optional `xlsxwriter` package
  wasn't installed**, with a bare `ModuleNotFoundError`, even though
  the PNG/TIFF/SVG/PDF graphs and the tab-delimited `.txt` file had
  already been written successfully by that point in `export_all()`.
  Now caught specifically and skipped gracefully -- a warning dialog
  names the exact `pip install` command needed, everything else still
  exports normally.
- **A rare `TclError` deep inside customtkinter's own widget-teardown
  cascade**, reproduced reliably with a specific sequence (open the
  About dialog, save the current plot as two different image formats
  back-to-back, then close the app). Root-caused to an interaction
  between the About dialog's modal `grab_set()` and matplotlib's TIFF
  save path that is not fully understood at the Tcl level (isolated to
  confirm it wasn't caused by this app's own code -- e.g. saving the
  same or different single format, or closing the About dialog first,
  never reproduced it). Fixed two ways: (1) `_shutdown()` now
  proactively closes any open child `Toplevel` windows before tearing
  down the main window, and (2) the final `super().destroy()` call is
  wrapped defensively, since an isolated failure to individually delete
  one internal Tcl command during a teardown that's happening
  regardless is inconsequential and must never surface as a crash.
  Verified clean across 5 repeated runs of the exact original failing
  sequence after the fix, with completely silent console output
  (the fallback log line is debug-level, invisible unless someone
  explicitly enables debug logging).

## [1.6.0] - 2026-07-12

### Added
- **`kinetics.confidence_interval()`**: 95%/99% CI for Eₐ(α) for
  Friedman/FWO/KAS/Starink, via regression slope standard error and a
  hardcoded Student's t table (no SciPy dependency). Validated with a
  Monte Carlo coverage check (see Fixed, below).
- **GUI**: "Show 95% confidence bands" checkbox (Kinetic Plot tab,
  FWO/KAS/Starink only); "Plot style" dropdown (6 journal presets); new
  **Model Fitting** tab showing Coats-Redfern results (sortable table +
  R² bar chart, green/red pass/fail coloring); Coats-Redfern now runs
  automatically on every analysis.
- **`plotter.PLOT_STYLES`**: ACS, RSC, Nature, Elsevier, Colorblind,
  Monochrome presets; `apply_plot_style()`.
- **TIFF export**: `exporter.save_graph()`/`export_all()` save a 300
  dpi, LZW-compressed `.tiff` by default alongside PNG/SVG/PDF, with
  author/sample/methods/date metadata embedded in the ImageDescription
  tag.
- 33 new tests (confidence intervals incl. Monte Carlo coverage check,
  journal styles, TIFF export/metadata, Model Fitting tab, CI/style GUI
  controls).

### Fixed (found during validation, before shipping)
- **Friedman's numerical differentiation makes its CI statistically
  valid but practically useless.** Discovered via Monte Carlo coverage
  testing: Friedman's CI achieves correct ~94.5% coverage but a mean
  width of ~1300 kJ/mol on realistically noisy data (vs. KAS's ~25
  kJ/mol at ~95% coverage) -- `confidence_interval()`'s docstring and
  the GUI checkbox steer users towards the integral methods instead.
- **The Colorblind/Monochrome plot styles silently did nothing.**
  Setting matplotlib's `axes.prop_cycle` rcParam has no effect when a
  plot call passes an explicit `color=` argument, which
  `update_plot()` always does (to keep each method's color consistent
  across every plot in the app). Fixed by adding a style-specific
  per-method color override table, consulted instead of the rcParam
  cycle; verified the rendered pixel colors actually change between
  styles before considering this done.
- **A real (not false-positive) Coats-Redfern recommendation on the
  bundled sample dataset differs from KTH's "unknown" verdict on the
  same data** (Coats-Redfern recommends A2; KTH's Malek identification
  says unknown). Investigated and confirmed this is not a bug: A2's
  Coats-Redfern fit passes both criteria (R\u00b2=0.991, 15.4% gap from
  the reference Eₐ, under the 20% threshold) — the two methods use
  different ambiguity criteria (a fixed 20% gap threshold vs. a
  runner-up margin ratio), so a genuinely borderline case can land on
  different sides of each. Documented rather than "fixed" (there was
  nothing wrong to fix).

## [1.5.0] - 2026-07-11

### Added
- **`kinetics.coats_redfern()`**: the Coats-Redfern model-fitting integral
  method (`ln(g(alpha)/T^2) = ln(AR/(beta*Ea)) - Ea/(RT)`), fit per
  heating rate and averaged, over 17 candidate models
  (`kinetics.COATS_REDFERN_MODELS` -- the 13 already used by
  `kth_full()`'s Malek identification, plus F4, D5, and the P1-P4
  power-law family, added to the shared `_model_f_g` library).
- 9 new tests, including a dedicated regression test for the R^2
  false-positive found during validation (below).

### Fixed (found during validation, before shipping)
- **R^2 alone is not a safe way to rank Coats-Redfern candidates.**
  Validated against synthetic F1 ground-truth data: model A2 achieves
  R^2 = 0.9999991 (as good as F1's own fit) while recovering an
  activation energy more than 2x too low (70 kJ/mol vs. the true
  150 kJ/mol) -- the well-known "kinetic compensation effect" that
  makes model-fitting methods notoriously unreliable for model
  discrimination on their own. `coats_redfern()` therefore screens
  every candidate's fitted Ea against a model-free reference Ea
  (computed internally via the already-validated `vyazovkin()`, or
  supplied explicitly) and only recommends a model that is both a good
  fit (R^2 > 0.99) *and* within 20% of that reference -- explicitly
  returning `recommended_model=None` rather than guessing when no
  candidate satisfies both. Confirmed this correctly identifies F1 and
  D3 on their respective synthetic datasets while explicitly rejecting
  A2's high-R^2 false positive, and declines to recommend anything on
  non-physical random data.

### Scope note
This release is a deliberately-scoped first slice of a much larger
professionalization request (PyQt6 rewrite, 41 reaction models, ML
prediction, literature database, cloud storage, plugin system, REST
API, etc.). Rather than build all of that shallowly, this release adds
one well-defined, fully-validated capability that fits the existing
architecture with no new hard dependencies. See the project's issue
tracker / conversation history for the full prioritisation discussion:
in short, a GUI framework rewrite conflicts with backward compatibility
and reintroduces the exact heavy-dependency install risk (PyQt6,
scikit-learn, xgboost) that an earlier release specifically fixed for
Windows users without a C++ compiler; a literature database needs real
citations, not fabricated ones; and cloud/ML integrations need real
credentials and real training data to be more than decoration.

## [1.4.0] - 2026-07-10

### Added
- **`kinetics.kth_full()`**: the full KTH kinetic triplet -- Eₐ(α),
  pre-exponential factor A(α), and the experimental reaction model
  f(α) -- via Malek's method (thesis, Chapter VII, §VII.4.1.3).
  - 13 candidate theoretical models (`kinetics.KTH_CANDIDATE_MODELS`):
    reaction order (F0-F3), geometric contraction (R2-R3), diffusion
    (D1-D4), and Avrami-Erofeev nucleation-and-growth (A2-A4).
  - Model identification compares experimental, normalised y(α)/z(α)
    curves against each candidate's theoretical f(α) / f(α)·g(α)
    curves (the standard "master plot" shape-equivalence), and
    explicitly returns `"unknown"` when the best match is either poor
    or too close to the runner-up to trust -- rather than force-fitting
    a model onto ambiguous data. Confirmed on the bundled real sample
    dataset, which legitimately identifies as `"unknown"` (its two best
    candidates differ by under 5%).
  - `kinetics.average_ea`-style outputs for `A(α)` and `f(α)`, plus
    `model_scores`, `y_alpha`/`z_alpha` (experimental) and
    `y_theory`/`z_theory` (identified model's own curves, for plotting).
- **GUI**: a "KTH mode" dropdown ("Eₐ only" / "Full triplet (A, f(α),
  model)"); a new **KTH Model** tab plotting the y(α)/z(α) diagnostic
  curves against the identified model; `KTH_A`/`KTH_f(α)` columns in
  the Results Table.
- **Exporter**: `KTH_A`/`KTH_f_alpha` columns in `Ea_data.txt` and
  `Ea_results.xlsx` (plus a comment line / dedicated `KTH_Model` sheet
  naming the identified model and its match scores), and
  `KTH_model_plot.png/.svg/.pdf` alongside the main graph, whenever
  KTH was run in "Full triplet" mode.
- `logging.debug()` calls throughout model identification (candidate
  scores, ambiguity decisions) via the existing `isokinetics.kinetics`
  logger.
- 24 new tests: ground-truth validation against *two* different
  synthetic datasets (F1 and D3, generated from genuinely different
  reaction models, guarding against a classifier that only ever guesses
  one answer), A(α)/ln(A) recovery, the `"unknown"` fallback on
  non-physical data, GUI full-triplet-mode integration, and exporter
  integration.

### Fixed (found during validation, before shipping)
- **z(α) formula was missing a factor.** A literal reading of
  `z(α) = p(x)·(dα/dt)·(T/β)` -- as commonly abbreviated, including in
  the source thesis -- is *not* proportional to the theoretical
  `f(α)·g(α)` master-plot function; deriving it from first principles
  shows an additional `x·exp(x)` factor is required
  (`z(α) = (dα/dt)·(T/β)·x·exp(x)·p(x)`). Without this fix, model
  identification confidently picked the *wrong* model on synthetic data
  with a known ground truth (e.g. D1 instead of F1). Caught by
  validating against synthetic data before integrating, exactly as with
  the earlier Vyazovkin fix.
- **`p(x)` (temperature integral) computed by direct numerical
  integration**, not the closed-form Senum-Yang-style rational
  approximation as originally specified: a commonly-seen version of
  that approximation (`exp(-x)/x^2` tacked onto the rational prefactor)
  has the wrong asymptotic power of `x` for large `x`, and was found to
  make model identification unreliable. Numerical integration
  (trapezoidal rule, `isokinetics.kinetics._temperature_integral_p`)
  sidesteps the ambiguity and was checked against the exact closed form
  (`p(x) = e^-x/x - E1(x)`) to within 1e-5 relative error during
  development.
- `_identify_model`'s best-model selection now explicitly filters out
  non-finite scores before ranking (Python's `min()` does not reliably
  skip `NaN` values, which could otherwise select a spurious "best"
  model).

## [1.3.0] - 2026-07-09

### Added
- **KTH method** (`kinetics.kth`): a sixth isoconversional method,
  original to the accompanying thesis (Chapter VII), named after its
  authors' initials. Structurally a differential method like Friedman,
  but uses a discrete-step reaction rate `(α_{i+1}-α_i)/Δt_i` over each
  interval on the alpha grid, paired with the mean temperature over
  that interval, rather than an instantaneous derivative at a single
  point. The time step is recovered from the temperature step under
  the linear-heating assumption (no explicit time column required).
  Registered in `kinetics.METHODS`, added to the GUI's method
  checkboxes (`gui.METHOD_NAMES`), and given its own colour/marker in
  `plotter.py` (pink/magenta, "x").
- Validated the same way as every other method here: recovers a known
  150 kJ/mol ground-truth activation energy to within 5% on synthetic
  data (`tests/test_kinetics.py`), plus dedicated tests for its
  boundary handling (the last alpha point, which has no "next" point to
  step to) and its minimum-valid-points requirement (returns NaN with
  fewer than 3 valid heating-rate pairs, per the method's own
  specification -- stricter than the 2-point minimum used by the other
  methods here).
- Added `logging` calls (`isokinetics.kinetics` logger) inside `kth()`
  at DEBUG level, to aid debugging when points are skipped.

## [1.2.0] - 2026-07-08

### Fixed
- **Stray console errors on window close.** Closing the main window
  (or the splash screen) while any Tk `after` callback was still
  scheduled -- our own progress-bar updates, the splash screen's
  safety-net timer, or customtkinter's own internal DPI-scaling /
  appearance-mode polling loops (`ScalingTracker.check_dpi_scaling`,
  `AppearanceModeTracker.update`, both of which reschedule themselves
  every ~100ms for the life of the window) -- could print errors like:
  ```
  invalid command name "...check_dpi_scaling"
    while executing
  "...check_dpi_scaling"
    ("after" script)
  ```
  after the widgets they referenced were destroyed. Root-caused and
  fixed with a new `isokinetics._tk_utils.cancel_all_pending_after_callbacks()`
  helper (queries Tcl's `after info` and cancels every pending id, ours
  and customtkinter's alike), called from both `SplashScreen.destroy()`
  and `KineticsApp`'s new `_shutdown()` sequence before the widget
  hierarchy is torn down. Verified via a subprocess-based regression
  test suite (`tests/test_shutdown_regression.py`) that asserts zero
  such errors across closing at 9 different timing offsets, including
  mid-computation.
- **Background analysis thread now stops cleanly on close.**
  `KineticsApp` gained an `_alive` flag (false as soon as shutdown
  begins) and a `threading.Event` `_stop_event`. The analysis worker
  thread checks `_stop_event` between and during method computations
  and abandons immediately (without touching any widget) if the window
  is closing; `_update_progress`/`_on_analysis_done` additionally check
  `_alive` as a second guard against an already-in-flight callback.
  `_shutdown()` signals the stop event and joins the (daemon) thread
  with a short timeout before destroying the window.
- **`destroy()` is now idempotent** on both `KineticsApp` and
  `SplashScreen`. `tkinter.Tk.destroy()` itself is *not* safe to call
  twice (raises `TclError: can't invoke "destroy" command: application
  has been destroyed` on the second call); both classes now guard
  against this so tests, external callers, or Tkinter's own teardown
  calling `destroy()` more than once can never raise.

## [1.1.0] - 2026-07-08

### Added
- Branded splash screen (`isokinetics.splash`) shown on startup: a
  frameless, centered, dark-themed window with a small vector logo, a
  determinate progress bar tied to real initialization stages (NumPy /
  pandas, Matplotlib's Tk backend, UI construction), and an author
  credit ("Oukil khaled ibn el-walid").
- `run.py`: a root-level convenience launcher for running the app
  straight from a source checkout without installing the package
  first (`python run.py`).
- `tests/test_splash_smoke.py` and `tests/test_main_entrypoint.py`.

### Fixed
- **NumPy / Python 3.13 install failure.** `requirements.txt` pinned
  `numpy==1.26.4`, which has no pre-built wheel for Python 3.13 (NumPy
  only gained 3.13 wheels starting at 2.1.0), causing pip to attempt a
  from-source build that fails without a C/C++ compiler installed.
  Re-pinned to `numpy==2.1.3`, verified compatible with the existing
  `pandas==2.2.3` / `matplotlib==3.9.2` pins (full test suite re-run
  against this exact combination).
- **Confusing relative-import error when running `main.py` directly.**
  `python src/isokinetics/main.py` cannot work because relative imports
  (`from .splash import ...`) require the file to be loaded as part of
  its package. `main()` now detects this case (`__package__` is empty)
  and prints clear, actionable instructions instead of surfacing the
  raw `ImportError: attempted relative import with no known parent
  package`.
- Fixed a latent race condition in the splash screen's background
  loading thread: `Tk.after()` calls issued from a worker thread are
  only valid once the Tk main loop is actually running. The worker
  thread is now started via an `after()` callback itself (guaranteeing
  the main loop has started first) and a 20-second safety-net timer
  ensures the app always proceeds to the main window even if a startup
  stage raises an unexpected exception.

## [1.0.0] - 2026-07-07

### Added
- Initial public release.
- Data loading/validation for text (space/tab/comma/semicolon-delimited)
  and Excel (`.xlsx`/`.xls`) input files with 5 columns
  (`alpha, T1, T2, T3, T4`).
- Uniform-alpha-grid interpolation (`interpolate_alpha`) so all methods
  are compared on the same conversion grid.
- Five isoconversional kinetic methods, implemented with NumPy/pandas
  only (no SciPy):
  - Friedman (differential)
  - Flynn-Wall-Ozawa (FWO)
  - Kissinger-Akahira-Sunose (KAS)
  - Starink
  - Vyazovkin (advanced/iterative, with a from-scratch golden-section
    minimiser and trapezoidal temperature integral)
- Publication-ready Matplotlib plot of Eₐ(α) with per-method legends
  showing the mean activation energy.
- Desktop GUI (customtkinter) with:
  - Data Preview, Kinetic Plot, and Results Table tabs.
  - Method selection checkboxes, progress bar, and status messages.
  - Background-threaded analysis so the UI never freezes.
- One-click export to a timestamped folder containing:
  - `Ea_vs_alpha.png` (300 dpi), `.svg`, and `.pdf`.
  - `Ea_data.txt` (tab-delimited, Origin-compatible).
  - `Ea_results.xlsx` (two sheets: full table + averages).
- Full pytest suite (37 tests), including a synthetic-data validation
  test that verifies every method recovers a known, ground-truth
  activation energy to within 5-10%.
- Packaging via `pyproject.toml` (installable, with an
  `isokinetics-gui` console-script entry point) and a pinned
  `requirements.txt` for fully reproducible environments.
- GitHub Actions CI workflow running the test suite under Xvfb.

### Notes
- The Vyazovkin objective function implemented here follows the
  standard formulation (S. Vyazovkin, *J. Comput. Chem.* 18 (1997) 393),
  which weights each pairwise integral ratio by the corresponding
  heating-rate ratio (βⱼ/βᵢ). This weighting is essential: without it,
  the objective function has no interior minimum on real data and is
  monotonically minimised as Eₐ → 0, which is not physically meaningful.
  This was verified numerically during development (see
  `tests/test_kinetics.py`).
