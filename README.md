# Isoconversional Kinetics Analyzer
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()
[![NumPy](https://img.shields.io/badge/powered%20by-NumPy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-2.2.3-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9.2-11557c.svg)](https://matplotlib.org/)
[![customtkinter](https://img.shields.io/badge/GUI-customtkinter-1f6feb.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![GitHub](https://img.shields.io/badge/GitHub-khadev-181717.svg?logo=github&logoColor=white)](https://github.com/khadev)
[![Release](https://img.shields.io/badge/release-v2.1.0-success.svg)](CHANGELOG.md)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)]()

A desktop application **and** Python library for **isoconversional kinetic
analysis** of non-isothermal thermal-decomposition data (TGA/DSC), computing
the activation energy Eₐ as a function of conversion α using six
methods:

- **Friedman** (differential)
- **Flynn-Wall-Ozawa (FWO)**
- **Kissinger-Akahira-Sunose (KAS)**
- **Starink**
- **Vyazovkin** (advanced / iterative, non-linear)
- **KTH** (original discretised-Arrhenius differential method)

Built entirely on **NumPy + pandas** (no SciPy dependency) with a modern
GUI powered by [`customtkinter`](https://github.com/TomSchimansky/CustomTkinter).

---

## Screenshots

| Splash screen | Kinetic Plot (w/ CI bands) | Model Fitting tab | About (top-right) |
|---|---|---|---|
| ![Splash screen](screenshots/splash_screen.png) | ![Confidence bands](screenshots/confidence_bands.png) | ![Model fitting](screenshots/model_fitting_tab.png) | ![Progress and About fixed](screenshots/progress_and_about_fixed.png) |

*(All screenshots were captured from a real, running instance of the app.)*

---

## Features

- **Branded splash screen** on startup: a frameless, centered loading
  window with a progress bar tied to real initialization stages
  (NumPy/pandas, Matplotlib backend, UI construction) and an author
  credit, before the main window appears.
- **Flexible input**: space/tab/comma/semicolon-delimited text files or
  Excel workbooks, 5 columns (`alpha, T1, T2, T3, T4`).
- **Alpha-grid interpolation** so every method is compared on exactly the
  same conversion grid, even if the raw data isn't perfectly aligned.
- **6 isoconversional methods**, implemented from scratch with NumPy only
  (linear regression via `numpy.polyfit`; the Vyazovkin method uses a
  hand-written trapezoidal temperature integral and golden-section search
  — no `scipy.optimize` required).
- **Publication-ready plot**: clean style, Unicode α / Eₐ labels, legend
  showing the mean Eₐ per method.
- **One-click export** to PNG (300 dpi) / SVG / PDF plus a tab-delimited
  `Ea_data.txt` (opens directly in Origin, Excel, etc.) and an optional
  multi-sheet `.xlsx` workbook.
- **Responsive GUI**: all computation runs in a background thread; the
  progress bar and status label update live, the UI never freezes.
- **Usable as a library**: every module (`data_loader`, `kinetics`,
  `plotter`, `exporter`) works standalone in scripts/notebooks, with no
  GUI dependency required for the computation itself.
- **Validated**: the test suite generates synthetic Arrhenius data with a
  *known* activation energy and checks that every method recovers it to
  within 5-10% (see [Testing](#testing)).

---

## Installation

### Option A — from source (recommended for now)

```bash
git clone https://github.com/your-org/isoconversional-kinetics.git
cd isoconversional-kinetics
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Option B — exact, pinned environment

If you want the *exact* dependency versions this project was built and
tested against (useful for PyInstaller builds or strict reproducibility):

```bash
pip install -r requirements.txt
```

(See `requirements.txt` for the pinned list: `customtkinter==5.2.2`,
`Pillow==10.4.0`, `matplotlib==3.9.2`, `pandas==2.2.3`, `numpy==1.26.4`,
`darkdetect==0.8.0`, `openpyxl==3.1.5`, `xlsxwriter==3.2.0`.)

### Option C — as a package (once published)

```bash
pip install isoconversional-kinetics
```

---

## Usage

### Launching the GUI

```bash
# after `pip install -e .` or `pip install isoconversional-kinetics`
isokinetics-gui

# or, without installing the console script (from the repo root):
python -m isokinetics

# or, the simplest option for a plain source checkout:
python run.py
```

On startup you'll see a branded splash screen (progress bar + author
credit) while the app initializes, then the main window opens.

> **Do not** run `python src/isokinetics/main.py` directly — that file
> lives *inside* the `isokinetics` package and uses relative imports
> (`from .gui import run`), which only resolve when the file is
> *imported as part of its package*, not executed as a standalone
> script. Running it directly now prints a clear message pointing you
> to the three options above instead of a confusing
> `ImportError: attempted relative import with no known parent package`.
> See [Troubleshooting](#troubleshooting) for details.

**Workflow:**

1. Click **Load Data** and select a `.txt`/`.csv`/`.xlsx` file: 5 columns
   (`alpha, T1, T2, T3, T4`, processed) or 4 columns (`T1, T2, T3, T4`, raw;
   a dialog then asks for the alpha step, see v1.9.0 below). The **Data
   Preview** tab fills in.
2. Enter the four heating rates **β₁…β₄** (K/min) in the top bar.
3. Tick the methods you want to run (all six are selected by default).
4. Click **Run Analysis**. The progress bar advances as each method
   completes; the **Kinetic Plot** and **Results Table** tabs update when
   done.
5. Click **Export**, choose a destination folder, and a timestamped
   subfolder (`Kinetics_Results_YYYYMMDD_HHMMSS`) is created containing
   the graph (PNG/SVG/PDF), `Ea_data.txt`, and `Ea_results.xlsx`.

### Using it as a library

```python
from isokinetics import data_loader, kinetics, plotter, exporter

# 1. Load & preprocess
df = data_loader.load_data("sample_data/matric_121459.txt")
df = data_loader.interpolate_alpha(df)  # uniform alpha grid, 0.01 step

betas = [5.0, 10.0, 20.0, 40.0]  # K/min

# 2. Run any/all methods
results = {
    "Friedman": kinetics.friedman(df, betas),
    "FWO": kinetics.fwo(df, betas),
    "KAS": kinetics.kas(df, betas),
    "Starink": kinetics.starink(df, betas),
    "Vyazovkin": kinetics.vyazovkin(df, betas),
    "KTH": kinetics.kth(df, betas),
}
averages = {name: kinetics.average_ea(res) for name, res in results.items()}
print(averages)  # {'Friedman': 98.28, 'FWO': 83.77, 'KAS': 78.11, ..., 'KTH': 98.68}

# 3. Plot
fig = plotter.build_figure()
plotter.update_plot(fig, results, averages)
fig.savefig("Ea_vs_alpha.png", dpi=300)

# 4. Or export everything (graph + tab-delimited + Excel) in one call
out_folder = exporter.export_all("/path/to/output_dir", results, fig)
print("Saved to:", out_folder)
```

---

## Input data format

A text file (whitespace/tab/comma/semicolon-separated) or an Excel file,
with **5 columns** for processed data (extra columns are ignored) or **4
columns** (`T1..T4`) for raw data whose alpha is generated by the application;
**no header required**:

| alpha | T1 (β₁) | T2 (β₂) | T3 (β₃) | T4 (β₄) |
|-------|---------|---------|---------|---------|
| 0.01  | 502.95  | 518.85  | 542.95  | 587.91  |
| 0.02  | 510.97  | 527.29  | 550.97  | 593.09  |
| ...   | ...     | ...     | ...     | ...     |

- `alpha`: fractional conversion, between 0 and 1.
- `T1..T4`: absolute temperature (**Kelvin**) at which that conversion was
  reached, one column per heating rate.
- The corresponding **β₁…β₄** (K/min) are entered separately in the GUI
  (or passed directly to the library functions).

A ready-to-use example is included at `sample_data/matric_121459.txt`.

---

## Output files

Every export run creates `Kinetics_Results_YYYYMMDD_HHMMSS/` containing:

```
Kinetics_Results_20260707_153000/
├── Ea_vs_alpha.png     # 300 dpi
├── Ea_vs_alpha.svg
├── Ea_vs_alpha.pdf
├── Ea_data.txt         # tab-delimited: alpha, Friedman, FWO, KAS, Starink, Vyazovkin, KTH
└── Ea_results.xlsx     # sheets: "Ea_vs_alpha" and "Averages"
```

`Ea_data.txt` layout:

```
alpha       Friedman   FWO      KAS      Starink  Vyazovkin  KTH
(average)   98.2783    83.7728  78.1076  78.4457  78.6324    98.6761
0.0100      72.0334    55.7721  49.6201  49.9423  50.2316    74.2859
0.0200      74.3880    59.4623  53.3918  53.7154  54.0027    75.0987
...
```

This opens directly in Origin, Excel, LibreOffice Calc, or any tool that
reads tab-delimited text.

---

## The methods, briefly

All six methods estimate Eₐ(α) from the same underlying data
(temperatures at which a given α is reached, at several heating rates β),
using standard closed-form or iterative isoconversional relations:

| Method     | Type        | Equation (slope vs. 1/T) |
|------------|-------------|---------------------------|
| Friedman   | Differential | `ln(β·dα/dT) = ln(A·f(α)) − Eₐ/(RT)` |
| FWO        | Integral (approx.) | `ln(β) = const − 1.052·Eₐ/(RT)` |
| KAS        | Integral (approx.) | `ln(β/T²) = const − Eₐ/(RT)` |
| Starink    | Integral (approx.) | `ln(β/T^1.92) = const − 1.0008·Eₐ/(RT)` |
| Vyazovkin  | Integral (exact, iterative, windowed) | minimises `Φ(Eₐ) = ΣᵢΣⱼ≠ᵢ [J(Eₐ,Tᵢ)·βⱼ] / [J(Eₐ,Tⱼ)·βᵢ]`, `J` integrated over `[α−Δα, α]` |
| KTH        | Differential (discrete-step) | `ln((α_{i+1}-α_i)/Δtᵢ) = ln(Φ(αᵢ)) − Eₐ/(R·T_mean,ᵢ)` |

For Vyazovkin, `J(E) = ∫ exp(−E/(RT')) dT'` is the temperature integral,
taken over the conversion window `[α−Δα, α]` (Vyazovkin 2001, the
"advanced" form, which stays accurate when Eₐ varies with α; pass
`advanced=False` for the original whole-range form of 1997). The integral
is evaluated by Gauss-Laguerre / Gauss-Legendre quadrature (machine
precision, no arbitrary reference temperature) and the minimisation uses
a from-scratch **golden-section search** (tolerance 0.01 kJ/mol) whose
bracket widens automatically — no `scipy.optimize` dependency.

> **Implementation note on Vyazovkin's objective function.** The
> heating-rate ratio `βⱼ/βᵢ` is essential: dropping it makes the objective
> monotonic in Eₐ with no interior minimum.

> **About KTH.** KTH is an original method (from the accompanying
> thesis, Chapter VII) named after its authors' initials. It is
> structurally a differential method like Friedman, but instead of an
> instantaneous derivative `dα/dT` at a single point, it uses a
> **discrete-step rate** `(α_{i+1} − αᵢ)/Δtᵢ` over the interval between
> two consecutive points on the α grid, paired with the **mean
> temperature** over that same interval, `T_mean,ᵢ = (Tᵢ + T_{i+1})/2`.
> The time step is recovered from the temperature step under the
> linear-heating assumption `T = T₀ + βt` (`Δtᵢ = (T_{i+1} − Tᵢ)/β`), so
> no explicit time column is needed in the input data. `kinetics.kth()`
> exposes only Eₐ(α), for drop-in compatibility with the other methods;
> `kinetics.kth_full()` (see below) recovers the complete kinetic
> triplet.

---

---

## Coats-Redfern: model-fitting method with honest model ranking

`kinetics.coats_redfern(df, beta_list)` fits every candidate reaction
model directly (`ln(g(α)/T²) = ln(AR/(βEₐ)) − Eₐ/(RT)`, per heating
rate, then averaged) over 17 candidates
(`kinetics.COATS_REDFERN_MODELS`), and recommends a best fit.

```python
result = kinetics.coats_redfern(df, betas)
print(result["recommended_model"])   # e.g. "F1", or None if inconclusive
print(result["results"]["F1"])       # {"Ea":..., "A":..., "r2":..., "aic":..., "bic":..., "ea_gap_pct":...}
```

> **Why this doesn't just rank by R².** Model-fitting methods like
> Coats-Redfern are well known to be poor at discriminating models on
> R² alone (the "kinetic compensation effect"): validated here against
> synthetic F1 ground-truth data, model **A2 scores R² = 0.9999991** —
> as good as F1's own fit — while recovering an activation energy more
> than **2× too low** (70 kJ/mol vs. the true 150 kJ/mol). Because of
> this, `coats_redfern()` screens every candidate's fitted Eₐ against a
> **model-free reference Eₐ** (computed internally via `vyazovkin()`,
> or supplied explicitly via `reference_ea_kj=`) and only recommends a
> model that is both a good fit (R² > 0.99) *and* within 20% of that
> reference. If nothing satisfies both, it returns
> `recommended_model=None` rather than guessing — the same
> "unknown"-over-guessing philosophy as `kth_full()`.

---

## KTH full kinetic triplet: A(α) and f(α) via Malek's method

Beyond Eₐ(α), `kinetics.kth_full(df, beta_list)` recovers the complete
kinetic triplet — activation energy, pre-exponential factor **A(α)**,
and the **experimental reaction model f(α)** — using Malek's method
(thesis, Chapter VII, §VII.4.1.3).

**In the GUI:** switch the **"KTH mode"** dropdown (top bar) from "Eₐ
only" to **"Full triplet (A, f(α), model)"** before running the
analysis with KTH selected. A new **KTH Model** tab shows the
diagnostic plot; the **Results Table** gains `KTH_A` and `KTH_f(α)`
columns; and exports gain the same two columns plus a `KTH_Model` sheet
(Excel) or comment line (text) naming the identified model.

**As a library:**

```python
result = kinetics.kth_full(df, betas)
print(result["model"])       # e.g. "F1", "D3", or "unknown"
print(result["A"])           # {alpha: A (s^-1)}
print(result["f_alpha"])     # {alpha: f(alpha) value}
print(result["model_scores"])  # {model_code: match_score, ...} -- lower is better
```

### How model identification works

1. Compute Malek's `y(α) = (dα/dt)·exp(Eₐ/RT)` and
   `z(α) = (dα/dt)·(T/β)·x·exp(x)·p(x)` (`x = Eₐ/RT`) from the
   experimental data, averaged across all four heating rates and each
   normalised to [0, 1].
2. Compare those curves against 13 standard candidate models — reaction
   order (F0–F3), geometric contraction (R2–R3), diffusion (D1–D4), and
   Avrami-Erofeev nucleation-and-growth (A2–A4) — using the well-known
   identity that, after normalisation, `y(α)` has the same shape as the
   model's `f(α)` and `z(α)` has the same shape as `f(α)·g(α)` (the
   "combined master plot").
3. The best-matching model (lowest combined RMSE, restricted to the
   α ∈ [0.2, 0.8] window where the thesis states Malek's method is
   valid) is accepted **only if** its match is both good (below a
   threshold) and unambiguous (clearly better than the runner-up).
   Otherwise the result is `"unknown"` — model identification is
   explicitly designed to say "I don't know" on data that doesn't
   confidently match a simple model, rather than force-fit one. (Real
   experimental data, e.g. the bundled sample dataset, legitimately
   comes back `"unknown"` — its two best candidates score within 5% of
   each other.)
4. Once a model is identified, `A(α) = Φ(α)/f(α)`, where `Φ(α)` is the
   intercept of the same per-step regression that gives Eₐ(α).

> **A note on getting the mathematics right.** The z(α) formula as
> commonly abbreviated (`z(α) = p(x)·(dα/dt)·(T/β)`) is missing a
> factor: deriving it from first principles (`g(α) = (AEₐ/Rβ)·p(x)`
> substituted into the Arrhenius rate law) shows the term needed for
> `z(α)` to actually be proportional to `f(α)·g(α)` is
> `x·exp(x)·p(x)·(dα/dt)·(T/β)` — an extra `x·exp(x)` factor. Without
> it, `z(α)` computed from real data doesn't peak near the
> theoretically-expected α ≈ 0.632 for a first-order process and
> model identification fails outright. This was caught by validating
> against synthetic data with a *known* underlying model (both F1 and
> D3 were tested) before trusting the result — see
> `tests/test_kinetics.py`. Similarly, `p(x)` (the temperature integral)
> is evaluated here by direct numerical integration rather than a
> closed-form rational approximation, since a commonly-seen version of
> that approximation has the wrong asymptotic power of `x` (see the
> docstring of `_temperature_integral_p` for the details) — numerical
> integration sidesteps the ambiguity entirely and was checked here to
> agree with the exact closed form to within 1e-5 relative error.

---

---

## v2.1.0: import wizard for raw instrument data

Select the **four raw instrument files** (one per heating rate) in *Load Data* and the wizard
replaces the manual Origin step: it cleans each file, lets you map the temperature and heat-flow
columns, lets you drag over the decomposition peak of every file, converts the peak to a
conversion alpha (cumulative baseline-corrected area), aligns the four curves on one alpha grid
and fills in the heating rates. Try it with `sample_data/dsc_instrument/`.

* **Check your data before trusting the numbers.** For a single-step process the peak must shift to
  a higher temperature as the heating rate rises, so alpha(T) curves must *not* cross. The wizard
  warns when the temperatures at alpha = 0.5 are not ordered by heating rate.
* **The baseline matters.** Compare *linear* and *sigmoidal*: with a heat-capacity step the linear
  baseline can silently bias Ea by 20-30 %.
* Tip: files are put in heating-rate order automatically (3, 5, 7, 10 - not 10, 3, 5, 7).

---

## v2.0.0: stabilization interval, lifetime sync, manual KTH triplet

* **Select an alpha range on the plot** (*Kinetic Plot* tab): drag with the mouse. The range is
  shaded and the legend shows each method's mean Ea inside it, `(mean Ea [0.20-0.80] = ... kJ/mol)`.
  *Reset to Full Range* (or double-click) goes back. Full-range results are never changed.
* **Lifetime tab follows the selection**: `Friedman mean (alpha 0.20-0.80)` etc.; an already
  calculated lifetime is recalculated automatically, and reverts when the range is reset.
* **Exports reflect the selection**: `(average)` row / `Averages` sheet = interval mean, files start
  with `# Selected stabilization interval: alpha 0.20 to 0.80`.
* **Calculate KTH Full Triplet**: one click on the *KTH Model* tab (background thread) gives
  Ea, A and f(alpha) without re-running the whole analysis.
* Library: `kinetics.calculate_interval_mean(results, alpha_min, alpha_max)`.

---

## v1.9.0: raw-data import (no Origin pre-processing)

Load either kind of file; the application recognises it by its columns.

| File | Columns | What happens |
|---|---|---|
| **Processed** | 5: `alpha  T1  T2  T3  T4` | analysed directly |
| **Raw** | 4: `T1  T2  T3  T4` | a dialog asks for the conversion step, alpha is generated, temperatures are interpolated |

For a raw file choose the step as `0.01`, `0.02`, a fraction `1/50`, or a number
of points `100` (a whole number N means a step of 1/N). alpha = 0 and 1 are
never included (Ea is undefined there), so `0.01` / `100` give 99 points
(0.01 ... 0.99) and `1/50` gives 49 points (0.02 ... 0.98).

**Assumption (read this).** A file of temperatures only contains no
conversion information. Each column is therefore treated as a curve whose rows
are *equally spaced in conversion*, and the temperature is linearly
interpolated at every alpha of the grid. The dialog lets you say where the rows
sit: *first row alpha = 0, last row alpha = 1* (default), or *interior points*
`alpha = i/(n+1)` (99 rows -> 0.01 ... 0.99, i.e. a processed file without its
alpha column). If your raw file is something else (e.g. temperature and mass
over time), it must be converted to alpha first.

*Change alpha step...* regenerates the table (results are cleared);
*Save processed table...* writes the generated `[alpha, T1..T4]` table for reuse.
Try `sample_data/sample_raw_4columns_F1_endpoints.txt` (101 rows, true Ea =
150 kJ/mol) or `sample_raw_4columns_interior.txt` (the bundled sample without
its alpha column; choose *interior points*).

---

## v1.8.0: lifetime prediction (Arrhenius shelf-life)

The **Lifetime** tab extrapolates how long a material lasts at its service
temperature from how long it lasts at a (usually hotter) storage temperature:

```
t_service = t_storage * exp[ Ea/R * (1/T_service - 1/T_storage) ]
```

* Inputs: storage temperature, time at storage temperature, service
  temperature (degC or K; time in years/days/hours/minutes).
* Ea comes from the **recommended Coats-Redfern model** by default; if none
  is recommended the first model-free mean is used and the tab says so.
  Any other method mean, Coats-Redfern model, or a manual value can be chosen.
* Output in years, days and minutes, plus the acceleration factor and the
  lifetime for Ea +/-10 %.
* Library use: `from isokinetics import lifetime; lifetime.predict_lifetime(ea_kj, T_storage_K, t_storage_days, T_service_K)`.

Caveat: this assumes one mechanism and a constant Ea over the whole
temperature range; Ea measured at 500-700 K is extrapolated to ambient, so
the result is an estimate (the tab warns when this applies).

---

## v1.7.0: redesigned interface and validated kinetics

### New interface
* **Sidebar workflow** — 1 Data → 2 Heating rates → 3 Methods → 4 Display;
  *Run Analysis*, *Export* and *Save Image* stay pinned at the bottom.
* **Dark / light theme** switch (header, or `Ctrl+T`); tables and plots
  follow it. Exported images are *always* white-background.
* **Tabs**: Data Preview · Summary · Kinetic Plot · Results Table ·
  Model Fitting · KTH Model · Validation.
* **Summary tab**: one card per method (mean Eₐ ± std, range) and a short
  plain-language reading of the results (e.g. whether Eₐ varies with α).
* **Informative progress bar**: current stage (“Running Vyazovkin, step
  3 of 7”), percentage, elapsed time and time remaining.
* **Consistent number format** everywhere (`isokinetics.formatting`):
  α 3 decimals, Eₐ 2 decimals (kJ/mol), T 2 decimals (K), A scientific,
  R² 4 decimals, `N/A` for undefined values.
* Ease of use: *Use bundled sample*, *All / None* for methods, red border on
  invalid β, warning if the β order contradicts the data, *Copy table*,
  shortcuts `Ctrl+O` / `Ctrl+R` / `F5` / `Ctrl+E` / `Ctrl+T`.
* Export adds `Summary.txt`, a `Model_Fitting` Excel sheet and the
  Coats-Redfern chart.

### Kinetics corrections (all verified against synthetic ground truth)
| Issue found | Fix |
|---|---|
| Vyazovkin integrated from a fixed 300 K → wrong whenever data are near/below 300 K or Eₐ is low (30 kJ/mol → 430) | exact integral from 0 K / over the α-window |
| Vyazovkin silently capped Eₐ at 500 kJ/mol | bracket widens automatically; NaN if no interior minimum |
| “Advanced” Vyazovkin was actually the 1997 whole-range form (biased when Eₐ varies) | windowed 2001 form (default), `advanced=False` for the old one |
| Coats-Redfern dropped the `(1 − 2RT/E)` term → A about 10 % too low | restored (fixed-point iteration) |
| Temperature integral for Malek's method accurate to only ~1e-4 | Gauss-Laguerre, ~1e-13 |
| R = 8.314 | CODATA 8.314462618 |

FWO is unchanged: its ±3 % deviation is the Doyle approximation behind the
constant 1.052 (exact only near Eₐ/RT ≈ 30-40), a property of the method.

### Validating the methods yourself
```bash
python -m isokinetics.validation      # prints the report, exit code 1 on failure
```
or press **Run validation** in the *Validation* tab. Five synthetic
experiments with a known Eₐ (constant, low, high, near-ambient and
variable Eₐ(α)) are generated independently of `kinetics.py` and every
method must recover Eₐ within its theoretical tolerance
(`tests/test_kinetics_ground_truth.py`).

---

## v1.6.0: GUI integration, confidence intervals, journal styles, TIFF export

### Confidence intervals

`kinetics.confidence_interval(df, betas, method_name="KAS")` computes a
95%/99% CI for Eₐ(α) using the regression slope's standard error and a
Student's t critical value (hardcoded table, no SciPy dependency).

> **Use KAS, FWO, or Starink — not Friedman.** Validated via Monte Carlo
> coverage check (adding realistic ±0.5 K instrument noise across many
> trials and checking how often the resulting 95% CI actually contains
> the known true Eₐ): KAS achieves ~95% empirical coverage with a
> practically useful ~25 kJ/mol width. Friedman is *also* statistically
> valid (~94.5% coverage) but its numerical differentiation step
> amplifies noise so much the mean width balloons to ~1300 kJ/mol —
> correct, but not useful to display. The GUI's confidence-band checkbox
> only applies to the three integral methods for this reason.

In the GUI: tick **"Show 95% confidence bands"** — shaded bands appear
behind FWO/KAS/Starink's lines on the Kinetic Plot.

### Journal-style plot presets

`plotter.PLOT_STYLES` provides 6 presets (ACS, RSC, Nature, Elsevier,
Colorblind, Monochrome), selectable from the **"Plot style"** dropdown.
Colorblind and Monochrome remap each method's line color (Okabe-Ito
palette / greyscale) rather than relying on matplotlib's color-cycle
rcParam, which has no effect once a plot call passes an explicit
`color=` — a real bug caught while testing this feature (the palette
was silently not applying at all).

### Model Fitting tab (Coats-Redfern)

Every analysis run also computes `kinetics.coats_redfern()` (fast, so
it always runs regardless of which methods are selected) and displays
it in a new **Model Fitting** tab: a sortable table (Eₐ, A, R², pass/
fail status per candidate) and an R² bar chart, colored green
(passed the reference-Eₐ screen) or red (rejected — including any
high-R² "kinetic compensation effect" false positives).

### TIFF export with metadata

`exporter.save_graph()`/`export_all()` now also save a 300 dpi,
LZW-compressed `.tiff` alongside PNG/SVG/PDF by default
(`include_tiff=False` to skip), with author/sample/methods/date
embedded in the TIFF's `ImageDescription` tag.

---

## Project structure

```
isoconversional-kinetics/
├── run.py                  # convenience launcher (no install needed)
├── src/isokinetics/
│   ├── __init__.py
│   ├── __main__.py        # enables `python -m isokinetics`
│   ├── main.py            # console entry point (+ friendly diagnostics)
│   ├── splash.py           # branded splash screen (progress bar + author)
│   ├── gui.py              # customtkinter application
│   ├── data_loader.py      # load / validate / interpolate
│   ├── kinetics.py         # Friedman, FWO, KAS, Starink, Vyazovkin, KTH, Coats-Redfern
│   ├── validation.py       # synthetic ground-truth generator + validation report
│   ├── lifetime.py         # Arrhenius shelf-life / service-lifetime prediction
│   ├── interval.py         # stabilization-interval helpers (pure, no GUI)
│   ├── interaction.py      # SpanSelector controller (drag to select, double-click to reset)
│   ├── formatting.py       # one number-format policy for GUI, plots and exports
│   ├── plotter.py          # matplotlib figures (light/dark theme aware)
│   └── exporter.py         # PNG/SVG/PDF/TIFF + txt + xlsx + Summary.txt export
├── tests/
│   ├── test_data_loader.py
│   ├── test_kinetics.py            # incl. synthetic ground-truth validation
│   ├── test_kinetics_ground_truth.py  # every method vs. known Ea (const. + variable)
│   ├── test_formatting.py
│   ├── test_raw_data.py
│   ├── test_lifetime.py
│   ├── test_gui_redesign.py        # theme, progress, tabs (needs a display)
│   ├── test_exporter_plotter.py
│   ├── test_gui_smoke.py           # requires a display (Xvfb on CI)
│   ├── test_splash_smoke.py        # requires a display (Xvfb on CI)
│   └── test_main_entrypoint.py     # run.py / -m isokinetics / error-path tests
├── examples/run_analysis_headless.py
├── sample_data/matric_121459.txt
├── screenshots/
├── .github/workflows/ci.yml
├── pyproject.toml          # PyPI packaging (src layout, console script)
├── requirements.txt        # pinned, exact versions
├── requirements-dev.txt    # pytest, build, twine, ...
├── LICENSE (MIT)
├── CHANGELOG.md
└── README.md
```

---

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# Headless machines (no display) need a virtual display for the GUI
# smoke tests -- everything else runs anywhere:
xvfb-run -a pytest tests/ -v

# Or, to skip the GUI smoke tests entirely:
pytest tests/ -v --ignore=tests/test_gui_smoke.py
```

The suite includes a **synthetic-data validation test**: it simulates a
first-order Arrhenius reaction with a *known* activation energy
(150 kJ/mol) at four heating rates, feeds the resulting α/T data through
all six methods, and asserts each one recovers ≈150 kJ/mol (within 5%
for the linear/differential methods, 10% for Vyazovkin). This is a much stronger check
than a plain "did it crash" test.

---

## Building the package

```bash
pip install -r requirements-dev.txt
python -m build          # creates dist/*.whl and dist/*.tar.gz
twine check dist/*
# twine upload dist/*    # publish to PyPI (requires credentials)
```

---

---

## v1.6.1: Save Image, About dialog, and reliability fixes

### Save Image (no full export needed)

Click **"Save Image..."** to save exactly the plot in the currently
active tab (Kinetic Plot, KTH Model, or Model Fitting) directly to a
file you choose — PNG, TIFF (300 dpi, LZW-compressed), PDF, or SVG.
This is separate from **Export**, which writes a whole timestamped
folder with every method's data; Save Image is for "I just want this
one picture."

### About dialog

Click **"About"** for a short summary of what the app does and the
developer's contact email, for feature requests or help.

### Reliability fixes (from real user reports)

- **Splash screen could crash on close, especially on Windows.** The
  background loading thread schedules a progress-bar update per stage;
  if the splash was destroyed between two stages, a freshly-scheduled
  update could still fire afterwards and touch already-destroyed
  widgets (cancelling *already-pending* callbacks on destroy, added
  previously, does not protect against a *new* one being scheduled
  after that point). Fixed with the same "check if still alive before
  touching widgets" guard `KineticsApp` already used, applied to the
  splash screen too.
- **A missing optional dependency (`xlsxwriter`) crashed the entire
  export**, even though the image files and the tab-delimited text file
  had already been written successfully. `export_all()` now catches
  this specifically, skips only the `.xlsx` file, and tells you exactly
  what to run (`pip install xlsxwriter`) to get it back — everything
  else still exports normally.
- **A rare `TclError` during shutdown** (About dialog left open, then
  two image saves in different formats back-to-back, then closing the
  app) was reproduced and fixed two ways: child windows are now closed
  before the main window tears down, and any residual Tcl-level
  teardown error is safely absorbed (the window is being destroyed
  regardless at that point) rather than surfacing as a crash.

---

## Troubleshooting

### `pip install` tries to compile NumPy from source and fails (Windows, no compiler)

**Symptom:**
```
ERROR: Unknown compiler(s): [['icl'], ['cl'], ['c++'], ['g++'], ['clang++'], ['clang-cl']]
```

**Cause:** `numpy==1.26.4` (an older pin) has **no pre-built wheel for
Python 3.13** — its wheels only go up to `cp312`, because 1.26.4 was
released in February 2024, before Python 3.13 existed. When pip can't
find a matching wheel, it falls back to building NumPy from source,
which needs a C/C++ compiler (Visual Studio Build Tools on Windows,
`g++`/`gcc` on Linux, `clang++` on macOS) — hence the error.

**Fix:** This repository's `requirements.txt` is already pinned to
`numpy==2.1.3`, which **does** ship a Python 3.13 wheel (NumPy added
3.13 support starting at 2.1.0) and has been verified compatible with
the other pinned versions here (`pandas==2.2.3`, `matplotlib==3.9.2`) —
the full test suite passes with this exact combination. Just make sure
you're installing from this project's `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you're on an older Python (3.9–3.12) and specifically need the
original `numpy==1.26.4` pin for some other reason, that version does
have wheels for those interpreters and will install normally there.

**General rule of thumb** when you hit this kind of error with *any*
package: check whether the pinned version actually ships a wheel for
your Python version before assuming your compiler setup is at fault —
```bash
pip index versions <package>          # see what's available
# or check https://pypi.org/project/<package>/#files
```

### `attempted relative import with no known parent package`

**Symptom:**
```
C:\...\isoconversional-kinetics> python src/isokinetics/main.py
Failed to import GUI dependencies: attempted relative import with no known parent package
```

**Cause:** `src/isokinetics/main.py` is a module *inside* the
`isokinetics` package, and it uses a relative import
(`from .splash import show_splash_then_launch`) to reach its sibling
modules. Relative imports only work when Python has loaded the file
*as part of a package* (e.g. via `import isokinetics.main`). Running
the file directly as a script (`python path/to/main.py`) makes Python
treat it as a standalone top-level module with **no** parent package,
so the leading `.` in the import has nothing to resolve against.

This is not specific to this project — it's how Python's import system
works for *any* package that uses relative imports.

**Fix — use one of these instead** (in order of convenience):

```bash
python run.py               # simplest: works from a plain source checkout
python -m isokinetics        # also works from a plain source checkout
isokinetics-gui              # after `pip install -e .`
```

All three give Python the package context it needs. `run.py` (at the
repository root) does this explicitly: it adds `src/` to `sys.path`
and then *imports* `isokinetics.main` as a module, rather than running
`main.py` as the top-level script — so every relative import inside the
package resolves correctly.

If you do run `main.py` directly by mistake, current version of the app
detects this and prints exactly this guidance instead of a raw
traceback.

### `invalid command name "...check_dpi_scaling"` / `"...update"` printed when closing the window

**Cause:** customtkinter schedules a couple of recurring internal
`after()` loops on every root window (DPI-scaling and appearance-mode
polling, both rescheduling themselves every ~100ms). If the window is
destroyed while one of these -- or any of the app's own `after`-based
callbacks -- is still pending, Tcl prints this error the next time it
tries to fire.

**Fix:** Already handled. Both `KineticsApp` and `SplashScreen`
override `destroy()` to cancel every pending `after`/`after_idle`
callback (see `isokinetics._tk_utils.cancel_all_pending_after_callbacks`)
before tearing down the widget hierarchy, and the background analysis
thread is signalled to stop via a `threading.Event` and joined briefly
before the window closes. See `tests/test_shutdown_regression.py` for
the regression tests covering this (including closing mid-analysis, at
9 different timing offsets).

### Best-practice summary for this kind of project structure

- Put your package under `src/<package_name>/` ("src layout") and use
  relative imports (`from . import foo`) between its modules freely —
  that's normal and correct.
- **Never** execute a file that lives inside the package directly as a
  script for day-to-day use. Instead, give the project one (or more) of:
  - a `pyproject.toml` console-script entry point (`isokinetics-gui`,
    here), for after installation;
  - a `__main__.py` inside the package, enabling `python -m <package>`;
  - a tiny launcher script at the repository root (`run.py`, here) that
    adjusts `sys.path` and imports the package properly, for people who
    haven't installed it yet.
- `pip install -e .` (editable install) is the most robust option during
  development — it makes the package importable from anywhere without
  copying files, so `isokinetics-gui` and `python -m isokinetics` both
  work regardless of your current directory.

---

## Contributing

Issues and pull requests are welcome. Please:

1. Add/update tests for any behavioural change.
2. Run `pytest` (and `xvfb-run -a pytest` for GUI changes) before
   submitting.
3. Keep functions documented with docstrings (NumPy/Google style).

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements / References

- Friedman, H. L. (1964). *J. Polym. Sci. C*, 6, 183-195.
- Flynn, J. H., & Wall, L. A. (1966). *J. Res. Natl. Bur. Stand. A*, 70, 487.
- Ozawa, T. (1965). *Bull. Chem. Soc. Jpn.*, 38, 1881-1886.
- Kissinger, H. E. (1957). *Anal. Chem.*, 29, 1702-1706.
- Akahira, T., & Sunose, T. (1971). *Res. Report Chiba Inst. Technol.*, 16, 22-31.
- Starink, M. J. (2003). *Thermochim. Acta*, 404, 163-176.
- Vyazovkin, S. (1997). *J. Comput. Chem.*, 18, 393-402.
- Malek, J. (1992). *Thermochimica Acta*, 200, 257-269.
- KTH: original method, accompanying thesis (Chapter VII, pp. 108-112).
