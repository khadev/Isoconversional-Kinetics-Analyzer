
<div align="center">

<img src="https://raw.githubusercontent.com/khadev/Isoconversional-Kinetics-Analyzer/main/screenshots/splash_screen.png" alt="Isoconversional Kinetics Analyzer" width="640"/>

# Isoconversional Kinetics Analyzer

**Professional desktop application & Python library for isoconversional kinetic analysis of non-isothermal thermal-decomposition data (TGA/DSC).**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()
[![NumPy](https://img.shields.io/badge/powered%20by-NumPy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-2.2.3-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9.2-11557c.svg)](https://matplotlib.org/)
[![customtkinter](https://img.shields.io/badge/GUI-customtkinter-1f6feb.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![GitHub](https://img.shields.io/badge/GitHub-khadev-181717.svg?logo=github&logoColor=white)](https://github.com/khadev)
[![Release](https://img.shields.io/badge/release-v1.9.0-success.svg)](CHANGELOG.md)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)]()

</div>

---

## Overview

**Isoconversional Kinetics Analyzer** is a complete, publication-grade toolkit for extracting the activation energy **Eₐ(α)** — and, optionally, the full kinetic triplet **A(α)**, **f(α)** — from non-isothermal TGA/DSC experiments. It ships as both a **modern desktop GUI** and a **scriptable Python library**, built entirely on **NumPy + pandas** with **no SciPy dependency**.

### Six isoconversional methods

| | Method | Type |
|---|---|---|
| 🔹 | **Friedman** | Differential |
| 🔹 | **Flynn-Wall-Ozawa (FWO)** | Integral (approx.) |
| 🔹 | **Kissinger-Akahira-Sunose (KAS)** | Integral (approx.) |
| 🔹 | **Starink** | Integral (approx.) |
| 🔹 | **Vyazovkin** | Integral (exact, iterative, windowed) |
| ⭐ | **KTH** | Differential (discrete-step, original) |

Plus **Coats-Redfern** model fitting (17 candidate models) with honest, reference‑Eₐ‑screened model ranking, and **Malek's method** for full kinetic‑triplet identification (13 candidate models).

---

## ✨ Features

- 🖥️ **Modern GUI** — dark/light theme, sidebar workflow, summary cards, live progress bar, keyboard shortcuts.
- 📊 **Publication-ready plots** — 6 journal presets (ACS, RSC, Nature, Elsevier, Colorblind, Monochrome), confidence bands, Unicode α/Eₐ labels.
- 📥 **Flexible input** — text (space/tab/comma/semicolon) or Excel; **5-column processed** (`α, T₁..T₄`) or **4-column raw** (`T₁..T₄`) with an interactive alpha-grid dialog.
- 📤 **One-click export** — timestamped folder with PNG (300 dpi) / SVG / PDF / TIFF, tab-delimited `Ea_data.txt` (Origin-compatible), multi-sheet `.xlsx`, and `Summary.txt`.
- 🧪 **Validated** — synthetic ground-truth tests verify every method recovers a known Eₐ within 5–10 %.
- ⏳ **Lifetime prediction** — Arrhenius shelf-life extrapolation with Eₐ ±10 % bounds and acceleration factor.
- 🐍 **Library-first** — every module (`data_loader`, `kinetics`, `plotter`, `exporter`) works standalone in scripts/notebooks.
- 🔒 **No SciPy** — linear regression via `numpy.polyfit`; Vyazovkin minimisation via a from-scratch golden-section search.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/khadev/Isoconversional-Kinetics-Analyzer.git
cd Isoconversional-Kinetics-Analyzer

# 2. Create environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt
pip install -e .

# 4. Launch
isokinetics-gui                    # or: python run.py
```

> **⚠️ Never run** `python src/isokinetics/main.py` directly — that file lives *inside* the package and uses relative imports. Use `python run.py`, `python -m isokinetics`, or `isokinetics-gui` instead.

---

## 🖼️ Screenshots

<div align="center">

| Splash screen | Kinetic plot (with CI bands) | Model Fitting tab | About (top-right) |
|:---:|:---:|:---:|:---:|
| ![Splash](screenshots/splash_screen.png) | ![CI](screenshots/confidence_bands.png) | ![Fit](screenshots/model_fitting_tab.png) | ![About](screenshots/progress_and_about_fixed.png) |

</div>

---

## 🐍 Library Usage

```python
from isokinetics import data_loader, kinetics, plotter, exporter

# Load & preprocess
df = data_loader.load_data("sample_data/matric_121459.txt")
df = data_loader.interpolate_alpha(df)          # uniform α grid, step 0.01

betas = [5.0, 10.0, 20.0, 40.0]                 # K/min

# Run all six methods
results = {
    "Friedman":   kinetics.friedman(df, betas),
    "FWO":        kinetics.fwo(df, betas),
    "KAS":        kinetics.kas(df, betas),
    "Starink":    kinetics.starink(df, betas),
    "Vyazovkin":  kinetics.vyazovkin(df, betas),
    "KTH":        kinetics.kth(df, betas),
}
averages = {name: kinetics.average_ea(res) for name, res in results.items()}
print(averages)

# Plot & export
fig = plotter.build_figure()
plotter.update_plot(fig, results, averages)
exporter.export_all("/path/to/output", results, fig)
```

---

## 📁 Input Format

A text file (whitespace/tab/comma/semicolon) or Excel workbook, **no header required**:

| α | T₁ (β₁) | T₂ (β₂) | T₃ (β₃) | T₄ (β₄) |
|------|---------|---------|---------|---------|
| 0.01 | 502.95  | 518.85  | 542.95  | 587.91  |
| 0.02 | 510.97  | 527.29  | 550.97  | 593.09  |
| ...  | ...     | ...     | ...     | ...     |

- **α** — fractional conversion, 0 ≤ α ≤ 1.
- **T₁..T₄** — absolute temperature (Kelvin) at which α was reached, one column per heating rate.
- **β₁..β₄** — heating rates (K/min), entered in the GUI or passed to the library.

A ready-to-use sample is provided at `sample_data/matric_121459.txt`.

---

## 📤 Output Files

Every export creates `Kinetics_Results_YYYYMMDD_HHMMSS/` containing:

```
Kinetics_Results_20260707_153000/
├── Ea_vs_alpha.png          # 300 dpi
├── Ea_vs_alpha.svg
├── Ea_vs_alpha.pdf
├── Ea_vs_alpha.tiff         # 300 dpi LZW, with metadata
├── Ea_data.txt              # tab-delimited — opens in Origin / Excel
├── Ea_results.xlsx          # sheets: Ea_vs_alpha, Averages, Model_Fitting, KTH_Model
├── Coats_Redfern_plot.png
└── Summary.txt
```

---

## 🧠 Method Summary

| Method | Type | Slope vs. 1/T |
|---|---|---|
| **Friedman** | Differential | `ln(β·dα/dT) = ln(A·f(α)) − Eₐ/(RT)` |
| **FWO** | Integral (approx.) | `ln(β) = const − 1.052·Eₐ/(RT)` |
| **KAS** | Integral (approx.) | `ln(β/T²) = const − Eₐ/(RT)` |
| **Starink** | Integral (approx.) | `ln(β/T^1.92) = const − 1.0008·Eₐ/(RT)` |
| **Vyazovkin** | Integral (exact, iterative) | Minimises Φ(Eₐ) = ΣᵢΣⱼ≠ᵢ [J(Eₐ,Tᵢ)·βⱼ]/[J(Eₐ,Tⱼ)·βᵢ] over `[α−Δα, α]` |
| **KTH** | Differential (discrete-step) | `ln((αᵢ₊₁−αᵢ)/Δtᵢ) = ln(Φ(αᵢ)) − Eₐ/(R·T̄ᵢ)` |

> **Vyazovkin's heating-rate ratio** `βⱼ/βᵢ` is essential — dropping it makes the objective monotonic in Eₐ with no interior minimum.

> **KTH** is an original method (accompanying thesis, Chapter VII): a differential method like Friedman, but using discrete-step rates `(αᵢ₊₁−αᵢ)/Δtᵢ` paired with mean temperatures — no explicit time column needed.

---

## 🔬 Honest Model Ranking (Coats-Redfern)

`kinetics.coats_redfern()` does **not** rank by R² alone — that suffers from the well-known **kinetic compensation effect**. Validated against synthetic F1 ground truth, model A2 scores **R² = 0.9999991** yet recovers an Eₐ **more than 2× too low** (70 vs 150 kJ/mol). So every candidate is screened against a model-free reference Eₐ (from Vyazovkin, or supplied via `reference_ea_kj=`) and recommended only if **R² > 0.99 AND within 20 % of the reference**. Otherwise it returns `None` rather than guess.

---

## 🧬 KTH Full Triplet (Malek's Method)

`kinetics.kth_full()` recovers **A(α)** and **f(α)** via Malek's `y(α)`/`z(α)` master-plot identification against 13 candidate models (F0–F3, R2–R3, D1–D4, A2–A4) over α ∈ [0.2, 0.8]. Returns `"unknown"` when the best match is poor or ambiguous.

```python
result = kinetics.kth_full(df, betas)
print(result["model"])         # "F1", "D3", or "unknown"
print(result["A"])             # {α: A (s⁻¹)}
print(result["f_alpha"])       # {α: f(α)}
print(result["model_scores"])  # lower is better
```

---

## ✅ Validation & Testing

```bash
# Full suite (headless machines need Xvfb for GUI tests)
xvfb-run -a pytest tests/ -v

# Skip GUI smoke tests
pytest tests/ -v --ignore=tests/test_gui_smoke.py

# Standalone validation report
python -m isokinetics.validation
```

The suite includes a **synthetic ground-truth test**: a first-order Arrhenius reaction with known Eₐ = 150 kJ/mol at four heating rates is fed through all six methods, each asserted to recover ≈150 kJ/mol within 5 % (linear/differential) or 10 % (Vyazovkin).

---

## 🗂️ Project Structure

```
Isoconversional-Kinetics-Analyzer/
├── run.py                    # convenience launcher
├── src/isokinetics/
│   ├── main.py               # entry point
│   ├── splash.py             # splash screen
│   ├── gui.py                # customtkinter GUI
│   ├── data_loader.py        # load / validate / interpolate
│   ├── kinetics.py           # 6 methods + Coats-Redfern + KTH full
│   ├── validation.py         # ground-truth validation
│   ├── lifetime.py           # Arrhenius shelf-life
│   ├── formatting.py         # number-format policy
│   ├── plotter.py            # matplotlib (theme-aware, journal styles)
│   └── exporter.py           # PNG/SVG/PDF/TIFF + txt + xlsx
├── tests/                    # pytest suite (incl. ground-truth)
├── sample_data/
├── screenshots/
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── CHANGELOG.md
└── README.md
```

---

## 🛠️ Troubleshooting

<details>
<summary><b>pip tries to compile NumPy from source and fails (Windows, no compiler)</b></summary>

An older pin like `numpy==1.26.4` has no Python 3.13 wheel. This project pins `numpy==2.1.3`, which ships a 3.13 wheel and is compatible with the other pins. Install from this project's `requirements.txt`.
</details>

<details>
<summary><b>attempted relative import with no known parent package</b></summary>

Running `python src/isokinetics/main.py` executes a module *inside* the package as a standalone script, so its relative imports have no context. Use one of:

```bash
python run.py
python -m isokinetics
isokinetics-gui        # after pip install -e .
```
</details>

<details>
<summary><b>invalid command name "...check_dpi_scaling" printed on close</b></summary>

Already handled — `destroy()` cancels pending `after`/`after_idle` callbacks on both `KineticsApp` and `SplashScreen`. See `tests/test_shutdown_regression.py`.
</details>

---

## 🤝 Contributing

Contributions are welcome! Please add/update tests for any behavioural change, run `pytest` (and `xvfb-run -a pytest` for GUI changes), and keep functions documented with NumPy/Google-style docstrings.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup details and how to add a new isoconversional method.

---

## 📜 License

Distributed under the **MIT License**. See **[LICENSE](LICENSE)** for details.

---

## 📚 References

- Friedman, H. L. (1964). *J. Polym. Sci. C*, 6, 183–195.
- Flynn, J. H., & Wall, L. A. (1966). *J. Res. Natl. Bur. Stand. A*, 70, 487.
- Ozawa, T. (1965). *Bull. Chem. Soc. Jpn.*, 38, 1881–1886.
- Kissinger, H. E. (1957). *Anal. Chem.*, 29, 1702–1706.
- Akahira, T., & Sunose, T. (1971). *Res. Report Chiba Inst. Technol.*, 16, 22–31.
- Starink, M. J. (2003). *Thermochim. Acta*, 404, 163–176.
- Vyazovkin, S. (1997). *J. Comput. Chem.*, 18, 393–402.
- Malek, J. (1992). *Thermochimica Acta*, 200, 257–269.
- KTH: original method, accompanying thesis (Chapter VII, pp. 108–112).

---

<div align="center">

**⭐ If this project helped your research, please consider giving it a star! ⭐**

Made with 🔬 and 🐍 by [khadev](https://github.com/khadev)

</div>
