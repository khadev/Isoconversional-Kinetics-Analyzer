"""Synthetic DSC instrument exports with a KNOWN activation energy (shared by the tests).

The signal is generated from the reaction itself, independently of the code under test:
alpha(T) comes from :func:`isokinetics.validation.simulate_constant_ea` (exact integral of the
rate law), the heat flow is ``HF = -dH * (beta/60) * d alpha / dT`` plus a sloping baseline and
noise, and the file layout copies the real exports: a pre-run block with HeatFlow exactly 0,
the heating ramp, then a cooling block with HeatFlow 0 and falling temperature.
"""

from __future__ import annotations

import os

import numpy as np

from isokinetics import validation

RATES = (3.0, 5.0, 7.0, 10.0)           # K/min
EA_TRUE = 150.0                          # kJ/mol
HEADER = "Temp\u00e9rature de l'\u00e9chantillon (\u00b0C) HeatFlow |c (mW)"


def synthetic_curve(beta, ea=EA_TRUE, a_pre=1e12, noise=0.002, seed=0, baseline=(-3.0, 0.004),
                    step_baseline=0.0):
    """Return ``(T_celsius, heat_flow_mW, (T_start_C, T_end_C))`` for one heating rate.

    The reaction runs between roughly alpha = 1e-4 and 0.9999, so the ends of a selection that
    covers it sit on the baseline.  ``baseline=(b0, slope)`` is a linear drift in mW and mW/K;
    ``step_baseline`` adds a heat-capacity step that follows the conversion."""
    grid = np.concatenate([[1e-5], np.arange(1e-4, 1.0, 1e-4), [0.99995]])
    df = validation.simulate_constant_ea(ea, a_pre, "F1", betas_k_per_min=(beta,) * 4, alphas=grid)
    t_of_alpha = df["T1"].to_numpy()                                  # kelvin, increasing
    t_lo, t_hi = t_of_alpha[0] - 25.0, t_of_alpha[-1] + 25.0
    t = np.arange(t_lo, t_hi, 0.02 * beta)                            # constant heating rate, ~0.02 min/point
    alpha = np.interp(t, t_of_alpha, grid, left=0.0, right=1.0)
    dadt = np.gradient(alpha, t)
    hf = -3000.0 * (beta / 60.0) * dadt                               # endothermic (negative) peak
    hf = hf + baseline[0] + baseline[1] * (t - t[0]) + step_baseline * alpha
    hf = hf + np.random.default_rng(seed + int(beta)).normal(0.0, noise, len(t))
    return t - 273.15, hf, (t_of_alpha[0] - 273.15, t_of_alpha[-1] - 273.15)


def _layout(t_c, hf):
    """Pre-run zeros + heating ramp + cooling zeros, like a real export."""
    pre_t = np.linspace(t_c[0] - 6.0, t_c[0] - 0.05, 120)
    cool_t = np.linspace(t_c[-1], t_c[0] - 60.0, 400)
    temp = np.concatenate([pre_t, t_c, cool_t])
    flow = np.concatenate([np.zeros(len(pre_t)), hf, np.zeros(len(cool_t))])
    return temp, flow


def write_utf16_with_header(path, t_c, hf, metadata=True):
    temp, flow = _layout(t_c, hf)
    lines = []
    if metadata:
        lines += ["Ech-DRD-TEST", "Date de cr\u00e9ation : 16/02/2026 15:08:33", "Utilisateur : admin", "",
                  "HeatFlow |dp |c :", " Masse initiale : 0.895 mg", " Masse molaire : N/A", "", HEADER]
    lines += [f"{a:.6f}                         {b:.6g}                   " for a, b in zip(temp, flow)]
    with open(path, "w", encoding="utf-16") as fh:
        fh.write("\n".join(lines) + "\n")


def write_utf16_no_header(path, t_c, hf):
    write_utf16_with_header(path, t_c, hf, metadata=False)


def write_three_columns_no_header(path, t_c, hf):
    """time [min], temperature, heat flow -- no header, so the columns must be mapped by hand."""
    temp, flow = _layout(t_c, hf)
    time_min = np.arange(len(temp)) * 0.02
    with open(path, "w", encoding="utf-16") as fh:
        fh.write("\n".join(f"{m:.4f}\t{a:.6f}\t{b:.6g}" for m, a, b in zip(time_min, temp, flow)) + "\n")


def write_utf8_decimal_comma(path, t_c, hf):
    temp, flow = _layout(t_c, hf)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Temp\u00e9rature (\u00b0C);HeatFlow (mW)\n")
        fh.write("\n".join(f"{a:.5f};{b:.6g}".replace(".", ",") for a, b in zip(temp, flow)) + "\n")


def write_xlsx(path, t_c, hf, header_row=3, kelvin=False):
    import pandas as pd
    temp, flow = _layout(t_c, hf)
    if kelvin:
        temp = temp + 273.15
    rows = [["Sample TEST", None], ["created 2026", None], [None, None]][:header_row]
    unit = "K" if kelvin else "\u00b0C"
    rows.append([f"Temp\u00e9rature de l'\u00e9chantillon ({unit})", "HeatFlow (mW)"])
    rows += [[a, b] for a, b in zip(temp, flow)]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)


def write_xls(path, t_c, hf):
    import xlwt
    temp, flow = _layout(t_c, hf)
    book = xlwt.Workbook()
    sheet = book.add_sheet("Data")
    sheet.write(0, 0, "Temp\u00e9rature de l'\u00e9chantillon (\u00b0C)")
    sheet.write(0, 1, "HeatFlow (mW)")
    for i, (a, b) in enumerate(zip(temp, flow), start=1):
        sheet.write(i, 0, float(a))
        sheet.write(i, 1, float(b))
    book.save(path)


def make_four_files(folder, writers=None, **curve_kw):
    """Write four files (one per rate, names like ``Ech-TEST_5_C_101010.txt``).
    Returns ``(paths, intervals_celsius)``; ``writers`` is one writer per rate."""
    writers = writers or [write_utf16_with_header] * 4
    paths, spans = [], []
    for rate, writer in zip(RATES, writers):
        t_c, hf, span = synthetic_curve(rate, **curve_kw)
        ext = ".xlsx" if writer is write_xlsx else ".xls" if writer is write_xls else ".txt"
        path = os.path.join(str(folder), f"Ech-TEST_{rate:g}_C_10{int(rate):02d}10{ext}")
        writer(path, t_c, hf)
        paths.append(path)
        spans.append(span)
    return paths, spans
