"""
data_loader.py
================
Utilities for loading, validating, and pre-processing isoconversional
kinetics data: conversion (alpha) vs. temperature (T), measured at four
different heating rates (beta).

Two kinds of input file are recognised automatically (see
:func:`load_table`):

**Processed file -- 5 columns** ::

    alpha   T1   T2   T3   T4

``T1..T4`` are the temperatures (Kelvin) at which the given alpha was
reached, one column per heating rate.  The data are ready for analysis.

**Raw file -- 4 columns** ::

    T1   T2   T3   T4

Only temperatures, no conversion column.  The conversion is then *generated*
(:func:`build_alpha_table`), reproducing the manual Origin workflow: every
column is treated as a curve ``T(alpha)`` whose rows are **equally spaced in
conversion**, and the temperature is interpolated linearly at each alpha of a
chosen uniform grid.  Two conventions for placing the rows on the alpha axis
are offered (see ``ROW_MAPPINGS``).  Because a temperature-only file contains
no conversion information, this equal-spacing assumption is *the* assumption
of the raw workflow -- check it matches how your file was produced.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

#: Canonical column names used throughout the rest of the application.
EXPECTED_COLUMNS = ["alpha", "T1", "T2", "T3", "T4"]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a data file containing conversion (alpha) and temperature values
    for four heating rates.

    Supports plain text files (space, tab, or comma/semicolon delimited)
    and Excel workbooks (.xlsx/.xls). Only the first five columns are
    used; any extra columns are ignored.

    Parameters
    ----------
    filepath : str
        Path to the data file.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with columns ``["alpha", "T1", "T2", "T3",
        "T4"]``, sorted by ascending alpha.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed, or does not contain at least five
        usable numeric columns / three usable rows.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".xlsx", ".xls"):
        raw = pd.read_excel(filepath, header=None)
    else:
        raw = _read_text_table(filepath)

    return validate_data(raw)


def _read_text_table(
    filepath: str, min_columns: int = 5, separators: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Attempt to parse a delimited text file trying several common
    separators used by thermal-analysis software (whitespace, tab,
    comma, semicolon).

    Parameters
    ----------
    filepath : str
        Path to the text file.
    min_columns : int, optional
        Minimum number of columns a parse must produce to be accepted.
    separators : sequence of str, optional
        Separators to try, in order (default: whitespace, tab, comma,
        semicolon).

    Returns
    -------
    pandas.DataFrame
        Raw (un-validated) parsed table.

    Raises
    ------
    ValueError
        If no separator produces a table with >= ``min_columns`` columns.
    """
    last_error: Optional[Exception] = None
    for sep in (separators if separators is not None else [r"\s+", "\t", ",", ";"]):
        try:
            candidate = pd.read_csv(
                filepath,
                sep=sep,
                engine="python",
                header=None,
                comment="#",
                skip_blank_lines=True,
            )
            if candidate.shape[1] >= min_columns:
                return candidate
        except Exception as exc:  # noqa: BLE001 - we want to try the next separator
            last_error = exc
            continue

    raise ValueError(
        f"Could not parse '{filepath}' as a delimited text file "
        f"(tried whitespace/tab/comma/semicolon). Last error: {last_error}"
    )


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalise a raw DataFrame loaded from disk.

    Steps performed:
        1. Keep only the first five columns.
        2. Rename them to ``alpha, T1, T2, T3, T4``.
        3. Coerce every value to numeric (non-numeric entries -> NaN).
        4. Drop rows containing any NaN.
        5. Sort by ascending alpha and reset the index.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw DataFrame as loaded from file (may have more than 5 columns).

    Returns
    -------
    pandas.DataFrame
        Cleaned, validated DataFrame.

    Raises
    ------
    ValueError
        If fewer than 5 columns are present, or fewer than 3 valid rows
        remain after cleaning.
    """
    if df.shape[1] < 5:
        raise ValueError(
            f"Expected at least 5 columns (alpha, T1..T4), got {df.shape[1]}."
        )

    clean = df.iloc[:, :5].copy()
    clean.columns = EXPECTED_COLUMNS

    for col in EXPECTED_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(axis=0, how="any")

    if len(clean) < 3:
        raise ValueError(
            "Not enough valid numeric rows found in the data file "
            "(at least 3 rows with alpha, T1, T2, T3, T4 are required)."
        )

    # Basic physical sanity checks (non-fatal -> raise with a clear message)
    if (clean["alpha"] < 0).any() or (clean["alpha"] > 1).any():
        raise ValueError("Column 'alpha' must contain values between 0 and 1.")
    for col in ["T1", "T2", "T3", "T4"]:
        if (clean[col] <= 0).any():
            raise ValueError(
                f"Column '{col}' must contain strictly positive temperatures (Kelvin)."
            )

    clean = clean.sort_values("alpha").reset_index(drop=True)
    return clean


def interpolate_alpha(
    df: pd.DataFrame,
    alpha_min: float = 0.01,
    alpha_max: float = 0.99,
    step: float = 0.01,
) -> pd.DataFrame:
    """
    Resample the temperature columns onto a uniform alpha grid using
    linear interpolation, so that all isoconversional methods are
    evaluated at exactly the same set of alpha values.

    If interpolation is not possible or not sensible (fewer than 4
    original points, or an empty overlap with the requested range), the
    original DataFrame is returned unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated data with columns alpha, T1, T2, T3, T4.
    alpha_min : float, optional
        Minimum alpha value of the output grid (default 0.01).
    alpha_max : float, optional
        Maximum alpha value of the output grid (default 0.99).
    step : float, optional
        Spacing of the output alpha grid (default 0.01).

    Returns
    -------
    pandas.DataFrame
        DataFrame resampled on the uniform alpha grid, or the original
        DataFrame if resampling could not be performed safely.
    """
    if len(df) < 4:
        return df.copy()

    a_orig = df["alpha"].to_numpy()

    lo = max(alpha_min, float(a_orig.min()))
    hi = min(alpha_max, float(a_orig.max()))
    if hi <= lo:
        return df.copy()

    n_points = int(round((hi - lo) / step)) + 1
    if n_points < 3:
        return df.copy()

    grid = np.linspace(lo, hi, n_points)

    out = {"alpha": grid}
    for col in ["T1", "T2", "T3", "T4"]:
        out[col] = np.interp(grid, a_orig, df[col].to_numpy())

    return pd.DataFrame(out)


# =========================================================================== #
# Automatic detection: processed (5 columns) vs raw (4 columns)
# =========================================================================== #
KIND_PROCESSED = "processed"
KIND_RAW = "raw"
RAW_COLUMNS = ["T1", "T2", "T3", "T4"]

#: How the rows of a raw curve are placed on the conversion axis.
#:
#: ``"endpoints"``: first row is alpha = 0 and last row is alpha = 1
#:   (row i of n -> alpha = i / (n - 1)), i.e. the curve spans the whole
#:   conversion range.
#: ``"interior"``: alpha = 0 and 1 are *not* in the file; row i of n ->
#:   alpha = i / (n + 1), i = 1..n (e.g. 99 rows -> 0.01 ... 0.99, exactly the
#:   layout of a processed file without its alpha column).
ROW_MAPPINGS = {
    "endpoints": "First row \u03b1 = 0, last row \u03b1 = 1",
    "interior": "Rows are interior points: \u03b1 = i/(n+1)  (99 rows \u2192 0.01 \u2026 0.99)",
}
DEFAULT_ROW_MAPPING = "endpoints"

#: Allowed range of the conversion step (fraction of conversion).
MIN_ALPHA_STEP = 0.001
MAX_ALPHA_STEP = 0.25


@dataclass
class LoadedTable:
    """Result of :func:`load_table`."""

    kind: str                      # KIND_PROCESSED or KIND_RAW
    data: pd.DataFrame             # processed: alpha,T1..T4 ; raw: T1..T4 (NaN-padded if ragged)
    n_columns: int                 # columns found in the file
    rows_per_column: List[int] = field(default_factory=list)  # raw only

    @property
    def is_raw(self) -> bool:
        return self.kind == KIND_RAW

    @property
    def n_rows(self) -> int:
        return int(max(self.rows_per_column)) if self.rows_per_column else len(self.data)


def load_table(filepath: str) -> LoadedTable:
    """
    Load a text/Excel file and decide what it contains from its number of
    columns:

    * **5 or more columns** -> *processed* file ``[alpha, T1, T2, T3, T4]``
      (extra columns ignored, exactly as :func:`load_data`).
    * **4 columns** -> *raw* file ``[T1, T2, T3, T4]``; the alpha column
      still has to be generated with :func:`build_alpha_table`.
    * anything else -> ``ValueError``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed or fails validation.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".xlsx", ".xls"):
        table = pd.read_excel(filepath, header=None)
        table = table.dropna(axis=1, how="all")  # ignore fully empty columns
    else:
        # Tab first: unlike whitespace splitting it keeps empty cells, so
        # columns of unequal length stay aligned.
        seps = ["\t", r"\s+", ",", ";"]
        try:
            table = _read_text_table(filepath, min_columns=4, separators=seps)
        except ValueError as exc:
            # Tell the user what was found instead of a generic parse error.
            try:
                few = _read_text_table(filepath, min_columns=2, separators=[r"\s+", "\t", ",", ";"])
            except ValueError:
                raise exc from None
            raise ValueError(
                f"Found {few.dropna(axis=1, how='all').shape[1]} column(s). Expected 5 "
                "(alpha, T1..T4: processed data) or 4 (T1..T4: raw data, alpha will be generated)."
            ) from None
        table = table.dropna(axis=1, how="all")

    n_cols = table.shape[1]
    if n_cols >= 5:
        return LoadedTable(KIND_PROCESSED, validate_data(table), n_cols)
    if n_cols == 4:
        raw = validate_raw_data(table)
        return LoadedTable(
            KIND_RAW, raw, n_cols, [int(raw[c].notna().sum()) for c in RAW_COLUMNS]
        )
    raise ValueError(
        f"Found {n_cols} column(s). Expected 5 (alpha, T1..T4: processed data) "
        "or 4 (T1..T4: raw data, alpha will be generated)."
    )


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a raw 4-column table of temperatures.

    * non-numeric cells (e.g. a header line) and blanks are dropped
      **per column**, so the four curves may have different lengths;
    * every curve needs at least 4 valid points;
    * temperatures must be strictly positive (kelvin) and must not
      decrease along a curve (heating experiments);
    * a first column that only contains values in [0, 1] is rejected: that
      is a conversion column, i.e. the file has only three temperature
      columns.

    Returns a DataFrame with columns ``T1..T4`` (shorter curves padded with
    NaN at the end).
    """
    if df.shape[1] < 4:
        raise ValueError(f"Expected 4 temperature columns (T1..T4), got {df.shape[1]}.")

    curves = {}
    for name, col in zip(RAW_COLUMNS, df.columns[:4]):
        values = pd.to_numeric(df[col], errors="coerce").dropna().reset_index(drop=True)
        if len(values) < 4:
            raise ValueError(
                f"Column {name} has only {len(values)} numeric value(s); at least 4 are needed."
            )
        if (values <= 0).any():
            raise ValueError(f"Column {name} must contain strictly positive temperatures (Kelvin).")
        drops = np.flatnonzero(np.diff(values.to_numpy()) < 0)
        if drops.size:
            k = int(drops[0]) + 1
            raise ValueError(
                f"Column {name} is not increasing at value #{k + 1} "
                f"({values.iloc[k - 1]:g} \u2192 {values.iloc[k]:g}). A heating curve must rise "
                "monotonically; smooth or trim the data first."
            )
        curves[name] = values

    first = curves["T1"].to_numpy()
    if first.min() >= 0 and first.max() <= 1.0:
        raise ValueError(
            "The first column contains only values between 0 and 1: it looks like a "
            "conversion (alpha) column, so the file seems to have just three temperature "
            "columns. Four heating rates (T1..T4) are required."
        )
    return pd.DataFrame(curves)


# =========================================================================== #
# Generating alpha and interpolating (the Origin step)
# =========================================================================== #
_FRACTION = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)\s*$")


def parse_alpha_step(text: str) -> float:
    """
    Convert what the user typed into a conversion step ``d_alpha``.

    =========  =====================================================
    ``0.02``   step 0.02
    ``1/50``   step 0.02
    ``50``     50 intervals -> step 1/50 = 0.02  (numbers >= 2)
    ``0,01``   decimal comma accepted
    =========  =====================================================

    Raises ``ValueError`` with a user-readable message otherwise.
    """
    raw = (text or "").strip().replace(",", ".")
    if not raw:
        raise ValueError("Enter a conversion step (e.g. 0.01, 1/50) or a number of points (e.g. 100).")

    match = _FRACTION.match(raw)
    try:
        if match:
            den = float(match.group(2))
            if den == 0:
                raise ValueError("Division by zero.")
            step = float(match.group(1)) / den
        else:
            value = float(raw)
            if value >= 2:
                if abs(value - round(value)) > 1e-9:
                    raise ValueError("A number of points must be a whole number.")
                step = 1.0 / round(value)
            else:
                step = value
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("could not convert"):
            msg = f"'{text.strip()}' is not a valid step. Use e.g. 0.01, 1/50 or 100."
        raise ValueError(msg) from None

    if not math.isfinite(step) or step <= 0 or step >= 1:
        raise ValueError(
            "The step must be between 0 and 1 (e.g. 0.01), or a number of points of 2 or more (e.g. 100)."
        )
    if step < MIN_ALPHA_STEP - 1e-12:
        raise ValueError(f"The step {step:g} is too small (minimum {MIN_ALPHA_STEP:g}, i.e. 1000 points).")
    if step > MAX_ALPHA_STEP + 1e-12:
        raise ValueError(f"The step {step:g} is too large (maximum {MAX_ALPHA_STEP:g}, i.e. 3 points).")
    return step


def alpha_grid(step: float) -> np.ndarray:
    """
    Uniform conversion grid ``step, 2*step, ...`` strictly below 1.

    alpha = 0 and alpha = 1 are excluded: the activation energy is undefined
    there (``ln(1 - alpha)``, ``d alpha/dt -> 0``).  So 0.01 gives 99 points
    (0.01 ... 0.99) and 1/50 gives 49 points (0.02 ... 0.98).
    """
    if not (MIN_ALPHA_STEP - 1e-12 <= step <= MAX_ALPHA_STEP + 1e-12):
        raise ValueError(f"step must be between {MIN_ALPHA_STEP:g} and {MAX_ALPHA_STEP:g}.")
    n = int(math.floor((1.0 - 1e-9) / step))
    return np.round(np.arange(1, n + 1) * step, 10)


def row_alpha(n_rows: int, mapping: str = DEFAULT_ROW_MAPPING) -> np.ndarray:
    """Conversion assigned to each of ``n_rows`` rows of a raw curve (see
    ``ROW_MAPPINGS``)."""
    if mapping == "endpoints":
        return np.linspace(0.0, 1.0, n_rows)
    if mapping == "interior":
        return np.arange(1, n_rows + 1) / (n_rows + 1.0)
    raise ValueError(f"Unknown row mapping {mapping!r}; choose one of {sorted(ROW_MAPPINGS)}.")


def build_alpha_table(
    raw: pd.DataFrame, step: float, mapping: str = DEFAULT_ROW_MAPPING
) -> pd.DataFrame:
    """
    Turn a raw ``T1..T4`` table into a processed ``[alpha, T1..T4]`` table.

    For each column the rows are placed on the conversion axis according to
    ``mapping`` and the temperature is **linearly interpolated** at every
    alpha of ``alpha_grid(step)`` -- the operation otherwise done by hand in
    Origin.  Alpha values outside the range covered by the data (possible
    with ``mapping="interior"``) are dropped instead of being extrapolated.

    Parameters
    ----------
    raw : DataFrame
        Columns T1..T4 (kelvin), as returned by :func:`validate_raw_data`.
    step : float
        Conversion step (see :func:`parse_alpha_step`).
    mapping : {"endpoints", "interior"}

    Returns
    -------
    DataFrame
        Columns alpha, T1..T4 on the uniform grid, ready for
        :mod:`isokinetics.kinetics` (no further resampling is needed).
    """
    curves = {}
    for name in RAW_COLUMNS:
        values = raw[name].dropna().to_numpy(dtype=float)
        if len(values) < 4:
            raise ValueError(f"Column {name} has fewer than 4 values.")
        curves[name] = (row_alpha(len(values), mapping), values)

    lo = max(c[0][0] for c in curves.values())
    hi = min(c[0][-1] for c in curves.values())
    grid = alpha_grid(step)
    grid = grid[(grid >= lo - 1e-12) & (grid <= hi + 1e-12)]
    if len(grid) < 3:
        raise ValueError(
            "The chosen step leaves fewer than 3 conversion points inside the range covered by the "
            "data. Use a larger step or the other row mapping."
        )

    table = {"alpha": grid}
    for name, (a_rows, values) in curves.items():
        table[name] = np.interp(grid, a_rows, values)
    return pd.DataFrame(table)


def resolution_note(raw: pd.DataFrame, step: float, mapping: str = DEFAULT_ROW_MAPPING) -> Optional[str]:
    """Warning text when ``step`` is finer than the spacing of the rows (the
    interpolation then adds no information), else ``None``."""
    n_min = min(int(raw[c].notna().sum()) for c in RAW_COLUMNS)
    spacing = 1.0 / (n_min - 1) if mapping == "endpoints" else 1.0 / (n_min + 1)
    if step < spacing * 0.999:
        return (
            f"Your shortest curve has {n_min} rows (\u2248 {spacing:.4f} conversion per row); "
            f"a step of {step:g} is finer than the data, so values are interpolated between rows."
        )
    return None


def describe_grid(step: float) -> str:
    """``"99 points, \u03b1 = 0.010 \u2026 0.990 (\u0394\u03b1 = 0.01)"`` for the dialog preview."""
    grid = alpha_grid(step)
    return (f"{len(grid)} points, \u03b1 = {grid[0]:.3f} \u2026 {grid[-1]:.3f} "
            f"(\u0394\u03b1 = {step:g})")


def save_processed_table(df: pd.DataFrame, path: str) -> str:
    """
    Write a processed ``[alpha, T1..T4]`` table (headerless, tab-delimited
    unless ``path`` ends with ``.csv``) that :func:`load_data` reads back.
    Returns ``path``.
    """
    sep = "," if path.lower().endswith(".csv") else "\t"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for _, row in df.iterrows():
            cells = [f"{row['alpha']:.4f}"] + [f"{row[c]:.5f}" for c in RAW_COLUMNS]
            fh.write(sep.join(cells) + "\n")
    return path


#: Conversion steps that "look intentional"; used to guess the row layout.
_NICE_STEPS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1)


def suggest_mapping(n_rows: int) -> str:
    """
    Guess how the rows of a raw file sit on the conversion axis, from the
    row count alone.

    A file with 99 / 49 / 199 ... rows is almost certainly the *interior*
    layout (``alpha = i/(n+1)`` = 0.01 ... 0.99, i.e. a processed file that
    lost its alpha column); using the wrong layout shifts every temperature
    by up to a step.  Any other row count returns the default
    (``"endpoints"``: first row alpha = 0, last row alpha = 1).
    """
    if n_rows >= 3 and any(abs(1.0 / (n_rows + 1) - s) < 1e-9 for s in _NICE_STEPS):
        return "interior"
    return DEFAULT_ROW_MAPPING


# =========================================================================== #
# Raw instrument exports (DSC): four files, one per heating rate  --  Import wizard backend
# =========================================================================== #
# What these files look like (checked against real exports):
#
#   * text files are often UTF-16 with a BOM, whitespace separated, with a metadata block and
#     a header row such as  "Temperature de l'echantillon (degC) HeatFlow |dp |c (mW)";
#     some exports have **no header at all**;
#   * every file holds three segments: a pre-run block with HeatFlow exactly 0, the heating
#     ramp (temperature strictly increasing) and a cooling block, again HeatFlow 0 with the
#     temperature falling back -- plotted naively this draws a flat line across the graph;
#   * the heating rate is only in the file name (``..._5_C_...`` = 5 degC/min).
#
# Conversion alpha is NOT the normalised heat flow.  The heat flow is a *rate* (power); the
# conversion is the cumulative *area* under the baseline-corrected peak divided by the total
# area, which rises monotonically from 0 to 1 (min-max scaling of a peak goes up *and* down,
# so it cannot be a conversion, and kinetic fits to it would be meaningless).

INSTRUMENT_EXTENSIONS = (".xls", ".xlsx", ".xlsm", ".txt", ".csv", ".dat", ".tsv")
TEMP_COL = "Temperature_K"
HEATFLOW_COL = "HeatFlow_mW"
KELVIN_OFFSET = 273.15
N_CURVES = 4                       # the kinetic engine works with exactly four heating rates

#: A selection must contain at least this many points in every curve.
MIN_SELECTION_POINTS = 10

BASELINES = {
    "linear": "Linear (straight line between the selection ends)",
    "sigmoidal": "Sigmoidal (baseline follows the conversion)",
}
DEFAULT_BASELINE = "linear"


class SelectionTooSmall(ValueError):
    """The temperature range holds too few data points to analyse."""


def _fold(text: str) -> str:
    """Lower-case, accent-free text for keyword matching."""
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFKD", str(text).lower()) if not unicodedata.combining(c)
    )


def _is_temp_header(text: str) -> bool:
    return "temp" in _fold(text)


def _is_flow_header(text: str) -> bool:
    folded = _fold(text)
    return "heat" in folded or "flow" in folded or "flux" in folded


def unit_from_name(name: str, default: str = "C") -> str:
    """``"K"`` if a column name says ``(K)``, ``"C"`` for ``(degC)`` / ``(C)`` / Celsius, else ``default``."""
    folded = _fold(name)
    if re.search(r"\(\s*k\s*\)|\bkelvin\b", folded):
        return "K"
    if re.search(r"\(\s*\u00b0?\s*c\s*\)|degc|celsius", folded):
        return "C"
    return default


def read_text_lines(path: str) -> List[str]:
    """
    All lines of a text file whatever its encoding: UTF-16 (with or without BOM; the usual
    DSC export), UTF-8 (with or without BOM), or a legacy 8-bit code page.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw[3:].decode("utf-8")
    elif raw.count(b"\x00") > len(raw) // 4:           # UTF-16 without a BOM
        text = raw.decode("utf-16-le" if raw[1:2] == b"\x00" else "utf-16-be", errors="replace")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
    return text.splitlines()


def _tokens(line: str) -> List[str]:
    """Split a line on tabs / spaces / ';' / ',' (CSV)."""
    s = line.strip()
    if not s:
        return []
    if ";" in s:
        return [p.strip() for p in s.split(";") if p.strip()]
    if re.search(r",\s", s) or (s.count(",") == 1 and not re.search(r"\s", s)):
        return [p for p in re.split(r"\s*,\s*", s) if p]
    return s.split()


def _numeric_row(line: str) -> Optional[List[float]]:
    """All numbers of a line if *every* token is numeric (decimal point or comma), else None.
    Strictness keeps metadata lines such as ``Masse initiale : 0.895 mg`` out of the data."""
    toks = _tokens(line)
    if len(toks) < 2:
        return None
    try:
        vals = [float(t.replace(",", ".")) for t in toks]
    except ValueError:
        return None
    return vals if all(math.isfinite(v) for v in vals) else None


def _split_header(text: str, ncols: int) -> Optional[List[str]]:
    """Split one header line into ``ncols`` column names (tabs, then 2+ spaces, then at the
    ``(unit)`` boundaries, then single spaces); ``None`` if it cannot be done reliably."""
    s = text.strip()
    candidates = [
        [p.strip() for p in s.split("\t") if p.strip()],
        [p.strip() for p in re.split(r"\s{2,}", s) if p.strip()],
        [m.strip() for m in re.findall(r"[^()]*?\([^()]*\)", s) if m.strip()],
        s.split(),
    ]
    for parts in candidates:
        if len(parts) == ncols:
            return parts
    return None


def _unique(names: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


_RATE_PATTERN = re.compile(r"(?<![0-9.,])(\d+(?:[.,]\d+)?)\s*_?\s*\u00b0?\s*[CK](?=[_\-\s./]|$)")


def heating_rate_from_name(path_or_text: str) -> Optional[float]:
    """
    The heating rate (K/min) written in a file name, e.g. ``Ech-DRD-E01-1_5_C_100711.txt``
    -> 5.0 (``..._10_C_...`` -> 10.0, ``sample 7\u00b0C.xlsx`` -> 7.0); ``None`` if there is none.
    """
    stem = os.path.splitext(os.path.basename(str(path_or_text)))[0]
    match = _RATE_PATTERN.search(stem)
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    return value if value > 0 else None


def order_by_heating_rate(paths: Sequence[str]) -> List[str]:
    """
    The files in ascending heating rate when every name carries a (distinct) rate -- the
    order T1..T4 must have -- otherwise in natural name order (``_3`` before ``_10``; a plain
    alphabetical sort would put ``_10`` first).
    """
    rates = [heating_rate_from_name(p) for p in paths]
    if all(r is not None for r in rates) and len(set(rates)) == len(rates):
        return [p for _, p in sorted(zip(rates, paths), key=lambda rp: rp[0])]

    def natural(p: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(p))]

    return sorted(paths, key=natural)


@dataclass
class RawTable:
    """Every numeric column of one instrument file, as read (original units, nothing converted)."""

    data: pd.DataFrame                  # columns named ``names``
    names: List[str]
    generic_names: bool                 # True when the file has no usable header ("Column 1", ...)
    header_row: Optional[int]
    source: str
    heating_rate: Optional[float]       # from the file name
    notes: List[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.data)

    def preview(self, n: int = 10) -> pd.DataFrame:
        """The first ``n`` rows (what the wizard shows)."""
        return self.data.head(n)


def _raw_table_text(path: str) -> RawTable:
    lines = read_text_lines(path)
    rows = [(i, _numeric_row(line)) for i, line in enumerate(lines)]
    numeric = [(i, r) for i, r in rows if r is not None]
    if len(numeric) < 3:
        raise ValueError(f"{os.path.basename(path)}: no table of numbers found.")
    counts: Dict[int, int] = {}
    for _, r in numeric:
        counts[len(r)] = counts.get(len(r), 0) + 1
    ncols = max(counts, key=lambda k: (counts[k], k))
    block = [(i, r) for i, r in numeric if len(r) == ncols]
    first_idx = block[0][0]

    notes: List[str] = []
    header_row: Optional[int] = None
    names: Optional[List[str]] = None
    candidates = [i for i in range(first_idx) if _is_temp_header(lines[i])]
    if candidates:
        header_row = candidates[-1]                       # the one nearest to the data
        names = _split_header(lines[header_row], ncols)
        if names is None:
            notes.append(f"header found but it does not split into {ncols} columns: generic names used")
    else:
        notes.append("no header row found: generic column names used")
    generic = names is None
    names = _unique(names) if names else [f"Column {k}" for k in range(1, ncols + 1)]
    data = pd.DataFrame([r for _, r in block], columns=names)
    return RawTable(data, names, generic, header_row, path, heating_rate_from_name(path), notes)


def _raw_table_excel(path: str) -> RawTable:
    try:
        sheet = pd.read_excel(path, header=None, sheet_name=0)
    except ImportError as exc:
        raise ValueError(
            f"Reading {os.path.splitext(path)[1]} files needs an extra package "
            f"(.xls: pip install xlrd; .xlsx: pip install openpyxl): {exc}") from exc
    numeric = sheet.apply(pd.to_numeric, errors="coerce")
    header_row: Optional[int] = None
    for r in range(min(len(sheet), 60)):                  # the header sits near the top
        if any(isinstance(v, str) and _is_temp_header(v) for v in sheet.iloc[r]):
            header_row = r
            break
    first = 0 if header_row is None else header_row + 1
    body = numeric.iloc[first:]
    usable = [c for c in body.columns if body[c].notna().mean() > 0.5]
    if len(usable) < 2:
        raise ValueError(f"{os.path.basename(path)}: need at least two numeric columns.")
    body = body[usable].dropna(how="any")
    if len(body) < 3:
        raise ValueError(f"{os.path.basename(path)}: fewer than 3 numeric rows.")

    notes: List[str] = []
    names: Optional[List[str]] = None
    if header_row is not None:
        cells = [sheet.iloc[header_row][c] for c in usable]
        if all(isinstance(v, str) and v.strip() for v in cells):
            names = [v.strip() for v in cells]
        else:                                              # one cell holding every name?
            joined = [v for v in sheet.iloc[header_row] if isinstance(v, str)]
            names = _split_header(joined[0], len(usable)) if len(joined) == 1 else None
    if names is None:
        notes.append("no usable header names: generic column names used")
    generic = names is None
    names = _unique(names) if names else [f"Column {k}" for k in range(1, len(usable) + 1)]
    data = body.copy()
    data.columns = names
    return RawTable(data.reset_index(drop=True), names, generic, header_row, path,
                    heating_rate_from_name(path), notes)


def load_raw_table(path: str) -> RawTable:
    """
    Read every numeric column of one raw instrument file (``.xls``/``.xlsx``, or a text
    export: ``.txt``/``.csv``/``.dat``; UTF-16, UTF-8 or a legacy code page) without
    interpreting it.  The header row is found dynamically; **if the file has none, generic
    names "Column 1", "Column 2" ... are used** so the columns can still be mapped by hand.
    """
    if os.path.splitext(path)[1].lower() in (".xls", ".xlsx", ".xlsm"):
        return _raw_table_excel(path)
    return _raw_table_text(path)


def detect_columns(names: Sequence[str]) -> Tuple[int, int]:
    """
    Indices ``(x, y)`` of the temperature and heat-flow columns: the first name containing
    "temp" and the first containing "heat" / "flow" / "flux".  With generic or unrecognised
    names this falls back to the first two columns.
    """
    n = len(names)
    if n < 2:
        raise ValueError("At least two columns are needed.")
    x = next((i for i, nm in enumerate(names) if _is_temp_header(nm)), None)
    y = next((i for i, nm in enumerate(names) if _is_flow_header(nm) and i != x), None)
    if x is None:
        x = next((i for i in range(n) if i != y), 0)
    if y is None:
        y = next((i for i in range(n) if i != x), 1)
    return x, y


def extract_heating_ramp(
    temp_k: np.ndarray, flow: np.ndarray
) -> "Tuple[np.ndarray, np.ndarray, List[str]]":
    """
    Keep only the heating ramp of an instrument export.

    1. cut everything after the temperature maximum (the cooling segment);
    2. strip leading / trailing rows whose heat flow is exactly 0 (the instrument is not
       recording yet / any more -- these draw the flat line across a naive plot);
    3. keep only strictly increasing temperatures (drops repeated points and jitter).

    Returns ``(T_K, heat_flow, notes)`` describing what was removed.
    """
    notes: List[str] = []
    n0 = len(temp_k)
    imax = int(np.argmax(temp_k))
    if imax < n0 - 1:
        notes.append(f"cooling segment removed ({n0 - 1 - imax} rows after the temperature maximum)")
    temp_k, flow = temp_k[: imax + 1], flow[: imax + 1]

    nonzero = np.flatnonzero(flow != 0)
    if nonzero.size == 0:
        raise ValueError("The heat flow is 0 everywhere: nothing was recorded.")
    lead, trail = int(nonzero[0]), int(len(flow) - 1 - nonzero[-1])
    if lead or trail:
        notes.append(f"{lead + trail} rows with heat flow exactly 0 removed at the ends")
    temp_k, flow = temp_k[nonzero[0]: nonzero[-1] + 1], flow[nonzero[0]: nonzero[-1] + 1]

    running_max = np.maximum.accumulate(temp_k)
    keep = np.r_[True, temp_k[1:] > running_max[:-1]]
    if not keep.all():
        notes.append(f"{int((~keep).sum())} rows with non-increasing temperature removed")
    temp_k, flow = temp_k[keep], flow[keep]
    if len(temp_k) < 3:
        raise ValueError("Fewer than 3 points remain on the heating ramp.")
    return temp_k, flow, notes


def curve_from_table(
    table: RawTable,
    x_col: Union[int, str],
    y_col: Union[int, str],
    unit: Optional[str] = None,
    heating_only: bool = True,
) -> pd.DataFrame:
    """
    Turn two mapped columns of a :class:`RawTable` into a clean curve
    (``Temperature_K``, ``HeatFlow_mW``): NaN/inf rows dropped, degC -> kelvin, and (with
    ``heating_only``) just the heating ramp (:func:`extract_heating_ramp`).

    ``unit`` is ``"C"`` or ``"K"``; ``None`` takes it from the X column's name (default degC).
    ``df.attrs`` carries ``source``, ``heating_rate`` (from the file name), ``notes`` and the
    mapping.
    """
    def index(key: Union[int, str]) -> int:
        return table.names.index(key) if isinstance(key, str) else int(key)

    xi, yi = index(x_col), index(y_col)
    if xi == yi:
        raise ValueError("The X and Y axes must be different columns.")
    unit = unit or unit_from_name(table.names[xi])
    temp = table.data.iloc[:, xi].to_numpy(dtype=float)
    flow = table.data.iloc[:, yi].to_numpy(dtype=float)
    ok = np.isfinite(temp) & np.isfinite(flow)
    temp, flow = temp[ok], flow[ok]
    if len(temp) < 3:
        raise ValueError("Fewer than 3 numeric rows in the mapped columns.")
    if unit.upper().startswith("C"):
        temp = temp + KELVIN_OFFSET
    notes = list(table.notes)
    if heating_only:
        temp, flow, extra = extract_heating_ramp(temp, flow)
        notes.extend(extra)
    curve = pd.DataFrame({TEMP_COL: temp, HEATFLOW_COL: flow})
    curve.attrs.update({
        "source": table.source, "heating_rate": table.heating_rate, "notes": notes,
        "x_name": table.names[xi], "y_name": table.names[yi], "temperature_unit": unit.upper()[:1],
    })
    return curve


def load_raw_instrument_excel(
    path: str, heating_only: bool = True, temperature_unit: Optional[str] = None
) -> pd.DataFrame:
    """
    One call: read an instrument export (``.xls``/``.xlsx`` and the ``.txt``/``.csv`` text
    exports), find the header row dynamically, auto-detect the temperature and heat-flow
    columns, drop NaNs, convert degC -> **kelvin** and keep the heating ramp.  Returns the
    two-column curve of :func:`curve_from_table`.  (The import wizard uses the finer-grained
    :func:`load_raw_table` / :func:`curve_from_table` so the user can override the mapping.)
    """
    table = load_raw_table(path)
    xi, yi = detect_columns(table.names)
    return curve_from_table(table, xi, yi, temperature_unit, heating_only)


def is_instrument_file(path: str) -> bool:
    """True if ``path`` has a supported extension and parses as an instrument curve."""
    if os.path.splitext(path)[1].lower() not in INSTRUMENT_EXTENSIONS:
        return False
    try:
        load_raw_instrument_excel(path)
    except Exception:  # noqa: BLE001
        return False
    return True


# --------------------------------------------------------------------------- #
# Selection -> conversion
# --------------------------------------------------------------------------- #
def select_range(curve: pd.DataFrame, t_lo_k: float, t_hi_k: float) -> pd.DataFrame:
    """Rows of ``curve`` with ``t_lo_k <= T <= t_hi_k`` (inclusive)."""
    lo, hi = sorted((float(t_lo_k), float(t_hi_k)))
    t = curve[TEMP_COL]
    return curve[(t >= lo) & (t <= hi)].reset_index(drop=True)


def count_in_range(curve: pd.DataFrame, t_lo_k: float, t_hi_k: float) -> int:
    """Number of data points of one curve inside the temperature range."""
    return len(select_range(curve, t_lo_k, t_hi_k))


def check_selection(
    curve: pd.DataFrame, t_lo_k: float, t_hi_k: float,
    min_points: int = MIN_SELECTION_POINTS, label: str = "this file",
) -> int:
    """Raise :class:`SelectionTooSmall` if the range holds fewer than ``min_points`` points of
    ``curve``; otherwise return the count (the fallback for "the selection is too small")."""
    n = count_in_range(curve, t_lo_k, t_hi_k)
    if n < min_points:
        raise SelectionTooSmall(
            f"The selected interval in {label} holds only {n} data point(s); at least "
            f"{min_points} are needed. Select a wider interval.")
    return n


def _cumulative_area(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of ``y`` over ``x`` (same length, starting at 0)."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def conversion_from_heat_flow(
    temp_k: np.ndarray, flow: np.ndarray, baseline: str = DEFAULT_BASELINE
) -> "Tuple[np.ndarray, np.ndarray, Dict[str, float]]":
    """
    Conversion alpha(T) of a DSC peak from its heat flow.

    ``alpha(T)`` is the area of the baseline-corrected peak up to ``T`` divided by its total
    area, so it rises monotonically from 0 to 1 whatever the sign of the peak (exothermic or
    endothermic).  The heating rate is constant, so ``dt = dT / beta`` and ``beta`` cancels in
    the ratio: integrating over temperature is exact.

    Baselines
    ---------
    ``"linear"``     straight line between the heat flow at the two ends of the selection (each
                     end level is the median of a few points, so noise does not tilt it).
    ``"sigmoidal"``  the baseline is ``h0 + (h1 - h0) * alpha(T)``: it moves from the pre-peak to
                     the post-peak level in step with the conversion (iterated to convergence),
                     the usual treatment of a heat-capacity step.

    Returns ``(alpha, baseline_curve, quality)``; ``quality`` has ``max_dip`` (largest decrease
    of the raw cumulative area as a fraction of the total -- large values mean the heat flow
    crosses the baseline, i.e. the selection or baseline is questionable) and ``area``.
    """
    t = np.asarray(temp_k, dtype=float)
    h = np.asarray(flow, dtype=float)
    n = len(t)
    if n < MIN_SELECTION_POINTS:
        raise SelectionTooSmall(f"Only {n} points: at least {MIN_SELECTION_POINTS} are needed.")
    if baseline not in BASELINES:
        raise ValueError(f"Unknown baseline {baseline!r}; choose one of {sorted(BASELINES)}.")
    span = t[-1] - t[0]
    if not span > 0:
        raise ValueError("The temperature must increase over the selected range.")

    k = max(1, min(5, n // 10))
    h0, h1 = float(np.median(h[:k])), float(np.median(h[-k:]))
    x = (t - t[0]) / span
    base = h0 + (h1 - h0) * x
    if baseline == "sigmoidal":
        for _ in range(50):
            cum = _cumulative_area(h - base, t)
            if abs(cum[-1]) < 1e-300:
                break
            new = h0 + (h1 - h0) * (cum / cum[-1])
            converged = np.max(np.abs(new - base)) < 1e-9 * max(1.0, abs(h1 - h0))
            base = new
            if converged:
                break
    cum = _cumulative_area(h - base, t)
    total = float(cum[-1])
    if not abs(total) > 1e-12 * max(1.0, float(np.max(np.abs(h))) * span):
        raise ValueError("The baseline-corrected peak has no measurable area in this interval.")
    raw_alpha = cum / total                      # the sign of the peak cancels in the ratio
    alpha = np.clip(np.maximum.accumulate(raw_alpha), 0.0, 1.0)
    quality = {"max_dip": float(np.max(np.maximum.accumulate(raw_alpha) - raw_alpha)), "area": abs(total)}
    return alpha, base, quality


def curve_conversion(
    curve: pd.DataFrame, t_lo_k: float, t_hi_k: float, baseline: str = DEFAULT_BASELINE
) -> "Optional[Tuple[np.ndarray, np.ndarray]]":
    """``(T_K, alpha)`` of ``curve`` inside the interval, or ``None`` where it cannot be
    computed (too few points, no peak area) -- used for the live preview."""
    sub = select_range(curve, t_lo_k, t_hi_k)
    try:
        alpha, _, _ = conversion_from_heat_flow(sub[TEMP_COL].to_numpy(), sub[HEATFLOW_COL].to_numpy(), baseline)
    except ValueError:
        return None
    return sub[TEMP_COL].to_numpy(), alpha


@dataclass
class InstrumentResult:
    """The outcome of cropping and converting four instrument curves."""

    table: pd.DataFrame                 # alpha, T1..T4 (kelvin); columns in ascending heating rate
    betas: List[float]                  # K/min, ascending (matches T1..T4)
    files: List[str]                    # source files in the same order
    ranges: List["Tuple[float, float]"]  # the interval selected in each file, kelvin
    step: float
    baseline: str
    counts: List[int] = field(default_factory=list)      # points inside each interval
    warnings: List[str] = field(default_factory=list)

    def describe(self) -> str:
        """One-line summary for the status area / exports."""
        rates = ", ".join(f"{b:g}" for b in self.betas)
        lo = min(r[0] for r in self.ranges) - KELVIN_OFFSET
        hi = max(r[1] for r in self.ranges) - KELVIN_OFFSET
        return (f"4 instrument files, \u03b2 = {rates} K/min; intervals within {lo:.1f}\u2013{hi:.1f} \u00b0C; "
                f"{self.baseline} baseline; {describe_grid(self.step)}")


def build_instrument_table(
    curves: Sequence[pd.DataFrame],
    betas: Sequence[float],
    ranges: Sequence["Tuple[float, float]"],
    step: float = 0.01,
    baseline: str = DEFAULT_BASELINE,
    labels: Optional[Sequence[str]] = None,
) -> InstrumentResult:
    """
    Crop four instrument curves to their own intervals (``ranges``, kelvin, one per curve) and
    assemble the standard ``[alpha, T1, T2, T3, T4]`` table the kinetic methods expect.

    For every curve: crop, compute ``alpha(T)`` from the baseline-corrected peak area
    (:func:`conversion_from_heat_flow`), then **invert** it -- interpolate the temperature at
    every alpha of one common grid (``0.01 ... 0.99`` for ``step=0.01``).  The four curves are
    thereby aligned on the same alpha axis, which is what isoconversional analysis needs.  The
    columns are ordered by ascending heating rate.

    Exactly four curves are required: the kinetic engine works with four heating rates
    (padding with NaN columns would only make every method fail).

    Raises
    ------
    SelectionTooSmall
        If an interval holds fewer than :data:`MIN_SELECTION_POINTS` points.
    ValueError
        For wrong numbers of curves/rates/ranges, equal or non-positive rates, an empty
        interval, or an interval without a measurable peak.
    """
    if not (len(curves) == len(betas) == len(ranges) == N_CURVES):
        raise ValueError(f"Exactly {N_CURVES} curves, heating rates and intervals are required "
                         "(the kinetic methods work with four heating rates).")
    rates = [float(b) for b in betas]
    if any(not (b > 0 and math.isfinite(b)) for b in rates):
        raise ValueError("Heating rates must be positive numbers (K/min).")
    if len(set(rates)) < N_CURVES:
        raise ValueError("The four heating rates must all be different.")
    names = list(labels) if labels is not None else [f"{b:g} K/min" for b in rates]

    grid = alpha_grid(step)
    order = list(np.argsort(rates))
    table = {"alpha": grid}
    warnings: List[str] = []
    counts: List[int] = []
    sorted_ranges: List["Tuple[float, float]"] = []
    for col, idx in enumerate(order, start=1):
        lo, hi = sorted((float(ranges[idx][0]), float(ranges[idx][1])))
        if not hi > lo:
            raise ValueError(f"The interval of {names[idx]} is empty.")
        counts.append(check_selection(curves[idx], lo, hi, label=names[idx]))
        sorted_ranges.append((lo, hi))
        sub = select_range(curves[idx], lo, hi)
        t = sub[TEMP_COL].to_numpy()
        alpha, _, quality = conversion_from_heat_flow(t, sub[HEATFLOW_COL].to_numpy(), baseline)
        strictly = alpha + 1e-12 * np.arange(len(alpha))          # strictly increasing for the inversion
        table[f"T{col}"] = np.interp(grid, strictly, t)
        if quality["max_dip"] > 0.05:
            warnings.append(
                f"{names[idx]}: the heat flow crosses the baseline (the cumulative area dips by "
                f"{quality['max_dip'] * 100:.0f} % of the total). Check the interval ends.")
    df = pd.DataFrame(table)
    mid = len(grid) // 2
    mids = [float(df[f"T{c}"].iloc[mid]) for c in range(1, N_CURVES + 1)]
    if any(b <= a for a, b in zip(mids, mids[1:])):
        warnings.append(
            "At \u03b1 \u2248 0.5 the temperature does not rise with the heating rate: check the heating "
            "rates and the selected intervals (a faster ramp must reach the same conversion at a higher T).")
    return InstrumentResult(
        table=df, betas=[rates[i] for i in order],
        files=[str(curves[i].attrs.get("source", "")) for i in order],
        ranges=sorted_ranges, step=float(step), baseline=baseline, counts=counts, warnings=warnings,
    )
