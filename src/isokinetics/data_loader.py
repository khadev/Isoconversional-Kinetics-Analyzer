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
from typing import List, Optional, Sequence

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
