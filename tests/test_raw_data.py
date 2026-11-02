"""Raw (4-column) vs processed (5-column) detection and alpha generation."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from isokinetics import data_loader as D
from isokinetics import kinetics, validation

BETAS = [5.0, 10.0, 20.0, 40.0]
SAMPLE = os.path.join(os.path.dirname(__file__), "..", "sample_data", "matric_121459.txt")


def _write(path, frame, sep="\t", header=False):
    frame.to_csv(path, sep=sep, header=header, index=False, float_format="%.5f")
    return str(path)


def _linear_raw(n=101):
    """Raw curves T_j(alpha) = base_j + 100*alpha with row i at alpha = i/(n-1)."""
    a = np.linspace(0, 1, n)
    return pd.DataFrame({f"T{j}": 300 + 20 * j + 100 * a for j in range(1, 5)})


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def test_five_columns_is_processed_and_unchanged():
    t = D.load_table(SAMPLE)
    assert t.kind == D.KIND_PROCESSED and not t.is_raw and t.n_columns == 5
    pd.testing.assert_frame_equal(t.data, D.load_data(SAMPLE))


def test_four_columns_is_raw(tmp_path):
    p = _write(tmp_path / "raw.txt", _linear_raw())
    t = D.load_table(p)
    assert t.is_raw and t.n_columns == 4
    assert list(t.data.columns) == ["T1", "T2", "T3", "T4"]
    assert t.rows_per_column == [101] * 4 and t.n_rows == 101


@pytest.mark.parametrize("sep", ["\t", ",", ";", " "])
def test_raw_text_delimiters(tmp_path, sep):
    p = _write(tmp_path / "raw.dat", _linear_raw(30), sep=sep)
    assert D.load_table(p).is_raw


def test_raw_with_header_line(tmp_path):
    frame = _linear_raw(20)
    p = tmp_path / "raw.txt"
    with open(p, "w") as fh:
        fh.write("T1 (K)\tT2 (K)\tT3 (K)\tT4 (K)\n")
        frame.to_csv(fh, sep="\t", header=False, index=False, float_format="%.4f")
    t = D.load_table(str(p))
    assert t.is_raw and t.rows_per_column == [20] * 4


def test_raw_excel(tmp_path):
    p = tmp_path / "raw.xlsx"
    _linear_raw(25).to_excel(p, header=False, index=False)
    t = D.load_table(str(p))
    assert t.is_raw and t.n_rows == 25


def test_processed_excel_still_detected(tmp_path):
    p = tmp_path / "proc.xlsx"
    D.load_data(SAMPLE).to_excel(p, header=False, index=False)
    assert D.load_table(str(p)).kind == D.KIND_PROCESSED


def test_ragged_columns_are_kept_per_column(tmp_path):
    frame = _linear_raw(40)
    frame.loc[30:, "T2"] = np.nan  # shorter second curve (empty tab-separated cells)
    frame.loc[35:, "T4"] = np.nan
    p = _write(tmp_path / "ragged.txt", frame)
    t = D.load_table(p)
    assert t.rows_per_column == [40, 30, 40, 35]
    table = D.build_alpha_table(t.data, 0.1)
    assert len(table) == 9 and table.notna().all().all()


@pytest.mark.parametrize("n_cols", [1, 2, 3])
def test_too_few_columns_rejected(tmp_path, n_cols):
    frame = _linear_raw(20).iloc[:, :n_cols]
    p = tmp_path / "few.txt"
    frame.to_csv(p, sep="\t", header=False, index=False)
    with pytest.raises(ValueError):
        D.load_table(str(p))


def test_alpha_like_first_column_is_rejected(tmp_path):
    frame = _linear_raw(20)
    frame["T1"] = np.linspace(0.01, 0.99, 20)  # looks like alpha -> only 3 temperature columns
    p = _write(tmp_path / "three.txt", frame)
    with pytest.raises(ValueError, match="conversion"):
        D.load_table(p)


def test_decreasing_temperature_rejected_with_position(tmp_path):
    frame = _linear_raw(20)
    frame.loc[10, "T3"] = 100.0
    p = _write(tmp_path / "bad.txt", frame)
    with pytest.raises(ValueError, match="T3.*not increasing"):
        D.load_table(p)


def test_non_positive_temperature_and_too_short_rejected():
    frame = _linear_raw(20)
    frame.loc[0, "T2"] = -5.0
    with pytest.raises(ValueError, match="positive"):
        D.validate_raw_data(frame)
    with pytest.raises(ValueError, match="at least 4"):
        D.validate_raw_data(_linear_raw(3))


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        D.load_table("/nonexistent/file.txt")


# --------------------------------------------------------------------------- #
# Step parsing and the alpha grid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,step", [
    ("0.01", 0.01), ("0.02", 0.02), ("1/50", 0.02), ("1/100", 0.01), (" 50 ", 0.02),
    ("100", 0.01), ("0,05", 0.05), ("2/100", 0.02), ("4", 0.25),
])
def test_parse_alpha_step_accepts(text, step):
    assert D.parse_alpha_step(text) == pytest.approx(step)


@pytest.mark.parametrize("text", ["", "abc", "1", "1.5", "7.5", "1/0", "0", "-0.1", "0.5", "0.0001", "5000", "nan"])
def test_parse_alpha_step_rejects(text):
    with pytest.raises(ValueError):
        D.parse_alpha_step(text)


def test_alpha_grid_excludes_zero_and_one():
    g = D.alpha_grid(0.01)
    assert len(g) == 99 and g[0] == 0.01 and g[-1] == 0.99
    g50 = D.alpha_grid(1 / 50)
    assert len(g50) == 49 and g50[0] == 0.02 and g50[-1] == 0.98
    assert 0.0 not in g50 and 1.0 not in g50
    odd = D.alpha_grid(0.03)  # does not divide 1 evenly
    assert odd[-1] == pytest.approx(0.99) and len(odd) == 33


def test_row_mappings():
    assert D.row_alpha(5, "endpoints").tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert D.row_alpha(3, "interior").tolist() == pytest.approx([0.25, 0.5, 0.75])
    with pytest.raises(ValueError):
        D.row_alpha(5, "nope")


# --------------------------------------------------------------------------- #
# build_alpha_table: the interpolation itself
# --------------------------------------------------------------------------- #
def test_linear_curves_are_interpolated_exactly():
    raw = _linear_raw(101)
    for step in (0.01, 0.02, 0.05, 1 / 3 * 0.75):
        out = D.build_alpha_table(raw, step)
        for j in range(1, 5):
            expected = 300 + 20 * j + 100 * out["alpha"]
            np.testing.assert_allclose(out[f"T{j}"], expected, atol=1e-9)


def test_interpolates_between_rows():
    raw = _linear_raw(11)                       # rows every 0.1 in alpha
    out = D.build_alpha_table(raw, 0.05)        # half-way points
    np.testing.assert_allclose(out["T1"], 320 + 100 * out["alpha"], atol=1e-9)
    assert len(out) == 19


def test_output_is_clean_processed_table():
    out = D.build_alpha_table(_linear_raw(101), 0.02)
    assert list(out.columns) == D.EXPECTED_COLUMNS
    assert out.alpha.is_monotonic_increasing and out.notna().all().all()
    D.validate_data(out)  # passes the processed-file validation unchanged


def test_interior_mapping_reproduces_the_bundled_processed_sample_exactly():
    proc = D.load_data(SAMPLE)
    raw = proc[["T1", "T2", "T3", "T4"]]
    rebuilt = D.build_alpha_table(raw, 0.01, "interior")
    np.testing.assert_allclose(rebuilt[["T1", "T2", "T3", "T4"]], raw, atol=1e-9)
    np.testing.assert_allclose(rebuilt["alpha"], proc["alpha"], atol=1e-12)


def test_interior_mapping_never_extrapolates():
    raw = _linear_raw(49)  # alpha 0.02 ... 0.98
    out = D.build_alpha_table(raw, 0.01, "interior")
    assert out["alpha"].min() >= 0.02 - 1e-9 and out["alpha"].max() <= 0.98 + 1e-9


def test_coarsest_step_still_gives_three_points():
    out = D.build_alpha_table(_linear_raw(4), 0.25, "interior")  # rows at 0.2/0.4/0.6/0.8
    assert out["alpha"].tolist() == [0.25, 0.5, 0.75]


def test_resolution_note():
    raw = _linear_raw(11)
    assert D.resolution_note(raw, 0.1) is None
    assert "finer than the data" in D.resolution_note(raw, 0.01)


def test_describe_grid():
    assert D.describe_grid(0.01) == "99 points, \u03b1 = 0.010 \u2026 0.990 (\u0394\u03b1 = 0.01)"
    assert D.describe_grid(1 / 50).startswith("49 points, \u03b1 = 0.020")


def test_save_processed_table_roundtrip(tmp_path):
    out = D.build_alpha_table(_linear_raw(101), 0.02)
    for name in ("p.txt", "p.csv"):
        path = D.save_processed_table(out, str(tmp_path / name))
        back = D.load_data(path)
        np.testing.assert_allclose(back.to_numpy(), out.to_numpy(), atol=1e-4)
        assert D.load_table(path).kind == D.KIND_PROCESSED


# --------------------------------------------------------------------------- #
# Ground truth: raw -> generated alpha -> kinetic analysis recovers the true Ea
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def raw_r3_150(tmp_path_factory):
    """Raw file whose row i sits exactly at alpha = i/500 (R3 model, Ea = 150 kJ/mol)."""
    a = np.clip(np.linspace(0, 1, 501), 1e-6, 1.0)
    sim = validation.simulate_constant_ea(150.0, 1e12, "R3", alphas=a)
    path = tmp_path_factory.mktemp("raw") / "raw.txt"
    sim[["T1", "T2", "T3", "T4"]].to_csv(path, sep="\t", header=False, index=False, float_format="%.5f")
    return str(path)


@pytest.mark.parametrize("step", [0.01, 0.02, 0.05])
@pytest.mark.parametrize("method,tol", [("Friedman", 0.5), ("KAS", 1.0), ("Starink", 0.5),
                                        ("Vyazovkin", 0.5), ("KTH", 0.5)])
def test_generated_alpha_recovers_true_ea(raw_r3_150, step, method, tol):
    table = D.load_table(raw_r3_150)
    df = D.build_alpha_table(table.data, step)
    res = kinetics.METHODS[method](df, BETAS)
    mean = np.mean([e for a, e in res.items() if 0.1 <= a <= 0.9])
    assert abs(mean / 150.0 - 1) * 100 <= tol, f"{method} step {step}: {mean:.2f}"


def test_coats_redfern_works_on_generated_grid(raw_r3_150):
    df = D.build_alpha_table(D.load_table(raw_r3_150).data, 0.01)
    cr = kinetics.coats_redfern(df, BETAS)
    assert cr["recommended_model"] == "R3"
    assert cr["results"]["R3"]["Ea"] == pytest.approx(150.0, rel=0.01)


# --------------------------------------------------------------------------- #
# Row-layout suggestion (99 rows -> interior; picking the wrong layout shifts T)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_rows,expected", [
    (99, "interior"), (49, "interior"), (199, "interior"), (19, "interior"),
    (101, "endpoints"), (100, "endpoints"), (500, "endpoints"), (50, "endpoints"), (2, "endpoints"),
])
def test_suggest_mapping_from_row_count(n_rows, expected):
    assert D.suggest_mapping(n_rows) == expected


def test_wrong_layout_really_matters_on_the_bundled_99_row_file():
    """Why the suggestion exists: same 99 rows, other layout -> several kelvin off."""
    proc = D.load_data(SAMPLE)
    raw = proc[["T1", "T2", "T3", "T4"]]
    right = D.build_alpha_table(raw, 0.01, D.suggest_mapping(len(raw)))
    wrong = D.build_alpha_table(raw, 0.01, "endpoints")
    cols = ["T1", "T2", "T3", "T4"]
    assert np.abs(right[cols].values - proc[cols].values).max() < 1e-9
    assert np.abs(wrong[cols].values - proc[cols].values).max() > 2.0
