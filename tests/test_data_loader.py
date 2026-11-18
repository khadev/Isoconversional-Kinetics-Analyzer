"""Unit tests for isokinetics.data_loader."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from isokinetics import data_loader

SAMPLE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "sample_data", "matric_121459.txt"
)


def test_load_sample_data() -> None:
    df = data_loader.load_data(SAMPLE_FILE)
    assert list(df.columns) == ["alpha", "T1", "T2", "T3", "T4"]
    assert len(df) == 99
    # Data must be sorted by ascending alpha.
    assert (df["alpha"].diff().dropna() > 0).all()
    # Temperatures should increase with alpha for each heating rate.
    for col in ["T1", "T2", "T3", "T4"]:
        assert (df[col].diff().dropna() > 0).all()


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        data_loader.load_data("/path/does/not/exist.txt")


def test_load_csv_with_commas(tmp_path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text(
        "alpha,T1,T2,T3,T4\n"
        "0.1,500,510,520,530\n"
        "0.2,510,520,530,540\n"
        "0.3,520,530,540,550\n"
        "0.4,530,540,550,560\n"
    )
    df = data_loader.load_data(str(path))
    assert len(df) == 4
    assert list(df.columns) == ["alpha", "T1", "T2", "T3", "T4"]


def test_validate_data_rejects_too_few_columns() -> None:
    df = pd.DataFrame({"alpha": [0.1, 0.2, 0.3], "T1": [500, 510, 520]})
    with pytest.raises(ValueError):
        data_loader.validate_data(df)


def test_validate_data_rejects_out_of_range_alpha() -> None:
    df = pd.DataFrame(
        {
            0: [1.5, 0.2, 0.3, 0.4],
            1: [500, 510, 520, 530],
            2: [505, 515, 525, 535],
            3: [510, 520, 530, 540],
            4: [515, 525, 535, 545],
        }
    )
    with pytest.raises(ValueError):
        data_loader.validate_data(df)


def test_validate_data_drops_nan_rows() -> None:
    df = pd.DataFrame(
        {
            0: [0.1, 0.2, np.nan, 0.4, 0.5],
            1: [500, 510, 520, 530, 540],
            2: [505, 515, 525, 535, 545],
            3: [510, 520, 530, 540, 550],
            4: [515, 525, 535, 545, 555],
        }
    )
    clean = data_loader.validate_data(df)
    assert len(clean) == 4  # the NaN row was dropped


def test_interpolate_alpha_produces_uniform_grid() -> None:
    df = data_loader.load_data(SAMPLE_FILE)
    interpolated = data_loader.interpolate_alpha(df, alpha_min=0.05, alpha_max=0.5, step=0.05)
    expected_alphas = np.round(np.arange(0.05, 0.5 + 1e-9, 0.05), 4)
    got_alphas = np.round(interpolated["alpha"].to_numpy(), 4)
    assert len(got_alphas) == len(expected_alphas)
    np.testing.assert_allclose(got_alphas, expected_alphas, atol=1e-6)


def test_interpolate_alpha_returns_original_when_too_few_points() -> None:
    df = pd.DataFrame(
        {
            "alpha": [0.1, 0.2, 0.3],
            "T1": [500, 510, 520],
            "T2": [505, 515, 525],
            "T3": [510, 520, 530],
            "T4": [515, 525, 535],
        }
    )
    result = data_loader.interpolate_alpha(df)
    pd.testing.assert_frame_equal(result, df)
