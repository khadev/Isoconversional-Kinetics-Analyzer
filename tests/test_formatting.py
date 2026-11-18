"""Tests for the shared number-formatting policy."""

from __future__ import annotations

import math

from isokinetics import formatting as f


def test_fixed_formats_follow_policy():
    assert f.fmt_alpha(0.25) == "0.250"
    assert f.fmt_ea(97.1534) == "97.15"
    assert f.fmt_ea(97.1534, unit=True) == "97.15 kJ/mol"
    assert f.fmt_temp(551.3649) == "551.36"
    assert f.fmt_r2(0.99237) == "0.9924"
    assert f.fmt_sci(1.23456e10) == "1.235e+10"
    assert f.fmt_pct(12.34) == "12.3 %"
    assert f.fmt_pct(-0.54, signed=True) == "-0.5 %"


def test_non_finite_values_are_na_everywhere():
    for fn in (f.fmt_alpha, f.fmt_ea, f.fmt_temp, f.fmt_sci, f.fmt_r2, f.fmt_pct, f.fmt_score):
        assert fn(float("nan")) == f.NA
        assert fn(None) == f.NA
    assert f.fmt_ea(math.inf, unit=True) == f.NA
    assert f.fmt_sci(float("nan"), na=f.EMPTY) == f.EMPTY


def test_beta_and_duration():
    assert f.fmt_beta(10.0) == "10"
    assert f.fmt_beta(7.5) == "7.5"
    assert f.fmt_duration(0.84) == "0.8 s"
    assert f.fmt_duration(42.2) == "42 s"
    assert f.fmt_duration(65) == "1 min 05 s"
    assert f.fmt_duration(None) == f.EMPTY


def test_alpha_range_text():
    assert f.alpha_range_text([0.99, 0.01, 0.5]) == "0.010 \u2013 0.990"
    assert f.alpha_range_text([]) == f.EMPTY
