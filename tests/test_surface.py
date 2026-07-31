"""Offline tests for src/surface.py. No network access, no DE runs
(slice parameter frames are constructed directly, so these tests are
fast and exact).
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.surface import (
    build_interpolator,
    check_fitted_calendar,
    evaluate_svi_grid,
    query_surface,
)
from src.svi import svi_total_variance

SPOT = 6500.0

# Two calendar-clean slices sharing rho/m/sigma with scaled (a, b):
# w_long(k) - w_short(k) = da + db*(rho*(k-m) + sqrt(...)) with da, db > 0
# and the bracket >= sigma*(1-|rho|)... verified explicitly in the test
# setup assertions rather than assumed.
P_SHORT = {"a": 0.0020, "b": 0.040, "rho": -0.40, "m": 0.02, "sigma": 0.20}
P_LONG = {"a": 0.0090, "b": 0.055, "rho": -0.40, "m": 0.02, "sigma": 0.20}


def _params_frame():
    rows = [
        {"expiry": pd.Timestamp("2026-08-08"), "T": 30 / 365, "F": 6510.0,
         "DF": 0.9970, **P_SHORT, "rmse": 0.0, "lee_flag": False},
        {"expiry": pd.Timestamp("2026-11-06"), "T": 120 / 365, "F": 6540.0,
         "DF": 0.9880, **P_LONG, "rmse": 0.0, "lee_flag": False},
    ]
    return pd.DataFrame(rows)


def _w(params, k):
    return svi_total_variance(np.atleast_1d(np.asarray(k, float)), **params)


def test_slices_are_calendar_clean():
    """Test-setup invariant: containment w_short <= w_long on a wide
    grid, so downstream monotonicity tests are meaningful."""
    k = np.linspace(-1.0, 1.0, 2001)
    assert np.all(_w(P_SHORT, k) <= _w(P_LONG, k))
    assert check_fitted_calendar(_params_frame(),
                                 np.linspace(-1, 1, 401)).empty


def test_grid_shape_and_values():
    """iv_matrix is (n_expiries, n_strikes) and matches direct SVI
    evaluation with each slice's own forward."""
    params = _params_frame()
    m_grid = np.linspace(0.7, 1.3, 121)
    grid = evaluate_svi_grid(params, m_grid, SPOT)
    assert grid.shape == (2, 121)
    assert np.all(np.isfinite(grid)) and np.all(grid > 0)
    for i, (_, row) in enumerate(params.sort_values("T").iterrows()):
        k = np.log(m_grid * SPOT / row["F"])
        iv_direct = np.sqrt(_w({p: row[p] for p in
                                ("a", "b", "rho", "m", "sigma")}, k)
                            / row["T"])
        np.testing.assert_allclose(grid[i], iv_direct, rtol=0, atol=1e-15)


def test_knot_exactness():
    """Querying exactly at a fitted maturity returns that slice's own
    value (forward and T included) to machine precision."""
    params = _params_frame()
    interp = build_interpolator(params, SPOT)
    for _, row in params.iterrows():
        for K in (6000.0, 6510.0, 7000.0):
            k = np.log(K / row["F"])
            expected = float(np.sqrt(
                _w({p: row[p] for p in ("a", "b", "rho", "m", "sigma")},
                   k)[0] / row["T"]))
            assert query_surface(interp, row["T"], K) == pytest.approx(
                expected, abs=1e-15)


def test_interior_finite_and_calendar_monotone():
    """Inside the fitted range: finite IVs, and total variance is
    non-decreasing in T at fixed k. Slices share the forward here so
    fixed K IS fixed k, making the invariant directly testable through
    the public API."""
    params = _params_frame()
    params["F"] = 6510.0  # equal forwards: k independent of T
    interp = build_interpolator(params, SPOT)
    T_grid = np.linspace(30 / 365, 120 / 365, 61)
    for K in (5900.0, 6510.0, 7100.0):
        w_prev = -np.inf
        for T in T_grid:
            iv = interp(T, K)
            assert np.isfinite(iv) and iv > 0
            w = iv * iv * T
            assert w >= w_prev - 1e-14, (
                f"calendar violated at K={K}, T={T}"
            )
            w_prev = w


def test_midpoint_linear_in_w_and_logF():
    """At the T midpoint: interpolated w is the average of the two
    slice w's at the query k, and F(T) is the geometric mean of the
    slice forwards."""
    params = _params_frame()
    T1, T2 = params["T"].min(), params["T"].max()
    T_mid = 0.5 * (T1 + T2)
    F_mid = float(np.sqrt(6510.0 * 6540.0))  # geometric mean under log-linear
    interp = build_interpolator(params, SPOT)
    K = 6400.0
    k = np.log(K / F_mid)
    w_expected = 0.5 * (_w(P_SHORT, k)[0] + _w(P_LONG, k)[0])
    assert interp(T_mid, K) == pytest.approx(
        float(np.sqrt(w_expected / T_mid)), abs=1e-14)


def test_clamp_no_extrapolation(caplog):
    """Outside the fitted range: value equals the edge slice's own IV
    (full clamp), warned exactly once per interpolator."""
    params = _params_frame()
    interp = build_interpolator(params, SPOT)
    K = 6450.0
    with caplog.at_level(logging.WARNING, logger="src.surface"):
        below = interp(7 / 365, K)
        below2 = interp(1 / 365, K)
        above = interp(300 / 365, K)
    assert below == interp(30 / 365, K) == below2
    assert above == interp(120 / 365, K)
    clamp_msgs = [r for r in caplog.records if "clamped" in r.message]
    assert len(clamp_msgs) == 1  # once per interpolator, not per query
    # a fresh interpolator warns again
    with caplog.at_level(logging.WARNING, logger="src.surface"):
        build_interpolator(params, SPOT)(1 / 365, K)
    assert len([r for r in caplog.records if "clamped" in r.message]) == 2


def test_single_slice_degenerates_to_that_slice():
    params = _params_frame().iloc[[0]].reset_index(drop=True)
    interp = build_interpolator(params, SPOT)
    K = 6300.0
    at_knot = interp(30 / 365, K)
    assert interp(60 / 365, K) == at_knot
    assert interp(10 / 365, K) == at_knot


def test_fitted_calendar_crossing_detected():
    """A short slice with a steepened b must be reported as crossing.
    NOTE: b multiplies BOTH wing slopes and lifts the ATM level via
    b*sigma, so the crossing is not confined to one wing -- assert the
    report's internal consistency, not a one-wing shape."""
    params = _params_frame()
    params.loc[params["T"].idxmin(), "b"] = 0.35
    k_grid = np.linspace(-1.0, 1.0, 801)
    flags = check_fitted_calendar(params, k_grid)
    assert len(flags) == 1
    row = flags.iloc[0]
    assert row["k_cross_lo"] <= row["k_cross_hi"]
    assert 0 < row["frac_crossed"] <= 1
    assert row["worst_gap"] > 0
    # report agrees with direct evaluation on the same grid
    p_short = {**P_SHORT, "b": 0.35}
    crossed = _w(p_short, k_grid) > _w(P_LONG, k_grid)
    assert row["frac_crossed"] == pytest.approx(float(np.mean(crossed)))
    assert row["k_cross_lo"] == pytest.approx(float(k_grid[crossed].min()))
    assert row["k_cross_hi"] == pytest.approx(float(k_grid[crossed].max()))


def test_query_validation():
    interp = build_interpolator(_params_frame(), SPOT)
    for bad_T, bad_K in ((-0.1, 6500.0), (0.1, -5.0),
                         (float("nan"), 6500.0), (0.1, float("inf"))):
        with pytest.raises(AssertionError):
            interp(bad_T, bad_K)