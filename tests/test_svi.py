"""Offline tests for src/svi.py. No network access anywhere.

Ground truth: synthetic smiles generated from known SVI parameters, so
parameter recovery has an exact target. DE is seeded, so results are
deterministic.
"""

import numpy as np
import pandas as pd
import pytest

from src.svi import (
    DEFAULT_SVI_CONFIG,
    calibrate_all_slices,
    calibrate_svi_slice,
    svi_loss,
    svi_total_variance,
)

TRUE = {"a": 0.02, "b": 0.40, "rho": -0.40, "m": 0.05, "sigma": 0.20}
K_GRID = np.linspace(-0.35, 0.35, 25)


def _slice_from_params(params, T, F=6510.0, DF=0.9970,
                       expiry="2026-08-08", k_grid=K_GRID):
    """Slice shaped like the Module 3 output, generated from exact SVI."""
    w = svi_total_variance(k_grid, **params)
    assert np.all(np.isfinite(w)) and np.all(w > 0)
    return pd.DataFrame({
        "expiry": pd.Timestamp(expiry), "k": k_grid,
        "iv": np.sqrt(w / T), "T": T, "F": F, "DF": DF,
    })


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------

def test_nonnegative_variance():
    """Valid params: w(k) >= 0 for all k in [-2, 2], including at the
    analytic minimum a + b*sigma*sqrt(1-rho^2)."""
    k = np.linspace(-2.0, 2.0, 4001)
    w = svi_total_variance(k, **TRUE)
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0.0)
    w_min_analytic = TRUE["a"] + TRUE["b"] * TRUE["sigma"] * np.sqrt(
        1.0 - TRUE["rho"] ** 2)
    assert w.min() >= w_min_analytic - 1e-12


def test_constraint_violation():
    """Each infeasible region returns inf everywhere (optimizer poison)."""
    k = np.linspace(-1.0, 1.0, 11)
    bad_sets = [
        {**TRUE, "rho": 1.0},         # |rho| >= 1
        {**TRUE, "rho": -1.5},
        {**TRUE, "b": -0.1},          # b < 0
        {**TRUE, "sigma": 0.0},       # sigma <= 0
        {**TRUE, "a": -1.0},          # a + b*sigma*sqrt(1-rho^2) < 0
        {**TRUE, "m": float("nan")},  # non-finite
    ]
    for bad in bad_sets:
        w = svi_total_variance(k, **bad)
        assert np.all(np.isinf(w)), f"expected inf for {bad}"
    params_vec = [TRUE[p] for p in ("a", "b", "rho", "m", "sigma")]
    params_vec[2] = 1.0
    assert svi_loss(np.array(params_vec), k,
                    np.full_like(k, 0.01)) == float("inf")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_parameter_recovery():
    """Exact synthetic smile: two-stage calibration recovers all five
    parameters within 1e-3."""
    T = 30 / 365
    res = calibrate_svi_slice(_slice_from_params(TRUE, T))
    for p, truth in TRUE.items():
        assert res[p] == pytest.approx(truth, abs=1e-3), (
            f"{p}: {res[p]} vs {truth}"
        )
    assert res["rmse"] < 1e-6
    assert not res["lee_flag"]
    assert res["n_points"] == len(K_GRID)


def test_rmse_threshold():
    """A well-behaved smile with small noise fits below RMSE_WARN."""
    T = 60 / 365
    rng = np.random.default_rng(11)
    df = _slice_from_params(TRUE, T, expiry="2026-09-07")
    w_noisy = (df["iv"] ** 2 * T) + rng.normal(0.0, 2e-4, len(df))
    df["iv"] = np.sqrt(np.maximum(w_noisy, 1e-8) / T)
    res = calibrate_svi_slice(df)
    assert res["rmse"] < DEFAULT_SVI_CONFIG["RMSE_WARN"]


def test_lee_flag():
    """A smile generated with b*(1+|rho|) = 2.25 > 2 must come back
    flagged (and the steep wing must actually be recovered, not clipped)."""
    steep = {"a": 0.005, "b": 1.5, "rho": 0.5, "m": 0.0, "sigma": 0.15}
    assert steep["b"] * (1 + abs(steep["rho"])) > 2.0
    T = 90 / 365
    res = calibrate_svi_slice(_slice_from_params(steep, T,
                                                 expiry="2026-10-07"))
    assert res["lee_flag"]
    assert res["b"] * (1 + abs(res["rho"])) == pytest.approx(2.25, abs=0.01)


def test_calibrate_all_slices_sorted_and_complete():
    frames = [
        _slice_from_params(TRUE, 120 / 365, F=6540.0, DF=0.9880,
                           expiry="2026-11-06"),
        _slice_from_params(TRUE, 30 / 365, F=6510.0, DF=0.9970,
                           expiry="2026-08-08"),
    ]
    out = calibrate_all_slices(pd.concat(frames, ignore_index=True))
    assert list(out["T"]) == sorted(out["T"])            # ascending T
    assert out.iloc[0]["expiry"] == pd.Timestamp("2026-08-08")
    assert (out["rmse"] < 1e-6).all()
    assert set(out.columns) >= {"a", "b", "rho", "m", "sigma", "F", "DF",
                                "rmse", "lee_flag"}


def test_calendar_preserved_by_interp():
    """Two clean slices (w_short(k) <= w_long(k) for all k) linearly
    interpolated in total variance produce no calendar violation at any
    intermediate T. This is the no-arbitrage argument for Module 5's
    time interpolation, provable pointwise: a convex combination of two
    ordered values stays between them and moves monotonically in t."""
    T1, T2 = 30 / 365, 120 / 365
    p_short = TRUE
    p_long = {**TRUE, "a": TRUE["a"] * 4.2, "b": TRUE["b"] * 1.1}
    k = np.linspace(-1.0, 1.0, 401)
    w1 = svi_total_variance(k, **p_short)
    w2 = svi_total_variance(k, **p_long)
    assert np.all(w1 <= w2), "test setup: slices must be calendar-clean"
    prev = w1
    for t in np.linspace(T1, T2, 41):
        lam = (t - T1) / (T2 - T1)
        w_t = (1 - lam) * w1 + lam * w2
        assert np.all(w_t >= prev - 1e-15), f"w decreased in T at t={t}"
        prev = w_t