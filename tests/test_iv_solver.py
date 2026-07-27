"""Offline tests for src/iv_solver.py. No network access anywhere.

Ground truth throughout is exact Black-76 pricing at a known sigma, so
roundtrip tolerances are meaningful, plus one external anchor against a
Hull textbook value.
"""

import numpy as np
import pandas as pd
import pytest

from src.iv_solver import (
    DEFAULT_SOLVER_CONFIG,
    b76_price,
    b76_vega,
    check_parity_iv,
    compute_iv_surface,
    implied_vol_brent,
    implied_vol_nr,
    mk_seed,
    solve_iv,
)

F, DF, T = 6510.0, 0.9970, 30 / 365


def _row(mid, K, typ, F_=F, DF_=DF, T_=T):
    return pd.Series({"mid_price": mid, "F": F_, "strike": K, "DF": DF_,
                      "T": T_, "option_type": typ})


# ---------------------------------------------------------------------------
# Pricing primitives
# ---------------------------------------------------------------------------

def test_put_call_parity_exact():
    """b76 put comes from parity, so C - P = DF*(F - K) to 1e-10."""
    for K in (5800.0, 6510.0, 7100.0):
        c = b76_price(F, K, DF, T, 0.22, "call")
        p = b76_price(F, K, DF, T, 0.22, "put")
        assert c - p == pytest.approx(DF * (F - K), abs=1e-10)


def test_vega_nonnegative_and_fd():
    """Vega >= 0 everywhere, and matches central finite difference."""
    for K in (5500.0, 6510.0, 7300.0):
        for sigma in (0.05, 0.2, 1.5):
            v = b76_vega(F, K, DF, T, sigma)
            assert v >= 0.0
            h = 1e-6
            fd = (b76_price(F, K, DF, T, sigma + h, "call")
                  - b76_price(F, K, DF, T, sigma - h, "call")) / (2 * h)
            assert v == pytest.approx(fd, rel=1e-6, abs=1e-8)


def test_nonfinite_inputs():
    """inf/nan in any input raises ValueError (isfinite guard)."""
    good = dict(F=F, K=6500.0, DF=DF, T=T, sigma=0.2)
    for field in good:
        for bad in (float("nan"), float("inf"), -float("inf")):
            kw = {**good, field: bad}
            with pytest.raises(ValueError):
                b76_price(kw["F"], kw["K"], kw["DF"], kw["T"], kw["sigma"],
                          "call")
            with pytest.raises(ValueError):
                b76_vega(kw["F"], kw["K"], kw["DF"], kw["T"], kw["sigma"])
    with pytest.raises(ValueError):
        b76_price(F, 6500.0, DF, T, -0.1, "call")  # non-positive sigma
    with pytest.raises(ValueError):
        b76_price(F, 6500.0, DF, T, 0.2, "straddle")  # bad type


# ---------------------------------------------------------------------------
# Seed and NR mechanics
# ---------------------------------------------------------------------------

def test_mk_seed():
    """Seed equals sqrt(2*|log(F/K)|/T), floored at 0.1 at the money."""
    K = 5800.0
    expected = np.sqrt(2.0 * abs(np.log(F / K)) / T)
    assert mk_seed(F, K, T) == pytest.approx(expected, rel=1e-14)
    assert mk_seed(F, F, T) == 0.1          # raw formula gives 0 at ATM
    assert mk_seed(F, F * 1.0001, 1.0) == 0.1  # tiny |k| still floored


def test_nr_convergence_order():
    """NR must return the sigma whose price met tolerance (check before
    update). Construct a market price within tolerance of the price at
    the seed: a correct implementation returns the seed EXACTLY; the r1
    ordering (update, then test the stale error) would return a nudged
    sigma instead."""
    K = 6510.0  # ATM-forward, so the seed is the 0.1 floor
    s0 = mk_seed(F, K, T)
    tol = DEFAULT_SOLVER_CONFIG["NR_TOL"]
    market = b76_price(F, K, DF, T, s0, "call") + 0.5 * tol
    out = implied_vol_nr(market, F, K, DF, T, "call")
    assert out == s0  # bitwise: no update may occur after convergence


def test_roundtrip_call():
    sigma_true = 0.2437
    for K in (6100.0, 6510.0, 6900.0):
        px = b76_price(F, K, DF, T, sigma_true, "call")
        iv = solve_iv(_row(px, K, "call"))
        assert iv == pytest.approx(sigma_true, abs=1e-6)


def test_roundtrip_put():
    sigma_true = 0.31
    for K in (6000.0, 6510.0, 6800.0):
        px = b76_price(F, K, DF, T, sigma_true, "put")
        iv = solve_iv(_row(px, K, "put"))
        assert iv == pytest.approx(sigma_true, abs=1e-6)


def test_nr_fallback():
    """NR failure hands off to Brent and the final IV is still correct.

    Deterministic trigger: a vega floor above any attainable dollar vega
    forces implied_vol_nr to signal fallback immediately; solve_iv must
    then recover the exact vol via Brent. A natural deep-OTM short-dated
    case is also exercised end to end.
    """
    K, sigma_true = 6510.0, 0.2
    px = b76_price(F, K, DF, T, sigma_true, "call")
    cfg = {**DEFAULT_SOLVER_CONFIG, "VEGA_FLOOR": 1e12}
    assert implied_vol_nr(px, F, K, DF, T, "call", cfg) is None
    iv = solve_iv(_row(px, K, "call"), cfg)
    assert iv == pytest.approx(sigma_true, abs=1e-6)

    # Natural stress: deepest OTM the pipeline can actually see (the
    # moneyness band caps |k|), short-dated, with a representable price.
    # NOTE: pushing further (e.g. 12% vol at K/S=0.71, T=7d) puts the
    # true price ~20 sigma OTM, below float64 resolution -- that regime
    # is excluded upstream by the bid>0 and moneyness filters, and full
    # deep-wing precision is the documented Jackel upgrade path.
    K2, T2, sigma2 = 5200.0, 14 / 365, 0.40
    px2 = b76_price(F, K2, DF, T2, sigma2, "put")
    iv2 = solve_iv(_row(px2, K2, "put", T_=T2))
    assert iv2 == pytest.approx(sigma2, abs=1e-6)


def test_parity():
    """Call and put IVs at the same (K, T) with shared (F, DF) agree
    within 1e-6 on exact prices."""
    sigma_true = 0.27
    for K in (6200.0, 6510.0, 6800.0):
        iv_c = solve_iv(_row(b76_price(F, K, DF, T, sigma_true, "call"),
                             K, "call"))
        iv_p = solve_iv(_row(b76_price(F, K, DF, T, sigma_true, "put"),
                             K, "put"))
        assert abs(iv_c - iv_p) < 1e-6


def test_hull_value():
    """Hull (11th ed.) reference: S=42, K=40, r=0.10, sigma=0.20, T=0.5
    gives call 4.76 and put 0.81. Map BS -> B76 via F = S*exp(rT),
    DF = exp(-rT), then verify pricing and inversion."""
    S, K, r, sigma, T_h = 42.0, 40.0, 0.10, 0.20, 0.5
    F_h = S * np.exp(r * T_h)
    DF_h = np.exp(-r * T_h)
    call = b76_price(F_h, K, DF_h, T_h, sigma, "call")
    put = b76_price(F_h, K, DF_h, T_h, sigma, "put")
    assert round(call, 2) == 4.76
    assert round(put, 2) == 0.81
    iv = solve_iv(_row(call, K, "call", F_=F_h, DF_=DF_h, T_=T_h))
    assert iv == pytest.approx(sigma, abs=1e-8)


def test_below_intrinsic():
    """Market price below discounted intrinsic violates static bounds:
    no root in the bracket, solver returns nan."""
    K = 6000.0  # ITM call: discounted intrinsic = DF*(F - K)
    intrinsic = DF * (F - K)
    bad_px = 0.9 * intrinsic
    assert np.isnan(implied_vol_brent(bad_px, F, K, DF, T, "call"))
    assert np.isnan(solve_iv(_row(bad_px, K, "call")))


# ---------------------------------------------------------------------------
# Chain-wide computation
# ---------------------------------------------------------------------------

def _synthetic_otm_chain(n_bad=0):
    """OTM-only frame shaped like attach_forwards output, exact B76 mids
    from a skewed smile; optionally corrupt n_bad rows below intrinsic."""
    rows = []
    strikes = np.arange(5900.0, 7150.0, 50.0)
    for K in strikes:
        iv = 0.18 + 0.15 * np.log(F / K)
        typ = "put" if K <= F else "call"
        rows.append({"expiry": pd.Timestamp("2026-08-08"),
                     "strike": K, "mid_price": b76_price(F, K, DF, T, iv, typ),
                     "F": F, "DF": DF, "T": T, "option_type": typ,
                     "k": np.log(K / F), "true_iv": iv})
    df = pd.DataFrame(rows)
    for i in range(n_bad):
        # Above the static upper bound (call <= DF*F, put <= DF*K): no
        # sigma can reproduce it, so inversion must fail. (A tiny positive
        # mid like 1e-12 is NOT invalid for an OTM option -- intrinsic is
        # zero, so Brent would legitimately invert it to a small vol.)
        df.loc[i, "mid_price"] = 2.0 * DF * max(F, df.loc[i, "strike"])
    return df


def test_compute_iv_surface_recovers_smile():
    df = _synthetic_otm_chain()
    out = compute_iv_surface(df)
    assert len(out) == len(df)
    np.testing.assert_allclose(out["iv"], out["true_iv"], atol=1e-6)


def test_compute_iv_surface_drops_bad_rows_and_asserts():
    df = _synthetic_otm_chain(n_bad=3)
    out = compute_iv_surface(df)          # 3/25 bad: above 80% threshold
    assert len(out) == len(df) - 3
    df_mostly_bad = _synthetic_otm_chain(n_bad=len(df) - 2)
    with pytest.raises(AssertionError, match="80%"):
        compute_iv_surface(df_mostly_bad)


def test_check_parity_iv_isolates_bad_quote():
    """Both-sides frame: clean quotes produce no breaches; one corrupted
    put mid is isolated at exactly that strike."""
    rows = []
    for K in np.arange(6300.0, 6750.0, 50.0):
        iv = 0.18 + 0.15 * np.log(F / K)
        for typ in ("call", "put"):
            rows.append({"expiry": pd.Timestamp("2026-08-08"), "strike": K,
                         "mid_price": b76_price(F, K, DF, T, iv, typ),
                         "F": F, "DF": DF, "T": T, "option_type": typ})
    df = pd.DataFrame(rows)
    assert check_parity_iv(df).empty
    bad_K = 6500.0
    mask = (df["strike"] == bad_K) & (df["option_type"] == "put")
    df.loc[mask, "mid_price"] += 5.0
    flags = check_parity_iv(df)
    assert list(flags["strike"]) == [bad_K]


def test_check_parity_iv_via_pipeline_seam():
    """Integration: the diagnostic must be reachable through the actual
    Module 1 API, i.e. attach_forwards(..., otm_only=False). (A fused
    attach+filter made this impossible in the first cut; this test pins
    the seam.)"""
    from src.data import attach_forwards, clean_chain
    from tests.test_data import ASOF, OFFLINE_CFG, SPOT, make_chain

    clean = clean_chain(make_chain(), SPOT, OFFLINE_CFG, asof=ASOF)
    both = attach_forwards(clean, SPOT, OFFLINE_CFG, otm_only=False)
    # both sides survive per strike, forwards attached
    sides = both.groupby(["expiry", "strike"])["option_type"].nunique()
    assert (sides == 2).all()
    assert {"F", "DF", "k"} <= set(both.columns)
    # exact synthetic mids with shared parity-implied (F, DF): no breaches
    assert check_parity_iv(both, OFFLINE_CFG).empty
    # and the default path still OTM-filters
    otm = attach_forwards(clean, SPOT, OFFLINE_CFG)
    assert (otm.groupby(["expiry", "strike"])["option_type"].nunique() == 1).all()