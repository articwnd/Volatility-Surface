"""Offline tests for src/data.py. No network access anywhere.

Synthetic chains are generated from exact Black-76 prices with known
(F, DF), so the parity regression has a machine-precision ground truth.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.data import (
    DEFAULT_CONFIG,
    attach_forwards,
    bey_to_continuous,
    clean_chain,
    implied_forward,
    load_snapshot,
    save_snapshot,
)

# Config that never touches the network: rate override short-circuits FRED.
OFFLINE_CFG = {**DEFAULT_CONFIG, "RISK_FREE_RATE": 0.04}


# ---------------------------------------------------------------------------
# Synthetic chain construction
# ---------------------------------------------------------------------------

def b76_price(F, K, DF, T, sigma, option_type):
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return call if option_type == "call" else call - DF * (F - K)


def synthetic_expiry(F, DF, T, spot, strikes, expiry, skew=0.15, atm_vol=0.18):
    """Both calls and puts at every strike, exact B76 mids, tight spreads."""
    rows = []
    for K in strikes:
        iv = atm_vol + skew * np.log(F / K)
        for typ in ("call", "put"):
            mid = b76_price(F, K, DF, T, iv, typ)
            rows.append({
                "strike": float(K),
                "lastPrice": mid,
                "bid": max(mid - 0.05, 0.01),
                "ask": mid + 0.05,
                "volume": 500.0,
                "openInterest": 1000.0,
                "impliedVolatility": iv,
                "option_type": typ,
                "expiry": pd.Timestamp(expiry),
            })
    return pd.DataFrame(rows)


SPOT = 6500.0
ASOF = pd.Timestamp("2026-07-09 14:00")


def make_chain():
    """Two expiries with known forwards/discount factors."""
    e1 = synthetic_expiry(
        F=6510.0, DF=0.9970, T=30 / 365, spot=SPOT,
        strikes=np.arange(6100, 6950, 50.0), expiry="2026-08-08",
    )
    e2 = synthetic_expiry(
        F=6540.0, DF=0.9880, T=120 / 365, spot=SPOT,
        strikes=np.arange(5900, 7150, 50.0), expiry="2026-11-06",
    )
    return pd.concat([e1, e2], ignore_index=True)


def cleaned_expiry(F, DF, T, expiry, **kw):
    df = synthetic_expiry(F, DF, T, SPOT, np.arange(6100, 6950, 50.0), expiry, **kw)
    return clean_chain(df, SPOT, OFFLINE_CFG, asof=ASOF)


# ---------------------------------------------------------------------------
# Spec-required tests
# ---------------------------------------------------------------------------

def test_forward_recovery():
    """Synthetic B76 prices with known (F, DF): recovery within 1e-8."""
    F_true, DF_true, T = 6510.0, 0.9970, 30 / 365
    df = cleaned_expiry(F_true, DF_true, T, "2026-08-08")
    F, DF, diag = implied_forward(df, SPOT, T, OFFLINE_CFG)
    assert not diag["fallback"]
    # Mid = exact price on both legs, so bid/ask offsets cancel in C - P.
    assert abs(F - F_true) < 1e-8
    assert abs(DF - DF_true) < 1e-8
    assert diag["resid_rms"] < 1e-8


def test_forward_fallback():
    """Fewer than FWD_MIN_PAIRS pairs triggers the rate fallback + flag."""
    T = 30 / 365
    df = cleaned_expiry(6510.0, 0.9970, T, "2026-08-08")
    # Keep only strikes outside the FWD_BAND so no near-the-money pairs exist.
    df = df[np.abs(df["strike"] / SPOT - 1.0) > DEFAULT_CONFIG["FWD_BAND"]]
    F, DF, diag = implied_forward(df, SPOT, T, OFFLINE_CFG)
    assert diag["fallback"] is True
    r = OFFLINE_CFG["RISK_FREE_RATE"]
    assert F == pytest.approx(SPOT * np.exp(r * T), rel=1e-12)
    assert DF == pytest.approx(np.exp(-r * T), rel=1e-12)


def test_forward_sanity_reject():
    """Corrupt prices producing an insane DF trigger fallback, not a bad
    forward."""
    T = 30 / 365
    df = cleaned_expiry(6510.0, 0.9970, T, "2026-08-08").copy()
    # Corrupt every call mid with a strike-proportional error: the fitted
    # slope becomes -(DF + 0.5), i.e. DF_hat ~ 1.5, failing the DF gate.
    is_call = df["option_type"] == "call"
    df.loc[is_call, "mid_price"] = (
        df.loc[is_call, "mid_price"] - 0.5 * df.loc[is_call, "strike"] + 4000.0
    )
    F, DF, diag = implied_forward(df, SPOT, T, OFFLINE_CFG)
    assert diag["fallback"] is True
    r = OFFLINE_CFG["RISK_FREE_RATE"]
    assert DF == pytest.approx(np.exp(-r * T), rel=1e-12)


def test_snapshot_roundtrip(tmp_path):
    """load(save(df)) preserves shape and required dtypes."""
    df = make_chain()
    path = save_snapshot(df, SPOT, "^SPX", snapshot_dir=str(tmp_path), asof=ASOF)
    df2, spot2, asof2 = load_snapshot(path)
    assert spot2 == SPOT
    assert asof2 == ASOF
    assert df2.shape == df.shape
    assert list(df2.columns) == list(df.columns)
    pd.testing.assert_series_equal(df2["strike"], df["strike"])
    pd.testing.assert_series_equal(df2["expiry"], df["expiry"])
    assert df2["strike"].dtype == np.float64
    assert df2["openInterest"].dtype == np.float64
    np.testing.assert_allclose(df2["bid"], df["bid"], rtol=0, atol=1e-12)


def test_rate_basis_conversion():
    """r = log(1 + y/4)/0.25, and r < y for positive yields."""
    y = 0.0385
    r = bey_to_continuous(y)
    assert r == pytest.approx(np.log(1 + y * 0.25) / 0.25, rel=1e-14)
    assert r < y
    # Growth-factor equivalence over the tenor:
    assert np.exp(r * 0.25) == pytest.approx(1 + y * 0.25, rel=1e-14)
    with pytest.raises(ValueError):
        bey_to_continuous(float("nan"))


# ---------------------------------------------------------------------------
# Cleaning and attach_forwards invariants
# ---------------------------------------------------------------------------

def test_clean_chain_filters_and_asof():
    df = make_chain()
    # Inject rows that each filter must remove.
    bad = df.iloc[:4].copy()
    bad.loc[bad.index[0], "bid"] = 0.0                       # filter 1
    bad.loc[bad.index[1], "openInterest"] = 5.0              # filter 2
    bad.loc[bad.index[2], "strike"] = SPOT * 2.0             # filter 4
    bad.loc[bad.index[3], "expiry"] = ASOF + pd.Timedelta(days=2)  # filter 5
    out = clean_chain(pd.concat([df, bad], ignore_index=True), SPOT,
                      OFFLINE_CFG, asof=ASOF)
    assert len(out) == len(df)  # exactly the injected rows are gone
    # days_to_expiry measured from ASOF, not wall clock
    assert set(out["days_to_expiry"].unique()) == {30, 120}
    assert np.allclose(out["T"], out["days_to_expiry"] / 365.0)


def test_clean_chain_drops_thin_expiry():
    df = make_chain()
    thin = synthetic_expiry(6520.0, 0.9940, 60 / 365, SPOT,
                            strikes=[6450.0, 6500.0], expiry="2026-09-07")
    out = clean_chain(pd.concat([df, thin], ignore_index=True), SPOT,
                      OFFLINE_CFG, asof=ASOF)
    assert pd.Timestamp("2026-09-07") not in set(out["expiry"])


def test_attach_forwards_invariants():
    df = clean_chain(make_chain(), SPOT, OFFLINE_CFG, asof=ASOF)
    out = attach_forwards(df, SPOT, OFFLINE_CFG)
    # One (F, DF) per expiry, near the synthetic truth
    fwd = out.groupby("expiry")[["F", "DF"]].first()
    assert fwd.loc["2026-08-08", "F"] == pytest.approx(6510.0, abs=1e-6)
    assert fwd.loc["2026-11-06", "F"] == pytest.approx(6540.0, abs=1e-6)
    assert not out["fwd_fallback"].any()
    # OTM-only with correct k signs
    calls = out[out["option_type"] == "call"]
    puts = out[out["option_type"] == "put"]
    assert (calls["k"] >= -1e-12).all()
    assert (puts["k"] <= 1e-12).all()
    # k is log(K/F)
    np.testing.assert_allclose(out["k"], np.log(out["strike"] / out["F"]),
                               rtol=0, atol=1e-15)