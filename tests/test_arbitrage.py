"""Offline tests for src/arbitrage.py. No network access anywhere.

Butterfly ground truth: Black-76 call prices are strictly convex in
strike, so any flag on exact B76 prices is a false positive by
construction, and injected non-convexity must be caught.
"""

import numpy as np
import pandas as pd
import pytest

from src.arbitrage import (
    DEFAULT_ARB_CONFIG,
    check_butterfly,
    check_calendar,
    exclude_flagged,
    run_arbitrage_checks,
    synthesize_call_curve,
)
from src.iv_solver import b76_price

F, DF, T = 6510.0, 0.9970, 30 / 365
EXPIRY = pd.Timestamp("2026-08-08")


def _otm_slice(strikes, F_=F, DF_=DF, T_=T, expiry=EXPIRY,
               atm_vol=0.18, skew=0.15):
    """OTM-only slice shaped like compute_iv_surface output: puts below
    the forward, calls above, exact B76 mids from a skewed smile."""
    rows = []
    for K in np.asarray(strikes, dtype=np.float64):
        iv = atm_vol + skew * np.log(F_ / K)
        typ = "put" if K <= F_ else "call"
        rows.append({
            "expiry": expiry, "strike": K, "option_type": typ,
            "mid_price": b76_price(F_, K, DF_, T_, iv, typ),
            "F": F_, "DF": DF_, "T": T_, "k": np.log(K / F_), "iv": iv,
        })
    return pd.DataFrame(rows)


# Deliberately irregular SPX-style spacing: 25s near the money, 50s and
# 100s in the wings.
UNEQUAL_STRIKES = np.concatenate([
    np.arange(5900.0, 6300.0, 100.0),
    np.arange(6300.0, 6450.0, 50.0),
    np.arange(6450.0, 6600.0, 25.0),
    np.arange(6600.0, 6800.0, 50.0),
    np.arange(6800.0, 7101.0, 100.0),
])


# ---------------------------------------------------------------------------
# Spec-required tests
# ---------------------------------------------------------------------------

def test_clean_slice_no_butterfly():
    """Exact B76 prices are strictly convex in K: no flags on unequally
    spaced strikes."""
    flags = check_butterfly(_otm_slice(UNEQUAL_STRIKES), EXPIRY)
    assert flags.empty


def test_butterfly_violation():
    """Bumping one middle quote above the chord must be flagged at that
    strike (and only near it)."""
    df = _otm_slice(UNEQUAL_STRIKES).copy()
    bad_K = 6500.0
    df.loc[df["strike"] == bad_K, "mid_price"] += 5.0  # put mid, K < F
    flags = check_butterfly(df, EXPIRY)
    assert not flags.empty
    assert bad_K in set(flags["K2"])  # the bumped strike is a middle leg
    assert (flags["B_value"] < 0).all()


def test_equal_spacing_formula_would_misfire():
    """A strictly convex curve on spacing (5, 25): the r1 unweighted
    formula reports a large negative 'butterfly' (false flag); the
    weighted check passes it clean."""
    K1, K2, K3 = 6400.0, 6405.0, 6430.0
    df = _otm_slice([K1, K2, K3], atm_vol=0.20, skew=0.0)
    curve = synthesize_call_curve(df)
    C = dict(zip(curve["strike"], curve["call_price"]))
    unweighted = C[K1] - 2.0 * C[K2] + C[K3]
    assert unweighted < -1.0  # r1 formula would scream violation here
    flags = check_butterfly(df, EXPIRY)
    assert flags.empty       # weighted check: correctly clean


def test_synthesized_calls():
    """Puts converted via C = P + DF*(F - K) match direct B76 call
    prices to 1e-10 (parity is exact for our pricer by construction)."""
    df = _otm_slice(UNEQUAL_STRIKES)
    curve = synthesize_call_curve(df)
    for K, C_synth in zip(curve["strike"], curve["call_price"]):
        iv = 0.18 + 0.15 * np.log(F / K)
        C_direct = b76_price(F, K, DF, T, iv, "call")
        assert C_synth == pytest.approx(C_direct, abs=1e-10)


def test_clean_calendar():
    """Non-decreasing ATM-forward total variance across maturities at
    fixed k: no flags."""
    frames = [
        _otm_slice(UNEQUAL_STRIKES, F_=6510.0, DF_=0.9970, T_=30 / 365,
                   expiry=pd.Timestamp("2026-08-08"), atm_vol=0.18),
        _otm_slice(UNEQUAL_STRIKES, F_=6525.0, DF_=0.9930, T_=60 / 365,
                   expiry=pd.Timestamp("2026-09-07"), atm_vol=0.18),
        _otm_slice(UNEQUAL_STRIKES, F_=6540.0, DF_=0.9880, T_=120 / 365,
                   expiry=pd.Timestamp("2026-11-06"), atm_vol=0.18),
    ]
    flags = check_calendar(pd.concat(frames, ignore_index=True))
    assert flags.empty


def test_calendar_violation():
    """A short slice with ATM total variance above a longer one must be
    flagged as (short, long) with the w values reported."""
    hot_short = _otm_slice(UNEQUAL_STRIKES, F_=6510.0, T_=30 / 365,
                           expiry=pd.Timestamp("2026-08-08"), atm_vol=0.40)
    cold_long = _otm_slice(UNEQUAL_STRIKES, F_=6525.0, DF_=0.9930,
                           T_=60 / 365, expiry=pd.Timestamp("2026-09-07"),
                           atm_vol=0.18)
    flags = check_calendar(pd.concat([hot_short, cold_long],
                                     ignore_index=True))
    assert len(flags) == 1
    row = flags.iloc[0]
    assert row["expiry_short"] == pd.Timestamp("2026-08-08")
    assert row["expiry_long"] == pd.Timestamp("2026-09-07")
    assert row["w_short"] > row["w_long"]
    # sanity on the magnitudes: 0.40^2*30/365 vs 0.18^2*60/365
    assert row["w_short"] == pytest.approx(0.40**2 * 30 / 365, rel=1e-2)
    assert row["w_long"] == pytest.approx(0.18**2 * 60 / 365, rel=1e-2)


# ---------------------------------------------------------------------------
# Runner, CSV output, exclusion policy
# ---------------------------------------------------------------------------

def _three_slice_frame(bad_strike=None, hot_expiry=None):
    frames = []
    for F_, DF_, days, exp, vol in [
        (6510.0, 0.9970, 30, "2026-08-08", 0.18),
        (6525.0, 0.9930, 60, "2026-09-07", 0.18),
        (6540.0, 0.9880, 120, "2026-11-06", 0.18),
    ]:
        vol = 0.40 if exp == hot_expiry else vol
        frames.append(_otm_slice(UNEQUAL_STRIKES, F_=F_, DF_=DF_,
                                 T_=days / 365, expiry=pd.Timestamp(exp),
                                 atm_vol=vol))
    df = pd.concat(frames, ignore_index=True)
    if bad_strike is not None:
        mask = ((df["expiry"] == pd.Timestamp("2026-08-08"))
                & (df["strike"] == bad_strike))
        df.loc[mask, "mid_price"] += 5.0
    return df


def test_run_arbitrage_checks_writes_csv(tmp_path):
    cfg = {**DEFAULT_ARB_CONFIG,
           "FLAGS_CSV": str(tmp_path / "arbitrage_flags.csv")}
    df = _three_slice_frame(bad_strike=6500.0, hot_expiry="2026-08-08")
    bf, cal = run_arbitrage_checks(df, cfg)
    assert not bf.empty and not cal.empty
    written = pd.read_csv(tmp_path / "arbitrage_flags.csv")
    assert set(written["check"]) == {"butterfly", "calendar"}
    assert len(written) == len(bf) + len(cal)


def test_exclude_flagged_butterfly_drops_middle_strike():
    df = _three_slice_frame(bad_strike=6500.0)
    bf, cal = run_arbitrage_checks(
        df, {**DEFAULT_ARB_CONFIG, "FLAGS_CSV": "outputs/_tmp_flags.csv"})
    assert cal.empty
    out = exclude_flagged(df, bf, cal)
    exp = pd.Timestamp("2026-08-08")
    dropped = set(bf["K2"])
    kept = set(out.loc[out["expiry"] == exp, "strike"])
    assert dropped.isdisjoint(kept)
    # other expiries untouched
    for other in ("2026-09-07", "2026-11-06"):
        assert (out["expiry"] == pd.Timestamp(other)).sum() == \
               (df["expiry"] == pd.Timestamp(other)).sum()


def test_exclude_flagged_calendar_drops_repeat_offender():
    """The hot short slice violates against BOTH longer maturities
    (2 pairs), so the exclusion rule drops the whole slice."""
    df = _three_slice_frame(hot_expiry="2026-08-08")
    bf, cal = run_arbitrage_checks(
        df, {**DEFAULT_ARB_CONFIG, "FLAGS_CSV": "outputs/_tmp_flags.csv"})
    assert len(cal) == 2
    out = exclude_flagged(df, bf, cal)
    assert pd.Timestamp("2026-08-08") not in set(out["expiry"])
    assert out["expiry"].nunique() == 2


def test_pipeline_end_to_end_clean():
    """Modules 1-3 on an exact synthetic chain: zero violations."""
    from src.data import attach_forwards, clean_chain
    from src.iv_solver import compute_iv_surface
    from tests.test_data import ASOF, OFFLINE_CFG, SPOT, make_chain

    clean = clean_chain(make_chain(), SPOT, OFFLINE_CFG, asof=ASOF)
    surf = compute_iv_surface(attach_forwards(clean, SPOT, OFFLINE_CFG),
                              OFFLINE_CFG)
    bf, cal = run_arbitrage_checks(
        surf, {**DEFAULT_ARB_CONFIG, "FLAGS_CSV": "outputs/_tmp_flags.csv"})
    assert bf.empty and cal.empty