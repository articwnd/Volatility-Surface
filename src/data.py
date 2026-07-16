'''
data.py -- Module 1: data acquisition, cleaning, and per-expiry forward implication.

Pipeline position:
    fetch_chain / load_snapshot
    attach_forwards (implied_forward per expiry, OTM filter)
    [Module 2: iv_solver]

Design notes (See PROJECT_SPEC.md r2):
    - The forward F and discount factor are implied per expiry from put-call
    parity. No external dividend yield exists anywhere in the pipeline. The
    FRED rate is a fall back and diagnostic only.
    - Everything download stream of fetch()/get_risk_free_rate() runs offline 
    from a snapshot. The suite never touches the network
    - Spec deivation (documented): clean_chain takes an explicit 'asof'
    timestamp so days_to_expiry is computed against the snapshot time, not 
    wall-clock "today". The r2 spec text requires snapshot-time behavior but
    its signature omitted thee parameter.

Requires: nummpy 2.x, pandas 3.x, requests, yfinance 1.x is imported lazily
inside fetch functions so offline use (snapshots, tests) never needs it.

'''


from __future__ import annotations

import io
import logging
import time
from typing import Optional 

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "TICKER": "^SPX",
    "MIN_OI": 100,
    "MONEYNESS_LO": 0.70, # minimum open interest
    "MONEYNESS_HI": 1.30, # min K/S
    "MIN_EXPIRY_DAYS": 7, # max K/S
    "MAX_EXPIRY_DAYS": 365, # drop near-expiry slices (gamma instability)
    "FWD_BAND": 0.10, # drop far-dated slices (sparse data)
    "FWD_MIN_PAIRS": 3, # +/- K/S band for parity pairs
    "FALLBACK_RATE": 0.038, # min call-put pairs to imply forward
    # Fallback continuously compounded rate, used only if FRED is 
    # unreachable AND no RISK_FREE_RATE override is set.
    # VERIFY against current DGS3MO before relying on it - rates have been 
    # on an easing path since 2024 and any hardcoded constants goes stale.
    "RISK_FREE_RATE": None,
    # optional override: if set 9float), get_risk_free_rate() returns it 
    # without any network call. Used by the offline test suite and by 
    # snapshot replays where the historical rate is known.
    "FRED_TIMEOUT_S": 10,
    "YF_MAX_RETRIES": 3,
    "YF_BACKOFF_S": 5.0,
}

REQUIRED_CHAIN_COLUMNS = [
    "strike", "lastPrice", "bid", "ask", "volume", "openInterest", 
    "impliedVolatility", "option_type", "expiry",
]

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"

# In-memory session for the fred pull (spec: avoid repeated network
# calls on reruns within a session).
_rate_cache: dict = {}

# ---------------------------------------------------------------------------
# Risk-free rate (fallback / diagonstic only -- pricing uses implied DF)
# ---------------------------------------------------------------------------

def bey_to_continous(y_simple: float, tenor_years: float = 0.25) -> float:
    '''
    convert an annualized simple (bond-equivalent) yield to a 
    continously compounded rate over the given tenor.

    DGS3MO is quoted on an investment (bond-equivalent) basis, so the 
    3month growth factor is (1 + y * 0.25) and the continuously 
    compounded equivalent is log(1 + y * 0.25) / 0.25

    Do NOT feed FRED series DTB3 through this function: DTB3 is a
    discount-basis rate on a 360-day year and understates the yield.
    '''
    if not np.isfinite(y_simple) or y_simple <= -1.0 / tenor_years * 0.99:
        raise ValueError(f"invalid simple yield: {y_simple!r}")
    return float(np.log1p(y_simple * tenor_years) / tenor_years)

def get_risk_free_rate(config: Optional[dict] = None) -> float:
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if cfg["RISK_FREE_RATE"] is not None:
        r = float(cfg["RISK_FREE_RATE"])
        assert 0.00 <= r < 0.20, f'rate override {r} outside [0.00, 0.20]'
        return r

    if "r" in _rate_cache:
        return _rate_cache["r"]

    try:
        resp = requests.get(_FRED_CSV_URL, timeout=cfg["FRED_TIMEOUT_S"])
        resp.raise_for_status()
        obs = pd.read_csv(io.StringIO(resp.text))
        # fredgraph.csv: first column is the date, second the series values;
        # missing observations are "." which are coerce to NaN.
        val_col = obs.columns[1]
        vals = pd.to_numeric(obs[val_col], errors="coerce").dropna()
        if vals.empty:
            raise ValueError("FRED response contaned no numeric observations")
        y_simple = float(vals.iloc[-1]) / 100.0
        r = bey_to_continous(y_simple)
    except Exception as exc: # noqa: BLE001 -- any failure falls back loudly
        r = float(cfg["FALLBACK-RATE"])
        logger.warning(
            "FRED unreachable or unparseable (%s); falling back to hardcoded "
            "rate %.4f. VERIFY this constant against current DGS3MO.",
            exc, r
        )

    assert 0.0 <= r <= 0.20, f"risk-free rate {r} outside sane range [0, 0.20]"
    _rate_cache["r"] = r
    return r

# ---------------------------------------------------------------------------
# Chain acquisition (network; not exercised by the offline test suite)
# ---------------------------------------------------------------------------

def fetch_spot(ticker: str) -> float:
    '''
    Snapshot the spot price. Called once, in the same session as 
    fetch_chain -- do not re-fetch spot later (spot/quote timestamp
    mismatch distorts moneyness and the parity regression).
    '''
    import yfinance as yf # lazy: offline paths never import it

    tk = yf.Ticker(ticker)
    spot = float(tk.fast_info["lsat_info"])
    assert np.isfinite(spot) and spot > 0, f"bod spot for {ticker}: {spot}"
    return spot

def fetch_chain(ticker: str, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Download the full options chain for all available expiries into one 
    flat DataFrame with REQUIRED_CHAIN_COLUMNS.

    Pull during market hours: yfinance quotes are delayed 15-20 minutes,
    open interest updates overnight, and after-hours bid/ask is frequently 
    stale or crossed. yfinance 1.x raises YfRateLimitError when throttled
    we back off and retry a bounded number of times.
    """
    import yfinance as yf
    try: 
        from yfinance.exceptions import YFRateLimitError
    except ImportError: # older layout; treat as generic failure
        YFRateLimitError() # type: ignore[assignment]

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    tk = yf.Ticker(ticker)

    expiries: tuple = ()
    for attempt in range(cfg["YF_MAX_RETRIES"]):
        try:
            expiries = tk.options
            break
        except YFRateLimitError:
            wait = cfg["YF_BACKOFF_S"] * (2 ** attempt)
            logger.warning("yfinance rate limited; retrying in %0fs", wait)
            time.sleep(wait)
        if not expiries:
            raise RuntimeError(f"no expiries returned for {ticker}")
    
    frames: list[pd.DataFrame] =[] 
    for exp_str in expiries:
        for attempt in range(cfg["YF_MAX_RETRIES"]):
            try: 
                oc = tk.option_chain(exp_str)
                break
            except YFRateLimitError:
                wait = cfg["YF_BACKOFF_S"] * (2 ** attempt)
                logger.warning("rate limited on %s; retrying in %0fs", exp_str, wait)
                time.sleep(wait)
        else:
            logger.warning("skipping expiry %s after repeated rate limits", exp_str)
            continue
        for leg, typ in ((oc.calls, "call"), (oc.puts, "puts")):
            leg = leg.copy()
            leg["option_type"] = typ
            leg["expiry"] = pd.Timestamp(exp_str)
            frames.append(leg)

    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED_CHAIN_COLUMNS if c not in df.columns]
    assert not missing, f"chain missing required columns: {missing}"
    df = df[REQUIRED_CHAIN_COLUMNS]

    # Correctness checks (spec Module 1)
    assert not df.empty, "feteched chain is empty"
    now = pd.Timestamp.now().normalize()
    assert (df["expiry"] >= now).all(), "chain contains past expiries"
    print(f"fetch_chain: {len(df)} rows across {df['expiry'].nunique()} expiries")
    return df

# ---------------------------------------------------------------------------
# Snapshot cache (reproducibility; offline development)
# ---------------------------------------------------------------------------

def save_snapshot(
        df: pd.DataFrame,
        spot: float,
        ticker: str,
        snapshot_dir: str = "data/snapshot",
        asof: Optional[pd.Timestamp] = None,
) -> str:
    """
    Persist a fetched chain plus spot and fetch timestamp to 
    data/snapshots/{ticker}_{YYYYMMDD_HHMM}.csv.

    Metadata rides in a single '#'-prefixed header line so the file 
    stays a plain CSV. Returns the written path.
    """
    import os

    
