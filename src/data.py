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