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

