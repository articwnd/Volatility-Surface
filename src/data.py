"""
data.py -- Module 1: data acquisition, cleaning, and per-expiry forward implication.

Pipeline position:
    fetch_chain / load_snapshot
        -> clean_chain
        -> attach_forwards (implied_forward per expiry, OTM filter)
        -> [Module 2: iv_solver]

Design notes (see PROJECT_SPEC.md r2):
- The forward F and discount factor DF are implied per expiry from put-call
  parity. No external dividend yield exists anywhere in the pipeline. The
  FRED rate is a fallback and diagnostic only.
- Everything downstream of fetch_chain()/get_risk_free_rate() runs offline
  from a snapshot. The test suite never touches the network.
- Spec deviation (documented): clean_chain takes an explicit `asof`
  timestamp so days_to_expiry is computed against the snapshot time, not
  wall-clock "today". The r2 spec text requires snapshot-time behavior but
  its signature omitted the parameter.

Requires: numpy 2.x, pandas 3.x, requests. yfinance 1.x is imported lazily
inside fetch functions so offline use (snapshots, tests) never needs it.
"""

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
    "MIN_OI": 100,            # minimum open interest
    "MONEYNESS_LO": 0.70,     # min K/S
    "MONEYNESS_HI": 1.30,     # max K/S
    "MIN_EXPIRY_DAYS": 7,     # drop near-expiry slices (gamma instability)
    "MAX_EXPIRY_DAYS": 365,   # drop far-dated slices (sparse data)
    "FWD_BAND": 0.10,         # +/- K/S band for parity pairs
    "FWD_MIN_PAIRS": 3,       # min call-put pairs to imply forward
    # Preferred option root when one expiry carries multiple products
    # (e.g. AM-settled SPX and PM-settled SPXW on third Fridays -- mixing
    # them interleaves two price curves and manufactures phantom
    # arbitrage; measured r5: 121/153 executable butterfly flags
    # clustered on third Fridays before this filter). SPXW's PM
    # settlement matches the close-of-day snapshot and the calendar-day
    # T convention. None disables root filtering.
    "OPTION_ROOT": "SPXW",
    # Fallback continuously compounded rate, used only if FRED is
    # unreachable AND no RISK_FREE_RATE override is set.
    # VERIFY against current DGS3MO before relying on it -- rates have been
    # on an easing path since 2024 and any hardcoded constant goes stale.
    "FALLBACK_RATE": 0.038,
    # Optional override: if set (float), get_risk_free_rate() returns it
    # without any network call. Used by the offline test suite and by
    # snapshot replays where the historical rate is known.
    "RISK_FREE_RATE": None,
    "FRED_TIMEOUT_S": 10,
    "YF_MAX_RETRIES": 3,
    "YF_BACKOFF_S": 5.0,
}

REQUIRED_CHAIN_COLUMNS = [
    "contractSymbol", "strike", "lastPrice", "bid", "ask", "volume",
    "openInterest", "impliedVolatility", "option_type", "expiry",
]

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"

# In-memory session cache for the FRED pull (spec: avoid repeated network
# calls on reruns within a session).
_rate_cache: dict = {}


# ---------------------------------------------------------------------------
# Risk-free rate (fallback / diagnostic only -- pricing uses implied DF)
# ---------------------------------------------------------------------------

def bey_to_continuous(y_simple: float, tenor_years: float = 0.25) -> float:
    """Convert an annualized simple (bond-equivalent) yield to a
    continuously compounded rate over the given tenor.

    DGS3MO is quoted on an investment (bond-equivalent) basis, so the
    3-month growth factor is (1 + y * 0.25) and the continuously
    compounded equivalent is log(1 + y * 0.25) / 0.25.

    Do NOT feed FRED series DTB3 through this function: DTB3 is a
    discount-basis rate on a 360-day year and understates the yield.
    """
    if not np.isfinite(y_simple) or y_simple <= -1.0 / tenor_years * 0.99:
        raise ValueError(f"invalid simple yield: {y_simple!r}")
    return float(np.log1p(y_simple * tenor_years) / tenor_years)


def get_risk_free_rate(config: Optional[dict] = None) -> float:
    """Most recent 3-month Treasury yield (DGS3MO) as a continuously
    compounded decimal.

    Used ONLY as the fallback when an expiry has too few call-put pairs to
    imply the forward, and as a diagnostic against implied financing.

    Resolution order:
    1. config['RISK_FREE_RATE'] override, if set (no network).
    2. Session cache from a prior successful FRED pull.
    3. Keyless FRED CSV endpoint (the JSON API requires an API key).
    4. config['FALLBACK_RATE'] with a loud warning.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if cfg["RISK_FREE_RATE"] is not None:
        r = float(cfg["RISK_FREE_RATE"])
        assert 0.0 <= r <= 0.20, f"rate override {r} outside [0, 0.20]"
        return r

    if "r" in _rate_cache:
        return _rate_cache["r"]

    try:
        resp = requests.get(_FRED_CSV_URL, timeout=cfg["FRED_TIMEOUT_S"])
        resp.raise_for_status()
        obs = pd.read_csv(io.StringIO(resp.text))
        # fredgraph.csv: first column is the date, second the series values;
        # missing observations are "." which coerce to NaN.
        val_col = obs.columns[1]
        vals = pd.to_numeric(obs[val_col], errors="coerce").dropna()
        if vals.empty:
            raise ValueError("FRED response contained no numeric observations")
        y_simple = float(vals.iloc[-1]) / 100.0
        r = bey_to_continuous(y_simple)
    except Exception as exc:  # noqa: BLE001 -- any failure falls back loudly
        r = float(cfg["FALLBACK_RATE"])
        logger.warning(
            "FRED unreachable or unparseable (%s); falling back to hardcoded "
            "rate %.4f. VERIFY this constant against current DGS3MO.",
            exc, r,
        )

    assert 0.0 <= r <= 0.20, f"risk-free rate {r} outside sane range [0, 0.20]"
    _rate_cache["r"] = r
    return r


# ---------------------------------------------------------------------------
# Chain acquisition (network; not exercised by the offline test suite)
# ---------------------------------------------------------------------------

def fetch_spot(ticker: str) -> float:
    """Snapshot the spot price. Called once, in the same session as
    fetch_chain -- do not re-fetch spot later (spot/quote timestamp
    mismatch distorts moneyness and the parity regression)."""
    import yfinance as yf  # lazy: offline paths never import it

    tk = yf.Ticker(ticker)
    spot = float(tk.fast_info["last_price"])
    assert np.isfinite(spot) and spot > 0, f"bad spot for {ticker}: {spot}"
    return spot


def fetch_chain(ticker: str, config: Optional[dict] = None) -> pd.DataFrame:
    """Download the full options chain for all available expiries into one
    flat DataFrame with REQUIRED_CHAIN_COLUMNS.

    Pull during market hours: yfinance quotes are delayed 15-20 minutes,
    open interest updates overnight, and after-hours bid/ask is frequently
    stale or crossed. yfinance 1.x raises YfRateLimitError when throttled;
    we back off and retry a bounded number of times.
    """
    import yfinance as yf
    try:
        from yfinance.exceptions import YfRateLimitError
    except ImportError:  # older layout; treat as generic failure
        YfRateLimitError = ()  # type: ignore[assignment]

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    tk = yf.Ticker(ticker)

    expiries: tuple = ()
    for attempt in range(cfg["YF_MAX_RETRIES"]):
        try:
            expiries = tk.options
            break
        except YfRateLimitError:
            wait = cfg["YF_BACKOFF_S"] * (2 ** attempt)
            logger.warning("yfinance rate limited; retrying in %.0fs", wait)
            time.sleep(wait)
    if not expiries:
        raise RuntimeError(f"no expiries returned for {ticker}")

    frames: list[pd.DataFrame] = []
    for exp_str in expiries:
        for attempt in range(cfg["YF_MAX_RETRIES"]):
            try:
                oc = tk.option_chain(exp_str)
                break
            except YfRateLimitError:
                wait = cfg["YF_BACKOFF_S"] * (2 ** attempt)
                logger.warning("rate limited on %s; retrying in %.0fs", exp_str, wait)
                time.sleep(wait)
        else:
            logger.warning("skipping expiry %s after repeated rate limits", exp_str)
            continue
        for leg, typ in ((oc.calls, "call"), (oc.puts, "put")):
            leg = leg.copy()
            leg["option_type"] = typ
            leg["expiry"] = pd.Timestamp(exp_str)
            frames.append(leg)

    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED_CHAIN_COLUMNS if c not in df.columns]
    assert not missing, f"chain missing required columns: {missing}"
    df = df[REQUIRED_CHAIN_COLUMNS]

    # Correctness checks (spec Module 1)
    assert not df.empty, "fetched chain is empty"
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
    snapshot_dir: str = "data/snapshots",
    asof: Optional[pd.Timestamp] = None,
) -> str:
    """Persist a fetched chain plus spot and fetch timestamp to
    data/snapshots/{ticker}_{YYYYMMDD_HHMM}.csv.

    Metadata rides in a single '#'-prefixed header line so the file stays
    a plain CSV. Returns the written path.
    """
    import os

    asof = pd.Timestamp.now() if asof is None else pd.Timestamp(asof)
    os.makedirs(snapshot_dir, exist_ok=True)
    safe_ticker = ticker.replace("^", "").replace("/", "-")
    path = os.path.join(
        snapshot_dir, f"{safe_ticker}_{asof.strftime('%Y%m%d_%H%M')}.csv"
    )
    with open(path, "w", newline="") as fh:
        fh.write(f"# ticker={ticker},spot={spot!r},asof={asof.isoformat()}\n")
        df.to_csv(fh, index=False)
    logger.info("snapshot written: %s (%d rows)", path, len(df))
    return path


def load_snapshot(path: str) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    """Load a snapshot written by save_snapshot.

    Returns (chain DataFrame, spot, asof timestamp). Required columns are
    restored to their canonical dtypes so the roundtrip is exact.
    """
    with open(path) as fh:
        header = fh.readline()
        assert header.startswith("# "), f"not a snapshot file: {path}"
        meta = dict(kv.split("=", 1) for kv in header[2:].strip().split(","))
        df = pd.read_csv(fh)

    df["expiry"] = pd.to_datetime(df["expiry"])
    df["option_type"] = df["option_type"].astype(str)
    if "contractSymbol" in df.columns:
        df["contractSymbol"] = df["contractSymbol"].astype(str)
    else:
        logger.warning(
            "load_snapshot: %s predates r5 (no contractSymbol column); "
            "the option-root filter cannot run and settlement-mixed "
            "expiries may carry phantom arbitrage. Re-pull to fix.", path,
        )
    for col in ("strike", "lastPrice", "bid", "ask", "impliedVolatility"):
        df[col] = df[col].astype(np.float64)
    for col in ("volume", "openInterest"):
        # yfinance emits these as floats with NaN for no-trade rows; keep
        # float64 so the roundtrip is dtype-stable.
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    spot = float(meta["spot"])
    asof = pd.Timestamp(meta["asof"])
    return df, spot, asof


# ---------------------------------------------------------------------------
# Option-root filtering (r5)
# ---------------------------------------------------------------------------

_ROOT_RE = __import__("re").compile(r"^([A-Z]+)")


def option_root(contract_symbol: str) -> str:
    """Root ticker from an OCC-style contract symbol: the leading alpha
    run, e.g. 'SPXW260821C05700000' -> 'SPXW', 'SPX...' -> 'SPX'."""
    m = _ROOT_RE.match(str(contract_symbol))
    return m.group(1) if m else ""


def filter_option_root(df: pd.DataFrame,
                       config: Optional[dict] = None) -> pd.DataFrame:
    """Keep exactly one option root per expiry.

    Per expiry: keep only OPTION_ROOT if present; otherwise, if a single
    root exists, keep it; if multiple non-preferred roots exist, keep the
    most-quoted one and warn. Rationale: one expiry date can carry
    multiple products with different settlement (AM-settled SPX vs
    PM-settled SPXW on third Fridays); their price curves differ by a
    settlement basis, and interleaving them by strike manufactures
    convexity violations that are not arbitrage.

    No-ops with a warning when contractSymbol is absent (pre-r5
    snapshots) or when OPTION_ROOT is None.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    preferred = cfg["OPTION_ROOT"]
    if preferred is None:
        return df
    if "contractSymbol" not in df.columns:
        logger.warning(
            "filter_option_root: no contractSymbol column; root filter "
            "skipped -- settlement-mixed expiries may carry phantom "
            "arbitrage"
        )
        return df

    df = df.copy()
    roots = df["contractSymbol"].map(option_root)
    kept: list[pd.DataFrame] = []
    for expiry, grp in df.groupby("expiry", sort=True):
        r = roots.loc[grp.index]
        present = r.value_counts()
        if preferred in present.index:
            chosen = preferred
        elif len(present) == 1:
            chosen = present.index[0]
        else:
            chosen = present.idxmax()
            logger.warning(
                "filter_option_root: expiry %s has roots %s, none "
                "preferred (%s); keeping most-quoted %s",
                expiry, list(present.index), preferred, chosen,
            )
        dropped = int((r != chosen).sum())
        if dropped:
            logger.info(
                "filter_option_root: expiry %s kept root %s, dropped %d "
                "row(s) from %s", expiry, chosen, dropped,
                [x for x in present.index if x != chosen],
            )
        kept.append(grp[r == chosen])
    return pd.concat(kept, ignore_index=True)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_chain(
    df: pd.DataFrame,
    spot: float,
    config: Optional[dict] = None,
    asof: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Apply all filters and compute derived columns.

    `asof` is the snapshot timestamp; days_to_expiry is computed against it
    (not wall-clock now) so snapshot replays are reproducible. Defaults to
    now() for live pulls.

    Adds: mid_price, moneyness, days_to_expiry, T.
    Drops (in order, with per-step counts printed): non-positive bids, low
    OI, non-positive mids, out-of-band moneyness, out-of-range expiries,
    then any expiry left with fewer than 3 strikes.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    asof = pd.Timestamp.now() if asof is None else pd.Timestamp(asof)
    assert np.isfinite(spot) and spot > 0, f"bad spot: {spot}"

    df = df.copy()
    # 0. one option root per expiry (settlement-mixing guard, r5)
    n_pre = len(df)
    df = filter_option_root(df, cfg)
    n0 = len(df)

    def _report(step: str, before: int, after: int) -> None:
        print(f"clean_chain [{step}]: dropped {before - after}, kept {after}")

    _report("option root", n_pre, n0)

    # 1. stale markets
    df = df[df["bid"] > 0]
    n1 = len(df); _report("bid>0", n0, n1)

    # 1b. crossed/locked quote guard (r5): a row with ask <= 0 or
    # ask < bid is not a market; its mid is meaningless and, worse, its
    # bid/ask poison the executable arbitrage tiers (measured r5: junk
    # quotes at non-standard strikes produced phantom $14-$72
    # "executable" violations).
    df = df[(df["ask"] > 0) & (df["ask"] >= df["bid"])]
    n1b = len(df); _report("ask>=bid>0", n1, n1b)
    n1 = n1b

    # 2. illiquid strikes
    df = df[df["openInterest"].fillna(0) >= cfg["MIN_OI"]]
    n2 = len(df); _report(f"OI>={cfg['MIN_OI']}", n1, n2)

    # 3. mid price
    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
    df = df[df["mid_price"] > 0]
    n3 = len(df); _report("mid>0", n2, n3)

    # 4. moneyness band
    df["moneyness"] = df["strike"] / spot
    df = df[df["moneyness"].between(cfg["MONEYNESS_LO"], cfg["MONEYNESS_HI"])]
    n4 = len(df); _report("moneyness band", n3, n4)

    # 5. expiry window (calendar days from snapshot time)
    df["days_to_expiry"] = (
        df["expiry"].dt.normalize() - asof.normalize()
    ).dt.days
    df = df[df["days_to_expiry"].between(
        cfg["MIN_EXPIRY_DAYS"], cfg["MAX_EXPIRY_DAYS"]
    )]
    n5 = len(df); _report("expiry window", n4, n5)

    # 6. year fraction
    df["T"] = df["days_to_expiry"] / 365.0

    # Drop expiries too thin for SVI to be meaningful.
    strikes_per_expiry = df.groupby("expiry")["strike"].nunique()
    thin = strikes_per_expiry[strikes_per_expiry < 3].index
    if len(thin) > 0:
        for exp in thin:
            logger.warning(
                "dropping expiry %s: fewer than 3 strikes after filtering "
                "(SVI needs 4-5+ points)", exp,
            )
        df = df[~df["expiry"].isin(thin)]
    _report("thin expiries", n5, len(df))

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Implied forward and discount factor (pricing-critical rate input)
# ---------------------------------------------------------------------------

def implied_forward(
    df_expiry: pd.DataFrame,
    spot: float,
    T: float,
    config: Optional[dict] = None,
) -> tuple[float, float, dict]:
    """Imply (F, DF) for one expiry from put-call parity.

    For European options C(K) - P(K) = DF * (F - K) exactly: the call-put
    mid spread is affine in strike with slope -DF and intercept DF*F. An
    OLS fit across near-the-money pairs recovers both. This replaces any
    external dividend yield input.

    Falls back to DF = exp(-r*T), F = spot*exp(r*T) (r from
    get_risk_free_rate) when fewer than FWD_MIN_PAIRS pairs exist or the
    fitted values fail sanity gates. The fallback ignores dividends and
    biases the slice -- it is flagged in diag, never silent.

    Returns (F, DF, diag) with diag keys:
        fallback (bool), n_pairs (int), resid_rms (float), resid_max (float)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    diag: dict = {"fallback": False, "n_pairs": 0,
                  "resid_rms": np.nan, "resid_max": np.nan}

    def _fallback(reason: str) -> tuple[float, float, dict]:
        r = get_risk_free_rate(cfg)
        F = spot * float(np.exp(r * T))
        DF = float(np.exp(-r * T))
        diag["fallback"] = True
        logger.warning(
            "implied_forward fallback (%s): using r=%.4f, F=%.2f, DF=%.6f. "
            "Dividends are ignored on this slice; IVs carry a documented bias.",
            reason, r, F, DF,
        )
        return F, DF, diag

    calls = df_expiry.loc[df_expiry["option_type"] == "call",
                          ["strike", "mid_price"]]
    puts = df_expiry.loc[df_expiry["option_type"] == "put",
                         ["strike", "mid_price"]]
    pairs = calls.merge(puts, on="strike", suffixes=("_c", "_p"))

    # Near-the-money pairs have the tightest spreads.
    band = cfg["FWD_BAND"]
    pairs = pairs[np.abs(pairs["strike"] / spot - 1.0) <= band]
    diag["n_pairs"] = len(pairs)

    if len(pairs) < cfg["FWD_MIN_PAIRS"]:
        return _fallback(f"{len(pairs)} parity pairs < {cfg['FWD_MIN_PAIRS']}")

    K = pairs["strike"].to_numpy(dtype=np.float64)
    y = (pairs["mid_price_c"] - pairs["mid_price_p"]).to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(K, y, 1)
    DF = float(-slope)
    if DF <= 0:
        return _fallback(f"non-positive fitted DF ({DF:.6f})")
    F = float(intercept / DF)

    # Sanity gates: corrupt quotes must trigger fallback, not a bad forward.
    if not (0.7 < DF <= 1.005):
        return _fallback(f"DF={DF:.6f} outside (0.7, 1.005]")
    if not (0.8 < F / spot < 1.2):
        return _fallback(f"F/spot={F / spot:.4f} outside (0.8, 1.2)")

    resid = y - DF * (F - K)
    diag["resid_rms"] = float(np.sqrt(np.mean(resid ** 2)))
    diag["resid_max"] = float(np.max(np.abs(resid)))
    worst = pairs.iloc[int(np.argmax(np.abs(resid)))]["strike"]
    if diag["resid_max"] > 0:
        logger.info(
            "implied_forward: n=%d, F=%.2f, DF=%.6f, resid_rms=%.4f, "
            "resid_max=%.4f (worst strike %.0f)",
            len(pairs), F, DF, diag["resid_rms"], diag["resid_max"], worst,
        )
    return F, DF, diag


def attach_forwards(
    df: pd.DataFrame,
    spot: float,
    config: Optional[dict] = None,
    otm_only: bool = True,
) -> pd.DataFrame:
    """Run implied_forward per expiry; attach F, DF, fwd_fallback and
    k = log(strike/F); then (by default) keep OTM rows only (puts with
    K <= F, calls with K >= F). ITM mids carry wide spreads and produce
    noisy IVs.

    `otm_only=False` attaches forwards WITHOUT filtering, yielding the
    both-sides frame that iv_solver.check_parity_iv needs (the parity
    diagnostic compares call vs. put IV at the same strike, which is
    impossible after the OTM filter keeps one side per strike). The
    pricing pipeline itself must use the default.

    Also prints the implied-financing vs. FRED diagnostic per expiry when a
    diagnostic rate is available.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    out_frames: list[pd.DataFrame] = []

    try:
        r_diag: Optional[float] = get_risk_free_rate(cfg)
    except Exception:  # diagnostic only; never block the pipeline on it
        r_diag = None

    for expiry, grp in df.groupby("expiry", sort=True):
        T = float(grp["T"].iloc[0])
        F, DF, diag = implied_forward(grp, spot, T, cfg)
        grp = grp.copy()
        grp["F"] = F
        grp["DF"] = DF
        grp["fwd_fallback"] = diag["fallback"]
        grp["k"] = np.log(grp["strike"].to_numpy(dtype=np.float64) / F)

        implied_financing = -np.log(DF) / T
        if r_diag is not None and not diag["fallback"]:
            print(
                f"{pd.Timestamp(expiry).date()}  T={T:.3f}  F={F:.2f}  "
                f"implied financing={implied_financing * 100:.2f}%  "
                f"FRED diag={r_diag * 100:.2f}%  "
                f"(gap absorbs dividends/borrow)"
            )

        if otm_only:
            is_call = grp["option_type"] == "call"
            otm = ((is_call & (grp["strike"] >= F))
                   | (~is_call & (grp["strike"] <= F)))
            grp = grp[otm]
        out_frames.append(grp)

    out = pd.concat(out_frames, ignore_index=True)

    # Correctness checks (spec Module 1)
    per_exp = out.groupby("expiry")[["F", "DF"]].nunique()
    assert (per_exp == 1).all().all(), "an expiry carries more than one (F, DF)"
    if otm_only:
        tol = 1e-12
        calls_bad = ((out["option_type"] == "call") & (out["k"] < -tol)).any()
        puts_bad = ((out["option_type"] == "put") & (out["k"] > tol)).any()
        assert not calls_bad and not puts_bad, "OTM filter sign invariant violated"

    return out


# ---------------------------------------------------------------------------
# CLI: python -m src.data --snapshot [--ticker ^SPX]
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Module 1: chain acquisition")
    parser.add_argument("--snapshot", action="store_true",
                        help="pull a fresh chain and write a snapshot CSV")
    parser.add_argument("--ticker", default=DEFAULT_CONFIG["TICKER"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.snapshot:
        asof = pd.Timestamp.now()
        spot = fetch_spot(args.ticker)
        chain = fetch_chain(args.ticker)
        path = save_snapshot(chain, spot, args.ticker, asof=asof)
        print(f"snapshot: {path}  spot={spot:.2f}  asof={asof.isoformat()}")
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()