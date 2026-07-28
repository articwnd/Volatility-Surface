"""
arbitrage.py -- Module 3: pre-fit no-arbitrage checks.

Pipeline position:
    compute_iv_surface output (OTM rows with iv, F, DF, k, T)
        -> run_arbitrage_checks
        -> [Module 4: svi]  (flagged strikes/slices excluded from the fit)

Design notes (see PROJECT_SPEC.md r2):
- Butterfly: call prices must be convex in strike. SPX strike spacing is
  irregular (5/10/25/50 point increments), so the check uses the
  spacing-weighted condition on each adjacent triple; the unweighted
  butterfly C(K1) - 2C(K2) + C(K3) is only valid for equal spacing and
  false-flags real chains (pinned by test_equal_spacing_formula_would_misfire).
- Only OTM quotes survive Module 1, so the call curve below the forward is
  synthesized from puts via parity C(K) = P(K) + DF*(F - K). The shared
  parity-implied (F, DF) makes this exact for parity-consistent quotes.
- Calendar: total variance must be non-decreasing in T at fixed
  log-forward-moneyness k (NOT fixed strike -- forwards differ by expiry).
  This pre-fit check covers a band around k = 0 only; global calendar
  monotonicity on the fitted surface is enforced by linear-in-total-
  variance time interpolation in Module 5. Documented limitation, not a
  hidden one.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ARB_CONFIG: dict = {
    "BUTTERFLY_EPS": 1e-8,   # price tolerance for float noise in convexity
    "CALENDAR_K_BAND": 0.02,  # |k| band defining "ATM-forward" for the check
    "CALENDAR_EPS": 0.0,      # strict: any decrease in total variance flags
    "FLAGS_CSV": os.path.join("outputs", "arbitrage_flags.csv"),
}

BUTTERFLY_COLUMNS = ["expiry", "K1", "K2", "K3", "C1", "C2", "C3", "B_value"]
CALENDAR_COLUMNS = ["k_bucket", "expiry_short", "expiry_long",
                    "T_short", "T_long", "w_short", "w_long", "k_short",
                    "k_long"]


# ---------------------------------------------------------------------------
# Call-curve construction from OTM quotes
# ---------------------------------------------------------------------------

def synthesize_call_curve(df_slice: pd.DataFrame) -> pd.DataFrame:
    """Build a single call-price curve C(K) for one expiry from OTM quotes.

    Actual call mids are used for K >= F; put mids are converted via
    parity C(K) = P(K) + DF*(F - K) for K < F. At a strike carrying both
    (possible exactly at K = F), the actual call is preferred. Output is
    sorted by strike with one row per strike.

    Requires columns: strike, mid_price, option_type, F, DF.
    """
    required = {"strike", "mid_price", "option_type", "F", "DF"}
    missing = required - set(df_slice.columns)
    assert not missing, f"synthesize_call_curve missing columns: {missing}"
    assert df_slice["F"].nunique() == 1 and df_slice["DF"].nunique() == 1, (
        "slice carries more than one (F, DF); pass one expiry at a time"
    )

    F = float(df_slice["F"].iloc[0])
    DF = float(df_slice["DF"].iloc[0])

    out = df_slice[["strike", "mid_price", "option_type"]].copy()
    is_put = out["option_type"] == "put"
    out["call_price"] = out["mid_price"]
    out.loc[is_put, "call_price"] = (
        out.loc[is_put, "mid_price"] + DF * (F - out.loc[is_put, "strike"])
    )

    # Prefer the actual call where both sides exist at one strike.
    out = (
        out.sort_values(["strike", "option_type"])  # 'call' < 'put'
        .drop_duplicates(subset="strike", keep="first")
        .reset_index(drop=True)
    )

    # Synthesized calls must be non-negative up to quote noise; a large
    # negative value means (F, DF) or the quote is bad.
    bad = out["call_price"] < -1e-6
    if bad.any():
        logger.warning(
            "synthesize_call_curve: %d negative synthesized call price(s); "
            "worst %.6f -- check quotes / implied forward",
            int(bad.sum()), float(out.loc[bad, "call_price"].min()),
        )
    return out[["strike", "call_price"]]


# ---------------------------------------------------------------------------
# Butterfly (strike) arbitrage
# ---------------------------------------------------------------------------

def check_butterfly(df_slice: pd.DataFrame, expiry,
                    config: Optional[dict] = None) -> pd.DataFrame:
    """Spacing-weighted convexity check for one expiry slice.

    For each adjacent strike triple K1 < K2 < K3:

        w = (K3 - K2) / (K3 - K1)
        B_value = w*C(K1) + (1 - w)*C(K3) - C(K2)

    B_value < -eps flags a violation (the middle price sits above the
    chord => non-convex => negative implied density). The unweighted form
    C(K1) - 2C(K2) + C(K3) is NOT equivalent under unequal spacing and
    must not be used.

    Returns a DataFrame (BUTTERFLY_COLUMNS) of violating triples; empty
    means clean.
    """
    cfg = {**DEFAULT_ARB_CONFIG, **(config or {})}
    curve = synthesize_call_curve(df_slice)
    K = curve["strike"].to_numpy(dtype=np.float64)
    C = curve["call_price"].to_numpy(dtype=np.float64)

    rows = []
    if len(K) >= 3:
        K1, K2, K3 = K[:-2], K[1:-1], K[2:]
        C1, C2, C3 = C[:-2], C[1:-1], C[2:]
        w = (K3 - K2) / (K3 - K1)
        B = w * C1 + (1.0 - w) * C3 - C2
        viol = B < -cfg["BUTTERFLY_EPS"]
        for i in np.flatnonzero(viol):
            rows.append({
                "expiry": expiry, "K1": K1[i], "K2": K2[i], "K3": K3[i],
                "C1": C1[i], "C2": C2[i], "C3": C3[i],
                "B_value": float(B[i]),
            })
    return pd.DataFrame(rows, columns=BUTTERFLY_COLUMNS)


# ---------------------------------------------------------------------------
# Calendar (time) arbitrage
# ---------------------------------------------------------------------------

def check_calendar(df: pd.DataFrame,
                   config: Optional[dict] = None) -> pd.DataFrame:
    """Calendar check at fixed log-forward-moneyness.

    For each expiry, the representative ATM-forward total variance is
    taken from the row with the smallest |k| inside the band
    |k| <= CALENDAR_K_BAND (expiries with no row in the band are skipped
    with a warning). Sorted by T, every ordered pair with
    w(T_short) > w(T_long) + eps is flagged.

    All pairs are reported (not only adjacent ones) so a single bad slice
    shows up against every longer maturity it violates, which makes the
    offender obvious in the diagnostics.

    Requires columns: expiry, T, k, iv.
    Returns a DataFrame (CALENDAR_COLUMNS); empty means clean.
    """
    cfg = {**DEFAULT_ARB_CONFIG, **(config or {})}
    required = {"expiry", "T", "k", "iv"}
    missing = required - set(df.columns)
    assert not missing, f"check_calendar missing columns: {missing}"

    band = cfg["CALENDAR_K_BAND"]
    reps = []
    for expiry, grp in df.groupby("expiry", sort=True):
        in_band = grp[np.abs(grp["k"]) <= band]
        if in_band.empty:
            logger.warning(
                "check_calendar: expiry %s has no quote with |k| <= %g; "
                "skipped (band-limited check, see spec limitation)",
                expiry, band,
            )
            continue
        row = in_band.loc[in_band["k"].abs().idxmin()]
        reps.append({
            "expiry": expiry, "T": float(row["T"]), "k": float(row["k"]),
            "w": float(row["iv"] ** 2 * row["T"]),
        })

    reps_df = pd.DataFrame(reps).sort_values("T").reset_index(drop=True)
    rows = []
    for i in range(len(reps_df)):
        for j in range(i + 1, len(reps_df)):
            short, long_ = reps_df.iloc[i], reps_df.iloc[j]
            if short["w"] > long_["w"] + cfg["CALENDAR_EPS"]:
                rows.append({
                    "k_bucket": f"|k|<={band}",
                    "expiry_short": short["expiry"],
                    "expiry_long": long_["expiry"],
                    "T_short": short["T"], "T_long": long_["T"],
                    "w_short": short["w"], "w_long": long_["w"],
                    "k_short": short["k"], "k_long": long_["k"],
                })
    return pd.DataFrame(rows, columns=CALENDAR_COLUMNS)


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def run_arbitrage_checks(df: pd.DataFrame,
                         config: Optional[dict] = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both checks across all slices; write the combined flags CSV;
    print a summary. Returns (butterfly_flags, calendar_flags)."""
    cfg = {**DEFAULT_ARB_CONFIG, **(config or {})}

    bf_frames = [
        check_butterfly(grp, expiry, cfg)
        for expiry, grp in df.groupby("expiry", sort=True)
    ]
    butterfly_flags = (
        pd.concat(bf_frames, ignore_index=True)
        if bf_frames else pd.DataFrame(columns=BUTTERFLY_COLUMNS)
    )
    calendar_flags = check_calendar(df, cfg)

    csv_path = cfg["FLAGS_CSV"]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    combined = pd.concat(
        [butterfly_flags.assign(check="butterfly"),
         calendar_flags.assign(check="calendar")],
        ignore_index=True, sort=False,
    )
    # 'check' first for readability; union of both column sets follows.
    cols = ["check"] + [c for c in combined.columns if c != "check"]
    combined[cols].to_csv(csv_path, index=False)

    n_exp_bf = butterfly_flags["expiry"].nunique() if not butterfly_flags.empty else 0
    print(
        f"run_arbitrage_checks: {len(butterfly_flags)} butterfly violations "
        f"across {n_exp_bf} expiries, {len(calendar_flags)} calendar "
        f"violations -> {csv_path}"
    )
    return butterfly_flags, calendar_flags


def exclude_flagged(df: pd.DataFrame, butterfly_flags: pd.DataFrame,
                    calendar_flags: pd.DataFrame) -> pd.DataFrame:
    """Drop flagged middle strikes (butterfly) and short legs of flagged
    calendar pairs from the frame handed to the SVI fit. The excluded
    rows remain in the flags CSV -- exclusion is from the FIT only, per
    spec ('flagged and excluded from the parametric fit').

    Butterfly: the middle strike K2 is the one priced above the chord, so
    it is the strike removed per flagged triple.
    Calendar: the SHORT expiry of a violating pair carries the excess
    variance; its ATM region is suspect, so the whole short slice is
    dropped only if it violates against 2+ longer maturities (one pair
    may be the long slice's fault); otherwise it is kept and merely
    logged. This is a pragmatic rule, stated here so it can be argued
    with rather than discovered.
    """
    out = df
    if not butterfly_flags.empty:
        bad = set(zip(butterfly_flags["expiry"], butterfly_flags["K2"]))
        mask = [
            (exp, k) not in bad
            for exp, k in zip(out["expiry"], out["strike"])
        ]
        n_drop = len(out) - int(np.sum(mask))
        out = out[np.array(mask)]
        logger.info("exclude_flagged: dropped %d butterfly-flagged strikes",
                    n_drop)
    if not calendar_flags.empty:
        counts = calendar_flags["expiry_short"].value_counts()
        drop_expiries = set(counts[counts >= 2].index)
        for exp in sorted(drop_expiries):
            logger.warning(
                "exclude_flagged: dropping expiry %s (calendar-violates "
                "against %d longer maturities)", exp, counts[exp],
            )
        singles = set(counts[counts < 2].index)
        for exp in sorted(singles):
            logger.warning(
                "exclude_flagged: expiry %s calendar-flagged once; kept "
                "(single pair may be the long slice's fault)", exp,
            )
        out = out[~out["expiry"].isin(drop_expiries)]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI: python -m src.arbitrage --from-snapshot data/snapshots/SPX_....csv
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    from src.data import attach_forwards, clean_chain, load_snapshot
    from src.iv_solver import compute_iv_surface

    parser = argparse.ArgumentParser(
        description="Module 3: arbitrage checks only (no fit)")
    parser.add_argument("--from-snapshot", required=True,
                        help="snapshot CSV written by src.data --snapshot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    chain, spot, asof = load_snapshot(args.from_snapshot)
    clean = clean_chain(chain, spot, asof=asof)
    surf = compute_iv_surface(attach_forwards(clean, spot))
    run_arbitrage_checks(surf)


if __name__ == "__main__":
    _main()