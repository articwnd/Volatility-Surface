"""
arbitrage.py -- Module 3: pre-fit no-arbitrage checks.

Pipeline position:
    compute_iv_surface output (OTM rows with iv, F, DF, k, T)
        -> run_arbitrage_checks
        -> [Module 4: svi]  (flagged strikes/slices excluded from the fit)

Design notes (see PROJECT_SPEC.md r2):
- Butterfly is TWO-TIER (r3). Tier 1 (diagnostic): strict convexity of
  MID call prices -- flags data-quality issues but, near tight strike
  spacing, mostly measures quote noise (the true chord gap is the same
  order as the bid-ask). Tier 2 (exclusion trigger): EXECUTABLE
  arbitrage only -- the long butterfly must be enterable at a credit
  against the touch, wings bought at the ask and body sold at the bid.
  A mid-price convexity breach inside the spread cannot be monetized and
  must not gut the fit input. Since ask >= mid >= bid implies
  B_exec >= B_mid, every executable violation is also a mid violation:
  one pass computes both.
- Only OTM quotes survive Module 1, so the call curve below the forward is
  synthesized from puts via parity C(K) = P(K) + DF*(F - K), applied to
  bid, mid, and ask alike (the parity offset is a cash-and-forward
  position; the forward leg is implied from mids and carries no modeled
  spread -- documented simplification).
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
    "BUTTERFLY_EPS": 1e-8,   # float-noise tolerance for the EXECUTABLE tier
    # Tier-1 (mid) violations smaller than half a tick are quantization,
    # not information (measured r5: 37% of live tier-1 flags were within
    # one $0.05 tick of zero). Reported flags require B_value below this.
    "REPORT_FLOOR": 0.025,
    "CALENDAR_K_BAND": 0.02,  # |k| band defining "ATM-forward" for the check
    "CALENDAR_EPS": 0.0,      # strict: any decrease in total variance flags
    "VERTICAL_MAX_DROPS": 50,  # cap on iterative offender removal per slice
    "FLAGS_CSV": os.path.join("outputs", "arbitrage_flags.csv"),
}

BUTTERFLY_COLUMNS = ["expiry", "K1", "K2", "K3", "C1", "C2", "C3",
                     "B_value", "B_exec", "executable"]
CALENDAR_COLUMNS = ["k_bucket", "expiry_short", "expiry_long",
                    "T_short", "T_long", "w_short", "w_long", "k_short",
                    "k_long"]
VERTICAL_COLUMNS = ["expiry", "K_low", "K_high", "spread_mid", "max_payoff",
                    "viol_type", "exec_amount", "executable", "offender"]


# ---------------------------------------------------------------------------
# Call-curve construction from OTM quotes
# ---------------------------------------------------------------------------

def synthesize_call_curve(df_slice: pd.DataFrame) -> pd.DataFrame:
    """Build call-price curves (bid/mid/ask) for one expiry from OTM
    quotes.

    Actual call quotes are used for K >= F; put quotes are converted via
    parity for K < F. The parity offset DF*(F - K) is a cash-and-forward
    position added to bid, mid, and ask alike: buying the synthetic call
    means buying the put (pay the ask), selling it means selling the put
    (receive the bid). The forward leg is implied from mids and carries
    no modeled spread -- a documented simplification that makes the
    executable tier slightly AGGRESSIVE (real synthetic-leg costs are
    higher), i.e. it errs toward flagging.

    At a strike carrying both sides (possible exactly at K = F), the
    actual call is preferred. Output is sorted by strike, one row per
    strike, columns: strike, call_price (mid), call_bid, call_ask.

    If bid/ask columns are absent, both collapse to the mid and the
    executable tier degenerates to the strict tier -- logged loudly,
    because exclusion then reverts to noise-sensitive mid behavior.
    """
    required = {"strike", "mid_price", "option_type", "F", "DF"}
    missing = required - set(df_slice.columns)
    assert not missing, f"synthesize_call_curve missing columns: {missing}"
    assert df_slice["F"].nunique() == 1 and df_slice["DF"].nunique() == 1, (
        "slice carries more than one (F, DF); pass one expiry at a time"
    )

    F = float(df_slice["F"].iloc[0])
    DF = float(df_slice["DF"].iloc[0])

    has_quotes = {"bid", "ask"} <= set(df_slice.columns)
    if not has_quotes:
        logger.warning(
            "synthesize_call_curve: no bid/ask columns; executable tier "
            "degenerates to the strict mid tier for this slice"
        )

    cols = ["strike", "mid_price", "option_type"] + (
        ["bid", "ask"] if has_quotes else [])
    out = df_slice[cols].copy()
    if not has_quotes:
        out["bid"] = out["mid_price"]
        out["ask"] = out["mid_price"]

    offset = pd.Series(0.0, index=out.index)
    is_put = out["option_type"] == "put"
    offset[is_put] = DF * (F - out.loc[is_put, "strike"])
    out["call_price"] = out["mid_price"] + offset
    out["call_bid"] = out["bid"] + offset
    out["call_ask"] = out["ask"] + offset

    # Prefer the actual call where both sides exist at one strike.
    out = (
        out.sort_values(["strike", "option_type"])  # 'call' < 'put'
        .drop_duplicates(subset="strike", keep="first")
        .reset_index(drop=True)
    )

    # Synthesized mids must be non-negative up to quote noise; a large
    # negative value means (F, DF) or the quote is bad.
    bad = out["call_price"] < -1e-6
    if bad.any():
        logger.warning(
            "synthesize_call_curve: %d negative synthesized call price(s); "
            "worst %.6f -- check quotes / implied forward",
            int(bad.sum()), float(out.loc[bad, "call_price"].min()),
        )
    return out[["strike", "call_price", "call_bid", "call_ask"]]


# ---------------------------------------------------------------------------
# Vertical spread (monotonicity / slope) static bounds  (added r5)
# ---------------------------------------------------------------------------

def check_vertical(df_slice: pd.DataFrame, expiry,
                   config: Optional[dict] = None
                   ) -> tuple[pd.DataFrame, list[float]]:
    """Vertical-spread static-bound check for one expiry slice, with
    iterative attribution of executable violations to specific strikes.

    A call spread C(K_low) - C(K_high), K_low < K_high, must lie in
    [0, DF*(K_high - K_low)]:

        rising: C(K_high) > C(K_low)          -- calls rising in strike
        steep:  C(K_low) - C(K_high) > DF*dK  -- spread above max payoff

    Tier 1 (diagnostic, mids, REPORT_FLOOR de-noised) reports both.
    Tier 2 (executable, at the touch):
        rising: C_bid(K_high) - C_ask(K_low) > eps   [buy low, sell high,
                enter at a credit for a non-negative payoff]
        steep:  C_bid(K_low) - C_ask(K_high) > DF*dK + eps  [sell the
                spread for more than its discounted max payoff]

    Why this exists: one junk quote (stale/crossed line at a non-standard
    strike) violates against BOTH neighbors; the butterfly check smears
    it across up to three triples, while the vertical check pinpoints it
    (measured r5: the $14-$72 "butterflies" on live data were single
    impossible quotes with adjacent call slopes of -6.9 then +7.8).

    Attribution: iteratively, the strike involved in the most executable
    pairs (ties: largest summed magnitude) is recorded as an offender
    and removed, then pairs are recomputed; capped at VERTICAL_MAX_DROPS.
    A lone junk quote is involved in 2 pairs and is removed first, so
    its healthy neighbors survive.

    Returns (flags DataFrame [VERTICAL_COLUMNS], offender strikes list).
    The `offender` column marks rows whose violation was attributed to
    that strike during removal (NaN on purely diagnostic rows).
    """
    cfg = {**DEFAULT_ARB_CONFIG, **(config or {})}
    eps = cfg["BUTTERFLY_EPS"]
    floor = cfg["REPORT_FLOOR"]
    DF = float(df_slice["DF"].iloc[0])
    curve = synthesize_call_curve(df_slice)

    def _pairs(c: pd.DataFrame) -> pd.DataFrame:
        K = c["strike"].to_numpy(dtype=np.float64)
        Cm = c["call_price"].to_numpy(dtype=np.float64)
        Cb = c["call_bid"].to_numpy(dtype=np.float64)
        Ca = c["call_ask"].to_numpy(dtype=np.float64)
        dK = K[1:] - K[:-1]
        return pd.DataFrame({
            "K_low": K[:-1], "K_high": K[1:],
            "spread_mid": Cm[:-1] - Cm[1:], "max_payoff": DF * dK,
            "exec_rising": Cb[1:] - Ca[:-1],
            "exec_steep": (Cb[:-1] - Ca[1:]) - DF * dK,
        })

    # Iterative executable attribution.
    offenders: list[float] = []
    work = curve.copy()
    for _ in range(cfg["VERTICAL_MAX_DROPS"]):
        if len(work) < 2:
            break
        p = _pairs(work)
        ex = p[(p["exec_rising"] > eps) | (p["exec_steep"] > eps)]
        if ex.empty:
            break
        strikes = pd.concat([ex["K_low"], ex["K_high"]])
        counts = strikes.value_counts()
        top = counts[counts == counts.max()].index
        if len(top) > 1:
            mag = {}
            for s in top:
                rows = ex[(ex["K_low"] == s) | (ex["K_high"] == s)]
                mag[s] = float(np.maximum(rows["exec_rising"], 0).sum()
                               + np.maximum(rows["exec_steep"], 0).sum())
            worst = max(mag, key=mag.get)
        else:
            worst = top[0]
        offenders.append(float(worst))
        work = work[work["strike"] != worst]

    if offenders:
        logger.warning(
            "check_vertical: expiry %s -- %d strike(s) violate vertical "
            "static bounds at the touch and were attributed as junk "
            "quotes: %s", expiry, len(offenders), sorted(offenders),
        )

    # Diagnostic tier on the ORIGINAL curve, de-noised by the floor.
    p0 = _pairs(curve)
    rows = []
    for _, r in p0.iterrows():
        rising_mid = -r["spread_mid"]                    # >0 => rising mids
        steep_mid = r["spread_mid"] - r["max_payoff"]    # >0 => too steep
        if rising_mid <= floor and steep_mid <= floor:
            continue
        viol_type = "rising" if rising_mid > steep_mid else "steep"
        exec_amt = float(max(r["exec_rising"], r["exec_steep"]))
        executable = bool(exec_amt > eps)
        off = np.nan
        if executable:
            for s in (r["K_low"], r["K_high"]):
                if s in offenders:
                    off = float(s)
                    break
        rows.append({
            "expiry": expiry, "K_low": r["K_low"], "K_high": r["K_high"],
            "spread_mid": float(r["spread_mid"]),
            "max_payoff": float(r["max_payoff"]),
            "viol_type": viol_type, "exec_amount": exec_amt,
            "executable": executable, "offender": off,
        })
    return pd.DataFrame(rows, columns=VERTICAL_COLUMNS), offenders


# ---------------------------------------------------------------------------
# Butterfly (strike) arbitrage
# ---------------------------------------------------------------------------

def check_butterfly(df_slice: pd.DataFrame, expiry,
                    config: Optional[dict] = None) -> pd.DataFrame:
    """Two-tier spacing-weighted convexity check for one expiry slice.

    For each adjacent strike triple K1 < K2 < K3 with w = (K3-K2)/(K3-K1):

        B_value = w*C_mid(K1) + (1-w)*C_mid(K3) - C_mid(K2)   [tier 1]
        B_exec  = w*C_ask(K1) + (1-w)*C_ask(K3) - C_bid(K2)   [tier 2]

    Tier 1 (B_value < -eps) is a DIAGNOSTIC: non-convex mids imply
    negative density at mid prices, but near tight spacing the chord gap
    is the same order as quote noise, so most tier-1 flags inside the
    spread are unmonetizable. Tier 2 (B_exec < -eps) is the EXCLUSION
    trigger: the long fly (buy wings at ask, sell body at bid) enters at
    a credit for a non-negative payoff -- executable arbitrage against
    the touch. Since ask >= mid >= bid, B_exec >= B_value, so executable
    violations are a subset of tier-1 flags; the returned frame contains
    all tier-1 rows with an `executable` boolean marking tier 2.

    The unweighted form C(K1) - 2C(K2) + C(K3) is NOT equivalent under
    unequal spacing and must not be used.

    Returns a DataFrame (BUTTERFLY_COLUMNS); empty means clean at tier 1
    (and therefore at tier 2).
    """
    cfg = {**DEFAULT_ARB_CONFIG, **(config or {})}
    curve = synthesize_call_curve(df_slice)
    K = curve["strike"].to_numpy(dtype=np.float64)
    C = curve["call_price"].to_numpy(dtype=np.float64)
    Cb = curve["call_bid"].to_numpy(dtype=np.float64)
    Ca = curve["call_ask"].to_numpy(dtype=np.float64)

    rows = []
    if len(K) >= 3:
        K1, K2, K3 = K[:-2], K[1:-1], K[2:]
        C1, C2, C3 = C[:-2], C[1:-1], C[2:]
        w = (K3 - K2) / (K3 - K1)
        B = w * C1 + (1.0 - w) * C3 - C2
        B_exec = w * Ca[:-2] + (1.0 - w) * Ca[2:] - Cb[1:-1]
        eps = cfg["BUTTERFLY_EPS"]
        # Tier-1 reporting de-noised to the floor; the executable tier
        # keeps the strict float-noise eps.
        viol = B < -max(cfg["REPORT_FLOOR"], eps)
        for i in np.flatnonzero(viol):
            rows.append({
                "expiry": expiry, "K1": K1[i], "K2": K2[i], "K3": K3[i],
                "C1": C1[i], "C2": C2[i], "C3": C3[i],
                "B_value": float(B[i]), "B_exec": float(B_exec[i]),
                "executable": bool(B_exec[i] < -eps),
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

    reps_df = pd.DataFrame(reps)
    if reps_df.empty:
        # No expiry had a quote inside the band (e.g. a wings-only
        # slice): nothing to compare. Return an empty, well-formed
        # frame rather than crashing on sort_values('T').
        return pd.DataFrame(columns=CALENDAR_COLUMNS)
    reps_df = reps_df.sort_values("T").reset_index(drop=True)
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
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all checks across all slices; write the combined flags CSV;
    print a summary. Returns (vertical_flags, butterfly_flags,
    calendar_flags).

    Ordering matters (r5): the vertical check runs FIRST and its
    executable offenders (junk quotes) are removed before butterfly and
    calendar are computed -- one impossible quote otherwise smears into
    up to three phantom butterfly triples and pollutes the ATM calendar
    representative. Offender rows remain in the flags CSV; removal here
    is from the downstream CHECK INPUT only (exclude_flagged applies the
    same removal to the fit input).
    """
    cfg = {**DEFAULT_ARB_CONFIG, **(config or {})}

    vert_frames: list[pd.DataFrame] = []
    offender_pairs: set[tuple] = set()
    for expiry, grp in df.groupby("expiry", sort=True):
        flags, offenders = check_vertical(grp, expiry, cfg)
        vert_frames.append(flags)
        offender_pairs.update((expiry, s) for s in offenders)
    vertical_flags = (
        pd.concat(vert_frames, ignore_index=True)
        if vert_frames else pd.DataFrame(columns=VERTICAL_COLUMNS)
    )

    if offender_pairs:
        keep = [
            (exp, k) not in offender_pairs
            for exp, k in zip(df["expiry"], df["strike"])
        ]
        df_checked = df[np.array(keep)]
    else:
        df_checked = df

    bf_frames = [
        check_butterfly(grp, expiry, cfg)
        for expiry, grp in df_checked.groupby("expiry", sort=True)
    ]
    butterfly_flags = (
        pd.concat(bf_frames, ignore_index=True)
        if bf_frames else pd.DataFrame(columns=BUTTERFLY_COLUMNS)
    )
    calendar_flags = check_calendar(df_checked, cfg)

    csv_path = cfg["FLAGS_CSV"]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    combined = pd.concat(
        [vertical_flags.assign(check="vertical"),
         butterfly_flags.assign(check="butterfly"),
         calendar_flags.assign(check="calendar")],
        ignore_index=True, sort=False,
    )
    cols = ["check"] + [c for c in combined.columns if c != "check"]
    combined[cols].to_csv(csv_path, index=False)

    n_off = len(offender_pairs)
    n_exec_bf = int(butterfly_flags["executable"].sum()) if not butterfly_flags.empty else 0
    print(
        f"run_arbitrage_checks: {len(vertical_flags)} vertical flags "
        f"({n_off} junk-quote strikes attributed), "
        f"{len(butterfly_flags)} butterfly mid-flags "
        f"({n_exec_bf} executable), "
        f"{len(calendar_flags)} calendar violations -> {csv_path}"
    )
    return vertical_flags, butterfly_flags, calendar_flags


def exclude_flagged(df: pd.DataFrame, vertical_flags: pd.DataFrame,
                    butterfly_flags: pd.DataFrame,
                    calendar_flags: pd.DataFrame) -> pd.DataFrame:
    """Drop vertical junk-quote offenders, EXECUTABLE butterfly middle
    strikes, and repeat-offender
    calendar slices from the frame handed to the SVI fit. All flagged
    rows (executable or not) remain in the flags CSV -- exclusion is
    from the FIT only, per spec.

    Butterfly (r3, two-tier): only triples with `executable=True` (fly
    enterable at a credit against the touch) trigger exclusion of the
    middle strike K2. Mid-only flags inside the bid-ask are quote noise,
    not arbitrage, and excluding them would gut the densest, most liquid
    region of real chains (measured: 22% of rows on a +/-$0.40-noise
    synthetic SPX chain).
    Calendar: the SHORT expiry of a violating pair carries the excess
    variance; the whole short slice is dropped only if it violates
    against 2+ longer maturities (one pair may be the long slice's
    fault); otherwise it is kept and merely logged. This is a pragmatic
    rule, stated here so it can be argued with rather than discovered.
    """
    out = df
    if not vertical_flags.empty:
        off = vertical_flags.dropna(subset=["offender"])
        if not off.empty:
            bad = set(zip(off["expiry"], off["offender"]))
            mask = np.array([
                (exp, k) not in bad
                for exp, k in zip(out["expiry"], out["strike"])
            ])
            n_drop = int((~mask).sum())
            out = out[mask]
            logger.info(
                "exclude_flagged: dropped %d junk-quote strike(s) failing "
                "vertical static bounds at the touch", n_drop,
            )
    if not butterfly_flags.empty:
        exec_flags = butterfly_flags[butterfly_flags["executable"]]
        n_noise = len(butterfly_flags) - len(exec_flags)
        if n_noise:
            logger.info(
                "exclude_flagged: %d mid-only butterfly flag(s) inside the "
                "spread retained in the fit (diagnostic tier, not "
                "executable)", n_noise,
            )
        if not exec_flags.empty:
            bad = set(zip(exec_flags["expiry"], exec_flags["K2"]))
            mask = [
                (exp, k) not in bad
                for exp, k in zip(out["expiry"], out["strike"])
            ]
            n_drop = len(out) - int(np.sum(mask))
            out = out[np.array(mask)]
            logger.info(
                "exclude_flagged: dropped %d strike(s) with EXECUTABLE "
                "butterfly arbitrage", n_drop,
            )
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
    run_arbitrage_checks(surf)  # returns (vertical, butterfly, calendar)


if __name__ == "__main__":
    _main()