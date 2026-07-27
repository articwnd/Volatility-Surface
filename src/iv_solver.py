"""
iv_solver.py -- Module 2: implied volatility inversion under Black-76.

Pipeline position:
    attach_forwards output (F, DF, k on every row)
        -> compute_iv_surface
        -> [Module 3: arbitrage]

Design notes (see PROJECT_SPEC.md r2):
- All pricing is Black-76 on the per-expiry parity-implied forward. There
  is no spot and no external rate anywhere in this module; F and DF ride
  on the rows from Module 1.
- Newton-Raphson is seeded with the Manaster-Koehler starting value
  (floored at 0.1 at the money) and checks convergence BEFORE updating,
  so the returned sigma is the one whose price met tolerance. Brent's
  method is the guaranteed-convergence fallback.
- The vega floor (default 1e-4 dollar vega) is a fallback trigger, not a
  division guard: it hands off to Brent while the NR iterate is still
  sane. A floor near machine epsilon never fires before the step size
  has already exploded.

Consistent with the companion C++ engine (ope::implied_vol): same seeding,
same two-tier philosophy (status-signaling core, convenience wrapper).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)

DEFAULT_SOLVER_CONFIG: dict = {
    "NR_TOL": 1e-8,        # absolute price convergence tolerance ($)
    "NR_MAX_ITER": 100,    # max NR iterations before Brent fallback
    "VEGA_FLOOR": 1e-4,    # dollar-vega threshold triggering Brent fallback
    "MK_SEED_FLOOR": 0.1,  # min NR seed (raw MK formula returns 0 at K = F)
    "IV_MAX": 5.0,         # upper sanity bound / Brent bracket ceiling
    "IV_MIN": 1e-6,        # Brent bracket floor
}

_BRENT = "brent"
_NR = "nr"
_FAIL = "nan"


# ---------------------------------------------------------------------------
# Black-76 pricing and vega
# ---------------------------------------------------------------------------

def _validate_inputs(F: float, K: float, DF: float, T: float,
                     sigma: float) -> None:
    """Shared input guard. isfinite catches inf and nan that a bare
    positivity comparison would pass (nan > 0 is False, but inf > 0 is
    True); the explicit check keeps failure modes loud and uniform."""
    vals = (F, K, DF, T, sigma)
    if not all(np.isfinite(v) for v in vals):
        raise ValueError(f"non-finite input to Black-76: "
                         f"F={F}, K={K}, DF={DF}, T={T}, sigma={sigma}")
    if F <= 0 or K <= 0 or DF <= 0 or T <= 0 or sigma <= 0:
        raise ValueError(f"non-positive input to Black-76: "
                         f"F={F}, K={K}, DF={DF}, T={T}, sigma={sigma}")


def b76_price(F: float, K: float, DF: float, T: float, sigma: float,
              option_type: str) -> float:
    """Black-76 price of a European option on the forward.

        d1   = (log(F/K) + 0.5*sigma^2*T) / (sigma*sqrt(T))
        d2   = d1 - sigma*sqrt(T)
        call = DF * (F*N(d1) - K*N(d2))
        put  = call - DF*(F - K)        [put-call parity]

    The put comes from parity rather than a separate formula: internal
    consistency is then exact by construction, which the parity tests and
    the synthesized-call butterfly path (Module 3) rely on.
    """
    _validate_inputs(F, K, DF, T, sigma)
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', "
                         f"got {option_type!r}")
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    call = DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    if option_type == "call":
        return float(call)
    return float(call - DF * (F - K))


def b76_vega(F: float, K: float, DF: float, T: float, sigma: float) -> float:
    """Black-76 vega (dPrice/dSigma), shared by calls and puts:

        vega = DF * F * phi(d1) * sqrt(T)

    Always non-negative (DF, F, phi, sqrt(T) all are).
    """
    _validate_inputs(F, K, DF, T, sigma)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return float(DF * F * norm.pdf(d1) * sqrt_T)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def mk_seed(F: float, K: float, T: float,
            floor: float = DEFAULT_SOLVER_CONFIG["MK_SEED_FLOOR"]) -> float:
    """Manaster-Koehler (1982) Newton-Raphson starting value, in forward
    terms:

        sigma_0 = sqrt(2 * |log(F/K)| / T)

    floored at `floor` (the raw formula returns 0 at K = F, and very small
    seeds waste iterations). Starting NR from the MK seed gives monotone
    convergence for vanilla prices. Mirrors ope::implied_vol seeding.
    """
    if not (np.isfinite(F) and np.isfinite(K) and np.isfinite(T)):
        raise ValueError(f"non-finite input to mk_seed: F={F}, K={K}, T={T}")
    if F <= 0 or K <= 0 or T <= 0:
        raise ValueError(f"non-positive input to mk_seed: F={F}, K={K}, T={T}")
    return float(max(np.sqrt(2.0 * abs(np.log(F / K)) / T), floor))


def implied_vol_nr(market_price: float, F: float, K: float, DF: float,
                   T: float, option_type: str,
                   config: Optional[dict] = None) -> Optional[float]:
    """Newton-Raphson IV solver. Returns sigma on convergence, or None to
    signal that the Brent fallback is needed (low vega, divergence, or
    iteration cap). Never raises on solver failure; input validation
    errors from b76_price still propagate.

    Ordering matters: convergence is checked BEFORE the update, so the
    returned sigma is exactly the one whose price met tolerance. (r1
    checked after updating, returning a post-update sigma against a
    pre-update price error -- an off-by-one; see test_nr_convergence_order.)
    """
    cfg = {**DEFAULT_SOLVER_CONFIG, **(config or {})}
    sigma = mk_seed(F, K, T, cfg["MK_SEED_FLOOR"])

    for _ in range(cfg["NR_MAX_ITER"]):
        price = b76_price(F, K, DF, T, sigma, option_type)
        if abs(price - market_price) < cfg["NR_TOL"]:
            return sigma
        vega = b76_vega(F, K, DF, T, sigma)
        if vega < cfg["VEGA_FLOOR"]:
            return None  # hand off to Brent while the iterate is still sane
        sigma = sigma - (price - market_price) / vega
        if sigma <= 0 or not np.isfinite(sigma):
            return None  # diverged
    return None  # iteration cap without convergence


def implied_vol_brent(market_price: float, F: float, K: float, DF: float,
                      T: float, option_type: str,
                      config: Optional[dict] = None) -> float:
    """Brent's method fallback on the bracket [IV_MIN, IV_MAX].

    If the market price lies outside the price range spanned by the
    bracket (e.g. below discounted intrinsic, a static-arbitrage
    violation in the quote), there is no root: return nan and log the
    offending inputs. Do not raise -- one bad quote must not kill the
    chain-wide pass.
    """
    cfg = {**DEFAULT_SOLVER_CONFIG, **(config or {})}
    lo, hi = cfg["IV_MIN"], cfg["IV_MAX"]

    def f(sigma: float) -> float:
        return b76_price(F, K, DF, T, sigma, option_type) - market_price

    f_lo, f_hi = f(lo), f(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        logger.warning(
            "implied_vol_brent: no root in [%g, %g] for %s K=%.2f T=%.4f "
            "price=%.6f (f_lo=%.3g, f_hi=%.3g) -- quote violates static "
            "bounds; returning nan", lo, hi, option_type, K, T,
            market_price, f_lo, f_hi,
        )
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-12))


def _solve_iv(row: pd.Series, config: Optional[dict] = None
              ) -> tuple[float, str]:
    """Core per-row solver: NR first, Brent on any NR failure.
    Returns (iv, method) with method in {'nr', 'brent', 'nan'}."""
    cfg = {**DEFAULT_SOLVER_CONFIG, **(config or {})}
    args = (float(row["mid_price"]), float(row["F"]), float(row["strike"]),
            float(row["DF"]), float(row["T"]), str(row["option_type"]))

    iv = implied_vol_nr(*args, cfg)
    if iv is not None and 0.0 < iv < cfg["IV_MAX"]:
        return iv, _NR

    iv = implied_vol_brent(*args, cfg)
    if np.isfinite(iv):
        return iv, _BRENT
    return float("nan"), _FAIL


def solve_iv(row: pd.Series, config: Optional[dict] = None) -> float:
    """Public per-row API per spec: NR then Brent; float or nan."""
    return _solve_iv(row, config)[0]


# ---------------------------------------------------------------------------
# Chain-wide computation
# ---------------------------------------------------------------------------

def compute_iv_surface(df: pd.DataFrame,
                       config: Optional[dict] = None) -> pd.DataFrame:
    """Apply the solver across the forward-attached, OTM-filtered chain.

    Adds column `iv`; drops rows where iv is nan or non-positive. Logs the
    NR / Brent / nan split and asserts at least 80% of rows produced a
    finite IV -- a lower rate means the data or the implied forwards are
    wrong, and continuing would poison every downstream module.
    """
    required = {"mid_price", "F", "strike", "DF", "T", "option_type"}
    missing = required - set(df.columns)
    assert not missing, f"compute_iv_surface missing columns: {missing}"
    assert len(df) > 0, "compute_iv_surface received an empty chain"

    results = df.apply(lambda row: _solve_iv(row, config), axis=1)
    df = df.copy()
    df["iv"] = [r[0] for r in results]
    methods = pd.Series([r[1] for r in results], index=df.index)

    n = len(df)
    n_nr = int((methods == _NR).sum())
    n_brent = int((methods == _BRENT).sum())
    n_fail = int((methods == _FAIL).sum())
    print(f"compute_iv_surface: {n} rows -> NR {n_nr} ({n_nr / n:.1%}), "
          f"Brent {n_brent} ({n_brent / n:.1%}), "
          f"failed {n_fail} ({n_fail / n:.1%})")

    finite_rate = float(np.isfinite(df["iv"]).mean())
    assert finite_rate >= 0.80, (
        f"only {finite_rate:.1%} of rows produced a finite IV (< 80%): "
        f"check the data and implied forwards before proceeding"
    )

    df = df[np.isfinite(df["iv"]) & (df["iv"] > 0)].reset_index(drop=True)
    return df


def check_parity_iv(df_both_sides: pd.DataFrame,
                    config: Optional[dict] = None,
                    tol: float = 1e-6) -> pd.DataFrame:
    """Diagnostic: invert BOTH legs at strikes where a call and a put
    both survived cleaning, and compare their IVs.

    Run this on `attach_forwards(df, spot, cfg, otm_only=False)` output:
    the frame must carry F/DF but must NOT be OTM-filtered (the filter
    keeps one side per strike, leaving nothing to compare). With a
    shared parity-implied (F, DF), call and put IVs agree near-exactly on
    clean quotes -- breaches isolate bad quotes, not model bias.

    Returns a DataFrame of (expiry, strike, iv_call, iv_put, iv_gap) for
    pairs breaching `tol`, sorted by gap descending. Empty means clean.
    """
    cfg = {**DEFAULT_SOLVER_CONFIG, **(config or {})}
    rows = []
    for (expiry, strike), grp in df_both_sides.groupby(["expiry", "strike"]):  # pyright: ignore[reportGeneralTypeIssues] -- pandas types a multi-column groupby key as bare Hashable; it is a tuple at runtime
        types = set(grp["option_type"])
        if types != {"call", "put"}:
            continue
        ivs = {}
        for _, row in grp.iterrows():
            ivs[row["option_type"]] = _solve_iv(row, cfg)[0]
        gap = abs(ivs["call"] - ivs["put"])
        if not np.isfinite(gap) or gap > tol:
            rows.append({"expiry": expiry, "strike": strike,
                         "iv_call": ivs["call"], "iv_put": ivs["put"],
                         "iv_gap": gap})
    out = pd.DataFrame(rows, columns=["expiry", "strike", "iv_call",
                                      "iv_put", "iv_gap"])
    if not out.empty:
        out = out.sort_values("iv_gap", ascending=False).reset_index(drop=True)
        logger.warning("check_parity_iv: %d strike(s) breach tol=%g; "
                       "inspect those quotes", len(out), tol)
    return out