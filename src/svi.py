"""
svi.py -- Module 4: per-slice SVI calibration.

Pipeline position:
    exclude_flagged output (OTM rows with iv, F, DF, k, T)
        -> calibrate_all_slices
        -> [Module 5: surface]

Design notes (see PROJECT_SPEC.md r2/r3):
- Raw SVI (Gatheral 2004) in log-forward-moneyness k = log(K/F):
      w(k) = a + b*(rho*(k - m) + sqrt((k - m)^2 + sigma^2))
  fitted per expiry to market total variance w_mkt = iv^2 * T.
- Two-stage optimizer: differential_evolution global search (the SVI
  loss surface has local minima; a single-pass local optimizer
  frequently converges to the wrong basin), then explicit L-BFGS-B
  polish from the DE solution. DE's built-in polish is disabled so the
  two stages are explicit and separately inspectable.
- k comes from Module 1's column (parity-implied forward). It is NOT
  recomputed here -- recomputing with a different forward would silently
  shift every slice.
- Post-fit Lee check: Lee's moment formula bounds the asymptotic slope
  of total variance in |k| by 2; SVI wing slopes are b*(1 -+ rho), so
  b*(1 + |rho|) > 2 implies the fit prices non-existent moments. Flagged,
  never silently clipped.
- Scope: plain 5-parameter fit. The Zeliade (2012) quasi-explicit "2+3"
  decomposition is a documented upgrade path, not part of this build.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

logger = logging.getLogger(__name__)

DEFAULT_SVI_CONFIG: dict = {
    "SVI_GLOBAL_SEED": 42,
    "SVI_BOUNDS": {            # box bounds, spec Module 4
        "a": (-0.5, 0.5),
        "b": (0.0, 2.0),
        "rho": (-0.999, 0.999),
        "m": (-1.0, 1.0),
        "sigma": (1e-4, 1.0),
    },
    "DE_MAXITER": 1000,
    "DE_TOL": 1e-8,
    "RMSE_WARN": 0.005,        # total-variance units
    "MIN_POINTS_WARN": 5,
    # Numerical guard: infeasible parameter sets price to +inf via
    # svi_total_variance (spec), but L-BFGS-B finite-difference gradients
    # choke on inf, so the OBJECTIVE clamps non-finite loss to this
    # large finite penalty. Documented deviation; svi_total_variance
    # itself returns inf exactly as specced.
    "LOSS_PENALTY": 1e12,
}

PARAM_ORDER = ("a", "b", "rho", "m", "sigma")
SLICE_COLUMNS = ["expiry", "T", "F", "DF", "a", "b", "rho", "m", "sigma",
                 "rmse", "lee_flag", "n_points"]


# ---------------------------------------------------------------------------
# SVI parameterization
# ---------------------------------------------------------------------------

def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float,
                       m: float, sigma: float) -> np.ndarray:
    """Raw-SVI total implied variance at log-forward-moneyness k.

        w(k) = a + b*(rho*(k - m) + sqrt((k - m)^2 + sigma^2))

    Feasibility (checked before evaluating; violation returns +inf so an
    optimizer immediately penalizes the parameter set):
        b >= 0, |rho| < 1, sigma > 0,
        a + b*sigma*sqrt(1 - rho^2) >= 0   [min of w over k => w >= 0]
    """
    k = np.asarray(k, dtype=np.float64)
    feasible = (
        b >= 0.0
        and abs(rho) < 1.0
        and sigma > 0.0
        and a + b * sigma * np.sqrt(1.0 - rho * rho) >= 0.0
        and np.isfinite([a, b, rho, m, sigma]).all()
    )
    if not feasible:
        return np.full_like(k, np.inf)
    d = k - m
    return a + b * (rho * d + np.sqrt(d * d + sigma * sigma))


def svi_loss(params: np.ndarray, k_market: np.ndarray,
             w_market: np.ndarray) -> float:
    """Sum of squared total-variance errors for the slice.

        loss = sum((w_model(k) - w_market)^2)

    Returns +inf for infeasible params (via svi_total_variance).
    """
    w_model = svi_total_variance(k_market, *params)
    if not np.all(np.isfinite(w_model)):
        return float("inf")
    diff = w_model - w_market
    return float(np.dot(diff, diff))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_svi_slice(df_slice: pd.DataFrame,
                        config: Optional[dict] = None) -> dict:
    """Calibrate raw SVI to one expiry slice.

    Input columns required: k, iv, T, F, DF (constant T/F/DF within the
    slice, attached in Module 1; k is NOT recomputed here).

    Stage 1: differential_evolution over the box bounds (global; DE's
             built-in polish disabled).
    Stage 2: L-BFGS-B from the DE solution, same bounds (local polish).
    Post:    RMSE in total-variance units, warned above RMSE_WARN;
             Lee wing-slope flag when b*(1 + |rho|) > 2.

    Returns dict with SLICE_COLUMNS keys.
    """
    cfg = {**DEFAULT_SVI_CONFIG, **(config or {})}
    required = {"k", "iv", "T", "F", "DF"}
    missing = required - set(df_slice.columns)
    assert not missing, f"calibrate_svi_slice missing columns: {missing}"
    assert df_slice["T"].nunique() == 1, "slice spans multiple maturities"

    T = float(df_slice["T"].iloc[0])
    F = float(df_slice["F"].iloc[0])
    DF = float(df_slice["DF"].iloc[0])
    expiry = df_slice["expiry"].iloc[0] if "expiry" in df_slice.columns else None

    k = df_slice["k"].to_numpy(dtype=np.float64)
    w = (df_slice["iv"].to_numpy(dtype=np.float64) ** 2) * T
    n = len(k)
    if n < cfg["MIN_POINTS_WARN"]:
        logger.warning(
            "calibrate_svi_slice: expiry %s has only %d points; the fit "
            "may be unreliable (spec limitation)", expiry, n,
        )
    assert np.all(np.isfinite(k)) and np.all(np.isfinite(w)), (
        "non-finite k or total variance entering the fit"
    )
    assert np.all(w >= 0.0), "negative market total variance"

    bounds = [cfg["SVI_BOUNDS"][p] for p in PARAM_ORDER]
    penalty = cfg["LOSS_PENALTY"]

    def objective(params: np.ndarray) -> float:
        loss = svi_loss(params, k, w)
        return loss if np.isfinite(loss) else penalty

    # Stage 1 -- global. rng is the current scipy parameter name (seed is
    # the legacy spelling). polish=False: stage 2 is explicit below.
    de = differential_evolution(
        objective, bounds=bounds, rng=cfg["SVI_GLOBAL_SEED"],
        maxiter=cfg["DE_MAXITER"], tol=cfg["DE_TOL"], polish=False,
    )

    # Stage 2 -- local polish from the DE solution.
    local = minimize(objective, x0=de.x, method="L-BFGS-B", bounds=bounds)
    best = local.x if local.fun <= de.fun else de.x
    if local.fun > de.fun:
        logger.warning(
            "calibrate_svi_slice: L-BFGS-B polish worsened the DE loss "
            "(%.3e -> %.3e) on expiry %s; keeping the DE solution",
            de.fun, local.fun, expiry,
        )

    a, b, rho, m, sigma = (float(v) for v in best)

    # Bound-rail diagnostic: a parameter pinned at its box bound means
    # the data does not identify it. The fit can still be accurate
    # INSIDE the observed strikes, but wing extrapolation is arbitrary.
    for name, val in zip(PARAM_ORDER, best):
        lo, hi = cfg["SVI_BOUNDS"][name]
        span = hi - lo
        if min(val - lo, hi - val) < 1e-3 * span:
            logger.warning(
                "calibrate_svi_slice: expiry %s parameter %s=%.4f railed "
                "at its bound [%.4g, %.4g] -- poorly identified; do not "
                "trust the fit outside the observed k range "
                "[%.3f, %.3f]", expiry, name, val, lo, hi,
                float(k.min()), float(k.max()),
            )

    # m-support diagnostic: the smile vertex m sitting OUTSIDE the
    # observed k range (railed or not) means the data contains no
    # curvature information locating it -- SVI is then locally
    # near-linear over the data and the fitted wings are arbitrary.
    # Typical cause: a slice with little/no smile convexity.
    if not (k.min() <= m <= k.max()):
        logger.warning(
            "calibrate_svi_slice: expiry %s fitted m=%.4f lies outside "
            "the observed k range [%.3f, %.3f] -- vertex unidentified; "
            "interpolation inside the range is usable, extrapolation is "
            "not", expiry, m, float(k.min()), float(k.max()),
        )

    loss = objective(best)
    rmse = float(np.sqrt(loss / n))
    if rmse > cfg["RMSE_WARN"]:
        logger.warning(
            "calibrate_svi_slice: expiry %s RMSE %.5f > %.5f -- poor fit "
            "(few liquid strikes, bad quotes, or optimizer failure)",
            expiry, rmse, cfg["RMSE_WARN"],
        )

    lee_slope = b * (1.0 + abs(rho))
    lee_flag = bool(lee_slope > 2.0)
    if lee_flag:
        logger.warning(
            "calibrate_svi_slice: expiry %s wing slope b*(1+|rho|)=%.3f "
            "exceeds Lee's moment bound of 2 -- surface prices "
            "non-existent moments; flagged, not clipped", expiry, lee_slope,
        )

    return {"expiry": expiry, "T": T, "F": F, "DF": DF,
            "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
            "rmse": rmse, "lee_flag": lee_flag, "n_points": n}


def calibrate_all_slices(df: pd.DataFrame,
                         config: Optional[dict] = None) -> pd.DataFrame:
    """Calibrate every expiry slice; return one row per slice sorted by
    T ascending (SLICE_COLUMNS). Logs RMSE per expiry."""
    results = []
    for expiry, grp in df.groupby("expiry", sort=True):
        res = calibrate_svi_slice(grp, config)
        print(
            f"SVI fit {pd.Timestamp(expiry).date()}: n={res['n_points']}, "
            f"rmse={res['rmse']:.2e}, a={res['a']:+.4f}, b={res['b']:.4f}, "
            f"rho={res['rho']:+.3f}, m={res['m']:+.4f}, "
            f"sigma={res['sigma']:.4f}"
            + ("  [LEE FLAG]" if res["lee_flag"] else "")
        )
        results.append(res)
    out = pd.DataFrame(results, columns=SLICE_COLUMNS)
    return out.sort_values("T").reset_index(drop=True)