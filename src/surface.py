"""
surface.py -- Module 5: continuous surface from fitted SVI slices.

Pipeline position:
    calibrate_all_slices output (one row per fitted expiry)
        -> evaluate_svi_grid (plotting matrix)
        -> build_interpolator (query API)
        -> [Module 6: viz]

Design notes (see PROJECT_SPEC.md r2):
- NO 2D spline of any kind. SVI is analytic in k, so the strike
  dimension needs no interpolation. In the time dimension total variance
  is interpolated LINEARLY between fitted slices: a cubic spline in T
  can overshoot between knots and reintroduce calendar arbitrage that
  the individual slices do not have, and bicubic fitting needs >= 4
  maturities, which post-filter data cannot always guarantee.
- Linear interpolation of total variance between slices that
  individually satisfy w_i(k) <= w_j(k) preserves calendar monotonicity
  at every intermediate T (a convex combination of ordered values stays
  between them and moves monotonically in the weight). That containment
  is checked, not assumed: check_fitted_calendar evaluates adjacent
  FITTED slices on a k grid, because Module 3's pre-fit check covers
  only an ATM band of the RAW quotes and fitted wings can still cross.
- Queries outside the fitted maturity range CLAMP to the nearest slice
  (full clamp: the returned value is that slice's IV at the query's k
  under the clamped slice's own T and forward). No time extrapolation.
  Warned once per interpolator, not per query.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.svi import svi_total_variance

logger = logging.getLogger(__name__)

REQUIRED_PARAM_COLUMNS = ["T", "F", "a", "b", "rho", "m", "sigma"]


def _validate_params(svi_params: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_PARAM_COLUMNS if c not in svi_params.columns]
    assert not missing, f"svi_params missing columns: {missing}"
    assert len(svi_params) >= 1, "no fitted slices"
    out = svi_params.sort_values("T").reset_index(drop=True)
    T = out["T"].to_numpy(dtype=np.float64)
    assert np.all(np.diff(T) > 0), "duplicate or non-increasing maturities"
    assert np.all(np.isfinite(out[REQUIRED_PARAM_COLUMNS].to_numpy(
        dtype=np.float64))), "non-finite slice parameters"
    return out


def _slice_w(row: pd.Series, k: np.ndarray) -> np.ndarray:
    return svi_total_variance(
        k, float(row["a"]), float(row["b"]), float(row["rho"]),
        float(row["m"]), float(row["sigma"]),
    )


# ---------------------------------------------------------------------------
# Plotting matrix
# ---------------------------------------------------------------------------

def evaluate_svi_grid(svi_params: pd.DataFrame,
                      moneyness_grid: np.ndarray,
                      spot: float) -> np.ndarray:
    """Evaluate each fitted slice on a common SPOT-moneyness grid.

    For each slice: k = log(moneyness * spot / F_slice) using that
    slice's own implied forward, w = SVI(k), iv = sqrt(w / T).

    Returns a 2D array of shape (n_expiries, n_strikes), IV in decimals,
    rows ordered by ascending T.
    """
    params = _validate_params(svi_params)
    m_grid = np.asarray(moneyness_grid, dtype=np.float64)
    assert np.isfinite(spot) and spot > 0, f"bad spot: {spot}"
    assert np.all(m_grid > 0), "moneyness grid must be positive"

    out = np.empty((len(params), len(m_grid)), dtype=np.float64)
    for i, (_, row) in enumerate(params.iterrows()):
        k = np.log(m_grid * spot / float(row["F"]))
        w = _slice_w(row, k)
        assert np.all(np.isfinite(w)) and np.all(w >= 0), (
            f"slice {i} produced non-finite/negative total variance; "
            f"was it Lee-flagged or bound-railed?"
        )
        out[i, :] = np.sqrt(w / float(row["T"]))
    return out


# ---------------------------------------------------------------------------
# Fitted-slice calendar diagnostic
# ---------------------------------------------------------------------------

def check_fitted_calendar(svi_params: pd.DataFrame,
                          k_grid: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Evaluate adjacent FITTED slices on a k grid and report crossings
    (w_short(k) > w_long(k)), which break the containment property the
    linear time interpolation relies on.

    Module 3 checks the RAW quotes in an ATM band only; fitted wings can
    still cross, especially where a slice was bound-railed or thin. Run
    this before trusting interpolated queries away from the money.

    Returns a DataFrame (expiry_short, expiry_long, T_short, T_long,
    k_cross_lo, k_cross_hi, frac_crossed, worst_gap); empty means clean
    on the grid.
    """
    params = _validate_params(svi_params)
    if k_grid is None:
        k_grid = np.linspace(-0.5, 0.5, 201)
    k_grid = np.asarray(k_grid, dtype=np.float64)

    rows = []
    for i in range(len(params) - 1):
        short, long_ = params.iloc[i], params.iloc[i + 1]
        w_s, w_l = _slice_w(short, k_grid), _slice_w(long_, k_grid)
        crossed = w_s > w_l
        if crossed.any():
            gap = w_s - w_l
            rows.append({
                "expiry_short": short.get("expiry"),
                "expiry_long": long_.get("expiry"),
                "T_short": float(short["T"]), "T_long": float(long_["T"]),
                "k_cross_lo": float(k_grid[crossed].min()),
                "k_cross_hi": float(k_grid[crossed].max()),
                "frac_crossed": float(np.mean(crossed)),
                "worst_gap": float(gap.max()),
            })
    out = pd.DataFrame(rows, columns=[
        "expiry_short", "expiry_long", "T_short", "T_long",
        "k_cross_lo", "k_cross_hi", "frac_crossed", "worst_gap",
    ])
    if not out.empty:
        logger.warning(
            "check_fitted_calendar: %d adjacent slice pair(s) cross on the "
            "k grid; interpolated queries in the crossing region are "
            "calendar-arbitrageable", len(out),
        )
    return out


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

def build_interpolator(svi_params: pd.DataFrame,
                       spot: float) -> Callable[[float, float], float]:
    """Return interp(T, K) -> iv built directly on the fitted slices.

    Query algorithm:
    1. T outside [T_min, T_max]: clamp to the nearest slice (its own T
       and forward are used -- full clamp, no time extrapolation),
       warned once per interpolator.
    2. Bracket T_i <= T <= T_j; interpolate log F linearly in T.
    3. k = log(K / F(T)); evaluate both slices analytically at k.
    4. w = linear in T between w_i(k) and w_j(k); iv = sqrt(w / T).

    `spot` is accepted for API symmetry/diagnostics; queries are in
    absolute strike K and use the interpolated forward, not spot.
    """
    params = _validate_params(svi_params)
    assert np.isfinite(spot) and spot > 0, f"bad spot: {spot}"

    T_arr = params["T"].to_numpy(dtype=np.float64)
    logF_arr = np.log(params["F"].to_numpy(dtype=np.float64))
    state = {"clamp_warned": False}

    def _clamp_warn(T_query: float, T_used: float) -> None:
        if not state["clamp_warned"]:
            logger.warning(
                "surface query T=%.4f outside fitted range [%.4f, %.4f]; "
                "clamped to nearest slice (no time extrapolation). "
                "Further clamps in this session will not be re-warned.",
                T_query, T_arr[0], T_arr[-1],
            )
            state["clamp_warned"] = True

    def _slice_iv(idx: int, K: float) -> float:
        row = params.iloc[idx]
        k = np.log(K / float(row["F"]))
        w = float(_slice_w(row, np.array([k]))[0])
        return float(np.sqrt(w / float(row["T"])))

    def interp(T: float, K: float) -> float:
        T = float(T)
        K = float(K)
        assert np.isfinite(T) and np.isfinite(K) and T > 0 and K > 0, (
            f"bad query: T={T}, K={K}"
        )
        if T <= T_arr[0]:
            if T < T_arr[0]:
                _clamp_warn(T, T_arr[0])
            return _slice_iv(0, K)
        if T >= T_arr[-1]:
            if T > T_arr[-1]:
                _clamp_warn(T, T_arr[-1])
            return _slice_iv(len(params) - 1, K)

        j = int(np.searchsorted(T_arr, T, side="right"))
        i = j - 1
        T_i, T_j = T_arr[i], T_arr[j]
        lam = (T - T_i) / (T_j - T_i)

        F_T = float(np.exp((1.0 - lam) * logF_arr[i] + lam * logF_arr[j]))
        k = np.array([np.log(K / F_T)])
        w_i = float(_slice_w(params.iloc[i], k)[0])
        w_j = float(_slice_w(params.iloc[j], k)[0])
        w = (1.0 - lam) * w_i + lam * w_j
        assert w >= 0.0 and np.isfinite(w), (
            f"non-finite/negative interpolated total variance at "
            f"T={T}, K={K}"
        )
        return float(np.sqrt(w / T))

    return interp


def query_surface(interp: Callable[[float, float], float],
                  T: float, K: float) -> float:
    """Convenience wrapper: implied volatility at one (T, K) point."""
    return interp(T, K)


# ---------------------------------------------------------------------------
# CLI: python -m src.surface --from-snapshot data/snapshots/SPX_....csv
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    from src.arbitrage import exclude_flagged, run_arbitrage_checks
    from src.data import attach_forwards, clean_chain, load_snapshot
    from src.iv_solver import compute_iv_surface
    from src.svi import calibrate_all_slices
    from src.viz import (plot_3d_surface, plot_smile_slices,
                         plot_term_structure)

    parser = argparse.ArgumentParser(
        description="Modules 1-6: build and render the full surface")
    parser.add_argument("--from-snapshot", required=True,
                        help="snapshot CSV written by src.data --snapshot "
                             "(live pulls happen in src.data, not here)")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--grid-lo", type=float, default=0.85)
    parser.add_argument("--grid-hi", type=float, default=1.15)
    parser.add_argument("--grid-n", type=int, default=120)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chain, spot, asof = load_snapshot(args.from_snapshot)
    clean = clean_chain(chain, spot, asof=asof)
    surf_df = compute_iv_surface(attach_forwards(clean, spot))
    vert, bf, cal = run_arbitrage_checks(surf_df)
    fit_input = exclude_flagged(surf_df, vert, bf, cal)
    params = calibrate_all_slices(fit_input)

    xc = check_fitted_calendar(params)
    if not xc.empty:
        print("FITTED-SLICE CALENDAR CROSSINGS (interpolation not "
              "arbitrage-free in these regions):")
        print(xc.to_string(index=False))

    # Restrict the plotted grid to roughly the observed moneyness range;
    # the identifiability diagnostics in Module 4 say wings beyond the
    # data are not to be trusted, so we do not render them.
    m_grid = np.linspace(args.grid_lo, args.grid_hi, args.grid_n)
    iv_matrix = evaluate_svi_grid(params, m_grid, spot)
    T_grid = params.sort_values("T")["T"].to_numpy(dtype=np.float64)

    html = plot_3d_surface(
        iv_matrix, m_grid, T_grid,
        output_path=f"{args.outputs}/vol_surface.html",
        title=f"IV surface -- spot {spot:.2f} @ {asof}",
    )
    smiles = plot_smile_slices(fit_input, params, f"{args.outputs}/smiles")
    ts = plot_term_structure(fit_input, f"{args.outputs}/term_structure.png")

    interp = build_interpolator(params, spot)
    T_mid = float(0.5 * (T_grid[0] + T_grid[-1]))
    print(f"sample query: iv(T={T_mid:.3f}, K={spot:.0f}) = "
          f"{interp(T_mid, spot):.4f}")
    print(f"outputs: {html}, {len(smiles)} smile plot(s), {ts}")


if __name__ == "__main__":
    _main()