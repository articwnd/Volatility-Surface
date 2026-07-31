"""
viz.py -- Module 6: rendering.

Pipeline position:
    evaluate_svi_grid matrix + fitted params + IV frame
        -> vol_surface.html, smiles/*.png, term_structure.png

Design notes (see PROJECT_SPEC.md r2):
- 3D surface: Plotly go.Surface written to a SELF-CONTAINED html file
  (plotly.js inlined, ~3MB, opens offline). X moneyness K/S, Y days to
  expiry, Z IV in percent, diverging colorscale so the skew is visible.
- Smile slices: per-expiry scatter of market IVs against the fitted SVI
  curve, on the k axis (log-forward-moneyness -- the fit's native
  coordinate), annotated with expiry, DTE, RMSE, and whether the
  forward was parity-implied or the rate fallback.
- Term structure: IV at the row with minimum |k| per expiry
  (ATM-FORWARD, not spot-ATM: k = log(K/F) = 0 is the forward).
  Selection logic lives in atm_forward_term_structure so it is testable
  without parsing a PNG.

matplotlib.pyplot is imported lazily inside functions so headless/test
environments can set a non-interactive backend before first use.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.svi import svi_total_variance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3D surface
# ---------------------------------------------------------------------------

def plot_3d_surface(iv_matrix: np.ndarray, moneyness_grid: np.ndarray,
                    T_grid: np.ndarray, output_path: str,
                    title: Optional[str] = None) -> str:
    """Interactive 3D Plotly surface, saved as a self-contained HTML file.

    Axes: X moneyness (K/S), Y days to expiry (T_grid*365), Z IV (%).
    `iv_matrix` must be shape (len(T_grid), len(moneyness_grid)) in
    DECIMAL vol (rows ordered like T_grid). Returns the written path.
    """
    import plotly.graph_objects as go

    iv_matrix = np.asarray(iv_matrix, dtype=np.float64)
    m_grid = np.asarray(moneyness_grid, dtype=np.float64)
    T_grid = np.asarray(T_grid, dtype=np.float64)
    assert iv_matrix.shape == (len(T_grid), len(m_grid)), (
        f"iv_matrix shape {iv_matrix.shape} != "
        f"({len(T_grid)}, {len(m_grid)})"
    )
    assert np.all(np.isfinite(iv_matrix)), "non-finite IVs in surface matrix"

    fig = go.Figure(data=[go.Surface(
        x=m_grid, y=T_grid * 365.0, z=iv_matrix * 100.0,
        colorscale="RdYlGn_r",
        colorbar=dict(title="IV (%)"),
    )])
    fig.update_layout(
        title=title or "Implied Volatility Surface",
        scene=dict(
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Days to expiry",
            zaxis_title="Implied vol (%)",
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # include_plotlyjs=True inlines plotly.js: the file opens with no
    # network access, per the README's "self-contained" promise.
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    logger.info("3D surface written: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Smile slices
# ---------------------------------------------------------------------------

def plot_smile_slices(df: pd.DataFrame, svi_params: pd.DataFrame,
                      output_dir: str) -> list[str]:
    """One PNG per expiry: market IVs (scatter) vs. fitted SVI (line) on
    the k axis. Returns the written paths.

    `df` requires: expiry, k, iv, days_to_expiry (fwd_fallback used if
    present). `svi_params` is calibrate_all_slices output.
    """
    import matplotlib.pyplot as plt

    required = {"expiry", "k", "iv", "days_to_expiry"}
    missing = required - set(df.columns)
    assert not missing, f"plot_smile_slices missing columns: {missing}"
    os.makedirs(output_dir, exist_ok=True)

    params_by_expiry = svi_params.set_index("expiry")
    paths: list[str] = []
    for expiry, grp in df.groupby("expiry", sort=True):
        if expiry not in params_by_expiry.index:
            logger.warning("plot_smile_slices: no fitted params for %s "
                           "(excluded slice?); skipped", expiry)
            continue
        p = params_by_expiry.loc[expiry]
        k_line = np.linspace(float(grp["k"].min()), float(grp["k"].max()), 200)
        w_line = svi_total_variance(k_line, float(p["a"]), float(p["b"]),
                                    float(p["rho"]), float(p["m"]),
                                    float(p["sigma"]))
        iv_line = np.sqrt(w_line / float(p["T"]))

        fwd_src = "parity-implied"
        if "fwd_fallback" in grp.columns and bool(grp["fwd_fallback"].iloc[0]):
            fwd_src = "RATE FALLBACK (dividend bias)"

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(grp["k"], grp["iv"] * 100.0, s=18, alpha=0.8,
                   label="market (OTM mids)", zorder=3)
        ax.plot(k_line, iv_line * 100.0, lw=1.8, label="SVI fit", zorder=2)
        dte = int(grp["days_to_expiry"].iloc[0])
        exp_str = pd.Timestamp(expiry).date()
        ax.set_title(f"{exp_str}  ({dte}d)   RMSE={float(p['rmse']):.2e}   "
                     f"forward: {fwd_src}"
                     + ("   [LEE FLAG]" if bool(p.get("lee_flag", False))
                        else ""))
        ax.set_xlabel("log-forward-moneyness  k = log(K/F)")
        ax.set_ylabel("Implied vol (%)")
        ax.legend()
        ax.grid(alpha=0.3)
        path = os.path.join(output_dir, f"smile_{exp_str}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    logger.info("smile plots written: %d file(s) in %s", len(paths), output_dir)
    return paths


# ---------------------------------------------------------------------------
# Term structure
# ---------------------------------------------------------------------------

def atm_forward_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Per expiry, the IV at the strike with minimum |k| (ATM-forward).

    Selection logic split out from the plot so it is testable directly.
    Returns (expiry, days_to_expiry, k, iv) sorted by days_to_expiry.
    """
    required = {"expiry", "k", "iv", "days_to_expiry"}
    missing = required - set(df.columns)
    assert not missing, f"atm_forward_term_structure missing: {missing}"
    rows = []
    for expiry, grp in df.groupby("expiry", sort=True):
        row = grp.loc[grp["k"].abs().idxmin()]
        rows.append({"expiry": expiry,
                     "days_to_expiry": int(row["days_to_expiry"]),
                     "k": float(row["k"]), "iv": float(row["iv"])})
    out = pd.DataFrame(rows).sort_values("days_to_expiry")
    return out.reset_index(drop=True)


def plot_term_structure(df: pd.DataFrame, output_path: str) -> str:
    """ATM-forward IV vs. days to expiry (line with markers). Returns
    the written path."""
    import matplotlib.pyplot as plt

    ts = atm_forward_term_structure(df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts["days_to_expiry"], ts["iv"] * 100.0, marker="o", lw=1.8)
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel("ATM-forward implied vol (%)")
    ax.set_title("Term structure (k = 0)")
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("term structure written: %s", output_path)
    return output_path