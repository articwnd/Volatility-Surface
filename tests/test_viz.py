"""Offline tests for src/viz.py. No network, headless (Agg backend
forced before pyplot's first import). File outputs go to tmp_path.
"""

import matplotlib

matplotlib.use("Agg")  # must precede any pyplot import (viz imports lazily)

import numpy as np
import pandas as pd
import pytest

from src.svi import svi_total_variance
from src.viz import (
    atm_forward_term_structure,
    plot_3d_surface,
    plot_smile_slices,
    plot_term_structure,
)

P1 = {"a": 0.0020, "b": 0.040, "rho": -0.40, "m": 0.02, "sigma": 0.20}
P2 = {"a": 0.0090, "b": 0.055, "rho": -0.40, "m": 0.02, "sigma": 0.20}


def _fit_frame_and_params():
    """A small (df, svi_params) pair consistent with the pipeline shapes."""
    slices = [
        (pd.Timestamp("2026-08-08"), 30, 6510.0, 0.9970, P1),
        (pd.Timestamp("2026-11-06"), 120, 6540.0, 0.9880, P2),
    ]
    df_rows, param_rows = [], []
    for expiry, dte, F, DF, p in slices:
        T = dte / 365.0
        k = np.linspace(-0.12, 0.10, 23)
        iv = np.sqrt(svi_total_variance(k, **p) / T)
        for ki, ivi in zip(k, iv):
            df_rows.append({"expiry": expiry, "k": ki, "iv": ivi, "T": T,
                            "days_to_expiry": dte, "fwd_fallback": False,
                            "F": F, "DF": DF,
                            "strike": F * np.exp(ki)})
        param_rows.append({"expiry": expiry, "T": T, "F": F, "DF": DF, **p,
                           "rmse": 1e-7, "lee_flag": False, "n_points": 23})
    return pd.DataFrame(df_rows), pd.DataFrame(param_rows)


def test_plot_3d_surface_selfcontained(tmp_path):
    m_grid = np.linspace(0.9, 1.1, 40)
    T_grid = np.array([30 / 365, 120 / 365])
    iv = np.full((2, 40), 0.18)
    path = plot_3d_surface(iv, m_grid, T_grid,
                           str(tmp_path / "vol_surface.html"),
                           title="test surface")
    html = open(path, encoding="utf-8").read()
    assert len(html) > 1_000_000          # plotly.js inlined => offline-open
    # NOTE: plotly JSON-escapes "/" as \u002f, so probe slash-free
    # substrings rather than the literal "Moneyness (K/S)".
    assert "Moneyness" in html
    assert "Days to expiry" in html
    assert "Implied vol" in html
    assert "test surface" in html


def test_plot_3d_surface_shape_guard(tmp_path):
    with pytest.raises(AssertionError):
        plot_3d_surface(np.zeros((3, 5)), np.zeros(4), np.zeros(3),
                        str(tmp_path / "x.html"))


def test_plot_smile_slices(tmp_path):
    df, params = _fit_frame_and_params()
    paths = plot_smile_slices(df, params, str(tmp_path / "smiles"))
    assert len(paths) == 2
    for p in paths:
        import os
        assert os.path.getsize(p) > 10_000  # a real rendered figure


def test_plot_smile_skips_unfitted_expiry(tmp_path, caplog):
    df, params = _fit_frame_and_params()
    params = params.iloc[[0]]             # second expiry has no fit row
    paths = plot_smile_slices(df, params, str(tmp_path / "smiles"))
    assert len(paths) == 1
    assert any("no fitted params" in r.message for r in caplog.records)


def test_atm_forward_selection():
    """The term structure must pick min |k| per expiry (ATM-forward),
    not min |K - spot| (spot-ATM). The k grid is asymmetric so the two
    choices differ, and the picked IV must match the picked k's row."""
    df, _ = _fit_frame_and_params()
    ts = atm_forward_term_structure(df)
    assert list(ts["days_to_expiry"]) == [30, 120]
    for _, row in ts.iterrows():
        grp = df[df["expiry"] == row["expiry"]]
        k_min = grp.loc[grp["k"].abs().idxmin()]
        assert row["k"] == k_min["k"]
        assert row["iv"] == k_min["iv"]
        assert abs(row["k"]) <= grp["k"].abs().min() + 1e-15


def test_plot_term_structure(tmp_path):
    import os
    df, _ = _fit_frame_and_params()
    path = plot_term_structure(df, str(tmp_path / "term_structure.png"))
    assert os.path.getsize(path) > 5_000