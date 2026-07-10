# Implied Volatility Surface (Python)

A production-oriented Python pipeline that ingests live options market data,
implies the forward and discount factor per expiry directly from put-call
parity, inverts Black-76 to recover implied volatilities across OTM strikes
and maturities, fits a per-slice SVI parametric model with arbitrage
enforcement, and renders an interactive 3D surface alongside diagnostic
slice plots.

---

## Problem Statement

Black-Scholes assumes a single constant volatility for all strikes and
maturities. Markets violate this empirically: equity index options consistently
show a pronounced negative skew (higher IV for lower strikes) and a term
structure that is rarely flat. A flat vol assumption misprices OTM puts, exotic
payoffs, and any instrument with path dependency.

The volatility surface is the market's encoding of that reality. It is the
central object in derivatives practice: every stochastic vol model (Heston,
SABR, rough vol) is calibrated to it, every exotic option is priced relative
to it, and every delta-hedging book uses it to compute consistent Greeks. This
project constructs a complete, arbitrage-checked surface from raw market data
and exposes the full pipeline so each stage can be inspected, tested, and
extended.

---

## Approach

### Pipeline Stages

**Stage 1 -- Data Acquisition**

Options chains are pulled via `yfinance` for a configurable underlying (default:
SPX). For each expiry, the full chain of calls and puts is collected. Mid-price
`(bid + ask) / 2` is used as the market price input. Quotes are filtered on:
- Open interest >= `MIN_OI` (default: 100) to remove illiquid strikes
- Bid > 0 to remove stale or crossed markets
- Moneyness within `[MONEYNESS_LO, MONEYNESS_HI]` (default: 0.70 to 1.30) to
  discard deep wings where IV inversion is numerically unreliable

Chains should be pulled during market hours: yfinance quotes are delayed
15-20 minutes, open interest updates overnight, and after-hours option
bid/ask is frequently stale or crossed. The spot price is snapshotted at the
same moment as the chain to avoid a spot/quote timestamp mismatch. Every
fetch is written to a timestamped CSV snapshot, and the entire downstream
pipeline can run from a snapshot instead of a live pull. This makes runs
reproducible and decouples model development from data availability.

**Stage 2 -- Forward Implication and IV Inversion**

Rather than sourcing an external risk-free rate and dividend yield, the
forward `F` and discount factor `DF` are implied per expiry directly from
put-call parity. For European options, `C(K) - P(K) = DF * (F - K)` holds
exactly, so an OLS regression of call-put mid spreads on strike across
near-the-money pairs recovers `DF` (negative slope) and `F` (intercept over
`DF`). This is the market-standard approach: it eliminates the dividend
yield as an input entirely and removes most sensitivity to the rate source.
SPX carries a nonzero dividend yield, so pricing off `F = S * exp(r*T)`
without a dividend adjustment would systematically split call and put IVs
and distort the skew.

With `F` and `DF` in hand, all pricing moves to Black-76 on the forward.
For each (strike, expiry) tuple on the OTM side (puts below the forward,
calls above -- ITM mids carry wide spreads and produce noisy IVs), implied
volatility is recovered by solving `B76_price(F, K, DF, T, sigma) =
market_price` numerically. The solver uses Newton-Raphson (NR) seeded with
the Manaster-Koehler (1982) starting value, with vega in the denominator
for fast convergence, and falls back to Brent's method when vega falls below
a threshold or NR fails to converge, guaranteeing a bracketed solution.

A FRED-sourced 3-month Treasury rate is retained only as a diagnostic and
as the fallback when an expiry has too few call-put pairs to imply the
forward. Put-call parity residuals from the forward regression are logged:
large residuals flag data quality issues rather than being silently
discarded.

**Stage 3 -- Arbitrage Checks**

Two no-arbitrage conditions are enforced before any parametric fit:

- **Butterfly (strike) arbitrage**: call prices must be convex in strike.
  Because SPX strike spacing is irregular (5/10/25/50 point increments), the
  check uses the spacing-weighted condition on each adjacent triple
  `K1 < K2 < K3`: flag when `C(K2) > w*C(K1) + (1-w)*C(K3)` with
  `w = (K3 - K2)/(K3 - K1)`. The unweighted butterfly `C(K1) - 2*C(K2) +
  C(K3)` is only valid for equally spaced strikes and would produce false
  flags on real SPX chains. Since only OTM quotes are retained, call prices
  below the forward are synthesized from puts via parity:
  `C(K) = P(K) + DF*(F - K)`.
- **Calendar spread (time) arbitrage**: total variance `sigma^2 * T` must be
  non-decreasing across maturities at fixed log-forward-moneyness
  `k = log(K/F)`, not at fixed strike (forwards differ across expiries).
  A surface that violates this allows riskless profit via calendar spreads.

Strikes or slices failing these checks are flagged and excluded from the
parametric fit. All violations are written to a diagnostic CSV for inspection.

**Stage 4 -- SVI Parametric Fit**

Each expiry slice is fitted independently using the Stochastic Volatility
Inspired (SVI) parameterization (Gatheral 2004). SVI expresses total implied
variance as a function of log-forward-moneyness `k = log(K/F)`:

```
w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
```

Parameters `{a, b, rho, m, sigma}` are calibrated per slice by minimizing the
sum of squared differences between model total variance and market total
variance, using `scipy.optimize.differential_evolution` for a global initial
search followed by `L-BFGS-B` for local refinement. The two-stage approach
(global then local) is essential: the SVI loss surface has local minima and a
single-pass local optimizer will frequently converge to the wrong solution.

After each fit, the wing slopes `b*(1 - rho)` and `b*(1 + rho)` are checked
against Lee's moment bound (total variance can grow at most linearly in `|k|`
with slope 2); violations are flagged. The Zeliade (2012) quasi-explicit
"2+3" decomposition, which reduces calibration to a 2-parameter outer loop,
is a documented upgrade path -- it is not implemented here.

**Stage 5 -- Surface Construction and Interpolation**

The surface is queried directly from the fitted slices; no 2D spline is
fitted on top of them. Within a slice, SVI is analytic in `k`, so strike
interpolation is unnecessary. For an arbitrary maturity `T` between two
fitted expiries, the forward is interpolated linearly in `log F`, both
bracketing slices are evaluated analytically at the query's
log-forward-moneyness, and total variance (not volatility) is interpolated
linearly in `T`. Linear interpolation of total variance preserves the
non-decreasing structure required by calendar no-arbitrage; a cubic spline
in the time dimension can overshoot and reintroduce calendar arbitrage
between clean slices. Queries outside the fitted maturity range are clamped
to the nearest slice with a logged warning rather than extrapolated.

For visualization, fitted slices are evaluated on a fine moneyness grid.
SVI wings are linear in total variance, consistent in form with Lee's
moment bounds; the explicit slope check in Stage 4 guards the bound.

**Stage 6 -- Visualization**

- **3D interactive surface**: Plotly `go.Surface` rendered to a self-contained
  HTML file. X-axis: moneyness (K/S), Y-axis: days to expiry, Z-axis: IV (%).
- **Smile slices**: per-expiry plots overlaying raw market IVs (scatter) against
  the fitted SVI curve (line) for visual residual inspection.
- **Term structure**: IV at `k = 0` (ATM-forward) vs. days-to-expiry, showing
  the market's pricing of forward uncertainty across time.

### Project Structure

```
vol_surface/
├── src/
│   ├── data.py              # yfinance ingestion, chain cleaning, snapshot
│   │                        #   cache, implied forward/DF, FRED fallback rate
│   ├── iv_solver.py         # Black-76 NR (Manaster-Koehler seed) + Brent
│   ├── arbitrage.py         # Weighted butterfly and fixed-k calendar checks
│   ├── svi.py               # SVI calibration (DE global + L-BFGS-B polish)
│   ├── surface.py           # Analytic slice evaluation, total-variance
│   │                        #   time interpolation, query API
│   └── viz.py               # Plotly 3D surface, smile slices, term structure
├── tests/
│   ├── test_data.py         # Implied forward/DF recovery on synthetic parity
│   ├── test_iv_solver.py    # NR/Brent roundtrip; known Hull values
│   ├── test_arbitrage.py    # Synthetic violations incl. unequal spacing
│   └── test_svi.py          # Parameter recovery on synthetic smiles
├── notebooks/
│   └── surface_explorer.ipynb  # Interactive walkthrough of full pipeline
├── outputs/
│   ├── vol_surface.html     # Interactive 3D surface (Plotly)
│   ├── smiles/              # Per-expiry SVI fit plots
│   └── arbitrage_flags.csv  # Violations log
├── data/
│   └── snapshots/           # Timestamped chain snapshots (CSV)
├── requirements.txt
└── README.md
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `TICKER` | `^SPX` | Underlying to pull options for |
| `MIN_OI` | 100 | Minimum open interest filter |
| `MONEYNESS_LO` | 0.70 | Minimum K/S to include |
| `MONEYNESS_HI` | 1.30 | Maximum K/S to include |
| `MIN_EXPIRY_DAYS` | 7 | Drop near-expiry slices (gamma instability) |
| `MAX_EXPIRY_DAYS` | 365 | Drop far-dated slices (sparse data) |
| `FWD_BAND` | 0.10 | K/S band (+/-) for parity pairs in forward regression |
| `FWD_MIN_PAIRS` | 3 | Minimum call-put pairs to imply forward; else fallback |
| `NR_TOL` | 1e-8 | Newton-Raphson price convergence tolerance |
| `NR_MAX_ITER` | 100 | Max NR iterations before Brent fallback |
| `VEGA_FLOOR` | 1e-4 | Vega threshold triggering Brent fallback |
| `SVI_GLOBAL_SEED` | 42 | Seed for differential evolution |

`VEGA_FLOOR` is set to 1e-4 (dollar vega) deliberately: a floor near machine
epsilon never triggers before the NR step size has already exploded, so it
guards nothing in practice. The floor is a fallback trigger, not just a
division guard.

---

## Tradeoffs

### Forward: Parity-Implied vs. Externally Sourced r and q

| Approach | Rate input needed | Dividend input needed | Consistency with quotes |
|---|---|---|---|
| Parity-implied F, DF | No | No | Exact (by construction) |
| `F = S*exp((r-q)*T)` | Yes (term structure ideally) | Yes | Only as good as r, q |

Implying the forward from put-call parity is standard desk practice for
European index options: the market's own prices embed the financing and
dividend information, and using them keeps the IV surface internally
consistent with the quotes it is built from. The external-input approach is
retained only as a fallback for expiries with too few parity pairs, and its
dividend omission bias is logged when used.

### IV Solver: NR vs. Brent vs. Bisection

| Solver | Speed | Convergence Guarantee | Vega Required |
|---|---|---|---|
| Newton-Raphson | Fastest (quadratic) | No (fails at low vega) | Yes |
| Brent's method | Fast (superlinear) | Yes | No |
| Bisection | Slowest (linear) | Yes | No |

NR is used with a Manaster-Koehler seed (rather than a fixed 20% guess): the
seed is cheap, and starting from it NR converges monotonically for vanilla
prices, which materially reduces fallback frequency. Brent's method is the
fallback for deep OTM strikes where vega approaches zero and NR either
diverges or oscillates. Bisection is not used: Brent's is strictly superior
in both speed and convergence guarantees. The same seeding approach is used
in the companion C++ pricing engine (`ope`), keeping the two projects
consistent.

### Parametric Model: SVI vs. SABR vs. SSVI

| Model | Per-slice | Global | Arbitrage-free by construction | Common use |
|---|---|---|---|---|
| SVI | Yes | No | No (requires separate check) | Equity vol surfaces |
| SSVI | No | Yes | Calendar: yes, Butterfly: partial | Equity (production) |
| SABR | Yes | No | No | Rates, FX |

SVI is implemented here because it is the most common equity surface
parameterization at the slice level and the calibration mechanics are
transparent. SSVI (Gatheral-Jacquier 2014) performs the fit globally across all
maturities simultaneously, enforcing calendar no-arbitrage by construction, and
is the preferred production upgrade. SABR is standard in rates and FX where the
model dynamics matter for exotic hedging; for equity surfaces it is less common.

### Time Interpolation: Linear-in-Total-Variance vs. Bicubic Spline

Interpolating IV directly in the time dimension can produce calendar arbitrage
between fitted slices even when individual slices are clean. Interpolating
total variance `sigma^2 * T` linearly in time is the correct approach: it
preserves the non-decreasing structure required by no-arbitrage and is the
industry standard. A bicubic spline (e.g. `RectBivariateSpline`) is smoother
but can overshoot between knots and reintroduce calendar arbitrage, and it
requires at least four maturities per spline degree, a constraint this
pipeline cannot always guarantee after filtering. In the strike dimension no
interpolation is needed at all: SVI is analytic in `k`.

### Data Source: yfinance vs. Bloomberg/CBOE

`yfinance` is free and sufficient for a research prototype. It returns
delayed (15-20 minute) option chains with mid-prices that are acceptable for
surface construction when pulled during market hours. Known limitations:
bid-ask spread noise inflates IV at illiquid strikes, data is not tick-level
(no intraday surface dynamics), open interest is updated overnight, and
deep-wing quotes are frequently stale. yfinance is an unofficial scraping
library and can break without warning when Yahoo changes endpoints; the
snapshot cache limits the blast radius of an outage. In production, Bloomberg
`OVDV` or CBOE bulk data would replace this layer entirely; the rest of the
pipeline is data-source-agnostic.

---

## Known Limitations

- **Single discount factor per expiry**: the parity regression recovers one
  `DF` per maturity, which is exactly what the pipeline needs, but no
  continuous rate term structure is bootstrapped. The FRED fallback uses a
  single 3-month tenor for all maturities and ignores dividends, so
  fallback-priced slices carry a documented systematic bias.
- **Discrete dividends not modeled**: the implied forward absorbs expected
  dividends in aggregate, which is correct for European index options, but
  no discrete dividend schedule is modeled. Single-name surfaces around
  ex-dividend dates would require it.
- **yfinance data quality**: quotes are delayed 15-20 minutes, OTM far-wing
  quotes are frequently stale or crossed, and open interest lags by a day.
  The OI and bid filters and market-hours pull discipline mitigate but do
  not eliminate this.
- **SVI local minima**: differential evolution with `L-BFGS-B` polish is robust
  but not guaranteed to find the global minimum for every slice. Slices with
  very few liquid strikes (fewer than 5 quotes) may produce unreliable fits.
- **Calendar check granularity**: the pre-fit calendar check compares total
  variance in a narrow band around `k = 0` rather than across all `k`. The
  fitted surface enforces calendar monotonicity globally via
  linear-in-total-variance time interpolation, but raw-quote calendar
  violations away from the money are not individually flagged.
- **No American option adjustment**: SPX options are European; this pipeline is
  not correct for single-name American options without an early exercise premium
  adjustment. The parity-implied forward is also only exact for European
  exercise.
- **Static surface**: the pipeline produces an end-of-day snapshot. Intraday
  surface dynamics require a streaming data source and incremental recalibration
  architecture not implemented here.

---

## Requirements

Python 3.11+.

```bash
pip install -r requirements.txt
```

```
numpy==2.4.4
pandas==3.0.2
scipy==1.17.1
yfinance==1.5.1
plotly==6.8.0
matplotlib==3.10.8
requests==2.33.1
pytest==9.1.1
```

Pins reflect the current releases as of July 2026. Note that pandas 3.x and
numpy 2.x are major versions with behavior changes relative to the 2.x/1.x
lines; the code targets these pins, not older releases. yfinance is pinned
at 1.x, whose options API (`Ticker.options`, `option_chain()`) is unchanged
from 0.2.x.

---

## Usage

```bash
# Pull a fresh chain snapshot (run during market hours)
python -m src.data --snapshot

# Build and display the full surface (live pull, or --from-snapshot <path>)
python -m src.surface

# Run arbitrage checks only (no fit)
python -m src.arbitrage

# Run the test suite (offline; uses synthetic data only)
pytest tests/ -v
```

The test suite and the full pipeline downstream of Stage 1 run offline from
snapshots; only `src.data` requires network access (Yahoo, and FRED for the
fallback rate).

---

## Revision Log

- **r2 (2026-07)**: Pre-build spec audit fixes. Replaced external r/q inputs
  with per-expiry parity-implied forward and discount factor; moved pricing
  to Black-76 on the forward (fixes systematic call/put IV split from the
  omitted SPX dividend yield). Replaced equal-spacing butterfly check with
  the spacing-weighted convexity condition (SPX strike increments are
  irregular). Calendar check moved from fixed strike to fixed
  log-forward-moneyness. Removed the Zeliade "2+3" calibration claim (not
  implemented; now an upgrade path). Replaced bicubic surface spline with
  analytic SVI in strike and linear total-variance interpolation in time.
  NR seeded with Manaster-Koehler instead of fixed 0.2; vega floor raised
  from 1e-10 to 1e-4; NR convergence now checked before the parameter
  update. Restricted IV inversion and fits to OTM quotes. Updated dependency
  pins from the stale 2024 set to current releases. Added snapshot caching
  and market-hours pull guidance. Documented FRED key requirement, the
  keyless CSV endpoint, and the DTB3 discount-basis conversion.
- **r1**: Initial spec.

---

## References

- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility
  parameterization with application to the valuation of volatility derivatives.*
  Global Derivatives, Madrid.
- Gatheral, J. & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.*
  Quantitative Finance, 14(1), 59-71.
- Zeliade Systems (2012). *Quasi-explicit calibration of Gatheral's SVI model.*
  Zeliade White Paper. (Documented upgrade path; not implemented.)
- Lee, R. (2004). *The moment formula for implied volatility at extreme strikes.*
  Mathematical Finance, 14(3), 469-480.
- Manaster, S. & Koehler, G. (1982). *The calculation of implied variances
  from the Black-Scholes model: A note.* Journal of Finance, 37(1), 227-230.
- Black, F. (1976). *The pricing of commodity contracts.* Journal of
  Financial Economics, 3(1-2), 167-179.
- Hull, J. (2022). *Options, Futures, and Other Derivatives.* 11th ed. Pearson.
- Data sourced via [yfinance](https://github.com/ranaroussi/yfinance)
- Fallback risk-free rate sourced from [FRED](https://fred.stlouisfed.org/)
  (3-month Treasury; see spec for series and basis-conversion details)
