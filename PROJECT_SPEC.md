# Volatility Surface -- Project Specification (r2)

**Language**: Python 3.11+
**Stack**: numpy 2.4.x, pandas 3.0.x, scipy 1.17.x, yfinance 1.5.x,
plotly 6.8.x, matplotlib 3.10.x (see `requirements.txt` for exact pins;
pandas 3 and numpy 2 are major versions with behavior changes, target them
directly)
**Local path**: `C:\Users\rchrd\Downloads\projects\vol_surface\`
**Build order**: follow modules top to bottom. Each module has clear inputs,
outputs, and correctness checks before you proceed to the next.

---

## Architecture Overview

```
data.py  -->  iv_solver.py  -->  arbitrage.py  -->  svi.py  -->  surface.py  -->  viz.py
  |                |                  |                |              |               |
fetch, clean,   invert B76         check no-       fit SVI        analytic        render
imply F & DF    per (K, T)         arbitrage       per slice      slices +        + plots
per expiry                                                        T-interp
```

Each module is independently testable. Do not let upstream bugs silently
propagate: add an assertion or explicit check at the output of each stage
before calling the next.

**Design change from r1**: no external dividend yield input and no reliance
on an external rate for pricing. The forward `F` and discount factor `DF`
are implied per expiry from put-call parity (Module 1) and all pricing is
Black-76 on the forward (Module 2). The FRED rate is a fallback and
diagnostic only. Rationale: SPX carries a nonzero dividend yield; pricing
against `F = S*exp(r*T)` with no `q` systematically splits call and put IVs
and fails the parity sanity check by construction.

---

## Module 1: `src/data.py`

**Purpose**: pull, clean, and standardize the raw options chain; imply the
per-expiry forward and discount factor; provide snapshot caching.

---

### `get_risk_free_rate() -> float`

**What it does**: fetches the most recent 3-month Treasury yield from FRED
as a continuously compounded decimal. Used ONLY as the fallback when an
expiry has too few call-put pairs to imply the forward (see
`implied_forward`), and as a diagnostic to compare against implied
financing.

**How**: HTTP GET to the keyless FRED CSV endpoint
`https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO`. (The FRED JSON
API at `api.stlouisfed.org` requires a registered API key; the fredgraph CSV
download does not.) Parse the CSV, take the most recent non-missing
observation, divide by 100.

**Basis conversion (do not skip)**: DGS3MO is quoted on an investment
(bond-equivalent) basis. Convert the annualized simple yield `y` to a
continuously compounded rate before use: `r = log(1 + y * 0.25) / 0.25`.
Do NOT use series DTB3 without conversion: DTB3 is a discount-basis rate on
a 360-day year and understates the yield by a systematic margin.

Cache the result in-memory for the session (avoid repeated network calls on
reruns).

**Returns**: `float` -- annualized, continuously compounded risk-free rate.

**Correctness check**: assert return value is between 0.0 and 0.20. If FRED
is unreachable, fall back to `FALLBACK_RATE` from config and print a
warning. Do not raise silently. `FALLBACK_RATE` must be reviewed against
the current DGS3MO level whenever it is touched -- rates have been on an
easing path since 2024 and any hardcoded constant goes stale. Store the
constant in one place with a `# VERIFY against current DGS3MO` comment.

---

### `fetch_chain(ticker: str) -> pd.DataFrame`

**What it does**: downloads the full options chain for all available expiries
and combines them into a single flat DataFrame.

**How**: call `yf.Ticker(ticker).options` to get the list of expiry strings.
Loop over each expiry, call `.option_chain(expiry)` to get calls and puts
separately, add a column `option_type` ("call" or "put"), add a column
`expiry` (the expiry string converted to `pd.Timestamp`), then concatenate
all into one DataFrame. Snapshot the spot price in the same fetch session
and store it alongside (see `save_snapshot`); do not re-fetch spot later.

yfinance 1.x raises `YfRateLimitError` when rate limited; catch it, back
off, and retry a bounded number of times. Pull during market hours: quotes
are delayed 15-20 minutes, open interest updates overnight, and after-hours
bid/ask is frequently stale or crossed.

**Columns required in output**:
`strike`, `lastPrice`, `bid`, `ask`, `volume`, `openInterest`,
`impliedVolatility`, `option_type`, `expiry`

**Note**: `impliedVolatility` from yfinance is included but is NOT used in
downstream IV inversion -- it is kept only for sanity comparison. We compute
our own IV from bid/ask mid-price.

**Returns**: `pd.DataFrame` with columns above.

**Correctness check**: assert the DataFrame is non-empty. Assert all expiry
values are in the future relative to today. Print the number of rows and
unique expiries found.

---

### `save_snapshot(df: pd.DataFrame, spot: float, ticker: str) -> str` / `load_snapshot(path: str) -> tuple[pd.DataFrame, float, pd.Timestamp]`

**What it does**: persists a fetched chain plus spot and fetch timestamp to
`data/snapshots/{ticker}_{YYYYMMDD_HHMM}.csv` (spot and timestamp in a
header comment or sidecar row), and loads it back.

**Why**: reproducibility. Every downstream stage must be runnable from a
snapshot with zero network access. This also decouples development from
yfinance outages (it is an unofficial scraping library and breaks without
warning) and lets the test/dev environment work offline.

**Correctness check**: `load_snapshot(save_snapshot(df, ...))` roundtrips
exactly (same shape, same dtypes for the required columns).

---

### `clean_chain(df: pd.DataFrame, spot: float, config: dict) -> pd.DataFrame`

**What it does**: applies all filtering rules and computes derived columns.

**Filters to apply (in order)**:
1. Drop rows where `bid <= 0` (stale or crossed markets).
2. Drop rows where `openInterest < config['MIN_OI']`.
3. Compute `mid_price = (bid + ask) / 2`. Drop rows where `mid_price <= 0`.
4. Compute `moneyness = strike / spot`. Drop rows outside
   `[config['MONEYNESS_LO'], config['MONEYNESS_HI']]`.
5. Compute `days_to_expiry` as calendar days from the SNAPSHOT timestamp
   (not wall-clock "today", which breaks snapshot replay) to expiry.
   Drop rows where `days_to_expiry < config['MIN_EXPIRY_DAYS']` or
   `days_to_expiry > config['MAX_EXPIRY_DAYS']`.
6. Compute `T = days_to_expiry / 365.0` (time to expiry in years).

**Returns**: filtered `pd.DataFrame` with all original columns plus
`mid_price`, `moneyness`, `days_to_expiry`, `T`.

**Correctness check**: print rows dropped at each filter step. If fewer
than 3 strikes remain for any expiry after filtering, drop that expiry
entirely and log a warning (SVI requires at least 4-5 points to be
meaningful).

---

### `implied_forward(df_expiry: pd.DataFrame, spot: float, T: float, config: dict) -> tuple[float, float, dict]`

**What it does**: implies the forward `F` and discount factor `DF` for one
expiry from put-call parity. This replaces any external dividend yield and
is the pricing-critical rate input.

**Theory**: for European options, `C(K) - P(K) = DF * (F - K)` exactly.
The call-put mid spread is affine in strike with slope `-DF` and intercept
`DF * F`.

**Algorithm**:
1. Inner-join call and put rows on `strike` (pairs where BOTH sides
   survived cleaning).
2. Restrict pairs to `abs(strike/spot - 1) <= config['FWD_BAND']`
   (default 0.10) -- near-the-money pairs have the tightest spreads.
3. If fewer than `config['FWD_MIN_PAIRS']` (default 3) pairs remain:
   return the fallback `DF = exp(-r*T)`, `F = spot * exp(r*T)` with
   `r = get_risk_free_rate()`, and set `diag['fallback'] = True`. Log a
   warning: the fallback ignores dividends and biases this slice.
4. Otherwise OLS-fit `y = C_mid - P_mid` against `x = strike`:
   `DF = -slope`, `F = intercept / DF`.
5. Sanity: assert `0.7 < DF <= 1.005` and `0.8 < F/spot < 1.2`. If either
   fails, fall back as in step 3 and flag.
6. Compute parity residuals `resid = (C - P) - DF*(F - K)` per pair; store
   max and RMS in `diag`. Large residuals are data-quality flags on those
   strikes -- log them, do not silently discard.

**Returns**: `(F, DF, diag)` where `diag` holds `fallback`, `n_pairs`,
`resid_rms`, `resid_max`.

**Correctness check**: on synthetic Black-76 prices generated with known
`(F, DF)`, recovery error < 1e-8 in both. On real SPX data, the implied
financing rate `-log(DF)/T` should be within ~100bp of the FRED diagnostic
rate minus a plausible dividend yield; print the comparison per expiry.

---

### `attach_forwards(df: pd.DataFrame, spot: float, config: dict) -> pd.DataFrame`

**What it does**: runs `implied_forward` per expiry and attaches columns
`F`, `DF`, `fwd_fallback`, and `k = log(strike / F)` (log-forward-moneyness)
to every row. Then applies the OTM filter: keep puts with `strike <= F` and
calls with `strike >= F`. ITM mids carry wide spreads and produce noisy
IVs; OTM-only fitting is standard practice.

**Returns**: `pd.DataFrame` with the new columns, OTM rows only.

**Correctness check**: every expiry has exactly one `(F, DF)`; no row has
`k` of the wrong sign for its type (puts `k <= 0`, calls `k >= 0`, allow
`|k| < 1e-12` at the boundary).

---

## Module 2: `src/iv_solver.py`

**Purpose**: recover implied volatility from a market price for a single
option, using Black-76 on the forward with NR (Manaster-Koehler seed) and
Brent fallback.

---

### `b76_price(F: float, K: float, DF: float, T: float, sigma: float, option_type: str) -> float`

**What it does**: computes the Black-76 price for a European call or put on
the forward. This is the forward function fed to the root-finder.

**Implementation**:
```
d1 = (log(F/K) + 0.5*sigma^2*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
call = DF * (F*N(d1) - K*N(d2))
put  = call - DF*(F - K)      [put-call parity]
```

Use `scipy.stats.norm.cdf` for N(). For the put, apply put-call parity
rather than re-deriving the formula separately -- it guarantees internal
consistency.

**Edge cases**:
- If any of `F, K, DF, T, sigma` fails `np.isfinite`: raise `ValueError`.
  (`isfinite` catches inf and nan that a simple `> 0` comparison passes.)
- If `sigma <= 0` or `T <= 0` or `F <= 0` or `K <= 0` or `DF <= 0`:
  raise `ValueError`.

**Returns**: `float` -- option price.

---

### `b76_vega(F: float, K: float, DF: float, T: float, sigma: float) -> float`

**What it does**: computes the Black-76 vega (dPrice/dSigma). Shared by
calls and puts.

**Formula**:
```
d1 = (log(F/K) + 0.5*sigma^2*T) / (sigma*sqrt(T))
vega = DF * F * phi(d1) * sqrt(T)
```

where `phi` is the standard normal PDF (`scipy.stats.norm.pdf`).

**Returns**: `float` -- vega. Always non-negative.

---

### `mk_seed(F: float, K: float, T: float) -> float`

**What it does**: Manaster-Koehler (1982) starting value for NR. In forward
terms:
```
sigma_0 = sqrt(2 * abs(log(F/K)) / T)
```
Floor the result at 0.1 (the raw formula returns 0 at `K = F`, and very
small seeds waste iterations). Starting NR from the MK seed gives monotone
convergence for vanilla prices, which is why it replaces the fixed 0.2
guess used in r1. This mirrors the seeding already validated in the C++
engine (`ope::implied_vol`).

**Returns**: `float` -- initial sigma.

---

### `implied_vol_nr(market_price: float, F: float, K: float, DF: float, T: float, option_type: str, config: dict) -> float | None`

**What it does**: Newton-Raphson solver for implied volatility.

**Algorithm**:
1. Initialize `sigma = mk_seed(F, K, T)`.
2. Loop up to `config['NR_MAX_ITER']` times:
   a. Compute `price = b76_price(F, K, DF, T, sigma, option_type)`.
   b. **Convergence check BEFORE any update**: if
      `abs(price - market_price) < config['NR_TOL']`: return `sigma`.
      (r1 checked tolerance after the update, returning a post-update
      sigma against a pre-update price error -- an off-by-one. The
      returned sigma must be the one whose price was tested.)
   c. Compute `vega = b76_vega(F, K, DF, T, sigma)`.
   d. If `vega < config['VEGA_FLOOR']`: return `None` to signal Brent
      fallback (do NOT raise here, the caller handles it).
   e. Update: `sigma = sigma - (price - market_price) / vega`.
   f. If `sigma <= 0` or not `np.isfinite(sigma)`: return `None`
      (NR diverged).
3. If max iterations reached without convergence: return `None`.

**Note on `VEGA_FLOOR`**: default is 1e-4 (dollar vega), raised from r1's
1e-10. A floor near machine epsilon never fires before the NR step has
already exploded; the floor exists to hand off to Brent while the iterate
is still sane, not merely to guard the division.

**Returns**: `float` or `None`. None means Brent fallback is needed.

---

### `implied_vol_brent(market_price: float, F: float, K: float, DF: float, T: float, option_type: str) -> float`

**What it does**: Brent's method fallback. Used when NR fails (low vega,
divergence, non-convergence).

**Algorithm**: use `scipy.optimize.brentq` with:
- `f = lambda sigma: b76_price(F, K, DF, T, sigma, option_type) - market_price`
- Bracket: `[1e-6, 5.0]` (covers all realistic vols).

**Edge case**: if `f(1e-6)` and `f(5.0)` have the same sign (no root in
bracket, i.e., market price violates static bounds -- e.g. below
discounted intrinsic), return `np.nan` and log the offending row. Do not
raise.

**Returns**: `float` or `np.nan`.

---

### `solve_iv(row: pd.Series, config: dict) -> float`

**What it does**: combines NR and Brent for a single option row. This is
the function mapped across the cleaned, forward-attached chain DataFrame.
`F` and `DF` come from the row (attached in Module 1), not from function
arguments -- there is no spot or external rate anywhere in this module.

**Algorithm**:
1. Call `implied_vol_nr(row['mid_price'], row['F'], row['strike'], row['DF'], row['T'], row['option_type'], config)`.
2. If result is not `None` and is in `(0, 5.0)`: return it.
3. Otherwise call `implied_vol_brent(...)` with the same inputs.
4. Return result (may be `np.nan` if bound-violated).

**Returns**: `float` (sigma) or `np.nan`.

---

### `compute_iv_surface(df: pd.DataFrame, config: dict) -> pd.DataFrame`

**What it does**: applies `solve_iv` across the entire chain and returns a
DataFrame with a new `iv` column.

**How**: use `df.apply(lambda row: solve_iv(row, config), axis=1)`.
Add the result as column `iv`. Drop rows where `iv` is `np.nan` or `iv <= 0`.

**Correctness check**: after solving, log what fraction of rows converged
via NR vs. Brent fallback vs. NaN. Assert at least 80% of rows produced a
finite IV -- if less, something is wrong with the data or the implied
forwards. Additionally, for strikes where both an OTM quote and its parity
counterpart existed pre-filter, spot-check that call and put IVs at the
same (K, T) agree within tolerance when both are inverted; with a shared
parity-implied `(F, DF)` this holds near-exactly for clean quotes, so
breaches isolate bad quotes rather than model bias.

**Returns**: `pd.DataFrame` with all prior columns plus `iv`.

---

## Module 3: `src/arbitrage.py`

**Purpose**: flag butterfly and calendar arbitrage violations before fitting.

---

### `check_butterfly(df_slice: pd.DataFrame, expiry: str) -> pd.DataFrame`

**What it does**: two-tier butterfly (strike) arbitrage check for a
single expiry slice.

**Approach (discrete check, unequal spacing, r3 two-tier)**:
SPX strike spacing is irregular (5/10/25/50 point increments), so the
equal-spacing butterfly `C(K1) - 2*C(K2) + C(K3)` from r1 is NOT a valid
convexity test and would produce false flags and missed violations. On
each adjacent triple `K1 < K2 < K3` with `w = (K3 - K2)/(K3 - K1)`:

```
B_value = w*C_mid(K1) + (1-w)*C_mid(K3) - C_mid(K2)   [tier 1: diagnostic]
B_exec  = w*C_ask(K1) + (1-w)*C_ask(K3) - C_bid(K2)   [tier 2: exclusion]
```

Tier 1 (`B_value < -eps`, eps ~ 1e-8 for float noise) flags non-convex
mids. It is a DIAGNOSTIC only: near tight spacing the convexity chord
gap is the same order as quote noise (measured ~$0.37 at 25-pt spacing
ATM vs. comparable realistic noise), and a mid breach inside the
bid-ask cannot be monetized. Tier 2 (`B_exec < -eps`) is the EXCLUSION
trigger: the long fly is enterable at a credit against the touch (wings
bought at the ask, body sold at the bid), i.e. executable arbitrage.
Since `ask >= mid >= bid`, `B_exec >= B_value`, so executable
violations are a subset of tier-1 flags; return all tier-1 rows with an
`executable` boolean marking tier 2.

**Input**: DataFrame filtered to one expiry with columns `strike`,
`mid_price`, `bid`, `ask`, `option_type`, `F`, `DF`. Because only OTM
quotes are retained, build the call curve as: actual call quotes for
`K >= F`, and parity-synthesized quotes `C(K) = P(K) + DF*(F - K)`
applied to bid, mid, and ask alike for `K < F` (the forward leg is
implied from mids and carries no modeled spread -- documented
simplification, errs toward flagging). If bid/ask are absent, both
collapse to the mid and tier 2 degenerates to tier 1 (logged loudly).
Sort by strike, dedupe (prefer the actual call), then run the triple
check.

**Returns**: `pd.DataFrame` of tier-1 flagged (K1, K2, K3) triples with
`B_value`, `B_exec`, `executable`, and `expiry` columns. Empty DataFrame
means clean at both tiers.

**Exclusion policy (implemented in `exclude_flagged`)**: only
`executable=True` triples trigger removal of the middle strike K2 from
the SVI fit input; mid-only flags stay in the fit and in the diagnostics
CSV. A calendar-flagged short slice is dropped only when it violates
against 2+ longer maturities (a single pair cannot attribute fault).

---

### `check_calendar(df: pd.DataFrame) -> pd.DataFrame`

**What it does**: checks across maturities for calendar spread arbitrage.
Calendar arbitrage means total variance decreases as maturity increases at
fixed log-forward-moneyness.

**Approach**:
Compare at fixed `k = log(K/F)`, NOT fixed strike -- forwards differ across
expiries, so the same strike sits at different moneyness on different
slices (r1 compared at fixed strike; documented as incorrect). For rows
with `abs(k) <= 0.02` (ATM-forward band), compute
`total_variance = iv^2 * T`. Sort by `T`. Flag any (T_short, T_long) pair
where `total_variance(T_short) > total_variance(T_long)`.

**Known limitation (document, do not hide)**: this pre-fit check covers a
band around `k = 0` only. Global calendar monotonicity on the fitted
surface is enforced separately by linear-in-total-variance time
interpolation in Module 5.

**Returns**: `pd.DataFrame` of flagged (k_bucket, T_short, T_long) pairs.
Empty DataFrame means clean.

---

### `run_arbitrage_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`

**What it does**: runs both checks across all slices and returns combined
results.

**How**:
1. For butterfly: group by expiry, call `check_butterfly` on each group,
   concatenate results.
2. For calendar: call `check_calendar` on the full DataFrame.
3. Write both result DataFrames to `outputs/arbitrage_flags.csv`.
4. Print a summary: "N butterfly violations across M expiries,
   K calendar violations."

**Returns**: `(butterfly_flags_df, calendar_flags_df)`.

---

## Module 4: `src/svi.py`

**Purpose**: fit the SVI parametric model per expiry slice.

**Scope note**: this module implements the plain 5-parameter raw-SVI fit
with a global-then-local optimizer. The Zeliade (2012) quasi-explicit
"2+3" decomposition is an upgrade path, not part of this build -- do not
claim it in any documentation until it exists.

---

### `svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray`

**What it does**: evaluates the SVI raw parameterization for a vector of
log-forward-moneyness values.

**Formula**:
```
w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))
```

where `k = log(K/F)` and `w(k)` is total implied variance.

**Constraints the parameters must satisfy** (check before returning):
- `b >= 0`
- `|rho| < 1`
- `sigma > 0`
- `a + b*sigma*sqrt(1 - rho^2) >= 0` (ensures w(k) >= 0 everywhere)

If constraints are violated, return `np.full_like(k, np.inf)` -- this
makes the optimizer immediately penalize infeasible parameter sets.

**Returns**: `np.ndarray` of total variance values, same shape as `k`.

---

### `svi_loss(params: np.ndarray, k_market: np.ndarray, w_market: np.ndarray) -> float`

**What it does**: objective function for SVI calibration. Computes sum of
squared differences between model and market total variances.

**Formula**:
```
loss = sum((svi_total_variance(k, *params) - w_market)^2)
```

**Inputs**:
- `params`: array `[a, b, rho, m, sigma]`
- `k_market`: log-forward-moneyness array for the slice (from Module 1's
  `k` column -- do NOT recompute with a different forward)
- `w_market`: market total variance array (`iv^2 * T`)

**Returns**: `float` -- scalar loss.

---

### `calibrate_svi_slice(df_slice: pd.DataFrame) -> dict`

**What it does**: calibrates SVI to one expiry slice using a two-stage
global-then-local optimizer.

**Inputs**:
- `df_slice`: DataFrame for one expiry with columns `k`, `iv`, `T`, `F`,
  `DF` (forward already attached in Module 1; no forward computation
  happens here)

**Algorithm**:
1. Take `k = df_slice['k']` and `w = iv^2 * T` (T is constant within a
   slice).
2. **Stage 1 -- Global search**: call
   `scipy.optimize.differential_evolution(svi_loss, bounds=...)` with:
   - `a` bounds: `[-0.5, 0.5]`
   - `b` bounds: `[0.0, 2.0]`
   - `rho` bounds: `[-0.999, 0.999]`
   - `m` bounds: `[-1.0, 1.0]`
   - `sigma` bounds: `[1e-4, 1.0]`
   - `rng=config['SVI_GLOBAL_SEED']`, `maxiter=1000`, `tol=1e-8`
     (`rng` is the current scipy parameter name; `seed` still works in
     1.17 but is the legacy spelling)
3. **Stage 2 -- Local polish**: pass global result as `x0` to
   `scipy.optimize.minimize(method='L-BFGS-B', ...)` with same bounds.
4. Compute RMSE: `sqrt(svi_loss / N)` where N is number of strikes.
   Log a warning if RMSE > 0.005 (poor fit, fewer than 5 points, or
   optimizer diverged).
5. **Lee wing-slope check**: flag if `b*(1 + abs(rho)) > 2`. Lee's moment
   formula bounds the asymptotic slope of total variance in `|k|` by 2;
   a fitted slope above it implies the surface prices non-existent
   moments and will misbehave under wing extrapolation. Record the flag
   in the output dict; do not silently clip.

**Returns**: `dict` with keys
`{a, b, rho, m, sigma, rmse, lee_flag, expiry, T, F, DF}`.

---

### `calibrate_all_slices(df: pd.DataFrame) -> pd.DataFrame`

**What it does**: iterates over all expiries and calibrates SVI to each.

**How**:
1. Group `df` by `expiry`.
2. Call `calibrate_svi_slice(group)` for each (forwards ride along on the
   rows).
3. Collect results into a list of dicts and convert to DataFrame.
4. Sort by `T` (ascending).

**Returns**: `pd.DataFrame` with columns
`[expiry, T, F, DF, a, b, rho, m, sigma, rmse, lee_flag]` -- one row per
fitted slice.

---

## Module 5: `src/surface.py`

**Purpose**: build a continuous surface from fitted SVI slices.

**Design change from r1**: no `RectBivariateSpline`, no 2D spline of any
kind. SVI is analytic in `k`, so the strike dimension needs no
interpolation at all. In the time dimension, total variance is interpolated
LINEARLY between fitted slices: a cubic spline in T can overshoot between
knots and reintroduce calendar arbitrage that the individual slices do not
have, and bicubic fitting requires at least 4 maturities, which post-filter
data cannot always guarantee. (r1's README and spec contradicted each other
on this point; linear-in-total-variance is the resolution.)

---

### `evaluate_svi_grid(svi_params: pd.DataFrame, moneyness_grid: np.ndarray, spot: float) -> np.ndarray`

**What it does**: evaluates each fitted SVI slice on a common
spot-moneyness grid to produce a 2D IV matrix for plotting.

**Inputs**:
- `svi_params`: output of `calibrate_all_slices` (contains per-slice `F`)
- `moneyness_grid`: 1D array of K/S values (e.g., `np.linspace(0.7, 1.3, 200)`)
- `spot`: snapshot spot for converting moneyness to strikes

**Algorithm**:
For each row (expiry) in `svi_params`:
1. Compute `k = log(moneyness_grid * spot / F)` using that slice's
   implied `F`.
2. Evaluate `w = svi_total_variance(k, a, b, rho, m, sigma)`.
3. Recover `iv_grid = sqrt(w / T)`.
4. Store as a row in the output matrix.

**Returns**: 2D `np.ndarray` of shape `(n_expiries, n_strikes)` containing
IV values in decimal form.

---

### `build_interpolator(svi_params: pd.DataFrame, spot: float) -> Callable`

**What it does**: returns a query function `interp(T, K) -> iv` built
directly on the fitted slices.

**Algorithm for a query (T, K)**:
1. If `T` is outside `[T_min, T_max]` of the fitted slices: clamp to the
   nearest slice and log a warning once per session. Do not extrapolate
   in time.
2. Find bracketing fitted maturities `T_i <= T <= T_j`.
3. Interpolate the forward: `log F(T)` linear in `T` between
   `log F_i` and `log F_j`.
4. Compute `k = log(K / F(T))`.
5. Evaluate both bracketing slices analytically at this `k`:
   `w_i = SVI_i(k)`, `w_j = SVI_j(k)`.
6. Interpolate total variance linearly in `T`:
   `w = w_i + (w_j - w_i) * (T - T_i)/(T_j - T_i)`.
7. Return `iv = sqrt(w / T)`.

Linear interpolation of total variance between slices that individually
satisfy `w_i(k) <= w_j(k)` preserves calendar monotonicity at every
intermediate `T` -- this is the no-arbitrage argument for the design.

**Returns**: a callable `interp(T, K) -> float`.

---

### `query_surface(interp, T: float, K: float) -> float`

**What it does**: convenience wrapper to query the surface at a single
(K, T) point.

**Returns**: `float` -- implied volatility at that point.

---

## Module 6: `src/viz.py`

**Purpose**: render all outputs.

---

### `plot_3d_surface(iv_matrix: np.ndarray, moneyness_grid: np.ndarray, T_grid: np.ndarray, output_path: str)`

**What it does**: renders an interactive 3D Plotly surface and saves it
to an HTML file.

**Axes**:
- X: moneyness (K/S)
- Y: days to expiry (`T_grid * 365`)
- Z: implied volatility in % (`iv_matrix * 100`)

**Styling**: use a diverging colorscale (e.g., `RdYlGn_r`) so the skew
is immediately visible. Add axis labels and a title showing the ticker
and snapshot date.

**Output**: saves to `output_path` (default: `outputs/vol_surface.html`).

---

### `plot_smile_slices(df: pd.DataFrame, svi_params: pd.DataFrame, output_dir: str)`

**What it does**: for each expiry, plots raw market IVs (scatter) against
the fitted SVI curve (line).

**One plot per expiry**. X-axis: log-forward-moneyness `k` (annotate a
secondary K/S axis if desired), Y-axis: IV (%). Label each plot with
expiry date, days to expiry, SVI RMSE, and whether the forward was
parity-implied or fallback. Save to `output_dir/smile_{expiry}.png`.

---

### `plot_term_structure(df: pd.DataFrame, output_path: str)`

**What it does**: plots ATM-forward implied volatility against days to
expiry.

**How**: for each expiry, find the row with `abs(k)` minimized (k is
log-forward-moneyness; this is ATM-forward, not spot-ATM), take its `iv`.
Plot (days_to_expiry, iv*100) as a line with point markers.

**Output**: saves to `output_path` (default: `outputs/term_structure.png`).

---

## Module 7: `tests/`

All tests run offline on synthetic data. No test may touch the network.

### `tests/test_data.py`

| Test | What to verify |
|---|---|
| `test_forward_recovery` | On synthetic Black-76 prices with known (F, DF), `implied_forward` recovers both within 1e-8 |
| `test_forward_fallback` | Fewer than FWD_MIN_PAIRS pairs triggers the rate-based fallback and sets the diag flag |
| `test_forward_sanity_reject` | Corrupt prices producing DF outside (0.7, 1.005] trigger fallback, not a bad forward |
| `test_snapshot_roundtrip` | `load_snapshot(save_snapshot(df))` preserves shape and required dtypes |
| `test_rate_basis_conversion` | Simple yield y converts to continuous r = log(1 + y/4)/0.25 correctly |

### `tests/test_iv_solver.py`

| Test | What to verify |
|---|---|
| `test_roundtrip_call` | `solve_iv` recovers sigma within 1e-6 when given a B76-generated price |
| `test_roundtrip_put` | Same for puts |
| `test_mk_seed` | Seed equals sqrt(2*abs(log(F/K))/T), floored at 0.1 at ATM |
| `test_nr_convergence_order` | NR returns the sigma whose price met tolerance (no post-update return) |
| `test_nr_fallback` | Deep OTM option triggers Brent fallback (NR returns None), final IV is still correct |
| `test_parity` | Call and put IVs at same (K, T) with shared (F, DF) agree within 1e-6 on exact prices |
| `test_hull_value` | Verify against a known Hull reference value (map BS inputs to B76 via F = S*exp(rT), DF = exp(-rT)) |
| `test_below_intrinsic` | `solve_iv` returns `np.nan` when market price < discounted intrinsic |
| `test_nonfinite_inputs` | inf/nan in any input raises ValueError (isfinite guard) |

### `tests/test_arbitrage.py`

| Test | What to verify |
|---|---|
| `test_clean_slice_no_butterfly` | Synthetic convex call prices on UNEQUALLY spaced strikes produce empty flags |
| `test_butterfly_violation` | Manually inject non-convexity, check flag is raised |
| `test_equal_spacing_formula_would_misfire` | A convex curve on spacing (5, 25) that the r1 unweighted formula flags is passed clean by the weighted check |
| `test_synthesized_calls` | Puts converted via C = P + DF*(F-K) match direct call prices |
| `test_synth_call_bid_ask_offset` | Parity offset applied to bid and ask alike; spread ordering survives |
| `test_exec_dominates_mid` | B_exec >= B_value on every triple (executable is a subset of mid flags) |
| `test_mid_flag_inside_spread_not_excluded` | Breach above the chord gap but inside the spread: tier-1 flag, executable False, strike kept |
| `test_violation_beyond_spread_excluded` | Breach beyond spread coverage: executable True, strike excluded |
| `test_degenerate_without_quotes_matches_strict` | Missing bid/ask collapses tier 2 to tier 1, logged |
| `test_clean_calendar` | Increasing total variance across maturities at fixed k produces empty flags |
| `test_calendar_violation` | Manually inject decreasing total variance at fixed k, check flag is raised |

### `tests/test_svi.py`

| Test | What to verify |
|---|---|
| `test_parameter_recovery` | Generate synthetic smile from known params, calibrate, recover params within 1e-3 |
| `test_nonnegative_variance` | SVI total variance is >= 0 for all k in [-2, 2] |
| `test_constraint_violation` | `svi_total_variance` returns `inf` when `|rho| >= 1` |
| `test_rmse_threshold` | RMSE on a well-behaved smile is below 0.005 |
| `test_lee_flag` | Params with b*(1+|rho|) > 2 set lee_flag |
| `test_calendar_preserved_by_interp` | Two clean slices linearly interpolated in w produce no calendar violation at any intermediate T |

---

## Build Order Checklist

Complete each item before moving to the next module.

```
[ ] Module 1: data.py
    [ ] get_risk_free_rate() passes rate range assertion, basis conversion tested
    [ ] fetch_chain() returns non-empty DataFrame with required columns
    [ ] save_snapshot()/load_snapshot() roundtrip exactly
    [ ] clean_chain() prints drop counts at each filter step
    [ ] implied_forward() recovers synthetic (F, DF) within 1e-8
    [ ] Implied financing vs. FRED diagnostic printed per expiry on real data
    [ ] At least 4 expiries survive filtering for SPX

[ ] Module 2: iv_solver.py
    [ ] b76_price() put-call parity holds to 1e-10
    [ ] b76_vega() is always non-negative
    [ ] implied_vol_nr() roundtrip error < 1e-6 on ATM option, MK-seeded
    [ ] Convergence checked before update (no off-by-one)
    [ ] implied_vol_brent() handles bracket failure gracefully (returns nan)
    [ ] compute_iv_surface() > 80% finite IV rate on clean SPX data

[ ] Module 3: arbitrage.py
    [ ] Weighted butterfly check passes convex synthetic data on unequal spacing
    [ ] Both checks run without error on real SPX data
    [ ] arbitrage_flags.csv written to outputs/
    [ ] At least one synthetic violation detected in tests

[ ] Module 4: svi.py
    [ ] svi_total_variance() returns inf for invalid params
    [ ] calibrate_svi_slice() logs RMSE for every expiry
    [ ] Parameter recovery test passes within 1e-3
    [ ] Lee wing-slope flag exercised in tests
    [ ] No uncaught exceptions on any SPX expiry slice

[ ] Module 5: surface.py
    [ ] iv_matrix shape is (n_expiries, n_strikes)
    [ ] Interpolator returns finite values inside the fitted T range
    [ ] Out-of-range T clamps with warning, does not extrapolate
    [ ] Calendar no-arbitrage holds at intermediate T (test_calendar_preserved_by_interp)

[ ] Module 6: viz.py
    [ ] vol_surface.html opens in browser and renders correctly
    [ ] One smile plot per expiry saved to outputs/smiles/
    [ ] Term structure plot uses k=0 (ATM-forward) and shows expected shape for SPX

[ ] Tests
    [ ] pytest tests/ -v passes with 0 failures, fully offline
    [ ] All roundtrip tests pass at stated tolerance
```

---

## Environment Notes

- **Version pins are current as of July 2026** (numpy 2.4.4, pandas 3.0.2,
  scipy 1.17.1, yfinance 1.5.1, plotly 6.8.0, matplotlib 3.10.8). The r1
  pins were ~2 years stale; pandas 3 and numpy 2 are majors with behavior
  changes, so code targets the new pins directly rather than straddling.
- **yfinance 1.x**: options API unchanged from 0.2.x; adds
  `YfRateLimitError` and an optional retry mechanism. It remains an
  unofficial scraper that can break without notice -- the snapshot layer is
  the mitigation.
- **FRED JSON API requires an API key**; this project uses the keyless
  `fredgraph.csv` endpoint instead, so no key management is needed.
- **Offline development**: everything downstream of `fetch_chain` /
  `get_risk_free_rate` must run from a snapshot. The test suite is fully
  offline by design.

---

## Upgrade Paths (Post-MVP, Interview-Relevant)

| Upgrade | Why it matters |
|---|---|
| Zeliade "2+3" quasi-explicit SVI calibration | Reduces the 5-parameter fit to a 2-parameter outer loop with a convex inner problem; faster and more stable than DE + polish |
| SSVI global calibration | Enforces calendar no-arbitrage by construction; preferred production parameterization (Gatheral-Jacquier 2014) |
| Bootstrapped rate term structure | Improves the fallback path and enables pricing beyond quoted expiries |
| Robust forward regression (Theil-Sen / WLS by spread) | OLS parity regression is sensitive to a single bad quote; robust estimators harden it |
| Dupire local vol extraction | `sigma_local(K, T)^2 = dw/dT / (...)` -- required for barrier and American option pricing |
| Heston calibration to surface | Full stochastic vol dynamics; needed for exotic pricing with vol-of-vol sensitivity |
| Rolling surface (intraday) | Streaming architecture with WebSocket data and incremental SVI recalibration per slice |
| Dispersion trade signal | Implied index vol vs. weighted sum of single-name vols; natural convergence of this project and the correlation heatmap |
