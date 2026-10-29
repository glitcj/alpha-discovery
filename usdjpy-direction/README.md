# USDJPY forward-return prediction (LightGBM)

Predict USDJPY from Yahoo Finance 1h bars (`yfinance`, `USDJPY=X`) with a leak-free walk-forward
evaluation. The project mirrors the experiment pipeline of `delivery-time-prediction`: versioned experiment
modules → one `execute()` call → leaderboard / diagnostics / experiments → refit → artefacts →
pre/post-analysis report.

There are three experiment arms; each changes one thing on top of the previous one:

| arm | objective | inputs | selection |
|---|---|---|---|
| `v003_cross_asset` (default) | **1-day forward log return**, `y_t = log(close[t+25] / close[t+1])` — the target of ml-pipe's `v001_logreturn_exp_v19_legacy_test` (`FWDLogReturn(48, 1)` on 30-min bars), regression | USDJPY + a 30-asset yfinance basket (FX, US yields, equity indices, VIX, commodities, BTC), ~7k features | Spearman |
| `v002_fwd_logret_1d` | same | USDJPY only, 234 features | Spearman |
| `v001_lgbm_walkforward` | next-hour direction, `close[t+1] > close[t]`, classification | USDJPY only | ROC AUC |

## Run it

Everything runs in Docker (`make help` lists the targets):

```bash
make build          # image with numpy/pandas/sklearn/lightgbm/yfinance/jupyter
make fetch          # ~2.8 years of 1h bars for USDJPY + the 30-asset basket -> dataset/ (the only network step)
make test           # unit tests on synthetic random-walk bars (no network)
make train          # v003 end-to-end -> outputs/ (leaderboard, holdout score, predictions.csv, model.joblib)
make train EXP=v002_fwd_logret_1d      # USDJPY-only features;  EXP=v001_lgbm_walkforward: next-hour direction
make predict        # forecast from the latest cached bar with the saved model
make experiments    # experiment ladder (TASK=return|direction; return = v003 data) -> outputs/experiments.json
make run-report     # re-execute this notebook; `make report` exports REPORT.html; `make readme` refreshes README.md
```

Options: `EXP=<module>`, `MODELS=persistence,ridge,lgbm_reg` (empty = the arm's set), `CV=5`, `SEED=0`,
`COST=0.5` (bp per unit position change).

## Layout

| path | role |
|---|---|
| `usdjpy/data.py` | yfinance download + CSV cache for USDJPY and the `BASKET`, `align_basket` (as-of join onto the FX-week grid + staleness), targets, purged chronological holdout |
| `usdjpy/features.py` | `BarCleaner` (optional FX-week grid), `BarFeatureEngineer` (v001), `MultiWindowFeatureEngineer` (ml-pipe feature set, per asset), `ColumnSubset` (by asset / regex), encoders |
| `usdjpy/models.py` | direction candidates (`majority`, `momentum`, `logreg`, `lgbm`), return regressors (`persistence`, `mean`, `best_feature`, `random`, `ridge`, `lgbm_reg`) |
| `usdjpy/evaluation.py` | `DirectionEvaluator` / `ReturnEvaluator` (walk-forward `TimeSeriesSplit` + gap), ml-pipe's metric set + `rank_score`, slices (incl. market regime), displays |
| `usdjpy/pipeline/classification.py` | `execute()`: the 10 phases, task-aware |
| `usdjpy/analysis/` | builders, display helpers, pre/post-analysis reporters (asset / feature-type importance, generalization gap, residual diagnostics) |
| `usdjpy/exp/` | the experiment arms (copy the latest to `v00N_*.py` to change one thing) |
| `usdjpy/train.py`, `usdjpy/experiments.py` | CLI and experiment ladder |

## Method (v003, aligned with ml-pipe)

| | ml-pipe `v001_logreturn_exp_v19_legacy_test` | here (`v003_cross_asset`) |
|---|---|---|
| bars | 30 min, FX week Sun 17:00 → Fri 16:59 New York | 1h (Yahoo keeps 30-min history for 60 days only), same FX-week grid, missing hours forward-filled |
| target | `FWDLogReturn(lookahead_window=48, lookahead_offset=1)` | `log(close[t+25] / close[t+1])`: the same 1-day window after a 1-bar buffer |
| universe | USDJPY + ~40 Tiingo/Sharadar series (FX, macro), non-FX series forward-filled onto the FX grid, then asset / feature-type selection by Spearman importance | USDJPY + 30 yfinance assets (14 FX crosses, DXY, US 13w/5y/10y/30y, N225/SPX/VIX/FTSE/DAX/HSI, gold/oil/silver/copper, BTC), as-of joined onto the FX-week grid with an hours-since-last-print column; no selection yet (v004) |
| features | 18 `tsf` types × windows 1.5h…63d × shifts {0, 1 day} per asset | the same 18 types and clock windows (2, 6, 12, 24, 120, 504, 1512 bars) × shifts {0, 23} per asset: ~7.3k features |
| model | feature selection → sparse mixture-of-experts net, robust-scaled/clipped target | `LGBMRegressor` (30% column sampling per tree) on a robust-scaled, ±5-clipped target |
| baselines | persistence, mean, best_feature, random_sample | persistence, mean, best_feature, random, + ridge |
| validation | rolling 1.5y train, 16-day skip, 6-month roll | 5-fold walk-forward, 16-day gap (274 bars); rolling 1.5y window as an experiment; last 15% held out, purged by 25 bars |
| metrics | MSE, MAE, R², Pearson, Spearman, Kendall, F1, ROC AUC, MCC (+ weighted τ, Wasserstein, precision/recall, sign-strategy Sharpe); leaderboard = weighted rank score | the same set: Spearman (selection), Pearson, Kendall, weighted τ, R², MSE, MAE, Wasserstein, MCC, F1, precision, recall, ROC AUC, binary accuracy, ml-pipe's `sharpe_ratio`, plus our daily non-overlapping sign trade and ml-pipe's `rank_score` |
| post-analysis | generalization gap (in- vs out-of-sample), rolling metrics, regime slices, residual report (ACF, trend, heteroskedasticity, normality) | the same: generalization gap on the 80/20 fit, rolling Spearman, trend × volatility regime slices, residual ACF / trend / rolling std / Jarque–Bera / ARCH-LM |

**Leakage guards.**
- Every feature is invariant to truncating or shocking the future (tested).
- A random walk scores Spearman ≈ 0 through the evaluator.
- No development label overlaps the holdout or a test fold.

The cells below re-run v003 and render the full report.

## Experiment ladder

`make experiments` evaluates a wider set of variants on v003's development rows and walk-forward folds
(holdout excluded): the v002 model variants plus asset-group ablations. The table is read from
`outputs/experiments.json`.

## Findings (data 2024-01-15 → 2026-10-29)

### v003: cross-asset inputs (the main gap vs. ml-pipe)

* **Cross-asset features are the first thing that moves the needle.** Out-of-fold Spearman for
  `lgbm_reg` is **+0.060** (holdout Jun–Oct 2026: **+0.067**, 50.4% sign accuracy, +3.2 bp/day on the
  daily sign trade after costs). The USDJPY-only control on the *same folds* scores −0.016, i.e. the
  entire gain comes from the basket. `ridge` on the same inputs reaches +0.044, so it is not a
  LightGBM artefact. ml-pipe's `rank_score` agrees with the Spearman ordering.
* **It is not yet a robust result.** Per-fold Spearman is 0.05 / 0.13 / 0.23 / 0.03 / −0.08: most of
  the signal sits in 2025 and the most recent fold is negative. With 5 folds that is a t-statistic of
  about 1.6. Treat it as "worth pursuing", not as an edge. The rolling 1.5-year-window variant
  (ml-pipe's training schedule) gives the same +0.061.
* **Seeds agree on the ranking, not on the P&L.** Re-running `lgbm_reg` with seeds 1 and 2 (30% column
  sampling makes seeds matter) gives fold patterns 0.07/0.14/0.15/0.04/−0.05 and 0.09/0.09/0.18/0.02/0.00
  and holdout Spearman +0.048 / +0.082 (seed 0: +0.067). The daily sign trade on the holdout, though,
  is +3.2 / −3.6 / −1.4 bp per day across the three seeds: the rank correlation is reproducible, a
  tradeable signal is not.
* **Where it comes from.** FX crosses alone give +0.040, non-FX assets alone +0.019, both together
  +0.060 (so they are complementary). Feature importance is diffuse: no column carries more than 1.3%
  of the gain, and the top groups are rolling kurtosis / skew / weighted volatility of JPY crosses,
  Hang Seng, Nikkei, DAX and US 5y/13w yields, mostly at the 5-day and 21-day windows and often at
  the 1-day-ago shift. That diffuseness is exactly what ml-pipe's feature-selection stage is for (v004).
* **Ladder (22 variants, same folds).** Keeping only the 5-day / 21-day / 63-day windows is the best
  variant at **+0.085** (fold std 0.085); the ≤ 1-day windows alone are *negative* (−0.043), so the
  signal is slow cross-market positioning, not hourly co-movement. The basket *without* USDJPY still
  scores +0.066 and USDJPY alone −0.016. A Huber objective gives +0.066 with the tightest fold spread
  (0.057). By group: equity/vol indices +0.053, FX +0.040, rates +0.015, commodities +0.004. Removing the
  staleness columns, adding calendar columns, 10% column sampling and library-default LightGBM all land
  between +0.04 and +0.06; ridge +0.044.
* **The model memorises.** On the chronological 80/20 diagnostic fit the in-sample Spearman is 0.94
  and the out-of-sample Spearman −0.02 (generalization gap ≈ 0.95; that 20% block is the same
  Jan–Jun 2026 period as the negative fold). 400 trees over 7k columns fit the training set almost
  perfectly, and the confidence slices confirm the magnitudes mean little: the largest |predictions|
  hit 48%, the smallest 51–55%. Residuals show the expected overlapping-label autocorrelation
  (lag-1 ≈ 0.96) and ARCH effects (p ≈ 0). All of this argues for far fewer, selected features and
  stronger shrinkage (v004) rather than more trees.
* **Leakage checks.** Basket series are as-of joined on bar *start* stamps, so an exchange bar stamped
  14:30 (closing 15:30) is the latest print visible at the 15:00 USDJPY bar, whose own close is the
  16:00 price; nothing from after the USDJPY close enters a feature. Random-walk inputs still score
  ≈ 0 through the whole pipeline (tests), every feature is invariant to truncating or shocking the
  future, and the holdout is purged by 25 bars.

### v002: USDJPY-only, same objective

* No model predicts the 1-day return from USDJPY's own history: all within ±0.05 Spearman noise, the
  `random` baseline wins, the 14-variant ladder spans −0.07 to +0.01.

### v001: next-hour direction (kept for reference)

* LightGBM CV AUC 0.519 (holdout 0.558); mostly time-of-day; too small to survive 0.5 bp costs.

## Next arms

* `v004`: ml-pipe-style selection. Asset / feature-type pre-selection on the development rows by mean
  |Spearman| (the tables in the pre-analysis), then a stability-ranked (rolling 63-day Spearman /
  Pearson, EWMA-weighted) top-N feature selector *inside* the pipeline with N chosen by inner
  walk-forward, plus a |ρ| > 0.8 multicollinearity filter. The ladder says where to start: slow
  windows (≥ 5 days), equity + FX groups, Huber loss. Goal: keep the +0.06–0.09 while cutting 7k
  columns to a few hundred, which should tighten the fold spread and close the 0.95 generalization gap.
* `v005`: ml-pipe's exact retrain schedule (1.5-year training window, 16-day skip, 6-month roll,
  out-of-sample pooled) so the numbers line up with ml-pipe's reports one-to-one.
* Data: the free Yahoo feed caps us at ~2.8 years and hourly bars; 5 folds on that is why the fold
  spread is wide. The Tiingo / Sharadar parquet files ml-pipe uses (or a documented API, see
  `planning-1hour-price-direction-classification.md`) would add 4 years and 30-min bars.
