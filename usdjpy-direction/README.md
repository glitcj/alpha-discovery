# USDJPY forward-return prediction (LightGBM)

Predict USDJPY from Yahoo Finance 1h bars (`yfinance`, `USDJPY=X`) with a leak-free walk-forward
evaluation. The project mirrors the experiment pipeline of `delivery-time-prediction`: versioned experiment
modules → one `execute()` call → leaderboard / diagnostics / experiments → refit → artefacts →
pre/post-analysis report.

There are two experiment arms:

| arm | objective | selection |
|---|---|---|
| `v002_fwd_logret_1d` (default) | **1-day forward log return**, `y_t = log(close[t+25] / close[t+1])`. This is the same target as ml-pipe's `v001_logreturn_exp_v19_legacy_test` (`FWDLogReturn(48, 1)` on 30-min bars), regression | Spearman |
| `v001_lgbm_walkforward` | next-hour direction, `close[t+1] > close[t]`, classification | ROC AUC |

## Run it

Everything runs in Docker (`make help` lists the targets):

```bash
make build          # image with numpy/pandas/sklearn/lightgbm/yfinance/jupyter
make fetch          # download ~2.8 years of 1h bars -> dataset/usdjpy_1h.csv (the only network step)
make test           # unit tests on synthetic random-walk bars (no network)
make train          # v002 end-to-end -> outputs/ (leaderboard, holdout score, predictions.csv, model.joblib)
make train EXP=v001_lgbm_walkforward   # the next-hour direction arm
make predict        # forecast from the latest cached bar with the saved model
make experiments    # experiment ladder (TASK=return|direction) -> outputs/experiments.json
make run-report     # re-execute this notebook; `make report` exports REPORT.html; `make readme` refreshes README.md
```

Options: `EXP=<module>`, `MODELS=persistence,ridge,lgbm_reg` (empty = the arm's set), `CV=5`, `SEED=0`,
`COST=0.5` (bp per unit position change).

## Layout

| path | role |
|---|---|
| `usdjpy/data.py` | yfinance download + CSV cache, targets (`make_target`, `make_forward_return_target`), purged chronological holdout |
| `usdjpy/features.py` | `BarCleaner` (optional FX-week grid), `BarFeatureEngineer` (v001), `MultiWindowFeatureEngineer` (ml-pipe feature set), encoders |
| `usdjpy/models.py` | direction candidates (`majority`, `momentum`, `logreg`, `lgbm`), return regressors (`persistence`, `mean`, `best_feature`, `random`, `ridge`, `lgbm_reg`) |
| `usdjpy/evaluation.py` | `DirectionEvaluator` / `ReturnEvaluator` (walk-forward `TimeSeriesSplit` + gap), metrics, slices, displays |
| `usdjpy/pipeline/classification.py` | `execute()`: the 10 phases, task-aware |
| `usdjpy/analysis/` | builders, display helpers, pre/post-analysis reporters |
| `usdjpy/exp/` | the experiment arms (copy the latest to `v00N_*.py` to change one thing) |
| `usdjpy/train.py`, `usdjpy/experiments.py` | CLI and experiment ladder |

## Method (v002, aligned with ml-pipe)

| | ml-pipe `v001_logreturn_exp_v19_legacy_test` | here (`v002_fwd_logret_1d`) |
|---|---|---|
| bars | 30 min, FX week Sun 17:00 → Fri 16:59 New York | 1h (Yahoo keeps 30-min history for 60 days only), same FX-week grid, missing hours forward-filled |
| target | `FWDLogReturn(lookahead_window=48, lookahead_offset=1)` | `log(close[t+25] / close[t+1])`: the same 1-day window after a 1-bar buffer |
| features | 18 `tsf` types × windows 1.5h…63d × shifts {0, 1 day}, over a selected cross-asset universe | the same 18 types and clock windows (2, 6, 12, 24, 120, 504, 1512 bars) × shifts {0, 23}, **USDJPY only**: 234 features |
| model | feature selection → sparse mixture-of-experts net, robust-scaled/clipped target | `LGBMRegressor` on a robust-scaled, ±5-clipped target |
| baselines | persistence, mean, best_feature, random_sample | persistence, mean, best_feature, random, + ridge |
| validation | rolling 1.5y train, 16-day skip, 6-month roll | 5-fold walk-forward, 16-day gap (274 bars); rolling 1.5y window as an experiment; last 15% held out, purged by 25 bars |
| metrics | Spearman, Pearson, MSE, R², MCC, … | Spearman (selection), Pearson, R², MSE, MCC, AUC/hit rate of the sign, daily sign trade on non-overlapping labels |

**Leakage guards.**
- Every feature is invariant to truncating or shocking the future (tested).
- A random walk scores Spearman ≈ 0 through the evaluator.
- No development label overlaps the holdout or a test fold.

The cells below re-run v002 and render the full report.

## Experiment ladder

`make experiments` evaluates a wider set of v002 variants on the same development rows and walk-forward
folds (holdout excluded). The table is read from `outputs/experiments.json`.

## Findings (data 2024-01-15 → 2026-10-28)

### v002: 1-day forward log return (ml-pipe objective)

* **No model predicts the 1-day return.** Out-of-fold Spearman: `lgbm_reg` −0.013, `ridge` −0.037,
  `persistence` −0.060, `best_feature` −0.063, `mean` −0.068, `random` −0.003. The labels overlap by
  24 bars, so 11k hourly rows hold only ~470 independent days, and Spearman's noise is about ±0.05.
  Every model is inside that band, and the leaderboard winner is the `random` baseline.
* **Variants don't change that.** Across 14 variants in the ladder, the range is −0.068 to +0.013:
  library-default LightGBM, a huber loss, short-only or long-only windows, shift-0 only, pct_change only,
  adding calendar features, a rolling 1.5-year window, and two ridge strengths. The best single feature
  in-sample (5-day skew/RSI/momentum, |ρ| ≈ 0.07) does not carry over out of sample.
* **The holdout agrees.** On Jun–Oct 2026, Spearman is ≈ −0.03 and the sign hit rate ≈ 0.50.
* **Comparison with ml-pipe.** The ml-pipe arm gets its edge, if any, from cross-asset inputs selected
  over a 40-asset universe and from 4 more years of history. Neither exists here: we have USDJPY only and
  ~2.8 years on Yahoo. A null result on USDJPY-only features is the expected baseline for that comparison.
* **Data sanity.** Labels match the raw closes. No feature is NaN or constant after warm-up. About 2%
  of FX-week hours are forward-filled gaps.

### v001: next-hour direction (kept for reference)

* LightGBM CV AUC 0.519 (holdout 0.558). Most of the signal is time-of-day; it is too small to survive
  0.5 bp costs.

## Next arms

* `v003`: add cross-asset inputs, the main difference from ml-pipe. Pull 1h bars for EURUSD, GBPJPY,
  AUDJPY, DXY, ^N225, ^GSPC and ^TNX, align them to the USDJPY grid, and build the same multi-window
  features on each.
* `v004`: ml-pipe-style feature selection (stability-ranked Spearman / Pearson, top-N grid) before
  LightGBM, once there are thousands of cross-asset features.
* For more history or 30-min bars, switch to a documented data source (see
  `planning-1hour-price-direction-classification.md`) or the Tiingo/Sharadar parquet files ml-pipe uses.
