# USDJPY 1-hour direction classification (LightGBM)

Predict whether the **next hourly close of USDJPY is above the current close**, from Yahoo Finance
1h bars (`yfinance`, `USDJPY=X`), with a leak-free walk-forward evaluation. The project mirrors the
experiment pipeline of `delivery-time-prediction`: versioned experiment modules → one `execute()`
call → leaderboard / diagnostics / experiments → refit → artefacts → pre/post-analysis report.

## Run it

Everything runs in Docker (`make help` lists the targets):

```bash
make build          # image with numpy/pandas/sklearn/lightgbm/yfinance/jupyter
make fetch          # download ~2 years of 1h bars -> dataset/usdjpy_1h.csv (the only network step)
make test           # unit tests on synthetic random-walk bars (no network)
make train          # v001 end-to-end -> outputs/ (leaderboard, holdout score, predictions.csv, model.joblib)
make predict        # P(up) for the hour after the latest cached bar
make experiments    # experiment ladder -> outputs/experiments.json
make run-report     # re-execute this notebook; `make report` exports REPORT.html; `make readme` refreshes README.md
```

Options: `EXP=<module>`, `MODELS=majority,logreg,lgbm`, `CV=5`, `SEED=0`, `COST=0.5` (bp per unit position change).

## Layout

| path | role |
|---|---|
| `usdjpy/data.py` | yfinance download + CSV cache, target (`y = next close > close`), chronological holdout |
| `usdjpy/features.py` | `BarCleaner`, `BarFeatureEngineer` (backward-looking only), encoders |
| `usdjpy/models.py` | candidates: `majority`, `momentum`, `logreg`, `lgbm` |
| `usdjpy/evaluation.py` | `DirectionEvaluator` (walk-forward `TimeSeriesSplit` + gap), metrics, slices, displays |
| `usdjpy/pipeline/classification.py` | `execute()`: the 10 phases |
| `usdjpy/analysis/` | builders, display helpers, pre/post-analysis reporters |
| `usdjpy/exp/v001_lgbm_walkforward.py` | the experiment arm (copy to `v002_*.py` to change one thing) |
| `usdjpy/train.py`, `usdjpy/experiments.py` | CLI and experiment ladder |

## Method

* **Target.** `y_t = 1` if `close_{t+1} > close_t`. Only labelled when bar `t+1` is exactly one hour after
  `t` (no labels across weekends or data gaps); unchanged closes are dropped.
* **Features** (all computed from bars ≤ t, checked by a truncate-the-future test): return lags 1–6/12/24,
  rolling mean/std of returns (6/24/120 bars), close z-score vs 24/120-bar SMA, RSI-14, ATR-14, short/long
  volatility ratio, candle body/wicks, hour (raw and sin/cos), weekday, Tokyo/London/NY session flags,
  bars since the last gap.
* **Split.** The last 15% of labelled rows is a holdout that is never looked at until the winner is refit.
  Model selection uses 5-fold walk-forward CV (expanding window) with a 24-bar gap between train and test.
* **Metrics.** ROC AUC (selection), log loss, Brier, accuracy, and a naive trading read-out: long when
  `P(up) > 0.5`, short otherwise, 0.5 bp per unit of position change (`strategy_bps`, annualised Sharpe).
* **Baselines.** `majority` (training up-share), `momentum` (repeat the last hour's sign), `logreg`.
* **Leakage guards.** A pure random walk must score AUC ≈ 0.5 through the full pipeline (tests); the
  `gap=0` experiment should match the `gap=24` score.

The cells below re-run the experiment and render the full report.

## Experiment ladder

`make experiments` evaluates a wider set of variants on the same development rows and walk-forward folds
(holdout excluded). The table is read from `outputs/experiments.json`.

## Findings (data 2024-01-15 → 2026-10-28, first run)

* **The signal is small.** LightGBM wins the walk-forward CV at AUC ≈ 0.52 (fold AUCs 0.50–0.55),
  ahead of logistic regression (≈ 0.51). Momentum and majority score ≤ 0.5. Hourly USDJPY returns have
  essentially no linear autocorrelation (|ACF| < 0.02); volatility clusters, direction does not.
* **Holdout AUC ≈ 0.56, accuracy ≈ 55%.** Much of that accuracy comes from the model leaning long
  (≈ 69% "up" calls) during a period in which USDJPY drifted up. One 5-month window is too short to
  call this an edge.
* **Costs eat it.** Before costs the sign strategy earns ≈ +0.2 bp/bar. The model flips position about every
  third bar (0.66 units of turnover per bar), so at 0.5 bp per unit of position change the net result is negative, both out-of-fold and
  on the holdout.
* **Where the edge is.** Removing calendar features drops CV AUC to ≈ 0.506, so most of what the model
  finds is time-of-day / session structure. AUC is highest in the NY session and in the most confident
  predictions (|P(up) − 0.5| ≥ 5pp).
* **Tuning doesn't matter (yet).** In the ladder every LightGBM variant (defaults, 7-leaf/800 trees, no
  candle features, ...) lands at AUC 0.515–0.523, within one fold-std (≈ 0.02) of each other. Using
  return lags only gives 0.502: past price moves alone carry essentially no direction information.
* **No leakage found.** Random-walk bars score AUC ≈ 0.5 through the full pipeline, features are invariant
  to truncating the future, and `gap=0` scores the same as `gap=24`.

## Next arms

* `v002`: trade only when |P(up) − 0.5| exceeds a threshold, to cut turnover. Choose the threshold on CV,
  then report it on the holdout.
* `v003`: predict a longer horizon (4h / 1d direction) where costs are a smaller share of the move.
* `v004`: add cross-asset context (DXY, US–JP rate proxies, Nikkei / S&P futures) at the same timestamps.
* For anything beyond research, switch to a documented data source (see `planning-1hour-price-direction-classification.md`).
