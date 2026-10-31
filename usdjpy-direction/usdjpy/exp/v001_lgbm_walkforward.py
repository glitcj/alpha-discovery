"""LightGBM vs. simple baselines for the direction of the next USDJPY hourly close,
selected by out-of-fold ROC AUC under 5-fold walk-forward cross-validation.

Arm
---
* Input: ``USDJPY=X`` 1h bars from Yahoo Finance (``make fetch``, ~730 days, the most Yahoo
  serves at 1h), cached as ``dataset/usdjpy_1h.csv``. ``BarCleaner`` drops impossible bars.
* Target: ``y = 1`` if the next hourly close is above this close. A bar is labelled only when
  the next bar is exactly 1h later (no labels across weekends/gaps); unchanged closes are dropped.
* Features (``BarFeatureEngineer``, backward-looking only, built once on the full series):
  return lags 1–6/12/24, rolling mean/std of returns (6/24/120), close z-score vs 24/120 SMA,
  RSI-14, ATR-14, short/long volatility ratio, candle body and wicks, hour (raw + sin/cos),
  weekday, Tokyo/London/NY session flags, bars since the last gap.
* Split: the last 15% of labelled rows is the holdout, untouched until the winner is refit.
* Candidates: ``lgbm`` = small, heavily regularised ``LGBMClassifier``. Baselines: ``majority``
  (training up-share, AUC 0.5), ``momentum`` (next hour repeats the last hour's sign), ``logreg``.
* Validation: ``TimeSeriesSplit(5, gap=24)`` (expanding window, 24 bars dropped between train and
  test). Selection = highest out-of-fold AUC; log loss, accuracy and a sign strategy with
  0.5 bp/position-change costs are reported next to it.
* Experiments on the expected winner: drop calendar features (is any edge just intraday
  seasonality?), a rolling ~6-month training window (regime drift), and ``gap=0`` (a leakage
  check: it should barely move the score).
* Pre-analysis: coverage and gaps, return distribution and class balance by hour, ACF of returns
  and |returns|, feature summary with Spearman ρ to the next return.
* Post-analysis: run summary, leaderboard + per-fold scores, 80/20 diagnostics with LightGBM
  curves, experiments, slices (session, time of day, weekday, volatility, confidence), feature
  importance, holdout scorecard with ROC / calibration / equity curve.

v001: initial arm.

How to add v002
---------------
Copy this file to ``v002_<slug>.py``, change exactly one thing (a feature, a candidate, the
selection rule, ...) and prepend a paragraph "v002 on top of v001: <change>, <why>;
everything else unchanged". Run it with ``make train EXP=v002_<slug>`` or from a notebook:
``from usdjpy.exp import v002_<slug> as exp; exp.run(display_preanalysis=True, display_postanalysis=True)``.
"""

from __future__ import annotations

from pathlib import Path

from ..analysis import BaselineBuilder, MetricsBuilder, PostAnalysisReporter, PreAnalysisReporter
from ..data import TICKER, load_bars
from ..evaluation import DEFAULT_COST_BPS, DirectionEvaluator
from ..features import CALENDAR_FEATURES, BarCleaner, BarFeatureEngineer
from ..models import CANDIDATE_NAMES, make_candidate
from ..pipeline import ExecutionResult, Experiment, execute
from ..pipeline.classification import OUTPUTS

# experiment name = file name (never hand-written); it labels the run record and the report
EXP_NAME = Path(__file__).stem

# DNA: how this arm descends from earlier arms. Read by usdjpy.lineage; keep it truthful, keep it short.
DNA = {
    "parents": [],                                 # the root arm
    "goal": "next-hour direction baseline: LightGBM vs majority / momentum / logreg under walk-forward AUC",
    "changes": [],
    "pipeline_commit": "5a31c9d",                  # pipeline sha this arm was authored/run against
    "note": "pre-lineage run 2026-10-29: lgbm CV AUC 0.519, holdout 0.558; edge mostly time-of-day",
}
BASELINES = ("majority", "momentum", "logreg")
CANDIDATES = ("lgbm",)
GAP = 24
HOLDOUT_FRAC = 0.15
ROLLING_WINDOW = 24 * 5 * 26        # ~6 months of trading hours


def make_cleaner() -> BarCleaner:
    return BarCleaner(max_abs_logret=0.05)


def make_feature_engineer() -> BarFeatureEngineer:
    return BarFeatureEngineer()


def make_candidates(seed: int, names=CANDIDATES) -> dict:
    return {name: make_candidate(name, random_state=seed) for name in names}


def make_baseline_builder(seed: int, names=BASELINES) -> BaselineBuilder:
    return BaselineBuilder(methods=names, random_state=seed)


def make_evaluator(cv: int, cost_bps: float = DEFAULT_COST_BPS) -> DirectionEvaluator:
    return DirectionEvaluator(cv=cv, gap=GAP, cost_bps=cost_bps)


def make_experiments(seed: int, cv: int, cost_bps: float = DEFAULT_COST_BPS,
                     subject: str = "lgbm") -> dict[str, Experiment]:
    """Variants of the expected winner: no calendar features, rolling window, no gap."""
    return {
        f"{subject} (no calendar features)": Experiment(
            pipeline=make_candidate(subject, random_state=seed, drop=CALENDAR_FEATURES)),
        f"{subject} (rolling ~6-month window)": Experiment(
            pipeline=make_candidate(subject, random_state=seed),
            evaluator=DirectionEvaluator(cv=cv, gap=GAP, max_train_size=ROLLING_WINDOW, cost_bps=cost_bps)),
        f"{subject} (gap=0)": Experiment(
            pipeline=make_candidate(subject, random_state=seed),
            evaluator=DirectionEvaluator(cv=cv, gap=0, cost_bps=cost_bps)),
    }


def make_partials() -> tuple[PreAnalysisReporter, PostAnalysisReporter]:
    """Pre- and post-analysis reporters with their defaults (sections can be swapped or disabled here)."""
    return PreAnalysisReporter(), PostAnalysisReporter()


def _split_models(models) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if models is None:
        return BASELINES, CANDIDATES
    names = [m.strip() for m in (models.split(",") if isinstance(models, str) else models) if m.strip()]
    unknown = set(names) - set(CANDIDATE_NAMES)
    if unknown:
        raise ValueError(f"unknown model(s) {sorted(unknown)}; choose from {CANDIDATE_NAMES}")
    return tuple(n for n in names if n in BASELINES), tuple(n for n in names if n not in BASELINES)


def run(
    display_preanalysis: bool = False,
    display_postanalysis: bool = False,
    cv: int = 5,
    seed: int = 0,
    models=None,
    n_rows: int | None = None,
    output_dir=None,
    refit: bool = True,
    predict: bool = True,
    diagnostics: bool = True,
    experiments: bool = True,
    figures: bool = True,
    cost_bps: float = DEFAULT_COST_BPS,
    bars=None,
    *args,
    **kwargs,
) -> ExecutionResult:
    """Load the cached bars, build the parts, execute once.

    ``models`` limits the candidate set (``"majority,lgbm"`` or a list); ``n_rows`` keeps only the
    most recent ``n_rows`` bars (smoke runs); ``bars`` overrides the cache (tests, notebooks).
    """
    # Prepare
    bars = load_bars() if bars is None else bars
    if n_rows:
        bars = bars.iloc[-n_rows:]
    baselines, candidates = _split_models(models)
    subject = "lgbm" if "lgbm" in candidates else (candidates or baselines)[0]
    preanalysis_reporter, postanalysis_reporter = make_partials()

    # Execute
    return execute(
        exp_name=EXP_NAME,
        # Input
        bars=bars,
        # Reporters: for analysis
        preanalysis_reporter=preanalysis_reporter,
        postanalysis_reporter=postanalysis_reporter,
        # Builders: for metrics and baselines
        metrics_builder=MetricsBuilder(cost_bps=cost_bps),
        baseline_builder=make_baseline_builder(seed, baselines),
        # Cleaning + features
        cleaner=make_cleaner(),
        feature_engineer=make_feature_engineer(),
        # Model(s)
        candidates=make_candidates(seed, candidates),
        # Validation
        evaluator=make_evaluator(cv, cost_bps),
        holdout_frac=HOLDOUT_FRAC,
        # Experiments
        extra_experiments=make_experiments(seed, cv, cost_bps, subject) if experiments else None,
        # Flags
        display_preanalysis=display_preanalysis,
        display_postanalysis=display_postanalysis,
        refit=refit,
        predict=predict,
        diagnostics=diagnostics,
        figures=figures,
        # Storage
        output_dir=output_dir or OUTPUTS,
        seed=seed,
        ticker=TICKER,
    )
