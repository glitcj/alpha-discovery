"""v002 on top of v001: the objective now matches ml-pipe's ``v001_logreturn_exp_v19_legacy_test``
(``zen/exp/alhabib/y2025/d1207/ready_experiments_v22``). It predicts the **1-day forward log return
of USDJPY** as a regression, selected by Spearman. Three things change, and all three come from that
ml-pipe arm:

* **Target**: ``FWDLogReturn(lookahead_window=1 day, lookahead_offset=1 bar)``, i.e.
  ``y_t = log(close[t+25] / close[t+1])`` on 1h bars. ml-pipe uses 30-min bars (48 + 1), but Yahoo only
  serves 30-min history for 60 days, so we keep 1h bars and the same clock spans.
* **Features**: ml-pipe's 18 ``basic_feature_defs`` (Diff, PCTChange, MADiv, WEMADiv, Volatility,
  ChangeVolatility, WEVolatility, ChangeWEVolatility, Skew, Kurt, ChangeSkew, ChangeKurt, MACDSimple,
  MACDiffSimple, RSISimple, Drawdown, Drawup, ZScore) over windows 2/6/12/24/120/504/1512 bars
  (1.5h→2h, 6h, 12h, 1d, 5d, 21d, 63d), each taken now and 23 bars ago (ml-pipe's ``[0, 47]``), on
  USDJPY only. The cross-asset universe and feature selection are left for a later arm. The bars are
  put on ml-pipe's FX-week grid (Sunday 17:00 → Friday 17:00 New York, missing hours forward-filled).
* **Validation**: walk-forward with a 16-calendar-day gap (274 FX-week bars, ml-pipe's
  ``skip_days_between_train_test``), and the holdout is purged by 25 bars so no development label
  looks into it.

Candidates mirror ml-pipe's baselines: ``persistence``, ``mean``, ``best_feature``, ``random`` (its
``baseline_methods``), plus ``ridge`` and the main candidate ``lgbm_reg``. That is an
``LGBMRegressor`` on a robust-scaled, clipped target, the same idea as ml-pipe's ``y_preprocessor``,
with gradient boosting in place of the mixture-of-experts net. Leaderboard metrics follow ml-pipe:
Spearman (primary), Pearson, R², MSE and MCC of the sign, plus a daily sign trade on
non-overlapping labels.

Experiments on ``lgbm_reg``:

* rolling ~1.5-year training window (ml-pipe's ``train_days = 365 * 1.5``)
* only the shift-0 features
* the v001 calendar/session columns added back

Everything else is unchanged from v001: the cleaner, the reporters, the 15% chronological holdout,
5 folds, the 0.5 bp cost, the seed handling, and the CLI.

v001 (kept for reference; earlier files are never edited)
---------------------------------------------------------
See ``v001_lgbm_walkforward.py``: next-hour direction classification, AUC selection.

How to add v003
---------------
Copy this file to ``v003_<slug>.py``, change exactly one thing and prepend a paragraph
"v003 on top of v002: <change>, <why>; everything else unchanged". Run it with ``make train EXP=v003_<slug>``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from ..analysis import BaselineBuilder, MetricsBuilder, PostAnalysisReporter, PreAnalysisReporter
from ..data import TICKER, load_bars, make_forward_return_target
from ..evaluation import DEFAULT_COST_BPS, ReturnEvaluator
from ..features import CALENDAR_FEATURES, BarCleaner, ColumnSubset, MultiWindowFeatureEngineer
from ..models import REGRESSION_CANDIDATES, make_candidate
from ..pipeline import ExecutionResult, Experiment, execute
from ..pipeline.classification import OUTPUTS

# experiment name = file name (never hand-written); it labels the run record and the report
EXP_NAME = Path(__file__).stem

# DNA: how this arm descends from earlier arms. Read by usdjpy.lineage; keep it truthful, keep it short.
DNA = {
    "parents": ["v001_lgbm_walkforward"],          # first entry = primary parent (tree edge)
    "goal": "match ml-pipe's 1-day FWDLogReturn objective on USDJPY-only inputs",
    "changes": [
        {"from": "v001_lgbm_walkforward",
         "what": "target = 1-day forward log return log(close[t+25] / close[t+1]), regression selected by Spearman",
         "why": "ml-pipe v001_logreturn_exp_v19_legacy_test predicts FWDLogReturn(48, 1) on 30-min bars"},
        {"from": "v001_lgbm_walkforward",
         "what": "ml-pipe's 18 tsf feature types x 7 windows x shifts {0, 23} on the FX-week grid",
         "why": "same feature family as ml-pipe so results are comparable"},
        {"from": "v001_lgbm_walkforward",
         "what": "16-calendar-day CV gap (274 bars) and a holdout purged by 25 bars",
         "why": "ml-pipe's skip_days_between_train_test; overlapping labels must not leak"},
    ],
    "pipeline_commit": "fdfc5ff",                  # pipeline sha this arm was authored/run against
    "note": "pre-lineage run 2026-10-29: every model within +-0.05 Spearman, the random baseline wins",
}
TASK = "return"
BASELINES = ("persistence", "mean", "best_feature", "random", "ridge")
CANDIDATES = ("lgbm_reg",)
HORIZON = 24                       # bars = 1 day (ml-pipe: 48 x 30 min)
OFFSET = 1                         # bar buffer before the return window starts
BARS_PER_DAY = 24 * 5 / 7          # FX-week hours per calendar day
GAP = round(16 * BARS_PER_DAY)     # 274 bars ~ ml-pipe skip_days_between_train_test = 16
HOLDOUT_FRAC = 0.15
ROLLING_WINDOW = round(365 * 1.5 * BARS_PER_DAY)   # ml-pipe train_days = 365 * 1.5


def make_cleaner() -> BarCleaner:
    return BarCleaner(max_abs_logret=0.05, regular_grid=True)


def make_feature_engineer() -> MultiWindowFeatureEngineer:
    return MultiWindowFeatureEngineer(calendar=True)


def make_target_builder():
    return partial(make_forward_return_target, horizon=HORIZON, offset=OFFSET)


def make_candidates(seed: int, names=CANDIDATES) -> dict:
    return {name: make_candidate(name, random_state=seed) for name in names}


def make_baseline_builder(seed: int, names=BASELINES) -> BaselineBuilder:
    return BaselineBuilder(methods=names, random_state=seed)


def make_evaluator(cv: int, cost_bps: float = DEFAULT_COST_BPS) -> ReturnEvaluator:
    return ReturnEvaluator(cv=cv, gap=GAP, cost_bps=cost_bps, horizon=HORIZON)


def make_experiments(seed: int, cv: int, cost_bps: float = DEFAULT_COST_BPS,
                     subject: str = "lgbm_reg") -> dict[str, Experiment]:
    """Variants of the expected winner: ml-pipe's rolling window, shift-0 only, + calendar."""
    return {
        f"{subject} (rolling ~1.5y window)": Experiment(
            pipeline=make_candidate(subject, random_state=seed),
            evaluator=ReturnEvaluator(cv=cv, gap=GAP, max_train_size=ROLLING_WINDOW, cost_bps=cost_bps,
                                      horizon=HORIZON)),
        f"{subject} (shift 0 only)": Experiment(
            pipeline=make_candidate(subject, random_state=seed,
                                    columns=ColumnSubset(exclude=CALENDAR_FEATURES, exclude_regex=r"_s23$"))),
        f"{subject} (+ calendar features)": Experiment(
            pipeline=make_candidate(subject, random_state=seed, columns=ColumnSubset())),
    }


def make_partials() -> tuple[PreAnalysisReporter, PostAnalysisReporter]:
    """Pre- and post-analysis reporters with their defaults (sections can be swapped or disabled here)."""
    return PreAnalysisReporter(), PostAnalysisReporter()


def _split_models(models) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if models is None:
        return BASELINES, CANDIDATES
    names = [m.strip() for m in (models.split(",") if isinstance(models, str) else models) if m.strip()]
    unknown = set(names) - set(REGRESSION_CANDIDATES)
    if unknown:
        raise ValueError(f"unknown model(s) {sorted(unknown)}; choose from {REGRESSION_CANDIDATES}")
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

    ``models`` limits the candidate set (``"persistence,lgbm_reg"`` or a list); ``n_rows`` keeps only
    the most recent ``n_rows`` bars (smoke runs); ``bars`` overrides the cache (tests, notebooks).
    """
    # Prepare
    bars = load_bars() if bars is None else bars
    if n_rows:
        bars = bars.iloc[-n_rows:]
    baselines, candidates = _split_models(models)
    subject = "lgbm_reg" if "lgbm_reg" in candidates else (candidates or baselines)[0]
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
        metrics_builder=MetricsBuilder(cost_bps=cost_bps, task=TASK, horizon=HORIZON),
        baseline_builder=make_baseline_builder(seed, baselines),
        # Cleaning + features
        cleaner=make_cleaner(),
        feature_engineer=make_feature_engineer(),
        # Model(s)
        candidates=make_candidates(seed, candidates),
        # Validation
        evaluator=make_evaluator(cv, cost_bps),
        holdout_frac=HOLDOUT_FRAC,
        # Target
        target_builder=make_target_builder(),
        purge=HORIZON + OFFSET,
        target_meta={"kind": f"forward log return over {HORIZON} bars after a {OFFSET}-bar buffer",
                     "horizon": HORIZON, "offset": OFFSET, "ml_pipe_equivalent": "FWDLogReturn(48, 1) on 30-min bars"},
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
