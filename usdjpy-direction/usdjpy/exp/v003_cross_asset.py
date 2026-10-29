"""v003 on top of v002: **cross-asset inputs**. The ~30-asset yfinance basket (``usdjpy.data.BASKET``:
14 FX crosses + DXY, US 13w/5y/10y/30y yields, N225/SPX/VIX/FTSE/DAX/HSI, gold/oil/silver/copper, BTC)
is forward-filled onto the USDJPY FX-week grid (as-of join, ml-pipe's ``reindex(..., method="ffill")``)
and every asset gets the same 18 feature types × 7 windows × 2 shifts as USDJPY, under the prefix
``<asset>__``, plus ``<asset>__stale`` (hours since its last print, so the model can tell a live
quote from a forward-filled one). This is the largest difference between v002 and ml-pipe's
``v001_logreturn_exp_v19_legacy_test``, which builds its features over a selected multi-asset universe.
LightGBM samples 30% of the ~7k columns per tree (``colsample_bytree``) to keep fits fast and
regularised. Everything else is unchanged from v002: target, FX-week grid, baselines, Spearman
selection, 16-day gap, purged 15% holdout.

Experiments on ``lgbm_reg``, all on identical folds:

* ``USDJPY only``: v002's column set, the apples-to-apples control for the cross-asset delta
* ``FX assets only`` / ``non-FX assets only``: which asset group carries any signal
* ``rolling ~1.5y window``: ml-pipe's ``train_days``

v004 (planned): ml-pipe-style asset / feature-type pre-selection and stability-ranked top-N feature
selection inside the pipeline.

How to add v004
---------------
Copy this file to ``v004_<slug>.py``, change exactly one thing and prepend a paragraph
"v004 on top of v003: <change>, <why>; everything else unchanged". Run it with ``make train EXP=v004_<slug>``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from ..analysis import BaselineBuilder, MetricsBuilder, PostAnalysisReporter, PreAnalysisReporter
from ..data import BASKET, BASKET_GROUPS, TICKER, load_bars, load_basket, make_forward_return_target
from ..evaluation import DEFAULT_COST_BPS, ReturnEvaluator
from ..features import CALENDAR_FEATURES, BarCleaner, ColumnSubset, MultiWindowFeatureEngineer
from ..models import REGRESSION_CANDIDATES, make_candidate
from ..pipeline import ExecutionResult, Experiment, execute
from ..pipeline.classification import OUTPUTS

# experiment name = file name (never hand-written); it labels the run record and the report
EXP_NAME = Path(__file__).stem
TASK = "return"
BASELINES = ("persistence", "mean", "best_feature", "random", "ridge")
CANDIDATES = ("lgbm_reg",)
ASSETS = tuple(BASKET)
FX_ASSETS = tuple(BASKET_GROUPS["fx"])
HORIZON = 24                       # bars = 1 day (ml-pipe: 48 x 30 min)
OFFSET = 1                         # bar buffer before the return window starts
BARS_PER_DAY = 24 * 5 / 7          # FX-week hours per calendar day
GAP = round(16 * BARS_PER_DAY)     # 274 bars ~ ml-pipe skip_days_between_train_test = 16
HOLDOUT_FRAC = 0.15
ROLLING_WINDOW = round(365 * 1.5 * BARS_PER_DAY)   # ml-pipe train_days = 365 * 1.5
COLSAMPLE = 0.3                    # fraction of the ~7k columns LightGBM samples per tree


def make_cleaner() -> BarCleaner:
    return BarCleaner(max_abs_logret=0.05, regular_grid=True)


def make_feature_engineer() -> MultiWindowFeatureEngineer:
    return MultiWindowFeatureEngineer(calendar=True, assets=ASSETS)


def make_target_builder():
    return partial(make_forward_return_target, horizon=HORIZON, offset=OFFSET)


def _lgbm(seed: int, columns=None):
    pipe = make_candidate("lgbm_reg", random_state=seed, columns=columns)
    return pipe.set_params(model__regressor__colsample_bytree=COLSAMPLE)


def make_candidates(seed: int, names=CANDIDATES) -> dict:
    return {name: (_lgbm(seed) if name == "lgbm_reg" else make_candidate(name, random_state=seed)) for name in names}


def make_baseline_builder(seed: int, names=BASELINES) -> BaselineBuilder:
    return BaselineBuilder(methods=names, random_state=seed)


def make_evaluator(cv: int, cost_bps: float = DEFAULT_COST_BPS) -> ReturnEvaluator:
    return ReturnEvaluator(cv=cv, gap=GAP, cost_bps=cost_bps, horizon=HORIZON)


def make_experiments(seed: int, cv: int, cost_bps: float = DEFAULT_COST_BPS,
                     subject: str = "lgbm_reg") -> dict[str, Experiment]:
    """Asset-group ablations on identical folds + ml-pipe's rolling window."""
    no_cal = CALENDAR_FEATURES
    return {
        f"{subject} (USDJPY only = v002)": Experiment(
            pipeline=make_candidate(subject, random_state=seed,
                                    columns=ColumnSubset(exclude=no_cal, assets=("USDJPY",)))),
        f"{subject} (FX assets only)": Experiment(
            pipeline=_lgbm(seed, ColumnSubset(exclude=no_cal, assets=("USDJPY",) + FX_ASSETS))),
        f"{subject} (non-FX assets only)": Experiment(
            pipeline=_lgbm(seed, ColumnSubset(exclude=no_cal, exclude_assets=FX_ASSETS))),
        f"{subject} (rolling ~1.5y window)": Experiment(
            pipeline=_lgbm(seed),
            evaluator=ReturnEvaluator(cv=cv, gap=GAP, max_train_size=ROLLING_WINDOW, cost_bps=cost_bps,
                                      horizon=HORIZON)),
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
    basket=None,
    *args,
    **kwargs,
) -> ExecutionResult:
    """Load the cached bars + basket, build the parts, execute once.

    ``models`` limits the candidate set; ``n_rows`` keeps only the most recent ``n_rows`` USDJPY bars
    (smoke runs); ``bars`` / ``basket`` override the caches (tests, notebooks).
    """
    # Prepare
    bars = load_bars() if bars is None else bars
    basket = load_basket(ASSETS) if basket is None else basket
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
        basket=basket,
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
