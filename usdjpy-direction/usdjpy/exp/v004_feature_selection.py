"""v004 on top of v003: **stability-ranked feature selection inside the model pipeline** (ml-pipe's
``FeatureSelector(method="spearman_stability")`` + ``MulticollinearityFilter``, simplified). v003 showed that
the cross-asset signal is real but diffuse (no column above 1.3% of the gain), that the model memorises 7k
columns (in-sample Spearman 0.94 vs out-of-sample −0.02) and that the slow (≥ 5-day) windows carry the signal.
ml-pipe's answer is to select a few hundred features whose correlation with the target is *stable* over time
before the model sees them.

``StabilityFeatureSelector`` (``usdjpy.features``) is fitted on the training rows of every fold: it cuts them
into 63-day blocks (1512 bars, ml-pipe's ``corr_stability_window``), computes each column's Spearman with the
target inside every block, scores the column by the half-life-weighted mean of the signed block correlations
(recent blocks weigh more; sign flips cancel), keeps the top ``N_FEATURES = 200`` by ``|score|`` after a greedy
``|ρ| > 0.8`` multicollinearity filter among the top 600. ``lgbm_reg`` and ``ridge`` both get the selector;
the baselines are unchanged. Everything else is unchanged from v003: the 30-asset basket, the target, the
FX-week grid, 16-day gap, purged 15% holdout, Spearman selection, LightGBM with 30% column sampling.

Experiments on ``lgbm_reg``, all on identical folds:

* ``no selection = v003``: the apples-to-apples control
* ``N=100`` / ``N=400``: how many stable features to keep
* ``asset/type pre-selection``: ml-pipe's group caps (best 15 assets, best 10 feature types) before the cut
* ``no multicollinearity filter``: is the |ρ| > 0.8 filter doing anything

v005 (planned): ml-pipe's exact retrain schedule (1.5-year training window, 16-day skip, 6-month roll).

How to add v005
---------------
Copy this file to ``v005_<slug>.py``, change exactly one thing and prepend a paragraph
"v005 on top of v004: <change>, <why>; everything else unchanged". Fill the ``DNA`` block below:
``parents=["v004_feature_selection"]``, a one-line ``goal``, one ``changes`` entry per change (``from`` / ``what`` /
``why``) and ``pipeline_commit`` = ``git rev-parse --short HEAD`` at authoring time. ``make train EXP=v005_<slug>``
records the run in ``experiment-lineage.json``; ``make lineage`` re-renders the tree (``outputs/figures/lineage.png``).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from ..analysis import BaselineBuilder, MetricsBuilder, PostAnalysisReporter, PreAnalysisReporter
from ..data import BASKET, TICKER, load_bars, load_basket, make_forward_return_target
from ..evaluation import DEFAULT_COST_BPS, ReturnEvaluator
from ..features import BarCleaner, MultiWindowFeatureEngineer, StabilityFeatureSelector
from ..models import REGRESSION_CANDIDATES, make_candidate
from ..pipeline import ExecutionResult, Experiment, execute
from ..pipeline.classification import OUTPUTS

# experiment name = file name (never hand-written); it labels the run record and the report
EXP_NAME = Path(__file__).stem

# DNA: how this arm descends from earlier arms. Read by usdjpy.lineage; keep it truthful, keep it short.
DNA = {
    "parents": ["v003_cross_asset"],               # first entry = primary parent (tree edge)
    "goal": "stability-ranked top-N feature selection inside the pipeline (ml-pipe FeatureSelector)",
    "changes": [
        {"from": "v003_cross_asset",
         "what": "StabilityFeatureSelector fitted per fold: Spearman with the target inside 63-day blocks, "
                 "half-life-weighted signed mean, top 200 by |score| after a |rho| > 0.8 multicollinearity filter; "
                 "applied to lgbm_reg and ridge",
         "why": "v003's signal is diffuse over 7k columns and the model memorises (generalization gap 0.95); "
                "ml-pipe selects a few hundred stable features before the net"},
    ],
    "pipeline_commit": "8cd361a",                  # pipeline sha this arm was authored/run against
    "note": None,
}
TASK = "return"
BASELINES = ("persistence", "mean", "best_feature", "random")
CANDIDATES = ("ridge", "lgbm_reg")   # both get the selector; ridge moves out of the baselines for that
ASSETS = tuple(BASKET)
HORIZON = 24                       # bars = 1 day (ml-pipe: 48 x 30 min)
OFFSET = 1                         # bar buffer before the return window starts
BARS_PER_DAY = 24 * 5 / 7          # FX-week hours per calendar day
GAP = round(16 * BARS_PER_DAY)     # 274 bars ~ ml-pipe skip_days_between_train_test = 16
HOLDOUT_FRAC = 0.15
ROLLING_WINDOW = round(365 * 1.5 * BARS_PER_DAY)   # ml-pipe train_days = 365 * 1.5
COLSAMPLE = 0.3                    # fraction of the selected columns LightGBM samples per tree (as v003)
N_FEATURES = 200                   # stable features kept per fold (ml-pipe grid-searches 50..300)
CORR_BLOCK = 24 * 63               # 63-day blocks for the stability score (ml-pipe corr_stability_window)


def make_cleaner() -> BarCleaner:
    return BarCleaner(max_abs_logret=0.05, regular_grid=True)


def make_feature_engineer() -> MultiWindowFeatureEngineer:
    return MultiWindowFeatureEngineer(calendar=True, assets=ASSETS)


def make_target_builder():
    return partial(make_forward_return_target, horizon=HORIZON, offset=OFFSET)


def make_selector(n_features: int | None = None, **kw) -> StabilityFeatureSelector:
    """The arm's selector; ``N_FEATURES`` / ``CORR_BLOCK`` are read at call time (tests shrink them)."""
    return StabilityFeatureSelector(n_features=N_FEATURES if n_features is None else n_features,
                                    block=CORR_BLOCK, **kw)


def _lgbm(seed: int, columns=None, selector=None):
    pipe = make_candidate("lgbm_reg", random_state=seed, columns=columns,
                          selector=make_selector() if selector is None else selector)
    return pipe.set_params(model__regressor__colsample_bytree=COLSAMPLE)


def _v003_lgbm(seed: int, columns=None):
    """v003's model: no selector, 30% column sampling over every column."""
    pipe = make_candidate("lgbm_reg", random_state=seed, columns=columns)
    return pipe.set_params(model__regressor__colsample_bytree=COLSAMPLE)


def make_candidates(seed: int, names=CANDIDATES) -> dict:
    return {name: (_lgbm(seed) if name == "lgbm_reg"
                   else make_candidate(name, random_state=seed, selector=make_selector())) for name in names}


def make_baseline_builder(seed: int, names=BASELINES) -> BaselineBuilder:
    return BaselineBuilder(methods=names, random_state=seed)


def make_evaluator(cv: int, cost_bps: float = DEFAULT_COST_BPS) -> ReturnEvaluator:
    return ReturnEvaluator(cv=cv, gap=GAP, cost_bps=cost_bps, horizon=HORIZON)


def make_experiments(seed: int, cv: int, cost_bps: float = DEFAULT_COST_BPS,
                     subject: str = "lgbm_reg") -> dict[str, Experiment]:
    """Selection ablations on identical folds: the v003 control, N, group caps, the collinearity filter."""
    return {
        f"{subject} (no selection = v003)": Experiment(pipeline=_v003_lgbm(seed)),
        f"{subject} (N=100)": Experiment(pipeline=_lgbm(seed, selector=make_selector(100))),
        f"{subject} (N=400)": Experiment(pipeline=_lgbm(seed, selector=make_selector(400))),
        f"{subject} (asset/type pre-selection)": Experiment(
            pipeline=_lgbm(seed, selector=make_selector(n_assets=15, n_types=10))),
        f"{subject} (no multicollinearity filter)": Experiment(
            pipeline=_lgbm(seed, selector=make_selector(corr_threshold=None))),
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
