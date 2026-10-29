"""Experiment ladder summarised in REPORT.ipynb.

    python -m usdjpy.experiments [--task return|direction] [--cv 5] [--seed 0] [--only lgbm] [--out ...]

``--task return`` (default) evaluates variants of the v002 forward-log-return regressors on v002's
data (FX-week grid, ml-pipe multi-window features, 1-day target, 16-day gap). ``--task direction``
runs the v001 next-hour direction ladder. Every variant sees the same development rows (holdout
excluded) and the same walk-forward folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.pipeline import Pipeline

from .data import load_bars
from .evaluation import DEFAULT_COST_BPS, DirectionEvaluator, ReturnEvaluator, md_table
from .exp import v001_lgbm_walkforward as v001
from .exp import v002_fwd_logret_1d as v002
from .features import CALENDAR_FEATURES, CANDLE_FEATURES, ROLLING_FEATURES, ColumnSubset
from .models import make_candidate
from .pipeline import prepare

COLUMNS = {
    "direction": ["auc", "auc_fold_std", "log_loss", "accuracy", "strategy_bps", "sharpe", "fit_time_s"],
    "return": ["spearman", "spearman_fold_std", "pearson", "r2", "mcc", "hit_rate", "strategy_bps", "sharpe",
               "fit_time_s"],
}


def direction_variants(seed: int) -> dict[str, Pipeline]:
    lgbm = lambda **kw: make_candidate("lgbm", seed, **kw)  # noqa: E731
    return {
        "majority (prior)": make_candidate("majority", seed),
        "momentum: repeat last hour": make_candidate("momentum", seed),
        "reversal: fade last hour": make_candidate("momentum", seed).set_params(model__reverse=True),
        "logreg: C=0.1": make_candidate("logreg", seed),
        "logreg: C=0.001": make_candidate("logreg", seed).set_params(model__C=0.001),
        "lgbm: library defaults (100 trees, 31 leaves)": make_candidate("lgbm", seed).set_params(
            model__n_estimators=100, model__learning_rate=0.1, model__num_leaves=31, model__min_child_samples=20,
            model__subsample=1.0, model__colsample_bytree=1.0, model__reg_lambda=0.0),
        "lgbm: tuned (v001)": lgbm(),
        "lgbm: stumps-ish (7 leaves, 800 trees, lr .01)": lgbm().set_params(
            model__num_leaves=7, model__n_estimators=800, model__learning_rate=0.01),
        "lgbm: 100 trees only": lgbm().set_params(model__n_estimators=100),
        "lgbm: without calendar": lgbm(drop=CALENDAR_FEATURES),
        "lgbm: without candles": lgbm(drop=CANDLE_FEATURES),
        "lgbm: returns only": lgbm(drop=ROLLING_FEATURES + CANDLE_FEATURES + CALENDAR_FEATURES),
    }


def return_variants(seed: int) -> dict[str, Pipeline]:
    no_cal = dict(exclude=CALENDAR_FEATURES)
    reg = lambda **kw: make_candidate("lgbm_reg", seed, **kw)  # noqa: E731
    return {
        "persistence (last day's return)": make_candidate("persistence", seed),
        "mean (21-day mean daily return)": make_candidate("mean", seed),
        "best_feature (1-D OLS)": make_candidate("best_feature", seed),
        "random (sampled targets)": make_candidate("random", seed),
        "ridge: alpha=100": make_candidate("ridge", seed),
        "ridge: alpha=1000": make_candidate("ridge", seed).set_params(model__regressor__alpha=1000.0),
        "lgbm_reg: tuned (v002)": reg(),
        "lgbm_reg: library defaults (100 trees, 31 leaves)": reg().set_params(
            model__regressor__n_estimators=100, model__regressor__learning_rate=0.1,
            model__regressor__num_leaves=31, model__regressor__min_child_samples=20,
            model__regressor__subsample=1.0, model__regressor__colsample_bytree=1.0,
            model__regressor__reg_lambda=0.0),
        "lgbm_reg: huber objective": reg().set_params(model__regressor__objective="huber"),
        "lgbm_reg: windows <= 1 day": reg(columns=ColumnSubset(**no_cal, include_regex=r"_w(2|6|12|24)_")),
        "lgbm_reg: windows >= 5 days": reg(columns=ColumnSubset(**no_cal, include_regex=r"_w(120|504|1512)_")),
        "lgbm_reg: shift 0 only": reg(columns=ColumnSubset(**no_cal, exclude_regex=r"_s23$")),
        "lgbm_reg: pct_change only": reg(columns=ColumnSubset(**no_cal, include_regex=r"^pct_change_")),
        "lgbm_reg: + calendar": reg(columns=ColumnSubset()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m usdjpy.experiments", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("return", "direction"), default="return")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap", type=int, default=None, help="bars between train and test (default: the arm's)")
    ap.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    ap.add_argument("--only", default=None, help="comma separated substring filters on variant names")
    ap.add_argument("--n-rows", type=int, default=None, help="keep only the most recent N bars")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "outputs" / "experiments.json"))
    args = ap.parse_args(argv)

    bars = load_bars()
    if args.n_rows:
        bars = bars.iloc[-args.n_rows:]
    if args.task == "return":
        gap = v002.GAP if args.gap is None else args.gap
        data = prepare(bars, v002.make_cleaner(), v002.make_feature_engineer(), v002.HOLDOUT_FRAC,
                       target_builder=v002.make_target_builder(), purge=v002.HORIZON + v002.OFFSET)
        ev = ReturnEvaluator(cv=args.cv, gap=gap, cost_bps=args.cost_bps, horizon=v002.HORIZON)
        todo = return_variants(args.seed)
    else:
        gap = v001.GAP if args.gap is None else args.gap
        data = prepare(bars, v001.make_cleaner(), v001.make_feature_engineer(), v001.HOLDOUT_FRAC)
        ev = DirectionEvaluator(cv=args.cv, gap=gap, cost_bps=args.cost_bps)
        todo = direction_variants(args.seed)
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        todo = {n: p for n, p in todo.items() if any(k in n for k in keys)}
    board, _ = ev.compare(todo, data.X, data.y, data.ret, verbose=True)
    cols = COLUMNS[args.task]
    print(f"\n=== {args.task} experiments (walk-forward out-of-fold, {args.cv} folds, gap {gap}) ===")
    print(md_table(board.reindex(columns=cols)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": args.task, "primary_metric": cols[0], "cv": args.cv, "gap": gap,
                               "seed": args.seed, "cost_bps": args.cost_bps, "n_dev": int(len(data.X)),
                               "dev_start": str(data.X.index[0]), "dev_end": str(data.X.index[-1]),
                               "results": board.reindex(columns=cols).reset_index().to_dict(orient="records")},
                              indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
