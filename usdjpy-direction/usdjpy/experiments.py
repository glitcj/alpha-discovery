"""Experiment ladder summarised in REPORT.ipynb.

    python -m usdjpy.experiments [--cv 5] [--seed 0] [--only lgbm] [--out outputs/experiments.json]

Each entry is a named variant of a candidate pipeline; all are evaluated on the same
development rows (holdout excluded) with the same walk-forward ``DirectionEvaluator``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.pipeline import Pipeline

from .data import load_bars
from .evaluation import DEFAULT_COST_BPS, DirectionEvaluator, md_table
from .features import CALENDAR_FEATURES, CANDLE_FEATURES, ROLLING_FEATURES, BarCleaner, BarFeatureEngineer
from .models import make_candidate
from .pipeline import prepare


def variants(seed: int) -> dict[str, Pipeline]:
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m usdjpy.experiments", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap", type=int, default=24)
    ap.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    ap.add_argument("--only", default=None, help="comma separated substring filters on variant names")
    ap.add_argument("--n-rows", type=int, default=None, help="keep only the most recent N bars")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "outputs" / "experiments.json"))
    args = ap.parse_args(argv)

    bars = load_bars()
    if args.n_rows:
        bars = bars.iloc[-args.n_rows:]
    data = prepare(bars, BarCleaner(), BarFeatureEngineer(), holdout_frac=0.15)
    ev = DirectionEvaluator(cv=args.cv, gap=args.gap, cost_bps=args.cost_bps)
    todo = variants(args.seed)
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        todo = {n: p for n, p in todo.items() if any(k in n for k in keys)}
    board, _ = ev.compare(todo, data.X, data.y, data.ret, verbose=True)
    cols = ["auc", "auc_fold_std", "log_loss", "accuracy", "strategy_bps", "sharpe", "fit_time_s"]
    print(f"\n=== experiments (walk-forward out-of-fold, {args.cv} folds, gap {args.gap}) ===")
    print(md_table(board[cols]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cv": args.cv, "gap": args.gap, "seed": args.seed, "cost_bps": args.cost_bps,
                               "n_dev": int(len(data.X)), "dev_start": str(data.X.index[0]),
                               "dev_end": str(data.X.index[-1]),
                               "results": board[cols].reset_index().to_dict(orient="records")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
