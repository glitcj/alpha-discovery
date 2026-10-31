"""Command line wrapper around an experiment module.

    python -m usdjpy.train                                   # v003 (cross-asset, 1-day forward log return) -> outputs/
    python -m usdjpy.train --exp v002_fwd_logret_1d          # USDJPY-only features
    python -m usdjpy.train --exp v001_lgbm_walkforward       # the next-hour direction arm
    python -m usdjpy.train --models ridge,lgbm_reg --evaluate-only
    python -m usdjpy.train --predict-only                    # forecast from the latest cached bar
    python -m usdjpy.train --n-rows 3000 --cv 3              # quick smoke run on the most recent bars

The same experiment runs in the notebook with
``from usdjpy.exp import v003_cross_asset as exp;
exp.run(display_preanalysis=True, display_postanalysis=True)``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import joblib

from . import lineage
from .data import BAR, align_basket, cache_path, load_bars, load_basket
from .evaluation import DEFAULT_COST_BPS, md_table
from .pipeline.classification import OUTPUTS

DEFAULT_EXP = "v003_cross_asset"
COLUMNS = {
    "direction": ["auc", "auc_fold_std", "log_loss", "accuracy", "strategy_bps", "sharpe", "n"],
    "return": ["spearman", "spearman_fold_std", "kendall", "pearson", "r2", "mcc", "f1", "bin_acc", "strategy_bps",
               "sharpe", "rank_score", "n"],
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m usdjpy.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=DEFAULT_EXP, help="module name under usdjpy/exp/ (default: %(default)s)")
    p.add_argument("--models", default=None, help="comma separated subset of the experiment's candidates")
    p.add_argument("--cv", type=int, default=5, help="number of walk-forward folds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS, help="cost per unit position change")
    p.add_argument("--n-rows", type=int, default=None, help="keep only the most recent N bars (smoke runs)")
    p.add_argument("--output-dir", default=str(OUTPUTS))
    p.add_argument("--evaluate-only", action="store_true", help="skip refit + holdout scoring")
    p.add_argument("--predict-only", action="store_true", help="only predict with outputs/model.joblib")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--preanalysis", action="store_true", help="print the pre-analysis section (text)")
    p.add_argument("--postanalysis", action="store_true", help="print the post-analysis section (text)")
    p.add_argument("--no-experiments", action="store_true", help="skip the feature/window/gap variants")
    p.add_argument("--no-diagnostics", action="store_true", help="skip the 80/20 fit + LightGBM curves")
    p.add_argument("--no-lineage", action="store_true",
                   help="do not record this run in experiment-lineage.json (smoke runs and custom --output-dir never are)")
    return p


def predict_only(output_dir: Path, last: int = 5) -> int:
    """Forecast from the most recent cached bars with the saved model.

    The features are rebuilt with the cleaner/feature engineer of the experiment that trained the
    model (read from ``best_model.json``). Direction arms print ``P(next close > close)``; return arms
    print the predicted forward log return.
    """
    model_path, meta_path = output_dir / "model.joblib", output_dir / "best_model.json"
    if not model_path.exists():
        print(f"no saved model at {model_path}; run without --predict-only first", file=sys.stderr)
        return 2
    try:
        bars = load_bars()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    info = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    module = importlib.import_module(f"usdjpy.exp.{info.get('exp_name', 'v001_lgbm_walkforward')}")
    cleaned = module.make_cleaner().fit_transform(bars)
    assets = getattr(module, "ASSETS", ())
    if assets:
        cleaned = cleaned.join(align_basket(load_basket(assets), cleaned.index))
    feats = module.make_feature_engineer().fit_transform(cleaned).tail(last)
    model = joblib.load(model_path)
    task = getattr(module, "TASK", "direction")
    print(f"[predict] {info.get('exp_name', '?')} / {info.get('best', '?')} ({model_path}) on {cache_path()}")
    if task == "return":
        pred = model.predict(feats)
        h, o = module.HORIZON, module.OFFSET
        for ts, p in zip(feats.index, pred):
            print(f"  bar {ts:%Y-%m-%d %H:%M} UTC  close {cleaned.loc[ts, 'close']:.3f}  predicted log return "
                  f"{ts + o * BAR:%m-%d %H:%M} -> +{h} bars: {p * 1e4:+.2f} bp  -> {'UP' if p > 0 else 'DOWN'}")
        print(f"(last line = forecast from the latest cached bar over the next {h} FX-week hours, starting "
              f"{o} bar later; `make fetch` refreshes the cache)")
        return 0
    p_up = model.predict_proba(feats)[:, 1]
    for ts, pu in zip(feats.index, p_up):
        nxt = ts + BAR
        print(f"  bar {ts:%Y-%m-%d %H:%M} UTC  close {cleaned.loc[ts, 'close']:.3f}  "
              f"P(close {nxt:%H:%M} > close {ts:%H:%M}) = {pu:.4f}  -> {'UP' if pu > 0.5 else 'DOWN'}")
    print("(the last line is the forecast for the bar after the latest cached one; `make fetch` refreshes the cache)")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if args.predict_only:
        return predict_only(output_dir)

    try:
        module = importlib.import_module(f"usdjpy.exp.{args.exp}")
    except ModuleNotFoundError as e:
        print(f"unknown experiment {args.exp!r} ({e}); see usdjpy/exp/", file=sys.stderr)
        return 2
    print(f"[train] experiment {args.exp} on {cache_path()}")
    result = module.run(
        display_preanalysis=args.preanalysis, display_postanalysis=args.postanalysis, cv=args.cv, seed=args.seed,
        models=args.models, n_rows=args.n_rows, output_dir=output_dir, refit=not args.evaluate_only,
        predict=not args.evaluate_only, diagnostics=not args.no_diagnostics, experiments=not args.no_experiments,
        figures=not args.no_figures, cost_bps=args.cost_bps,
    )

    task, pm = result.best.task, result.best.primary_metric
    cols = COLUMNS[task]
    unit = "bp/day" if task == "return" else "bp"
    print(f"\n=== leaderboard (walk-forward out-of-fold, higher {pm} is better) ===")
    print(md_table(result.board.reindex(columns=cols)))
    print(f"\nbest model: {result.best_name}  ({pm} {result.best.primary:.4f})")
    for s in ("session", "confidence"):
        print(f"\n--- {result.best_name} by {s} ---")
        print(md_table(result.best.slice_table(s)))
    if result.experiments:
        print("\n--- experiments ---")
        for name, rep in result.experiments.items():
            print(f"  {name}: {pm}={rep.primary:.4f} strategy={rep.overall['strategy_bps']:+.3f}{unit}")
    if result.holdout is not None:
        h = result.holdout
        print(f"\n=== holdout {h.index[0]:%Y-%m-%d} -> {h.index[-1]:%Y-%m-%d} ({len(h.y_true):,} bars) ===")
        print(md_table(h.summary_frame().reindex(columns=cols)))
    print("\nartefacts:")
    for k, v in result.paths.items():
        print(f"  {k}: {v}")
    if not args.no_lineage and args.n_rows is None and output_dir.resolve() == OUTPUTS.resolve():
        run = lineage.record_run(result, vars(args))
        shown = ", ".join(f"{m} {run['cv_metrics'][m]:.4f}" for m in (lineage.DEFAULT_METRIC, pm) if m in run["cv_metrics"])
        print(f"\nlineage: recorded {args.exp} run ({shown}) -> {lineage.LINEAGE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
