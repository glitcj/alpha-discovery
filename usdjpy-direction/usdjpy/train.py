"""Command line wrapper around an experiment module.

    python -m usdjpy.train                                   # v001 end-to-end -> outputs/
    python -m usdjpy.train --exp v001_lgbm_walkforward       # explicit experiment
    python -m usdjpy.train --models logreg,lgbm --evaluate-only
    python -m usdjpy.train --predict-only                    # P(up) for the next hour from the latest cached bar
    python -m usdjpy.train --n-rows 3000 --cv 3              # quick smoke run on the most recent bars

The same experiment runs in the notebook with
``from usdjpy.exp import v001_lgbm_walkforward as exp;
exp.run(display_preanalysis=True, display_postanalysis=True)``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import joblib

from .data import BAR, cache_path, load_bars
from .evaluation import DEFAULT_COST_BPS, md_table
from .features import make_feature_frame
from .pipeline.classification import OUTPUTS

DEFAULT_EXP = "v001_lgbm_walkforward"


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
    return p


def predict_only(output_dir: Path, last: int = 5) -> int:
    """Print P(up) for the most recent cached bars: the last row is the forecast for the next hour."""
    model_path = output_dir / "model.joblib"
    if not model_path.exists():
        print(f"no saved model at {model_path}; run without --predict-only first", file=sys.stderr)
        return 2
    try:
        bars = load_bars()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    feats = make_feature_frame(bars).tail(last)
    p = joblib.load(model_path).predict_proba(feats)[:, 1]
    print(f"[predict] model {model_path} on {cache_path()}")
    for ts, pu in zip(feats.index, p):
        nxt = ts + BAR
        print(f"  bar {ts:%Y-%m-%d %H:%M} UTC  close {bars.loc[ts, 'close']:.3f}  "
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

    cols = ["auc", "auc_fold_std", "log_loss", "accuracy", "strategy_bps", "sharpe", "n"]
    print("\n=== leaderboard (walk-forward out-of-fold, higher AUC is better) ===")
    print(md_table(result.board.reindex(columns=cols)))
    print(f"\nbest model: {result.best_name}  (AUC {result.best.auc:.4f})")
    for s in ("session", "confidence"):
        print(f"\n--- {result.best_name} by {s} ---")
        print(md_table(result.best.slice_table(s)))
    if result.experiments:
        print("\n--- experiments ---")
        for name, rep in result.experiments.items():
            print(f"  {name}: auc={rep.auc:.4f} strategy={rep.overall['strategy_bps']:+.3f}bp")
    if result.holdout is not None:
        h = result.holdout
        print(f"\n=== holdout {h.index[0]:%Y-%m-%d} -> {h.index[-1]:%Y-%m-%d} ({len(h.y_true):,} bars) ===")
        print(md_table(h.summary_frame().reindex(columns=cols)))
    print("\nartefacts:")
    for k, v in result.paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
