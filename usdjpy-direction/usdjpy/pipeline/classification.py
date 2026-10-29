"""Generic pipeline for the direction (classification) and forward-return (regression) tasks.

An experiment module (``usdjpy/exp/vNNN_*.py``) builds its inputs and calls :func:`execute`
exactly once. ``execute`` runs the numbered phases below, writes the artefacts under
``output_dir`` and, when ``display_postanalysis`` is on, hands the :class:`ExecutionResult`
to the post-analysis reporter.

Phases
------
1. Validation       – DatetimeIndex, OHLC columns, candidate names, holdout fraction
2. Preparation      – cleaner -> feature engineer on the *full* bar series (backward-looking
                      only), ``target_builder`` (next-bar direction or forward log return),
                      chronological split into development rows and holdout (``purge`` rows apart)
3. Pre-analysis     – reporters on the development bars/features (gated by ``display_preanalysis``)
4. Baselines        – ``baseline_builder.build()`` merged with the experiment's candidates
5. Cross-validation – walk-forward ``evaluator.compare`` → leaderboard; winner = best primary
                      metric (AUC for direction, Spearman for return)
6. Diagnostics      – one chronological 80/20 fit of the non-baseline candidates + LightGBM
                      per-iteration validation curves
7. Experiments      – extra variants evaluated with their own evaluator (feature subsets, windows, gaps)
8. Refit + holdout  – winner on all development rows → scored once on the holdout →
                      predictions.csv, model.joblib, best_model.json
9. Dump             – metrics.json, figures/, run_meta.json
10. Post-analysis   – reporters on the results (gated by ``display_postanalysis``)
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from .. import __version__
from ..analysis.builders import BaselineBuilder, MetricsBuilder
from ..analysis.reporters import PostAnalysisReporter, PreAnalysisReporter
from ..data import OHLC, align_basket, make_target, split_holdout, write_predictions
from ..evaluation import DirectionEvaluator, EvaluationReport, leaderboard, save_report
from ..features import BarCleaner, BarFeatureEngineer

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS = ROOT / "outputs"
PREDICTIONS = "predictions.csv"


@dataclass
class Experiment:
    """An extra variant evaluated next to the winner (``evaluator=None`` reuses the main one)."""

    pipeline: Pipeline
    evaluator: DirectionEvaluator | None = None


@dataclass
class Dataset:
    """Prepared inputs: features/target on development rows and on the holdout."""

    bars: pd.DataFrame            # cleaned bars (all)
    X: pd.DataFrame               # development features
    y: pd.Series
    ret: pd.Series                # next-bar log return (trading metrics)
    X_hold: pd.DataFrame
    y_hold: pd.Series
    ret_hold: pd.Series


@dataclass
class ExecutionResult:
    exp_name: str
    board: pd.DataFrame
    reports: dict[str, EvaluationReport]
    best_name: str
    data: Dataset
    experiments: dict[str, EvaluationReport] = field(default_factory=dict)
    diagnostics: dict | None = None
    model: Pipeline | None = None
    holdout: EvaluationReport | None = None
    predictions: pd.DataFrame | None = None
    paths: dict[str, Path] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def best(self) -> EvaluationReport:
        return self.reports[self.best_name]


def _log(exp_name: str, msg: str) -> None:
    print(f"[{exp_name}] {msg}", flush=True)


def prepare(bars: pd.DataFrame, cleaner: BarCleaner, feature_engineer, holdout_frac: float,
            target_builder=make_target, purge: int = 0, basket: dict | None = None) -> Dataset:
    """Clean, engineer (full series), label, drop warm-up rows, split chronologically.

    ``target_builder(cleaned_bars)`` returns a frame with ``y`` and ``next_logret``; ``purge`` rows
    are dropped between the development rows and the holdout (labels that look into the holdout).
    ``basket`` (name -> OHLC) is aligned onto the cleaned USDJPY grid (``align_basket``) and joined as
    ``close_<name>`` / ``stale_<name>`` columns before the feature engineer runs.
    """
    cleaned = clone(cleaner).fit_transform(bars)
    if basket:
        cleaned = cleaned.join(align_basket(basket, cleaned.index))
    feats = clone(feature_engineer).fit_transform(cleaned)
    target = target_builder(cleaned)
    # warm-up: drop rows before the USDJPY rolling windows are filled (basket columns may start later
    # than USDJPY on Yahoo and stay NaN for a while; LightGBM handles that natively)
    own = [c for c in feats.columns if "__" not in c]
    warm = feats[own].notna().all(axis=1)
    first_full = warm.idxmax() if warm.any() else feats.index[-1]
    target = target[target.index >= first_full]
    if len(target) < 50:
        raise ValueError(f"only {len(target)} labelled rows after warm-up; need more bars")
    dev_idx, hold_idx = split_holdout(target.index, holdout_frac, purge=purge)
    X = feats.loc[target.index]
    return Dataset(bars=cleaned, X=X.loc[dev_idx], y=target.loc[dev_idx, "y"],
                   ret=target.loc[dev_idx, "next_logret"], X_hold=X.loc[hold_idx],
                   y_hold=target.loc[hold_idx, "y"], ret_hold=target.loc[hold_idx, "next_logret"])


def lgbm_curves(pipe: Pipeline, X_tr, y_tr, X_val, y_val) -> dict | None:
    """Refit the LightGBM step with an eval set to record per-iteration validation curves.

    Returns ``{"curves": {metric: {"train": [...], "valid": [...]}}, "best_metric", "best_iter",
    "best_valid", "n_estimators"}``: AUC (higher is better) for classifiers, l2 on the transformed
    target (lower is better) for regressors (a ``TransformedTargetRegressor`` is unwrapped).
    """
    model = pipe[-1]
    y_tr, y_val = np.asarray(y_tr), np.asarray(y_val)
    if isinstance(model, TransformedTargetRegressor) and isinstance(model.regressor, LGBMRegressor):
        tf = clone(model.transformer).fit(y_tr.reshape(-1, 1).astype(float))
        y_tr = tf.transform(y_tr.reshape(-1, 1).astype(float)).ravel()
        y_val = tf.transform(y_val.reshape(-1, 1).astype(float)).ravel()
        model = model.regressor
    if isinstance(model, LGBMClassifier):
        metrics, best, higher = ["binary_logloss", "auc"], "auc", True
    elif isinstance(model, LGBMRegressor):
        metrics, best, higher = ["l2", "l1"], "l2", False
    else:
        return None
    enc = clone(pipe[:-1]).fit(X_tr, y_tr)
    Xt_tr, Xt_val = enc.transform(X_tr), enc.transform(X_val)
    m = clone(model).fit(Xt_tr, y_tr, eval_X=(Xt_tr, Xt_val), eval_y=(y_tr, y_val),
                         eval_names=["train", "valid"], eval_metric=metrics)
    ev = m.evals_result_
    curves = {k: {"train": list(ev["train"][k]), "valid": list(ev["valid"][k])} for k in metrics}
    val = np.asarray(curves[best]["valid"])
    i = int(np.argmax(val) if higher else np.argmin(val))
    return {"curves": curves, "best_metric": best, "best_iter": i + 1, "best_valid": float(val[i]),
            "n_estimators": int(model.n_estimators)}


def execute(
    exp_name: str,
    # Input: raw hourly bars (OHLC, UTC DatetimeIndex)
    bars: pd.DataFrame,
    # Reporters: for analysis
    preanalysis_reporter: PreAnalysisReporter | None,
    postanalysis_reporter: PostAnalysisReporter,
    # Builders: for metrics and baselines
    metrics_builder: MetricsBuilder,
    baseline_builder: BaselineBuilder,
    # Cleaning + features (run once on the full series)
    cleaner: BarCleaner,
    feature_engineer: BarFeatureEngineer,
    # Model(s): name -> features -> P(up) or predicted-return pipeline
    candidates: dict[str, Pipeline],
    # Validation
    evaluator: DirectionEvaluator,
    holdout_frac: float = 0.15,
    # Target: cleaned bars -> frame(y, next_logret); purge = rows between dev and holdout
    target_builder=make_target,
    purge: int = 0,
    target_meta: dict | None = None,
    # Cross-asset inputs: name -> OHLC frames joined onto the USDJPY grid (v003+)
    basket: dict | None = None,
    # Experiments: name -> Experiment
    extra_experiments: dict[str, Experiment] | None = None,
    # Flags
    display_preanalysis: bool = False,
    display_postanalysis: bool = False,
    refit: bool = True,
    predict: bool = True,
    diagnostics: bool = True,
    figures: bool = True,
    # Storage
    output_dir: str | Path = OUTPUTS,
    seed: int = 0,
    ticker: str = "USDJPY=X",
) -> ExecutionResult:
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step1: Validation
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must have a DatetimeIndex")
    missing = set(OHLC) - set(bars.columns)
    if missing:
        raise ValueError(f"bars are missing columns {sorted(missing)}")
    if not candidates and not baseline_builder.methods:
        raise ValueError("no candidates and no baselines: nothing to evaluate")

    # Step2: Preparation
    data = prepare(bars, cleaner, feature_engineer, holdout_frac, target_builder=target_builder, purge=purge,
                   basket=basket)
    task, pm = evaluator.task, evaluator.primary_metric
    if not data.X.index.max() < data.X_hold.index.min():
        raise ValueError("holdout must be strictly after the development rows")
    import lightgbm, sklearn  # noqa: E401  (versions for the run record)

    meta = {
        "exp_name": exp_name, "ticker": ticker, "seed": seed, "task": task, "primary_metric": pm,
        "target": target_meta or {"kind": "next-bar direction"}, "purge": purge,
        "cv": repr(evaluator._cv()), "cost_bps": evaluator.cost_bps, "n_features": int(data.X.shape[1]),
        "basket": sorted(basket) if basket else [],
        "basket_used": sorted({c.split("__")[0] for c in data.X.columns if "__" in c}),
        "n_bars": int(len(data.bars)), "bars_start": str(data.bars.index[0]), "bars_end": str(data.bars.index[-1]),
        "n_dev": int(len(data.X)), "dev_start": str(data.X.index[0]), "dev_end": str(data.X.index[-1]),
        "n_holdout": int(len(data.X_hold)), "holdout_start": str(data.X_hold.index[0]),
        "holdout_end": str(data.X_hold.index[-1]),
        "dev_up_share": float((data.y > 0).mean()), "holdout_up_share": float((data.y_hold > 0).mean()),
        "dev_y_mean": float(data.y.mean()), "dev_y_std": float(data.y.std()),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"usdjpy": __version__, "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                     "pandas": pd.__version__, "python": platform.python_version()},
    }
    _log(exp_name, f"Step2: {meta['n_bars']:,} bars -> {meta['n_dev']:,} dev rows + "
                   f"{meta['n_holdout']:,} holdout rows (from {meta['holdout_start']})")

    # Step3: Pre-analysis (development rows only)
    if display_preanalysis and preanalysis_reporter is not None:
        dev_bars = data.bars[data.bars.index <= data.X.index[-1]]
        preanalysis_reporter.display(exp_name=exp_name, bars=dev_bars, X=data.X, y=data.y, ret=data.ret,
                                     task=task)

    # Step4: Baselines + candidates
    all_candidates = {**baseline_builder.build(), **candidates}
    meta["candidates"] = list(all_candidates)
    _log(exp_name, f"Step4: {len(all_candidates)} candidates: {list(all_candidates)}")

    # Step5: Cross-validation
    _log(exp_name, f"Step5: walk-forward CV with {meta['cv']}")
    board, reports = evaluator.compare(all_candidates, data.X, data.y, data.ret, verbose=True)
    best_name = str(board.index[0])
    _log(exp_name, f"Step5: winner = {best_name} ({pm} {board.loc[best_name, pm]:.4f})")

    # Step6: Diagnostics - chronological 80/20 fit + LightGBM curves
    diag = None
    if diagnostics and candidates:
        cut = int(len(data.X) * 0.8)
        gap = getattr(evaluator, "gap", 0) or 0
        X_tr, y_tr = data.X.iloc[:cut], data.y.iloc[:cut]
        X_val, y_val, r_val = data.X.iloc[cut + gap:], data.y.iloc[cut + gap:], data.ret.iloc[cut + gap:]
        rows, gen_rows, curves = [], [], None
        r_tr = data.ret.iloc[:cut]
        for name, pipe in candidates.items():
            _log(exp_name, f"Step6: holdout fit of {name}")
            rep = evaluator.holdout(pipe, X_tr, y_tr, X_val, y_val, r_val, name=name)
            oos = metrics_builder.build(rep.y_true, rep.y_pred, rep.next_logret, name=name)
            rows.append(oos.assign(fit_time_s=rep.fit_time))
            # ml-pipe "generalization metrics": in-sample minus out-of-sample on the same fit
            ins = metrics_builder.build(y_tr, evaluator.predict_scores(rep.estimator_, X_tr), r_tr, name=name)
            gen_rows.append((ins - oos).assign(**{"in_sample_" + pm: float(ins[pm].iloc[0])}))
            if curves is None:
                curves = lgbm_curves(pipe, X_tr, y_tr, X_val, y_val)
                if curves:
                    curves["name"] = name
        diag = {"split": f"chronological 80/20 (gap {gap})", "holdout": pd.concat(rows), "lgbm": curves,
                "generalization": pd.concat(gen_rows)}

    # Step7: Experiments
    experiments: dict[str, EvaluationReport] = {}
    for name, exp in (extra_experiments or {}).items():
        _log(exp_name, f"Step7: experiment {name!r}")
        ev = exp.evaluator or evaluator
        experiments[name] = ev.evaluate(exp.pipeline, data.X, data.y, data.ret, name=name)
        _log(exp_name, f"Step7: {name}: {pm}={experiments[name].primary:.4f}")

    # Step8: Refit + holdout
    model = hold = preds = None
    paths: dict[str, Path] = {"output_dir": output_dir}
    if refit:
        _log(exp_name, f"Step8: refit {best_name} on all {len(data.X):,} development rows")
        model = clone(all_candidates[best_name]).fit(data.X, data.y.to_numpy())
        joblib.dump(model, output_dir / "model.joblib")
        paths["model"] = output_dir / "model.joblib"
        (output_dir / "best_model.json").write_text(json.dumps({
            "exp_name": exp_name, "task": task, "best": best_name, f"cv_{pm}": float(board.loc[best_name, pm]),
            "params": {k: repr(v) for k, v in model[-1].get_params().items()}}, indent=2))
        paths["best_model"] = output_dir / "best_model.json"
        if predict:
            p = evaluator.predict_scores(model, data.X_hold)
            hold = evaluator.score(best_name, data.y_hold, p, data.ret_hold, data.X_hold, method="holdout",
                                   estimator=model)
            preds = write_predictions(data.X_hold.index, p, output_dir / PREDICTIONS, y_true=data.y_hold,
                                      next_logret=data.ret_hold, task=task)
            paths["predictions"] = output_dir / PREDICTIONS
            o = hold.overall
            hit = o.get("accuracy", o.get("hit_rate"))
            unit = "bp/day" if task == "return" else "bp/bar"
            _log(exp_name, f"Step8: holdout {pm} {hold.primary:.4f}, hit {hit:.4f}, "
                           f"strategy {o['strategy_bps']:+.3f} {unit}")

    # Step9: Dump
    meta["wall_time_s"] = round(time.perf_counter() - t_start, 1)
    all_reports = {**reports, **experiments}
    paths["metrics"] = save_report(all_reports, leaderboard(all_reports), output_dir, best=best_name,
                                   figures=figures, holdout=hold,
                                   extra={"exp_name": exp_name, "task": task, "seed": seed, "cv": meta["cv"],
                                          "cost_bps": meta["cost_bps"], "n_dev": meta["n_dev"]})
    if figures:
        paths["figures"] = output_dir / "figures"
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    paths["run_meta"] = output_dir / "run_meta.json"
    _log(exp_name, f"Step9: artefacts under {output_dir} ({meta['wall_time_s']}s)")

    result = ExecutionResult(exp_name=exp_name, board=board, reports=reports, best_name=best_name, data=data,
                             experiments=experiments, diagnostics=diag, model=model, holdout=hold,
                             predictions=preds, paths=paths, meta=meta)

    # Step10: Post-analysis
    if display_postanalysis:
        postanalysis_reporter.display(result)
    return result
