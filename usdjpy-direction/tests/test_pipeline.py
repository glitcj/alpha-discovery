import json

import numpy as np
import pytest

from usdjpy.analysis import BaselineBuilder, MetricsBuilder, PostAnalysisReporter, PreAnalysisReporter
from usdjpy.evaluation import DirectionEvaluator
from usdjpy.features import BarCleaner, BarFeatureEngineer
from usdjpy.models import make_candidate
from usdjpy.pipeline import Experiment, execute, prepare


def test_prepare_split(bars):
    d = prepare(bars, BarCleaner(), BarFeatureEngineer(), holdout_frac=0.2)
    assert d.X.index.max() < d.X_hold.index.min()
    assert d.X.notna().all().all()                       # warm-up removed
    n = len(d.X) + len(d.X_hold)
    assert abs(len(d.X_hold) / n - 0.2) < 0.01
    assert set(np.unique(d.y)) == {0, 1}


def _run(bars, tmp_path, **kw):
    return execute(
        exp_name="test", bars=bars, preanalysis_reporter=PreAnalysisReporter(),
        postanalysis_reporter=PostAnalysisReporter(), metrics_builder=MetricsBuilder(),
        baseline_builder=BaselineBuilder(("majority", "momentum")), cleaner=BarCleaner(),
        feature_engineer=BarFeatureEngineer(), candidates={"lgbm": make_candidate("lgbm")},
        evaluator=DirectionEvaluator(cv=3, gap=24), output_dir=tmp_path, **kw)


def test_execute_end_to_end(bars, tmp_path, capsys):
    res = _run(bars, tmp_path, figures=True, display_preanalysis=True, display_postanalysis=True,
               extra_experiments={"lgbm (gap=0)": Experiment(make_candidate("lgbm"), DirectionEvaluator(cv=3, gap=0))})
    assert set(res.board.index) == {"majority", "momentum", "lgbm"}
    assert res.holdout is not None and len(res.predictions) == len(res.data.X_hold)
    assert abs(res.holdout.auc - 0.5) < 0.06                 # random walk
    assert res.diagnostics["lgbm"]["best_iter"] >= 1
    for f in ("metrics.json", "run_meta.json", "best_model.json", "model.joblib", "predictions.csv"):
        assert (tmp_path / f).exists(), f
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["dev_end"] < meta["holdout_start"]
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert "holdout" in metrics and "lgbm (gap=0)" in metrics["leaderboard"]
    out = capsys.readouterr().out
    assert "Pre-analysis" in out and "Post-analysis" in out and "Holdout" in out


def test_execute_evaluate_only(bars, tmp_path):
    res = _run(bars, tmp_path, refit=False, predict=False, diagnostics=False, figures=False)
    assert res.model is None and res.holdout is None and not (tmp_path / "model.joblib").exists()


def test_execute_validates(bars, tmp_path):
    with pytest.raises(ValueError):
        _run(bars.drop(columns="close"), tmp_path)
    with pytest.raises(TypeError):
        _run(bars.reset_index(drop=True), tmp_path)
