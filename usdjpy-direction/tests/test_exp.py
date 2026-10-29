import json

import pytest

from usdjpy import experiments, train
from usdjpy.exp import v001_lgbm_walkforward as v001


def test_exp_name_and_factories():
    assert v001.EXP_NAME == "v001_lgbm_walkforward"
    assert set(v001.make_candidates(0)) == {"lgbm"}
    assert v001.make_baseline_builder(0).methods == ("majority", "momentum", "logreg")
    assert v001._split_models("majority,lgbm") == (("majority",), ("lgbm",))
    assert v001._split_models(["logreg"]) == (("logreg",), ())
    assert len(v001.make_experiments(0, 3)) == 3
    with pytest.raises(ValueError):
        v001._split_models("nope")


def test_run_smoke(bar_cache, tmp_path):
    res = v001.run(cv=2, models="majority,momentum", output_dir=tmp_path, experiments=False,
                   diagnostics=False, figures=False)
    assert res.exp_name == "v001_lgbm_walkforward" and res.best_name in {"majority", "momentum"}
    assert (tmp_path / "predictions.csv").exists()
    assert json.loads((tmp_path / "run_meta.json").read_text())["n_dev"] == len(res.data.X)


def test_cli_train_and_predict(bar_cache, tmp_path, capsys):
    rc = train.main(["--models", "majority,lgbm", "--cv", "2", "--output-dir", str(tmp_path),
                     "--no-experiments", "--no-figures", "--no-diagnostics"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "=== leaderboard" in out and "=== holdout" in out
    assert train.main(["--predict-only", "--output-dir", str(tmp_path)]) == 0
    assert "P(close" in capsys.readouterr().out
    assert train.main(["--exp", "does_not_exist"]) == 2
    assert train.main(["--predict-only", "--output-dir", str(tmp_path / "empty")]) == 2


def test_cli_evaluate_only(bar_cache, tmp_path, capsys):
    rc = train.main(["--models", "majority", "--cv", "2", "--output-dir", str(tmp_path), "--evaluate-only",
                     "--no-experiments", "--no-figures"])
    assert rc == 0 and not (tmp_path / "predictions.csv").exists()
    assert "best model: majority" in capsys.readouterr().out


def test_experiment_ladder(bar_cache, tmp_path):
    out = tmp_path / "experiments.json"
    assert experiments.main(["--cv", "2", "--only", "majority,reversal,returns only", "--out", str(out)]) == 0
    res = json.loads(out.read_text())["results"]
    assert len(res) == 3
