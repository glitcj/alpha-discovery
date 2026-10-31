import json

import pytest

from usdjpy import lineage, train
from usdjpy.exp import v001_lgbm_walkforward as v001


def test_every_arm_has_valid_dna():
    arms = lineage.scan_experiments()
    assert set(arms) >= {"v001_lgbm_walkforward", "v002_fwd_logret_1d", "v003_cross_asset"}
    assert not any(a.get("dna_missing") for a in arms.values())
    roots = [n for n, a in arms.items() if not a["parents"]]
    assert roots == ["v001_lgbm_walkforward"]
    for name, a in arms.items():
        assert all(p in arms for p in a["parents"]), name
        assert all(c["from"] in a["parents"] for c in a["changes"]), name
        assert a["file"].startswith("usdjpy/exp/") and a["goal"]


def test_validate_dna_rejects_bad_blocks():
    known = {"v001_a", "v002_b"}
    ok = {"parents": ["v001_a"], "goal": "g", "changes": [{"from": "v001_a", "what": "w", "why": "y"}],
          "pipeline_commit": "abc1234"}
    assert lineage.validate_dna(ok, "v002_b", known)["note"] is None
    bad = [
        {**ok, "parents": ["v999_x"]},                                   # unknown parent
        {**ok, "changes": [{"from": "v002_b", "what": "w", "why": "y"}]},  # change from a non-parent
        {**ok, "changes": []},                                           # parents but no change
        {**ok, "extra": 1},                                              # unknown key
        {**ok, "goal": ""},
        {**ok, "parents": ["v002_b"]},                                   # self parent
    ]
    for dna in bad:
        with pytest.raises(ValueError):
            lineage.validate_dna(dna, "v002_b", known)


def test_sync_best_path_mermaid_render(tmp_path):
    path = tmp_path / "lineage.json"
    lin = lineage.sync(path, import_outputs_dir=None)
    assert path.exists() and set(lin["nodes"]) == set(lineage.list_arms())
    edges = {(e["from"], e["to"], e["kind"]) for e in lin["edges"]}
    assert ("v001_lgbm_walkforward", "v002_fwd_logret_1d", "primary") in edges
    assert ("v002_fwd_logret_1d", "v003_cross_asset", "primary") in edges
    gens = lineage.generations(lin)
    assert gens["v001_lgbm_walkforward"] == 0 and gens["v003_cross_asset"] == 2
    assert lineage.best_path(lin) == []                      # nothing recorded yet
    mermaid = lineage.to_mermaid(lin)
    assert mermaid.startswith("graph TD") and all(n in mermaid for n in lin["nodes"]) and "(not run)" in mermaid
    png = lineage.render(lin, out=tmp_path / "lineage.png")
    assert png.exists() and png.stat().st_size > 0
    # a second sync keeps the file stable (runs preserved, nothing duplicated)
    lin2 = lineage.sync(path, import_outputs_dir=None)
    assert {n: len(v["runs"]) for n, v in lin2["nodes"].items()} == {n: 0 for n in lin["nodes"]}


def test_record_run_and_cli(bar_cache, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "deadbee")
    monkeypatch.setenv("GIT_DIRTY", "1")
    res = v001.run(cv=2, models="majority,momentum", output_dir=tmp_path / "out", experiments=False,
                   diagnostics=False, figures=False)
    path = tmp_path / "lineage.json"
    run = lineage.record_run(res, {"cv": 2}, path=path)
    assert run["git_commit"] == "deadbee" and run["git_dirty"] is True and run["cv"] == 2
    assert isinstance(run["cv_metrics"]["auc"], float) and run["best"] in {"majority", "momentum"}
    assert run["holdout"] and "auc" in run["holdout"]
    lin = lineage.load(path)
    node = lin["nodes"]["v001_lgbm_walkforward"]
    assert node["parents"] == [] and len(node["runs"]) == 1 and node["pipeline_commit"]
    assert lineage.best_path(lin, metric="auc") == ["v001_lgbm_walkforward"]
    assert lineage.metric_of(node, "auc") == run["cv_metrics"]["auc"]
    # importing an outputs/ folder adds the run once
    lin = lineage.sync(path, import_outputs_dir=tmp_path / "out")
    assert len(lin["nodes"]["v001_lgbm_walkforward"]["runs"]) == 1          # same started_at -> not duplicated
    lineage.sync(path, import_outputs_dir=tmp_path / "out")
    assert len(lineage.load(path)["nodes"]["v001_lgbm_walkforward"]["runs"]) == 1
    tbl = lineage.table(lin, metric="auc")
    assert tbl.loc["v001_lgbm_walkforward", "runs"] == 1 and tbl.loc["v001_lgbm_walkforward", "git_commit"] == "deadbee"
    rc = lineage.main(["--path", str(path), "--out", str(tmp_path / "tree.png"), "--metric", "auc",
                       "--outputs", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert rc == 0 and "best path: v001_lgbm_walkforward" in out and "```mermaid" in out
    assert (tmp_path / "tree.png").exists()
    assert lineage.main(["--path", str(path), "--no-sync", "--mermaid"]) == 0
    assert capsys.readouterr().out.startswith("graph TD")


def test_import_outputs_skips_foreign_runs(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "run_meta.json").write_text(json.dumps({"exp_name": "v999_nope", "started_at": "x"}))
    (out / "metrics.json").write_text(json.dumps({"exp_name": "v999_nope", "best": "m", "leaderboard": {}}))
    lin = lineage.sync(tmp_path / "l.json", import_outputs_dir=out)
    assert all(not n["runs"] for n in lin["nodes"].values())


def test_train_cli_does_not_touch_real_lineage(bar_cache, tmp_path, monkeypatch):
    real = lineage.LINEAGE_PATH
    before = real.read_text() if real.exists() else None
    rc = train.main(["--exp", "v001_lgbm_walkforward", "--models", "majority", "--cv", "2", "--output-dir", str(tmp_path),
                     "--no-experiments", "--no-figures", "--no-diagnostics", "--no-lineage"])
    assert rc == 0
    after = real.read_text() if real.exists() else None
    assert before == after


def test_git_info_env(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "abc1234")
    monkeypatch.setenv("GIT_DIRTY", "0")
    assert lineage.git_info() == {"commit": "abc1234", "dirty": False}
    monkeypatch.delenv("GIT_COMMIT")
    monkeypatch.delenv("GIT_DIRTY")
    info = lineage.git_info()
    assert set(info) == {"commit", "dirty"}
