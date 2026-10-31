"""v004: stability-ranked feature selection inside the pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from tests.test_v003 import SMALL, SYN_ASSETS, _synthetic_basket
from usdjpy.data import basket_path, cache_path
from usdjpy.exp import v004_feature_selection as v004
from usdjpy.features import StabilityFeatureSelector
from usdjpy.models import make_candidate


def _frame(n=3000, p=40, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)],
                     index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
    X["AAA__stable_w24_s0"] = rng.normal(size=n)
    X["BBB__flip_w24_s0"] = rng.normal(size=n)
    X["dup_w24_s0"] = X["AAA__stable_w24_s0"] + 0.3 * rng.normal(size=n)        # near-duplicate (rho ~ 0.96), noisier
    sign = np.where(np.arange(n) < n // 2, 1.0, -1.0)                           # flips sign half-way
    y = 0.5 * X["AAA__stable_w24_s0"] + 0.5 * sign * X["BBB__flip_w24_s0"] + rng.normal(size=n)
    X.loc[X.index[:50], "f0"] = np.nan                                           # NaNs are tolerated
    return X, y.to_numpy()


def test_selector_ranks_stable_over_flipping_and_drops_duplicates():
    X, y = _frame()
    sel = StabilityFeatureSelector(n_features=5, block=500, half_life=2.0).fit(X, y)
    assert sel.block_corr_.shape == (6, X.shape[1])
    assert sel.selected_[0] == "AAA__stable_w24_s0"
    assert "dup_w24_s0" in sel.dropped_collinear_ and "dup_w24_s0" not in sel.selected_
    assert abs(sel.scores_["AAA__stable_w24_s0"]) > abs(sel.scores_["BBB__flip_w24_s0"])
    # without the collinearity filter the duplicate is kept
    raw = StabilityFeatureSelector(n_features=5, block=500, corr_threshold=None).fit(X, y)
    assert "dup_w24_s0" in raw.selected_ and not raw.dropped_collinear_
    out = sel.transform(X)
    assert list(out.columns) == sel.selected_ and len(out) == len(X)
    assert list(sel.get_feature_names_out()) == sel.selected_


def test_selector_group_caps_and_sizes():
    X, y = _frame()
    sel = StabilityFeatureSelector(n_features=3, block=500, n_assets=1, n_types=1).fit(X, y)
    assert all(c.startswith("AAA__") or "__" not in c for c in sel.selected_)   # USDJPY is always kept
    assert len(sel.selected_) <= 3
    big = StabilityFeatureSelector(n_features=1000, block=500, corr_threshold=None).fit(X, y)
    assert len(big.selected_) == X.shape[1]
    with pytest.raises(ValueError):
        StabilityFeatureSelector().fit(X)


def test_selector_uses_training_rows_only():
    X, y = _frame()
    a = StabilityFeatureSelector(n_features=5, block=500).fit(X.iloc[:2000], y[:2000])
    y2 = y.copy()
    y2[2000:] = -y2[2000:]                                                       # change the future only
    b = StabilityFeatureSelector(n_features=5, block=500).fit(X.iloc[:2000], y2[:2000])
    assert a.selected_ == b.selected_
    pd.testing.assert_series_equal(a.scores_, b.scores_)


def test_selector_inside_candidates():
    X, y = _frame()
    sel = StabilityFeatureSelector(n_features=4, block=500)
    for name in ("lgbm_reg", "ridge"):
        pipe = make_candidate(name, columns=lambda df: list(df.columns), selector=sel)
        assert "select" in dict(pipe.steps) and pipe.named_steps["select"] is not sel      # cloned
        pipe.fit(X, y)
        assert len(pipe.named_steps["select"].selected_) == 4
        assert pipe.predict(X).shape == (len(X),)
    plain = make_candidate("lgbm_reg", columns=lambda df: list(df.columns))
    assert "select" not in dict(plain.steps)


@pytest.fixture
def v004_cache(tmp_path, monkeypatch):
    bars, basket = make_bars(n_days=220, seed=3), _synthetic_basket()
    monkeypatch.setenv("USDJPY_DATA_DIR", str(tmp_path / "dataset"))
    cache_path().parent.mkdir(parents=True)
    bars.to_csv(cache_path(), index_label="timestamp")
    basket_path("AAA").parent.mkdir(parents=True)
    for name, b in basket.items():
        b.to_csv(basket_path(name), index_label="timestamp")
    monkeypatch.setattr(v004, "ASSETS", SYN_ASSETS)
    monkeypatch.setattr(v004, "CORR_BLOCK", 300)
    monkeypatch.setattr(v004, "N_FEATURES", 20)
    return tmp_path


def test_v004_dna_and_factories():
    assert v004.EXP_NAME == "v004_feature_selection" and v004.DNA["parents"] == ["v003_cross_asset"]
    assert v004.CANDIDATES == ("ridge", "lgbm_reg") and "ridge" not in v004.BASELINES
    cands = v004.make_candidates(0)
    assert all("select" in dict(p.steps) for p in cands.values())
    assert cands["lgbm_reg"].get_params()["model__regressor__colsample_bytree"] == v004.COLSAMPLE
    exps = v004.make_experiments(0, 3)
    assert len(exps) == 5 and "select" not in dict(exps["lgbm_reg (no selection = v003)"].pipeline.steps)
    assert exps["lgbm_reg (N=400)"].pipeline.named_steps["select"].n_features == 400


def test_v004_run_smoke(v004_cache, tmp_path):
    res = v004.run(cv=2, models="random,ridge,lgbm_reg", output_dir=tmp_path / "out", experiments=False,
                   figures=False, diagnostics=False)
    meta = json.loads((tmp_path / "out" / "run_meta.json").read_text())
    assert meta["exp_name"] == "v004_feature_selection" and res.best_name in {"random", "ridge", "lgbm_reg"}
    sel = res.model.named_steps.get("select") if res.model is not None and res.best_name != "random" else None
    if sel is not None:
        assert len(sel.selected_) <= 20 < meta["n_features"]
