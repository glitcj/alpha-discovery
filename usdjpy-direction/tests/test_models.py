import numpy as np
import pytest

from usdjpy.data import make_target
from usdjpy.features import make_feature_frame
from usdjpy.models import CANDIDATE_NAMES, MomentumClassifier, make_candidate


@pytest.fixture(scope="module")
def Xy(small_bars):
    f = make_feature_frame(small_bars)
    t = make_target(small_bars)
    X = f.loc[t.index].iloc[130:]
    return X, t.loc[X.index, "y"].to_numpy()


@pytest.mark.parametrize("name", CANDIDATE_NAMES)
def test_candidates_fit_predict_proba(Xy, name):
    X, y = Xy
    p = make_candidate(name, random_state=0).fit(X, y).predict_proba(X)
    assert p.shape == (len(X), 2) and np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(1), 1)


def test_momentum_follows_last_return(Xy):
    X, y = Xy
    m = MomentumClassifier().fit(X, y)
    p = m.predict_proba(X)[:, 1]
    s = np.sign(X["ret_lag1"].to_numpy())
    assert np.all((p > 0.5) == (s > 0)) and np.all((p < 0.5) == (s < 0))
    assert np.all(MomentumClassifier(reverse=True).fit(X, y).predict(X) == (s < 0))


def test_unknown_candidate():
    with pytest.raises(ValueError):
        make_candidate("nope")
