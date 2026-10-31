"""Candidate registry: baselines + LightGBM.

Every candidate is a Pipeline ``encode -> model`` that takes the engineered feature frame
(``usdjpy.features.ALL_FEATURES``) and exposes ``predict_proba``. Features are built once on
the full bar series before splitting (they only look backwards), so the pipelines contain
no cleaning/feature steps.

``majority``  ``DummyClassifier(prior)``: always the training up-share (AUC 0.5 by design)
``momentum``  ``MomentumClassifier``: the next hour continues the last hour's direction
``logreg``    L2 logistic regression on the imputed/standardised features
``lgbm``      ``LGBMClassifier``, small and heavily regularised (the signal is tiny)

Return regressors (v002+, ml-pipe's ``FWDLogReturn`` objective; baselines mirror ml-pipe's
``baseline_methods``) take the ``MultiWindowFeatureEngineer`` frame:

``persistence``   the last day's return carries on (``pct_change_w24_s0``)
``mean``          the trailing 21-day mean daily return (``pct_change_w504_s0 / 21``)
``best_feature``  1-D OLS on the training feature with the largest |Spearman ρ| to y
``random``        random draws from the training targets (the "no skill" floor)
``ridge``         ridge regression on the imputed/standardised features
``lgbm_reg``      ``LGBMRegressor`` (same regularised settings as ``lgbm``) on a robust-scaled,
                  clipped target, like ml-pipe's ``y_preprocessor``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .features import CALENDAR_FEATURES, ColumnSubset, make_preprocessor

CANDIDATE_NAMES = ("majority", "momentum", "logreg", "lgbm")
REGRESSION_CANDIDATES = ("persistence", "mean", "best_feature", "random", "ridge", "lgbm_reg")
ALL_CANDIDATE_NAMES = CANDIDATE_NAMES + REGRESSION_CANDIDATES

LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


class MomentumClassifier(ClassifierMixin, BaseEstimator):
    """Predict that the next return has the sign of ``column`` (default the last return).

    ``P(up) = 0.5 + strength * sign(x)``; ``x`` missing or 0 -> 0.5.
    """

    def __init__(self, column: str = "ret_lag1", strength: float = 0.05, reverse: bool = False):
        self.column = column
        self.strength = strength
        self.reverse = reverse

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame) or self.column not in X:
            raise ValueError(f"MomentumClassifier needs a DataFrame with column {self.column!r}")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        s = np.sign(np.nan_to_num(np.asarray(X[self.column], dtype=float)))
        if self.reverse:
            s = -s
        p = 0.5 + self.strength * s
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def make_candidate(name: str, random_state: int = 0, drop=(), columns=None, selector=None) -> Pipeline:
    """Build a fresh ``encode [-> select] -> model`` pipeline.

    ``drop`` removes v001 bar-feature columns; ``columns`` (a :class:`ColumnSubset`) selects the
    input columns of the return regressors (default: every column except the calendar ones);
    ``selector`` (a fitted-per-fold transformer such as :class:`~usdjpy.features.StabilityFeatureSelector`)
    is inserted between the encoder and ``ridge`` / ``lgbm_reg`` (v004+).
    """
    if name in REGRESSION_CANDIDATES:
        return make_regressor(name, random_state=random_state, columns=columns, selector=selector)
    if name == "majority":
        return Pipeline([("encode", make_preprocessor("tree", drop)),
                         ("model", DummyClassifier(strategy="prior"))])
    if name == "momentum":
        return Pipeline([("encode", make_preprocessor("tree", drop)), ("model", MomentumClassifier())])
    if name == "logreg":
        return Pipeline([("encode", make_preprocessor("dense", drop)),
                         ("model", LogisticRegression(C=0.1, max_iter=2000))])
    if name == "lgbm":
        return Pipeline([("encode", make_preprocessor("tree", drop)),
                         ("model", LGBMClassifier(random_state=random_state, **LGBM_PARAMS))])
    raise ValueError(f"unknown candidate {name!r}; choose from {ALL_CANDIDATE_NAMES}")


# ---------------------------------------------------------------------------
# return regressors
# ---------------------------------------------------------------------------
class FeatureRegressor(RegressorMixin, BaseEstimator):
    """Predict ``scale * X[column]`` (NaN -> 0). Used for the persistence / mean baselines."""

    def __init__(self, column: str = "pct_change_w24_s0", scale: float = 1.0):
        self.column = column
        self.scale = scale

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame) or self.column not in X:
            raise ValueError(f"FeatureRegressor needs a DataFrame with column {self.column!r}")
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        check_is_fitted(self, "n_features_in_")
        return np.nan_to_num(np.asarray(X[self.column], dtype=float)) * self.scale


class BestFeatureRegressor(RegressorMixin, BaseEstimator):
    """1-D least squares on the training column with the largest |Spearman ρ| to y."""

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float)
        best, best_rho = None, -1.0
        for c in X.columns:
            x = X[c].to_numpy(float)
            ok = np.isfinite(x)
            if ok.sum() < 10 or np.nanstd(x[ok]) == 0:
                continue
            rho = abs(spearmanr(x[ok], y[ok]).statistic)
            if np.isfinite(rho) and rho > best_rho:
                best, best_rho = c, rho
        if best is None:
            raise ValueError("no usable feature column")
        x = X[best].to_numpy(float)
        ok = np.isfinite(x)
        self.slope_, self.intercept_ = np.polyfit(x[ok], y[ok], 1)
        self.fill_ = float(np.nanmedian(x))
        self.column_, self.rho_ = best, float(best_rho)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "column_")
        x = np.asarray(pd.DataFrame(X)[self.column_], dtype=float)
        return self.intercept_ + self.slope_ * np.where(np.isfinite(x), x, self.fill_)


class RandomSampleRegressor(RegressorMixin, BaseEstimator):
    """Random draws from the training targets (deterministic per ``random_state``)."""

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X, y):
        self.sample_ = np.asarray(y, dtype=float).copy()
        self.n_features_in_ = np.shape(X)[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "sample_")
        return check_random_state(self.random_state).choice(self.sample_, size=len(X), replace=True)


def _clip5(z):
    return np.clip(z, -5.0, 5.0)


def target_transformer() -> Pipeline:
    """Robust-scale then clip the target to ±5 IQR-units (ml-pipe's y_preprocessor, simplified).
    The inverse undoes the scaling only, so predictions come back in log-return units."""
    return Pipeline([("scale", RobustScaler()), ("clip", FunctionTransformer(_clip5, check_inverse=False))])


LGBM_REG_PARAMS = {k: v for k, v in LGBM_PARAMS.items()}


def make_regressor(name: str, random_state: int = 0, columns=None, selector=None) -> Pipeline:
    cols = columns if columns is not None else ColumnSubset(exclude=CALENDAR_FEATURES)
    select = [("select", clone(selector))] if selector is not None else []
    if name == "persistence":
        return Pipeline([("encode", make_preprocessor("tree", columns=cols)),
                         ("model", FeatureRegressor("pct_change_w24_s0", 1.0))])
    if name == "mean":
        return Pipeline([("encode", make_preprocessor("tree", columns=cols)),
                         ("model", FeatureRegressor("pct_change_w504_s0", 1 / 21))])
    if name == "best_feature":
        return Pipeline([("encode", make_preprocessor("tree", columns=cols)), ("model", BestFeatureRegressor())])
    if name == "random":
        return Pipeline([("encode", make_preprocessor("tree", columns=cols)),
                         ("model", RandomSampleRegressor(random_state=random_state))])
    if name == "ridge":
        return Pipeline([("encode", make_preprocessor("dense", columns=cols)), *select,
                         ("model", TransformedTargetRegressor(regressor=Ridge(alpha=100.0),
                                                              transformer=target_transformer(),
                                                              check_inverse=False))])
    if name == "lgbm_reg":
        return Pipeline([("encode", make_preprocessor("tree", columns=cols)), *select,
                         ("model", TransformedTargetRegressor(
                             regressor=LGBMRegressor(random_state=random_state, **LGBM_REG_PARAMS),
                             transformer=target_transformer(), check_inverse=False))])
    raise ValueError(f"unknown regressor {name!r}; choose from {REGRESSION_CANDIDATES}")
