"""Candidate registry: baselines + LightGBM.

Every candidate is a Pipeline ``encode -> model`` that takes the engineered feature frame
(``usdjpy.features.ALL_FEATURES``) and exposes ``predict_proba``. Features are built once on
the full bar series before splitting (they only look backwards), so the pipelines contain
no cleaning/feature steps.

``majority``  ``DummyClassifier(prior)``: always the training up-share (AUC 0.5 by design)
``momentum``  ``MomentumClassifier``: the next hour continues the last hour's direction
``logreg``    L2 logistic regression on the imputed/standardised features
``lgbm``      ``LGBMClassifier``, small and heavily regularised (the signal is tiny)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .features import make_preprocessor

CANDIDATE_NAMES = ("majority", "momentum", "logreg", "lgbm")

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


def make_candidate(name: str, random_state: int = 0, drop=()) -> Pipeline:
    """Build a fresh ``encode -> model`` pipeline. ``drop`` removes feature columns."""
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
    raise ValueError(f"unknown candidate {name!r}; choose from {CANDIDATE_NAMES}")
