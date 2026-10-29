"""Builders: construct auxiliary objects for the pipeline (``build(...)``)."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from ..evaluation import DEFAULT_COST_BPS, compute_metrics, compute_return_metrics
from ..models import ALL_CANDIDATE_NAMES as CANDIDATE_NAMES
from ..models import make_candidate


class MetricsBuilder:
    """Turn ``(y_true, scores, next_logret)`` into a one-row metrics frame.

    ``task="direction"``: scores are ``P(up)``; ``task="return"``: predicted forward log returns.
    """

    def __init__(self, cost_bps: float = DEFAULT_COST_BPS, task: str = "direction", horizon: int = 24):
        self.cost_bps = cost_bps
        self.task = task
        self.horizon = horizon

    def build(self, y_true, scores, next_logret=None, name: str = "model") -> pd.DataFrame:
        if self.task == "return":
            row = compute_return_metrics(y_true, scores, next_logret, self.cost_bps, self.horizon)
        else:
            row = compute_metrics(y_true, scores, next_logret, self.cost_bps)
        return pd.DataFrame([row], index=pd.Index([name], name="model"))


class BaselineBuilder:
    """Build the reference pipelines every experiment is compared against.

    Parameters
    ----------
    methods : tuple of str
        Names from ``usdjpy.models.CANDIDATE_NAMES`` (default majority, momentum, logreg).
    random_state : int
    """

    def __init__(self, methods=("majority", "momentum", "logreg"), random_state: int = 0):
        self.methods = tuple(methods)
        self.random_state = random_state
        unknown = set(self.methods) - set(CANDIDATE_NAMES)
        if unknown:
            raise ValueError(f"unknown baseline(s) {sorted(unknown)}; choose from {CANDIDATE_NAMES}")

    def build(self) -> dict[str, Pipeline]:
        return {m: make_candidate(m, random_state=self.random_state) for m in self.methods}
