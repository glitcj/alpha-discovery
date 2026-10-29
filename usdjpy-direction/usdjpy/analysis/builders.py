"""Builders: construct auxiliary objects for the pipeline (``build(...)``)."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from ..evaluation import DEFAULT_COST_BPS, compute_metrics
from ..models import CANDIDATE_NAMES, make_candidate


class MetricsBuilder:
    """Turn ``(y_true, p_up, next_logret)`` into a one-row metrics frame."""

    def __init__(self, cost_bps: float = DEFAULT_COST_BPS):
        self.cost_bps = cost_bps

    def build(self, y_true, p_up, next_logret=None, name: str = "model") -> pd.DataFrame:
        row = compute_metrics(y_true, p_up, next_logret, self.cost_bps)
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
