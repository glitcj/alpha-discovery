"""USDJPY 1-hour price-direction classification.

Target: ``y_t = 1`` if the close of the next hourly bar is above the close of bar ``t``.

Layers (ml-pipe style, mirrors ``delivery-time-prediction``)
-----------------------------------------------------------
exp/           versioned experiment modules; each ``run()`` makes one ``execute()`` call
pipeline/      ``execute``: the generic phase runner (validate → prepare → CV → diagnostics →
               experiments → refit/holdout → dump → post-analysis)
analysis/      display helpers, builders (metrics, baselines) and pre/post-analysis reporters
data           yfinance download + CSV cache, target construction, chronological holdout split
features       sklearn transformers: bar cleaning, backward-looking feature engineering, encoding
models         candidate registry (majority, momentum, logreg, lgbm)
evaluation     walk-forward evaluator, classification + trading metrics, slices, displays
train          CLI wrapper around an experiment module; ``experiments`` = offline ladder
"""

__version__ = "0.1.0"
