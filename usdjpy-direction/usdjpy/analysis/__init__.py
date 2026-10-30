"""Analysis layer: notebook display helpers, builders and pre/post-analysis reporters.

Reporters own *how* results are shown (one ``display(...)`` method each, no data in the
constructor), builders own *how* auxiliary objects are constructed (``build(...)``), and
``display`` wraps IPython so the same code prints readable text when run from the shell.
Colours and matplotlib chrome live in :mod:`usdjpy.style` (re-exported here as ``style``).
"""

from .. import style
from .builders import BaselineBuilder, MetricsBuilder
from .display import (display_banner, display_caption, display_dataframe, display_figure, display_info_box,
                      display_info_grid, display_markdown, display_status, display_title, in_notebook)
from .reporters import (AutocorrelationReporter, CoverageReporter, CrossAssetReporter, DataReporter,
                        ExperimentsReporter, FeatureSummaryReporter, HoldoutReporter, ImportanceReporter,
                        LeaderboardReporter, MarketReporter, PerformanceTimelineReporter, PostAnalysisReporter,
                        PreAnalysisReporter, RegimePerformanceReporter, RegimeReporter, Reporter, ResidualReporter,
                        SliceReporter, TargetReporter, TrainingReporter, VerdictReporter)

__all__ = [
    "style",
    "BaselineBuilder", "MetricsBuilder",
    "display_banner", "display_caption", "display_dataframe", "display_figure", "display_info_box",
    "display_info_grid", "display_markdown", "display_status", "display_title", "in_notebook",
    "AutocorrelationReporter", "CoverageReporter", "CrossAssetReporter", "DataReporter", "ExperimentsReporter",
    "FeatureSummaryReporter", "HoldoutReporter", "ImportanceReporter", "LeaderboardReporter", "MarketReporter",
    "PerformanceTimelineReporter", "PostAnalysisReporter", "PreAnalysisReporter", "RegimePerformanceReporter",
    "RegimeReporter", "Reporter", "ResidualReporter", "SliceReporter", "TargetReporter", "TrainingReporter",
    "VerdictReporter",
]
