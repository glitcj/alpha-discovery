"""Analysis layer: notebook display helpers, builders and pre/post-analysis reporters.

Reporters own *how* results are shown (one ``display(...)`` method each, no data in the
constructor), builders own *how* auxiliary objects are constructed (``build(...)``), and
``display`` wraps IPython so the same code prints readable text when run from the shell.
"""

from .builders import BaselineBuilder, MetricsBuilder
from .display import (display_dataframe, display_figure, display_info_box, display_markdown,
                      display_title, in_notebook)
from .reporters import (AutocorrelationReporter, CoverageReporter, DataReporter, ExperimentsReporter, ResidualReporter,
                        FeatureSummaryReporter, HoldoutReporter, ImportanceReporter, LeaderboardReporter,
                        PostAnalysisReporter, PreAnalysisReporter, Reporter, SliceReporter, TargetReporter,
                        TrainingReporter)

__all__ = [
    "BaselineBuilder", "MetricsBuilder",
    "display_dataframe", "display_figure", "display_info_box", "display_markdown", "display_title",
    "in_notebook",
    "AutocorrelationReporter", "CoverageReporter", "DataReporter", "ExperimentsReporter",
    "FeatureSummaryReporter", "HoldoutReporter", "ImportanceReporter", "LeaderboardReporter",
    "PostAnalysisReporter", "PreAnalysisReporter", "Reporter", "ResidualReporter", "SliceReporter", "TargetReporter",
    "TrainingReporter",
]
