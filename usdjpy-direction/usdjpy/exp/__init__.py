"""Versioned experiment modules.

One file per experiment arm, ``vNNN_<slug>.py``, each exposing ``run(...)``. A new
arm is a new file whose docstring says "vNNN on top of vMMM: <the one change and why>;
everything else unchanged" — earlier files are never edited, so every leaderboard delta
is attributable to one change.
"""
