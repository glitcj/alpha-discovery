"""Versioned experiment modules.

One file per experiment arm, ``vNNN_<slug>.py``, each exposing ``run(...)``. A new
arm is a new file whose docstring says "vNNN on top of vMMM: <the one change and why>;
everything else unchanged" — earlier files are never edited, so every leaderboard delta
is attributable to one change.

Every arm also carries its **DNA**, a module-level dict read by ``usdjpy.lineage``::

    DNA = {
        "parents": ["v002_fwd_logret_1d"],   # [] for the root; first = primary parent (tree edge)
        "goal": "cross-asset inputs",         # one line
        "changes": [{"from": "v002_fwd_logret_1d", "what": "...", "why": "..."}],
        "pipeline_commit": "9572607",        # pipeline sha the arm was authored/run against
        "note": None,                         # optional free text
    }

``make train EXP=vNNN_<slug>`` records the run (metrics, git commit) under that arm in
``experiment-lineage.json``; ``make lineage`` syncs every arm, renders the tree
(``outputs/figures/lineage.png``) and prints the best-MCC path. Several parents are allowed
when an arm combines two earlier ones (the ml-pipe "Key Changes from v20_1 / from v21_6" style).
"""
