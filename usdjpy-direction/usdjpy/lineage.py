"""Experiment lineage: the DNA of every arm and the runs that scored it.

    python -m usdjpy.lineage                        # sync + render + table + best path + mermaid
    python -m usdjpy.lineage --metric spearman --split holdout
    python -m usdjpy.lineage --no-sync --mermaid    # just print the mermaid graph of the current JSON

Each ``usdjpy/exp/vNNN_*.py`` carries a ``DNA`` dict (``parents``, ``goal``, ``changes``,
``pipeline_commit``, optional ``note``). ``experiment-lineage.json`` (project root) stores one node per
arm with that DNA plus every recorded run (metrics, git commit, seed, ...). ``train.py`` records a run after
each full ``make train``; :func:`sync` adds arms that were never run and imports the latest ``outputs/``
result; :func:`render` draws the tree (``outputs/figures/lineage.png``) and :func:`to_mermaid` prints it for
the README.

The pipeline is still moving: ``pipeline_commit`` (authored against) and each run's ``git_commit`` (ran
against) are the only ties between an arm and the pipeline version. They are informative, not exact.
Lineage depends on the pipeline only through :class:`~usdjpy.pipeline.ExecutionResult`, so the experiments
(and this file) can move to their own repository later.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import re
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import exp as exp_pkg
from . import style
from .evaluation import md_table
from .pipeline.classification import OUTPUTS, ROOT, ExecutionResult

PROJECT = "usdjpy-direction"                  # not ROOT.name: inside Docker the project is mounted as /app
LINEAGE_PATH = ROOT / "experiment-lineage.json"
LINEAGE_FIGURE = OUTPUTS / "figures" / "lineage.png"
SCHEMA = 1
DEFAULT_METRIC = "mcc"
DNA_KEYS = {"parents", "goal", "changes", "pipeline_commit"}
DNA_OPTIONAL = {"note"}
CHANGE_KEYS = {"from", "what", "why"}
METRICS = ("spearman", "pearson", "r2", "mcc", "f1", "roc_auc", "auc", "bin_acc", "accuracy", "log_loss",
           "strategy_bps", "sharpe", "rank_score")
ARM_RE = re.compile(r"^v\d{3}_")


# --------------------------------------------------------------------------- git
def git_info() -> dict:
    """Short sha + dirty flag: from ``GIT_COMMIT`` / ``GIT_DIRTY`` (set by the Makefile; the container does
    not see ``.git``), else from ``git`` in the parent repo, else ``None``."""
    commit, dirty = os.environ.get("GIT_COMMIT"), os.environ.get("GIT_DIRTY")
    if commit:
        return {"commit": commit, "dirty": None if dirty is None else dirty not in ("", "0", "false", "False")}
    try:
        kw = dict(cwd=ROOT, capture_output=True, text=True, timeout=5)
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kw)
        if head.returncode != 0:
            return {"commit": None, "dirty": None}
        status = subprocess.run(["git", "status", "--porcelain"], **kw)
        return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


# --------------------------------------------------------------------------- DNA
def list_arms() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules(exp_pkg.__path__) if ARM_RE.match(m.name))


def load_arm(name: str):
    return importlib.import_module(f"usdjpy.exp.{name}")


def validate_dna(dna, name: str, known: set[str] | None = None) -> dict:
    """Check a DNA block; ``known`` = arm names that may be parents (default: the modules in usdjpy/exp)."""
    where = f"usdjpy/exp/{name}.py DNA"
    if not isinstance(dna, dict):
        raise ValueError(f"{where} must be a dict")
    keys = set(dna)
    if not DNA_KEYS <= keys or keys - DNA_KEYS - DNA_OPTIONAL:
        raise ValueError(f"{where} keys must be {sorted(DNA_KEYS)} (+ optional {sorted(DNA_OPTIONAL)}), got {sorted(keys)}")
    parents = dna["parents"]
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
        raise ValueError(f"{where}['parents'] must be a list of arm names")
    if name in parents:
        raise ValueError(f"{where}: an arm cannot be its own parent")
    known = set(list_arms()) if known is None else known
    unknown = [p for p in parents if p not in known]
    if unknown:
        raise ValueError(f"{where}: unknown parent(s) {unknown}")
    if not isinstance(dna["goal"], str) or not dna["goal"].strip():
        raise ValueError(f"{where}['goal'] must be a non-empty string")
    changes = dna["changes"]
    if not isinstance(changes, list):
        raise ValueError(f"{where}['changes'] must be a list")
    for c in changes:
        if not isinstance(c, dict) or set(c) != CHANGE_KEYS:
            raise ValueError(f"{where}['changes'] entries need exactly the keys {sorted(CHANGE_KEYS)}")
        if c["from"] not in parents:
            raise ValueError(f"{where}: change from {c['from']!r} but parents are {parents}")
    if parents and not changes:
        raise ValueError(f"{where}: an arm with parents must list at least one change")
    commit = dna["pipeline_commit"]
    if commit is not None and not isinstance(commit, str):
        raise ValueError(f"{where}['pipeline_commit'] must be a short git sha or None")
    return {"parents": list(parents), "goal": dna["goal"].strip(), "changes": [dict(c) for c in changes],
            "pipeline_commit": commit, "note": dna.get("note")}


def dna_of(module) -> dict:
    name = Path(module.__file__).stem
    return validate_dna(getattr(module, "DNA", None), name)


def _arm_meta(module) -> dict:
    return {"file": str(Path(module.__file__).resolve().relative_to(ROOT)),
            "task": getattr(module, "TASK", "direction"),
            "primary_metric": "spearman" if getattr(module, "TASK", "direction") == "return" else "auc"}


def scan_experiments() -> dict[str, dict]:
    """name -> DNA (+ file/task) for every arm in usdjpy/exp; arms without a DNA block are kept with
    ``dna_missing=True`` and a warning so the tree still shows them."""
    arms, names = {}, set(list_arms())
    for name in sorted(names):
        module = load_arm(name)
        meta = _arm_meta(module)
        if getattr(module, "DNA", None) is None:
            warnings.warn(f"{name} has no DNA block; add one (see usdjpy/exp/__init__.py)", stacklevel=2)
            arms[name] = {**meta, "parents": [], "goal": "", "changes": [], "pipeline_commit": None, "note": None,
                          "dna_missing": True}
            continue
        arms[name] = {**meta, **validate_dna(module.DNA, name, names)}
    return arms


# --------------------------------------------------------------------------- storage
def _empty() -> dict:
    return {"schema": SCHEMA, "project": PROJECT, "metric": DEFAULT_METRIC, "updated_at": None,
            "nodes": {}, "edges": []}


def load(path: Path | str = LINEAGE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return _empty()
    data = json.loads(path.read_text())
    for key, default in _empty().items():
        data.setdefault(key, default)
    return data


def save(lineage: dict, path: Path | str = LINEAGE_PATH) -> Path:
    path = Path(path)
    lineage["edges"] = edges_of(lineage)
    lineage["nodes"] = dict(sorted(lineage["nodes"].items()))
    lineage["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lineage, indent=2, default=_jsonable) + "\n")
    return path


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (Path, pd.Timestamp)):
        return str(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return str(v)


def edges_of(lineage: dict) -> list[dict]:
    edges = []
    for name, node in sorted(lineage["nodes"].items()):
        for i, parent in enumerate(node.get("parents", [])):
            edges.append({"from": parent, "to": name, "kind": "primary" if i == 0 else "secondary"})
    return edges


def upsert_node(lineage: dict, name: str, dna: dict) -> dict:
    """Refresh the DNA fields of a node (the file wins), keep its runs."""
    node = lineage["nodes"].setdefault(name, {"runs": []})
    node.update({k: v for k, v in dna.items()})
    node.setdefault("runs", [])
    # a fixed key order keeps the JSON diff-friendly
    order = ["file", "task", "primary_metric", "parents", "goal", "changes", "pipeline_commit", "note",
             "dna_missing", "runs"]
    lineage["nodes"][name] = {k: node[k] for k in order if k in node} | {k: v for k, v in node.items() if k not in order}
    return lineage["nodes"][name]


# --------------------------------------------------------------------------- runs
def _pick(metrics, keys=METRICS) -> dict:
    out = {}
    for k in keys:
        if k in metrics:
            v = metrics[k]
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            out[k] = None if np.isnan(v) else v
    return out


def summarise(result: ExecutionResult, cli: dict | None = None) -> dict:
    """One run record from an :class:`ExecutionResult` (+ the CLI args it was launched with)."""
    cli = cli or {}
    meta, best = result.meta, result.best_name
    git = git_info()
    row = result.board.loc[best].to_dict() if best in result.board.index else {}
    hold = result.holdout
    return {
        "at": meta.get("started_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git["commit"], "git_dirty": git["dirty"],
        "seed": meta.get("seed"), "cv": cli.get("cv"), "cost_bps": meta.get("cost_bps"),
        "models": meta.get("candidates", list(result.reports)),
        "best": best, "task": meta.get("task"), "primary_metric": meta.get("primary_metric"),
        "cv_metrics": _pick(row),
        "holdout": _pick(hold.overall) if hold is not None else None,
        "variants": {name: _pick(rep.overall) for name, rep in result.experiments.items()},
        "n_dev": meta.get("n_dev"), "n_features": meta.get("n_features"),
        "wall_time_s": meta.get("wall_time_s"), "versions": meta.get("versions"),
    }


def record_run(result: ExecutionResult, cli: dict | None = None, path: Path | str = LINEAGE_PATH) -> dict:
    """Append a run to the arm's node (creating the node from the module's DNA) and save."""
    lineage = load(path)
    module = load_arm(result.exp_name)
    dna = {**_arm_meta(module), **dna_of(module)}
    node = upsert_node(lineage, result.exp_name, dna)
    run = summarise(result, cli)
    node["runs"].append(run)
    save(lineage, path)
    return run


def import_outputs(lineage: dict, output_dir: Path | str = OUTPUTS) -> dict | None:
    """Turn the latest ``outputs/run_meta.json`` + ``metrics.json`` into a run of the matching node (if that
    run is not recorded yet). Returns the run or ``None``."""
    output_dir = Path(output_dir)
    meta_p, metrics_p = output_dir / "run_meta.json", output_dir / "metrics.json"
    if not (meta_p.exists() and metrics_p.exists()):
        return None
    meta, metrics = json.loads(meta_p.read_text()), json.loads(metrics_p.read_text())
    name = metrics.get("exp_name")
    node = lineage["nodes"].get(name)
    if node is None or meta.get("exp_name") != name:
        return None
    if any(r.get("at") == meta.get("started_at") for r in node["runs"]):
        return None
    best = metrics.get("best")
    board = metrics.get("leaderboard", {})
    candidates = set(meta.get("candidates", []))
    run = {
        "at": meta.get("started_at"), "git_commit": None, "git_dirty": None,
        "seed": meta.get("seed"), "cv": None, "cost_bps": meta.get("cost_bps"),
        "models": meta.get("candidates"), "best": best, "task": meta.get("task"),
        "primary_metric": meta.get("primary_metric"),
        "cv_metrics": _pick(board.get(best, {})),
        "holdout": _pick(metrics["holdout"]) if metrics.get("holdout") else None,
        "variants": {n: _pick(m) for n, m in board.items() if n not in candidates},
        "n_dev": meta.get("n_dev"), "n_features": meta.get("n_features"),
        "wall_time_s": meta.get("wall_time_s"), "versions": meta.get("versions"),
        "imported_from": str(output_dir.relative_to(ROOT)) if output_dir.is_relative_to(ROOT) else str(output_dir),
    }
    node["runs"].append(run)
    return run


def sync(path: Path | str = LINEAGE_PATH, import_outputs_dir: Path | str | None = OUTPUTS) -> dict:
    """Add/refresh every arm from usdjpy/exp (runs preserved), import the latest outputs/, save."""
    lineage = load(path)
    arms = scan_experiments()
    for name, dna in arms.items():
        upsert_node(lineage, name, dna)
    for name, node in lineage["nodes"].items():
        if name not in arms:
            node["file"] = None
    if import_outputs_dir is not None:
        import_outputs(lineage, import_outputs_dir)
    save(lineage, path)
    return lineage


# --------------------------------------------------------------------------- queries
def latest_run(node: dict) -> dict | None:
    runs = node.get("runs") or []
    return runs[-1] if runs else None


def metric_of(node: dict, metric: str = DEFAULT_METRIC, split: str = "cv") -> float | None:
    run = latest_run(node)
    if run is None:
        return None
    block = run.get("cv_metrics") if split == "cv" else run.get("holdout")
    v = (block or {}).get(metric)
    return None if v is None else float(v)


def generations(lineage: dict) -> dict[str, int]:
    """Longest path from a root, per node (roots = 0); cycles raise."""
    nodes = lineage["nodes"]
    memo: dict[str, int] = {}

    def depth(name: str, stack=()) -> int:
        if name in memo:
            return memo[name]
        if name in stack:
            raise ValueError(f"cycle in lineage through {name}")
        parents = [p for p in nodes.get(name, {}).get("parents", []) if p in nodes]
        memo[name] = 0 if not parents else 1 + max(depth(p, stack + (name,)) for p in parents)
        return memo[name]

    return {n: depth(n) for n in nodes}


def best_node(lineage: dict, metric: str = DEFAULT_METRIC, split: str = "cv") -> str | None:
    scored = {n: metric_of(node, metric, split) for n, node in lineage["nodes"].items()}
    scored = {n: v for n, v in scored.items() if v is not None}
    return max(scored, key=scored.get) if scored else None


def best_path(lineage: dict, metric: str = DEFAULT_METRIC, split: str = "cv") -> list[str]:
    """Root -> ... -> best node along primary parents (``[]`` when nothing has that metric)."""
    name = best_node(lineage, metric, split)
    if name is None:
        return []
    path, seen = [name], {name}
    while True:
        parents = lineage["nodes"].get(path[-1], {}).get("parents", [])
        if not parents or parents[0] in seen or parents[0] not in lineage["nodes"]:
            break
        path.append(parents[0])
        seen.add(parents[0])
    return path[::-1]


def table(lineage: dict, metric: str = DEFAULT_METRIC, split: str = "cv") -> pd.DataFrame:
    gens = generations(lineage)
    rows = []
    for name, node in lineage["nodes"].items():
        run = latest_run(node)
        rows.append({
            "arm": name, "gen": gens[name], "parent": (node.get("parents") or [None])[0] or "",
            "goal": node.get("goal", ""), f"cv_{metric}": metric_of(node, metric, "cv"),
            f"holdout_{metric}": metric_of(node, metric, "holdout"),
            "best": run["best"] if run else "", "runs": len(node.get("runs") or []),
            "pipeline_commit": node.get("pipeline_commit") or "",
            "git_commit": (run or {}).get("git_commit") or "",
        })
    df = pd.DataFrame(rows).set_index("arm") if rows else pd.DataFrame()
    return df.sort_values(["gen", "arm"]) if len(df) else df


# --------------------------------------------------------------------------- views
def _label_metric(node: dict, metric: str, split: str) -> str:
    v = metric_of(node, metric, split)
    if v is None:
        return "(not run)" if latest_run(node) is None else f"{metric} n/a"
    pm = node.get("primary_metric")
    extra = ""
    if pm and pm != metric:
        pv = metric_of(node, pm, split)
        if pv is not None:
            extra = f" · {pm} {pv:.3f}"
    return f"{metric} {v:.3f}{extra}"


def to_mermaid(lineage: dict, metric: str = DEFAULT_METRIC, split: str = "cv") -> str:
    best = set(best_path(lineage, metric, split))
    lines = ["graph TD"]
    for name, node in lineage["nodes"].items():
        lines.append(f'    {name}["{name}<br/>{_label_metric(node, metric, split)}"]')
    for e in edges_of(lineage):
        arrow = "-->" if e["kind"] == "primary" else "-.->"
        lines.append(f"    {e['from']} {arrow} {e['to']}")
    if best:
        lines.append(f"    classDef best fill:{style.SEQUENTIAL[2]},stroke:{style.STATUS['good']},stroke-width:2px")
        lines.append(f"    class {','.join(sorted(best))} best")
    return "\n".join(lines)


def render(lineage: dict, metric: str = DEFAULT_METRIC, split: str = "cv",
           out: Path | str = LINEAGE_FIGURE) -> Path:
    """Tree of the arms, generations top-down, nodes coloured by ``metric``, best path highlighted."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Patch

    nodes = lineage["nodes"]
    gens = generations(lineage)
    by_gen: dict[int, list[str]] = {}
    for name, g in sorted(gens.items()):
        by_gen.setdefault(g, []).append(name)
    n_gen, width = (max(by_gen) + 1) if by_gen else 1, max([len(v) for v in by_gen.values()] + [1])
    pos = {}
    for g, names in by_gen.items():
        for i, name in enumerate(names):
            pos[name] = ((i + 1) / (len(names) + 1) * width, -g)
    values = {n: metric_of(node, metric, split) for n, node in nodes.items()}
    scored = [v for v in values.values() if v is not None]
    norm = Normalize(min(scored + [0.0]), max(scored + [0.0]) or 1.0) if scored else Normalize(0, 1)
    cmap = style.cmap_sequential()
    path = best_path(lineage, metric, split)
    best_edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}

    with style.plot_style():
        fig, ax = plt.subplots(figsize=(max(7.0, 3.2 * width + 2), max(3.5, 1.7 * n_gen + 1.2)))
        ax.set_axis_off()
        ax.set_xlim(0, width)
        ax.set_ylim(-n_gen + 0.4, 0.6)
        for e in edges_of(lineage):
            if e["from"] not in pos or e["to"] not in pos:
                continue
            on_path = (e["from"], e["to"]) in best_edges
            ax.add_patch(FancyArrowPatch(pos[e["from"]], pos[e["to"]], arrowstyle="-|>", mutation_scale=14,
                                         shrinkA=22, shrinkB=22, lw=2.4 if on_path else 1.0,
                                         linestyle="-" if e["kind"] == "primary" else (0, (4, 3)),
                                         color=style.STATUS["good"] if on_path else style.INK["axis"], zorder=1))
        for name, (x, y) in pos.items():
            node, v = nodes[name], values[name]
            fill = cmap(norm(v)) if v is not None else style.INK["neutral"]
            dark = v is not None and norm(v) > 0.55
            edge = style.STATUS["good"] if name in path else style.INK["axis"]
            commit = node.get("pipeline_commit") or "?"
            text = f"{name}\n{_label_metric(node, metric, split)}\npipeline @{commit}"
            ax.text(x, y, text, ha="center", va="center", fontsize=8, zorder=3,
                    color="white" if dark else style.INK["primary"],
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=fill, edgecolor=edge,
                              linewidth=2.0 if name in path else 0.9))
        if scored:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label(f"{split} {metric} (latest run)")
        handles = [Line2D([], [], color=style.INK["axis"], lw=1.0, label="primary parent"),
                   Line2D([], [], color=style.INK["axis"], lw=1.0, linestyle=(0, (4, 3)), label="secondary parent"),
                   Line2D([], [], color=style.STATUS["good"], lw=2.4, label=f"best-{metric} path"),
                   Patch(facecolor=style.INK["neutral"], edgecolor=style.INK["axis"], label="not run")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2)
        style.finish(fig, title=f"Experiment lineage · {lineage.get('project') or PROJECT}",
                     subtitle=f"{len(nodes)} arms, {n_gen} generations; best {split} {metric}: "
                              + (f"{path[-1]} ({values[path[-1]]:.3f})" if path else "no scored arm yet"))
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        plt.close(fig)
    return out


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m usdjpy.lineage", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default=None, help=f"metric to rank / colour by (default: the JSON's, {DEFAULT_METRIC})")
    ap.add_argument("--split", choices=("cv", "holdout"), default="cv")
    ap.add_argument("--path", default=str(LINEAGE_PATH), help="lineage JSON (default: %(default)s)")
    ap.add_argument("--out", default=str(LINEAGE_FIGURE), help="PNG path (default: %(default)s)")
    ap.add_argument("--outputs", default=str(OUTPUTS), help="outputs/ folder whose latest run is imported on sync")
    ap.add_argument("--no-sync", action="store_true", help="do not rescan usdjpy/exp or import outputs/")
    ap.add_argument("--no-render", action="store_true", help="skip the PNG")
    ap.add_argument("--mermaid", action="store_true", help="print only the mermaid graph")
    args = ap.parse_args(argv)

    path = Path(args.path)
    lineage = load(path) if args.no_sync else sync(path, args.outputs)
    metric = args.metric or lineage.get("metric") or DEFAULT_METRIC
    if args.mermaid:
        print(to_mermaid(lineage, metric, args.split))
        return 0
    if not lineage["nodes"]:
        print("no arms found; add usdjpy/exp/vNNN_*.py modules with a DNA block")
        return 1
    print(f"=== experiment lineage ({path}; {len(lineage['nodes'])} arms, ranked by {args.split} {metric}) ===")
    print(md_table(table(lineage, metric, args.split)))
    bp = best_path(lineage, metric, args.split)
    if bp:
        v = metric_of(lineage["nodes"][bp[-1]], metric, args.split)
        print(f"\nbest path: {' -> '.join(bp)}  ({args.split} {metric} {v:.4f})")
    else:
        print(f"\nno arm has a recorded {args.split} {metric} yet (run `make train`)")
    if not args.no_render:
        print(f"figure: {render(lineage, metric, args.split, args.out)}")
    print("\n```mermaid\n" + to_mermaid(lineage, metric, args.split) + "\n```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
