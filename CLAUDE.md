# alpha-discovery

Research repo for alpha signals. Each experiment project lives in its own top-level folder
(`usdjpy-direction/` is the first) and mirrors the `delivery-time-prediction` layout: versioned experiment
arms calling one `pipeline.execute()`, analysis builders + pre/post reporters, a `train.py` CLI, an
`experiments.py` ladder, a `REPORT.ipynb` whose markdown cells become `README.md`, Docker + Makefile.
`ignored/` holds reference repos (ml-pipe, delivery-time-prediction) and is not tracked.

## Experiment arms and their DNA

- One arm = one file `usdjpy/exp/vNNN_<slug>.py` exposing `run(...)`. A new arm copies the latest one and
  changes one thing (or one coherent set of things); its docstring starts "vNNN on top of vMMM: ...".
  Earlier arms are never edited (metadata-only retrofits of the DNA block are the single exception).
- Every arm has a module-level `DNA` dict right after `EXP_NAME`, validated by `usdjpy.lineage`:

  ```python
  DNA = {
      "parents": ["v002_fwd_logret_1d"],   # [] for the root; first entry = primary parent (tree edge)
      "goal": "cross-asset inputs",         # one line: what this arm tries to improve
      "changes": [                          # the stack of changes, each against one parent
          {"from": "v002_fwd_logret_1d", "what": "...", "why": "..."},
      ],
      "pipeline_commit": "9572607",        # `git rev-parse --short HEAD` when the arm was authored
      "note": None,                         # optional free text (pre-lineage results, caveats)
  }
  ```

  Rules: exactly those keys; every parent is an existing arm; every `changes[].from` is a parent; an arm with
  parents lists at least one change; several parents are allowed when an arm merges two earlier ones (the
  ml-pipe "Key Changes from v20_1 / from v21_6" style, see
  `ignored/ml-pipe/src/alpaca_ml_projects/zen/exp/alhabib/y2025/d1207/ready_experiments_v22/experiment-dna-sample-v22-6.py`).

## Lineage tracking

- `experiment-lineage.json` (project root, tracked) holds one node per arm (its DNA + file/task) and every
  recorded run: timestamp, git commit + dirty flag, seed, CV metrics of the winner, holdout metrics, the
  in-arm variants, versions. `edges` is derived from `parents` for visualisers.
- `make train EXP=vNNN_<slug>` records the run automatically (full runs only: no `--n-rows`, default
  `outputs/`; `--no-lineage` opts out). The Makefile passes `GIT_COMMIT` / `GIT_DIRTY` into the container
  because `.git` is not mounted.
- `make lineage [METRIC=mcc]` rescans `usdjpy/exp/`, imports the latest `outputs/` result if it is not recorded
  yet, writes `outputs/figures/lineage.png`, prints the table, the best-METRIC path (root → best arm along primary
  parents) and a Mermaid block to paste into `REPORT.ipynb` / `README.md`.
- Loose by design while the pipeline is still moving: `pipeline_commit` (authored against) and each run's
  `git_commit` (ran against) are the only ties between an arm and the pipeline version. Experiments and the
  pipeline will later be versioned in separate repos; keep `usdjpy/lineage.py` dependent on the pipeline only
  through `ExecutionResult` so it moves with the experiments.

## Adding an arm (the workflow)

1. Copy the latest `usdjpy/exp/vNNN_*.py` to `v(NNN+1)_<slug>.py`; prepend the "on top of" paragraph.
2. Fill `DNA`: `parents`, `goal`, one `changes` entry per change, `pipeline_commit` from
   `git rev-parse --short HEAD`.
3. `make train EXP=v(NNN+1)_<slug>` (run recorded), then `make lineage`; check the tree and the best path.
4. Add findings to `REPORT.ipynb` (markdown) and refresh `README.md` (`make readme`), update the Mermaid block.
5. `make test` (synthetic data, no network) must stay green: `tests/test_lineage.py` validates every DNA block.
