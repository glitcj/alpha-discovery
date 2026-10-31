# USDJPY 1-day forward log return, cross-asset (v003) / USDJPY-only (v002) / 1-hour direction (v001) — everything runs inside the Docker image.
IMAGE  ?= usdjpy-direction
MOUNTS := -v $(CURDIR):/app
# the container cannot see ../.git: pass the commit in so experiment-lineage.json records which pipeline ran
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null)
GIT_DIRTY  := $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo 1 || echo 0)
GIT_ENV    := -e GIT_COMMIT=$(GIT_COMMIT) -e GIT_DIRTY=$(GIT_DIRTY)
RUN    := docker run --rm $(GIT_ENV) $(MOUNTS) $(IMAGE)
RUN_IT := docker run --rm -it $(GIT_ENV) $(MOUNTS) $(IMAGE)

# what `make run` executes inside the container, e.g. make run FILE="-m usdjpy.train --n-rows 3000 --cv 3"
FILE   ?= -m usdjpy.train --help
# experiment module under usdjpy/exp/ and candidate subset for `make train` / `make evaluate` (empty = the arm's set)
EXP    ?= v003_cross_asset
MODELS ?=
CV     ?= 5
SEED   ?= 0
COST   ?= 0.5
TASK   ?= return
# metric that ranks / colours the experiment lineage tree (`make lineage`)
METRIC ?= mcc
PORT   ?= 8899
# yfinance download settings for `make fetch` (Yahoo serves 1h bars for the last ~730 days only)
TICKER ?= USDJPY=X
PERIOD ?= 729d
# the report notebook: `make run-report` re-executes it in place, `make report` exports it to ./REPORT.html
NB       ?= REPORT.ipynb
NO_INPUT ?= 1

.PHONY: help build fetch run shell notebook test lint evaluate train predict experiments lineage \
        run-report report readme all clean

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## build the docker image (rerun after touching requirements.txt)
	docker build -t $(IMAGE) .

fetch: ## download 1h bars for USDJPY + the cross-asset basket into dataset/ (the only target that needs network)
	$(RUN) python -m usdjpy.data --ticker "$(TICKER)" --period $(PERIOD) --basket

run: ## run python inside the container: make run FILE="-m usdjpy.train --cv 3"
	$(RUN) python $(FILE)

shell: ## drop into a shell in the container
	$(RUN_IT) bash

notebook: ## JupyterLab at http://localhost:8899 (no token; PORT=<n> to override)
	@echo "jupyter lab -> http://localhost:$(PORT)"
	docker run --rm -it -p $(PORT):8888 $(GIT_ENV) $(MOUNTS) $(IMAGE) \
		jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --IdentityProvider.token=''

test: ## run the unit tests (pytest, synthetic data, no network)
	$(RUN) python -m pytest -q tests

lint: ## byte-compile the package and tests
	$(RUN) python -m compileall -q usdjpy tests

evaluate: ## walk-forward CV of the experiment's candidates, print the leaderboard (no refit/holdout)
	$(RUN) python -m usdjpy.train --exp $(EXP) $(if $(MODELS),--models $(MODELS),) --cv $(CV) --seed $(SEED) --cost-bps $(COST) --evaluate-only

train: ## run experiment EXP end-to-end: CV, pick best, refit, score the holdout, write outputs/ + record the run in experiment-lineage.json
	$(RUN) python -m usdjpy.train --exp $(EXP) $(if $(MODELS),--models $(MODELS),) --cv $(CV) --seed $(SEED) --cost-bps $(COST) \
		--preanalysis --postanalysis

predict: ## forecast from the latest cached bar with the saved model (run `make fetch` first)
	$(RUN) python -m usdjpy.train --predict-only

experiments: ## run the experiment ladder (TASK=return|direction) -> outputs/experiments.json
	$(RUN) python -m usdjpy.experiments --task $(TASK) --cv $(CV) --seed $(SEED) --cost-bps $(COST)

lineage: ## sync experiment-lineage.json from usdjpy/exp/ + outputs/, render outputs/figures/lineage.png, print the best-METRIC path + mermaid
	$(RUN) python -m usdjpy.lineage --metric $(METRIC)

run-report: ## re-execute NB (default REPORT.ipynb) in place so its outputs are fresh; fails on any error output
	$(RUN) jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$(NB)"
	$(RUN) python -c "import json,sys; nb=json.load(open('$(NB)')); \
		errs=[o for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='error']; \
		sys.exit(f'{len(errs)} error output(s) in $(NB): '+', '.join(e.get('ename','?') for e in errs) if errs else 0)"

HTML     ?= REPORT.html
report: ## export NB with its saved outputs to ./$(HTML) (code hidden unless NO_INPUT=0)
	$(RUN) jupyter nbconvert --to html $(if $(filter 1,$(NO_INPUT)),--no-input,) \
		--output-dir . --output "$(HTML)" "$(NB)"
	@echo "-> $(HTML)"

readme: ## README.md = the markdown cell(s) of NB, verbatim
	$(RUN) python -c "import json; nb=json.load(open('$(NB)')); \
		md='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='markdown'); \
		open('README.md','w').write(md.strip('\n')+'\n'); print('README.md <-', '$(NB)')"

all: build test train lineage run-report report readme ## build, test, train, lineage, re-execute + export the report, refresh README

clean: ## remove generated outputs (keeps metrics.json, experiments.json, figures) and the image
	rm -rf outputs/model.joblib outputs/predictions.csv .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	docker rmi -f $(IMAGE) 2>/dev/null || true
