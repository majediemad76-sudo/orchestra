# Multi-Model Orchestrator -- development entry points.
#
# Everything runs through the project venv explicitly. There is no `activate`
# step: a Make recipe is its own shell, so an activated venv would not survive
# from one line to the next anyway.

PY      := ./.venv/bin/python
PIP     := ./.venv/bin/pip
SOURCES := $(shell find . -name '*.py' -not -path './.venv/*')

# Small enough to be honest about, large enough to finish a short task.
RUN_BUDGET ?= 0.10
RUN_ROUNDS ?= 2
# Ceiling for the whole eval suite, not per task.
EVAL_BUDGET ?= 1.00
# Fixture drafting: how many accepted runs to mine, and the ceiling for the job.
FIXTURE_RUNS   ?= 10
FIXTURE_BUDGET ?= 0.50
# Extra flags, e.g. FIXTURE_FLAGS=--yes for a non-interactive shell.
FIXTURE_FLAGS  ?=
GOAL       ?= Write a two-sentence description of what this orchestrator does.

.DEFAULT_GOAL := help
.PHONY: help venv install test check lint run ui eval fixture clean distclean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv if it is missing
	@test -d .venv || python3 -m venv .venv

install: venv ## Install pinned dependencies
	$(PIP) install -r requirements.txt

test: ## Run the offline self check (no API keys, no network)
	@$(PY) scripts/self_check.py

check: lint test ## Compile every source file, then run the self check

lint: ## Byte-compile every source file
	@$(PY) -m py_compile $(SOURCES) && echo "py_compile: OK"

run: ## Live smoke run against the real APIs -- costs money, needs .env
	@test -f .env || { echo "no .env -- copy .env.example and fill in the keys"; exit 1; }
	$(PY) controller.py "$(GOAL)" --budget $(RUN_BUDGET) --max-rounds $(RUN_ROUNDS)

ui: ## Launch the Streamlit observer UI
	./.venv/bin/streamlit run app.py

eval: ## Run the eval suite against the live APIs -- costs money, needs .env
	@test -f .env || { echo "no .env -- copy .env.example and fill in the keys"; exit 1; }
	$(PY) scripts/eval_suite.py --budget $(EVAL_BUDGET)

fixture: ## Draft critic fixtures from accepted runs -- costs money, needs .env
	@test -f .env || { echo "no .env -- copy .env.example and fill in the keys"; exit 1; }
	$(PY) scripts/make_fixtures.py --limit $(FIXTURE_RUNS) --budget $(FIXTURE_BUDGET) $(FIXTURE_FLAGS)

clean: ## Remove bytecode caches and self-check run logs
	@find . -name '__pycache__' -type d -not -path './.venv/*' -prune -exec rm -rf {} +
	@find . -name '*.pyc' -not -path './.venv/*' -delete
	@rm -f runs/selfcheck-*.jsonl runs/eval-*.jsonl
	@echo "cleaned (real run logs in runs/ were kept)"

distclean: clean ## Also delete every run log and the virtualenv
	@rm -f runs/*.jsonl
	@rm -rf .venv
