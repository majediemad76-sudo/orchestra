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
# Working directory for the code worker. Empty means "wherever make was run",
# which for this target is the repository itself -- so point it somewhere
# disposable before asking for a task that writes files.
RUN_CWD    ?=
# Ceiling for the whole eval suite, not per task.
EVAL_BUDGET ?= 1.00
# Fixture drafting: how many accepted runs to mine, and the ceiling for the job.
FIXTURE_RUNS   ?= 10
FIXTURE_BUDGET ?= 0.50
# Extra flags, e.g. FIXTURE_FLAGS=--yes for a non-interactive shell.
FIXTURE_FLAGS  ?=
# Critic grading run.
CRITIC_BUDGET  ?= 0.50
CRITIC_FLAGS   ?=
GOAL       ?= Write a two-sentence description of what this orchestrator does.
# HTTP surface. Loopback by default and deliberately so: keys travel in the
# request body, so any other interface needs TLS terminated in front of it.
API_HOST   ?= 127.0.0.1
API_PORT   ?= 8000

.DEFAULT_GOAL := help
.PHONY: help venv install test check ruff lint typecheck run ui serve eval fixture eval-critic clean distclean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv if it is missing
	@test -d .venv || python3 -m venv .venv

install: venv ## Install pinned dependencies
	$(PIP) install -r requirements.txt

test: ## Run the offline self check (no API keys, no network)
	@$(PY) scripts/self_check.py

check: ruff test lint ## The gate: ruff, then the self check, then py_compile

ruff: ## Lint and style-check every source file
	@$(PY) -m ruff check .

typecheck: ## Run mypy over the core modules (not part of `check` yet)
	@$(PY) -m mypy

lint: ## Byte-compile every source file
	@$(PY) -m py_compile $(SOURCES) && echo "py_compile: OK"

run: ## Live smoke run against the real APIs -- costs money, needs .env
	@test -f .env || { echo "no .env -- copy .env.example and fill in the keys"; exit 1; }
	$(PY) controller.py "$(GOAL)" --budget $(RUN_BUDGET) --max-rounds $(RUN_ROUNDS) \
		$(if $(RUN_CWD),--cwd $(RUN_CWD),)

ui: ## Launch the Streamlit observer UI
	./.venv/bin/streamlit run app.py

serve: ## Serve the orchestrator over HTTP on 127.0.0.1 (keys come per request, not from .env)
	$(PY) api.py --host $(API_HOST) --port $(API_PORT)

eval: ## Run the eval suite against the live APIs -- costs money, needs .env
	@test -f .env || { echo "no .env -- copy .env.example and fill in the keys"; exit 1; }
	$(PY) scripts/eval_suite.py --budget $(EVAL_BUDGET)

fixture: ## Draft critic fixtures from accepted runs -- costs money, needs .env
	@test -f .env || { echo "no .env -- copy .env.example and fill in the keys"; exit 1; }
	$(PY) scripts/make_fixtures.py --limit $(FIXTURE_RUNS) --budget $(FIXTURE_BUDGET) $(FIXTURE_FLAGS)

eval-critic: ## Grade the Critic against reviewed fixtures -- costs money, needs .env
	@test -f .env || { echo "no .env -- copy .env.example and fill in the keys"; exit 1; }
	$(PY) scripts/eval_critic.py --budget $(CRITIC_BUDGET) $(CRITIC_FLAGS)

clean: ## Remove bytecode caches and self-check run logs
	@find . -name '__pycache__' -type d -not -path './.venv/*' -prune -exec rm -rf {} +
	@find . -name '*.pyc' -not -path './.venv/*' -delete
	@rm -f runs/selfcheck-*.jsonl runs/eval-*.jsonl
	@echo "cleaned (real run logs in runs/ were kept)"

distclean: clean ## Also delete every run log and the virtualenv
	@rm -f runs/*.jsonl
	@rm -rf .venv
