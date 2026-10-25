# Common tasks. Run from the repository root; override the interpreter with PYTHON=...
PYTHON ?= python
RUFF   ?= ruff

.PHONY: help setup setup-pip setup-dev test lint validate tables tables-check tables-provenance figures check clean

help:  ## list targets
	@grep -E "^[a-z-]+:.*## " $(MAKEFILE_LIST) | sed -E "s/:.*## /\t/" | expand -t 20

setup:  ## create the conda env `bdsl_graph` (GPU, CUDA 11.8)
	conda env create -f environment.yml

setup-pip:  ## pip install runtime deps into the active env (install torch first, see requirements.txt)
	$(PYTHON) -m pip install -r requirements.txt

setup-dev:  ## pip install test/lint tooling and the pre-commit hook
	$(PYTHON) -m pip install "pytest>=7" "ruff==0.6.9" pre-commit
	pre-commit install

test:  ## unit + smoke tests (CPU, no data needed)
	CUDA_VISIBLE_DEVICES="" $(PYTHON) -m pytest

lint:  ## ruff (rules in pyproject.toml)
	$(RUFF) check .

validate:  ## repository integrity (configs, model classes, registries)
	$(PYTHON) tools/validate_project.py --skip-dependencies

tables:  ## regenerate the paper table fragments into results/tables/generated/
	$(PYTHON) tools/p2_make_tables.py

tables-check:  ## regenerate and diff against results/tables/reference/ cell by cell
	$(PYTHON) tools/p2_make_tables.py --check

tables-provenance:  ## print the source file of every generated table cell
	$(PYTHON) tools/p2_make_tables.py --dry-run --provenance

figures:  ## render the paper figures into results/figures/
	MPLBACKEND=Agg $(PYTHON) tools/make_paper_figures.py

check: lint test validate tables-check  ## everything CI runs

clean:  ## remove regenerated outputs and caches
	rm -rf results/tables/generated results/figures .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
