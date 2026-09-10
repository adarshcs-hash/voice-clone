.PHONY: help install install-dev lint fmt typecheck test test-all cov serve docker clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.venv:
	python3 -m venv .venv

install: .venv ## Install the package (runtime deps only)
	$(PIP) install -e .

install-dev: .venv ## Install with dev + eval extras (no model runtimes)
	$(PIP) install -e ".[dev,eval]"

lint: ## Ruff lint + format check
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

fmt: ## Autoformat
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

typecheck: ## mypy --strict
	$(PY) -m mypy

test: ## Fast suite (no model weights required)
	$(PY) -m pytest -m "not slow"

test-all: ## Full suite including model-dependent tests
	$(PY) -m pytest

cov: ## Coverage report
	$(PY) -m pytest -m "not slow" --cov --cov-report=term-missing --cov-report=xml

check: lint typecheck test ## Everything CI runs

serve: ## Run the API locally with reload
	$(PY) -m uvicorn mlvoice.api.app:create_app --factory --reload --port 8000

docker: ## Build the runtime image
	docker build -f docker/Dockerfile -t mlvoice:local .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
