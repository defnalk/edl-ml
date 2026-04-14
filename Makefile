.PHONY: install install-dev lint format typecheck test test-cov docs docs-serve clean generate train tune evaluate

PYTHON ?= python
PIP ?= pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[all]"
	pre-commit install

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy --config-file mypy.ini

test:
	pytest

test-cov:
	pytest --cov=edl_ml --cov-report=term-missing --cov-report=xml

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

clean:
	rm -rf build dist *.egg-info .coverage coverage.xml .mypy_cache .pytest_cache .ruff_cache
	rm -rf site docs/_build
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

generate:
	$(PYTHON) -m edl_ml.cli.main generate --n-samples 500 --output data/processed/dataset.parquet

train: generate
	$(PYTHON) -m edl_ml.cli.main train --data data/processed/dataset.parquet \
	  --checkpoint data/models/model.pt --epochs 200 \
	  --mlflow-experiment edl-ml

tune:
	$(PYTHON) -m edl_ml.cli.main tune --data data/processed/dataset.parquet \
	  --n-trials 30 --out data/models/best_config.json

evaluate:
	$(PYTHON) -m edl_ml.cli.main evaluate --checkpoint data/models/model.pt \
	  --data data/processed/dataset.parquet --figures-dir data/reports/figures
