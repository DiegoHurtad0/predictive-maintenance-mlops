# =============================================================================
# Makefile — CLI shortcuts for the Predictive Maintenance Pipeline
# =============================================================================
# I designed these targets so that every common operation is a single command.
# `make train` runs the full pipeline. `make serve` starts the API. Simple.

.PHONY: help setup train test lint serve docker-train docker-serve clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtual environment and install dependencies
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Activate with: source .venv/bin/activate"

train: ## Run the full training + inference pipeline
	PYTHONPATH=. python -m src.pipeline

test: ## Run the test suite with pytest
	PYTHONPATH=. pytest tests/ -v --tb=short

lint: ## Run Ruff linter and MyPy type checker
	ruff check src/ api/ tests/
	mypy src/ api/ --ignore-missing-imports

serve: ## Start the FastAPI prediction server locally
	PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-train: ## Run training pipeline in Docker
	docker-compose --profile train run pipeline

docker-serve: ## Start the full serving stack (API + MLflow + monitoring)
	docker-compose up api mlflow

docker-monitor: ## Start full stack including Prometheus + Grafana
	docker-compose --profile monitoring up

clean: ## Remove generated artifacts and caches
	rm -rf outputs/predictions/* outputs/metrics/* logs/*.log
	rm -rf data/processed/*
	rm -rf __pycache__ src/__pycache__ api/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
