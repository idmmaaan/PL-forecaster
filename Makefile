.DEFAULT_GOAL := help
SHELL := /bin/bash

UV ?= uv
NPM ?= npm
WEB_DIR := apps/web
API_DIR := apps/api
COMPOSE := docker compose -f infra/compose.yaml

.PHONY: help install format lint lint-py lint-web typecheck typecheck-py typecheck-web \
        test test-api test-ml test-web check db-up db-down db-logs db-reset migrate \
        revision seed sync-fixtures ingest features baselines candidates train api web clean

help: ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## --- Setup ---

install: ## Install Python and Node dependencies
	$(UV) sync --python 3.12
	cd $(WEB_DIR) && $(NPM) install

## --- Quality ---

format: ## Format Python sources
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

lint: lint-py lint-web ## Lint everything

lint-py: ## Lint Python sources
	$(UV) run ruff check .
	$(UV) run ruff format --check .

lint-web: ## Lint the frontend
	cd $(WEB_DIR) && $(NPM) run lint

typecheck: typecheck-py typecheck-web ## Type-check everything

typecheck-py: ## Type-check Python sources
	$(UV) run mypy $(API_DIR)/app ml/src

typecheck-web: ## Type-check the frontend
	cd $(WEB_DIR) && $(NPM) run typecheck

## --- Tests ---

test: test-api test-ml test-web ## Run every test suite

test-api: ## Run the API tests
	$(UV) run pytest $(API_DIR)/tests

test-ml: ## Run the ML tests
	$(UV) run pytest ml/tests

test-web: ## Run the frontend tests
	cd $(WEB_DIR) && $(NPM) test

check: lint typecheck test ## Lint, type-check, and test

## --- Database ---

db-up: ## Start PostgreSQL and wait until it is healthy
	$(COMPOSE) up -d --wait

db-down: ## Stop PostgreSQL, keeping the data volume
	$(COMPOSE) down

db-logs: ## Tail the PostgreSQL logs
	$(COMPOSE) logs -f postgres

db-reset: ## Destroy the database volume and recreate an empty schema
	$(COMPOSE) down -v
	$(MAKE) db-up
	$(MAKE) migrate

migrate: ## Apply all migrations
	cd $(API_DIR) && $(UV) run alembic upgrade head

revision: ## Autogenerate a migration: make revision m="add table"
	cd $(API_DIR) && $(UV) run alembic revision --autogenerate -m "$(m)"

seed: ## Load the development fixtures into PostgreSQL
	$(UV) run python -m app.db.seed

sync-fixtures: ## Import real fixtures from football-data.org (needs FOOTBALL_DATA_API_TOKEN)
	$(UV) run python -m app.db.sync_fixtures $(if $(season),--season $(season),)

## --- ML pipeline (offline, CLI only) ---

ingest: ## Download and canonicalise historical seasons: make ingest seasons=2010:2025
	$(UV) run python -m epl_predictor.data.ingest $(if $(seasons),--seasons $(seasons),)

features: ## Build the v1 feature table from the canonical matches
	$(UV) run python -m epl_predictor.features.builder

## --- Run ---

api: ## Serve the API with reload on http://localhost:8000
	cd $(API_DIR) && $(UV) run uvicorn app.main:app --reload --port 8000

web: ## Serve the web app on http://localhost:5173
	cd $(WEB_DIR) && $(NPM) run dev

clean: ## Remove caches and build output
	find . -name '__pycache__' -type d -prune -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache $(WEB_DIR)/dist

baselines: ## Train and compare the mandatory baselines and calibration methods
	$(UV) run python -m epl_predictor.training.baselines

candidates: ## Benchmark the Hugging Face candidates against CatBoost: make candidates only="tabicl"
	$(UV) run python -m epl_predictor.training.candidates $(if $(only),--only $(only),)

train: ## Train a candidate artifact: make train adapter=catboost version=1.0.0
	$(UV) run python -m epl_predictor.training.train \
		$(if $(adapter),--adapter $(adapter),) \
		$(if $(version),--version $(version),) \
		$(if $(calibration),--calibration $(calibration),)
