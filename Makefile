.DEFAULT_GOAL := help
.PHONY: help install db-up db-down metrics-up metrics-down generate validate load reset serve test lint docker-build docker-run deploy

PY := .venv/bin/python
PIP := .venv/bin/pip
PORT ?= 8000

help: ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install dependencies
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

db-up: ## Start local Postgres and Redis
	docker compose up -d postgres redis

db-down: ## Stop local Postgres and Redis, keeping volumes
	docker compose stop postgres redis

metrics-up: ## Start local Prometheus + Grafana (dashboard on http://localhost:3000)
	docker compose --profile observability up -d
	@echo "Grafana: http://localhost:3000   Prometheus: http://localhost:9090"

metrics-down: ## Stop the local metrics stack
	docker compose --profile observability down

generate: ## Simulate the dataset into data/*.parquet
	$(PY) -m src.data_generation.build_all

validate: ## Check the generated data against the business invariants
	$(PY) -m src.data_generation.validate_data

load: ## Create the schema and COPY the parquet files into Postgres
	$(PY) scripts/load_to_postgres.py

reset: generate validate load ## Regenerate, validate and reload from scratch

serve: ## Run the API and UI locally with reload
	.venv/bin/uvicorn src.api.app:app --reload --port $(PORT)

test: ## Run the test suite
	.venv/bin/pytest -q

lint: ## Check formatting and lint
	.venv/bin/ruff format --check .
	.venv/bin/ruff check .

docker-build: ## Build the container image
	docker build -t ride-hailing-analytics .

docker-run: ## Run the whole stack in Docker
	docker compose --profile full up --build

deploy: ## Deploy to fly.io
	fly deploy
