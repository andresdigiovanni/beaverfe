.PHONY: help install clean lint format type-check test integration coverage tox commit bump build docs

# ── Help ──────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ───────────────────────────────────────────────────────────────

install: ## Install dependencies and pre-commit hooks
	uv sync
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .tox/ outputs/

# ── Quality ───────────────────────────────────────────────────────────────────

lint: ## Run all linters and checks
	uv run pre-commit run --all-files

format: ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .

type-check: ## Run type checking with mypy
	uv run mypy src/

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## Run unit tests
	uv run pytest

integration: ## Run integration tests (slow — full Optuna search loops)
	uv run pytest tests_integration/

coverage: ## Run tests with coverage report
	uv run pytest --cov --cov-report=term-missing --cov-report=html

tox: ## Run tests across Python versions
	uv run tox -p auto

# ── Release ───────────────────────────────────────────────────────────────────

commit: ## Interactive commit with commitizen
	uv run cz commit

bump: ## Bump version and update CHANGELOG
	uv run cz bump

build: ## Build distribution package
	uv build
