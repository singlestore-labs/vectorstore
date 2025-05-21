.PHONY: init install clean test unit-test integration-test lint format format-check help

# Default target executed when no arguments are given to make
default: help

# Initialize the development environment
init:
	pip install --upgrade pip
	pip install poetry
	poetry install --all-extras

# Install only production dependencies
install:
	poetry install --with test,dev

# Clean up build artifacts and caches
clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +

# Run all tests
test: unit-test integration-test

# Run only unit tests
unit-test:
	poetry run pytest -xvs tests/unit_tests

# Run integration tests (includes Docker-based tests)
integration-test:
	poetry run pytest -xvs tests/integration_tests

# Code quality checks
lint:
	poetry run ruff check .

# Format the code
format:
	poetry run ruff format .

# Check formatting
format-check:
	poetry run ruff format --check .

# Spell check
spell-check:
	poetry run codespell .

# Check all code quality aspects
check: format-check lint spell-check

# Build package distributions
build: clean
	poetry build

# Publish to PyPI
publish: build
	poetry publish

# Show help
help:
	@echo "Available targets:"
	@echo "  init          - Initialize development environment"
	@echo "  install       - Install production dependencies"
	@echo "  clean         - Clean build artifacts"
	@echo "  test          - Run all tests"
	@echo "  unit-test     - Run unit tests"
	@echo "  integration-test - Run integration tests"
	@echo "  format        - Format the code"
	@echo "  format-check  - Check code formatting"
	@echo "  lint          - Run linter"
	@echo "  spell-check   - Check for spelling errors"
	@echo "  check         - Run all code quality checks"
	@echo "  build         - Build package distributions"
	@echo "  publish       - Publish to PyPI"
	@echo "  help          - Show this help message"
