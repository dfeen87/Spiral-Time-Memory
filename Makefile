.PHONY: help install test lint format clean docs notebooks run-protocol-a

# Default target
help:
	@echo "Spiral-Time with Memory - Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install package and dependencies"
	@echo "  make install-dev      Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-fast        Run tests without slow ones"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make test-watch       Watch and rerun tests on changes"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linting (flake8, mypy)"
	@echo "  make format           Format code (black, isort)"
	@echo "  make format-check     Check formatting without changes"
	@echo "  make type-check       Run type checking (mypy)"
	@echo ""
	@echo "Notebooks:"
	@echo "  make notebooks        Execute all notebooks"
	@echo "  make notebooks-clean  Clear notebook outputs"
	@echo ""
	@echo "Experiments:"
	@echo "  make run-protocol-a   Run Protocol A (reset test)"
	@echo "  make run-protocol-b   Run Protocol B (process tensor)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs             Build documentation"
	@echo "  make docs-serve       Serve docs locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove build artifacts"
	@echo "  make clean-all        Deep clean including caches"
	@echo "  make check            Run all checks (format, lint, test)"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebooks]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-cov:
	pytest tests/ -v --cov=theory --cov=analysis --cov-report=html --cov-report=term

test-watch:
	pytest-watch tests/ -v

test-falsification:
	pytest tests/ -v -m falsification

# Code quality
lint:
	flake8 theory/ analysis/ simulations/ experiments/ --max-line-length=100 --extend-ignore=E203,W503
	mypy theory/ analysis/ --ignore-missing-imports

format:
	black theory/ analysis/ simulations/ experiments/ tests/ --line-length=100
	isort theory/ analysis/ simulations/ experiments/ tests/ --profile=black

format-check:
	black --check theory/ analysis/ simulations/ experiments/ --line-length=100
	isort --check-only theory/ analysis/ simulations/ experiments/ --profile=black

type-check:
	mypy theory/ analysis/ --ignore-missing-imports --pretty

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Notebooks
notebooks:
	jupyter nbconvert --to notebook --execute --inplace examples/*.ipynb

notebooks-clean:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace examples/*.ipynb

notebook-server:
	jupyter notebook examples/

# Experiments
run-protocol-a:
	python experiments/reset_tests/protocol_a.py

run-protocol-b:
	@echo "Protocol B implementation: experiments/process_tensor/protocol_b.py"
	@echo "Coming soon..."

# Documentation
docs:
	@echo "Building documentation..."
	@echo "Note: Full docs setup requires docs/ directory"

docs-serve:
	@echo "Serving documentation..."
	python -m http.server --directory docs/_build/html 8000

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean
	rm -rf .mypy_cache/
	rm -rf .tox/
	rm -rf .eggs/
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

# Full check pipeline (run before commits)
check: format-check lint test-fast
	@echo ""
	@echo "✓ All checks passed!"
	@echo "  - Formatting: OK"
	@echo "  - Linting: OK"
	@echo "  - Tests: OK"

# CI simulation
ci: format-check lint test-cov
	@echo ""
	@echo "✓ CI pipeline complete!"

# Quick development cycle
dev: format test-fast
	@echo ""
	@echo "✓ Development cycle complete!"

# Reproducibility check
reproducibility:
	@echo "Testing reproducibility..."
	python -c "import numpy as np; from theory.dynamics import *; np.random.seed(42); cfg = MemoryKernelConfig('exponential', tau_mem=1.0); K1 = memory_kernel(np.array([0,1,2]), cfg); np.random.seed(42); K2 = memory_kernel(np.array([0,1,2]), cfg); assert np.allclose(K1, K2); print('✓ Reproducibility verified')"

# Show package info
info:
	@echo "Package: spiral-time-memory"
	@echo "Version: 0.1.0"
	@echo "Python: $(shell python --version)"
	@echo "Location: $(shell pwd)"
	@pip list | grep -E "numpy|scipy|matplotlib|pytest" || echo "Dependencies not installed"

# Generate requirements from environment (for development)
freeze:
	pip freeze > requirements-frozen.txt
	@echo "✓ Requirements frozen to requirements-frozen.txt"
