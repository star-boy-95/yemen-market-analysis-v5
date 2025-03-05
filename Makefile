.PHONY: setup install clean test lint format docs

# Default target
all: setup install

# Create virtual environment
setup:
	python -m venv venv
	@echo "Virtual environment created. Activate with 'source venv/bin/activate'"

# Install package
install:
	pip install -e .

# Install development dependencies
dev:
	pip install -e ".[dev]"

# Clean up
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run tests
test:
	pytest

# Run tests with coverage
coverage:
	pytest --cov=src tests/

# Lint code
lint:
	flake8 src tests

# Format code
format:
	black src tests

# Generate documentation
docs:
	sphinx-build -b html docs/source docs/build

# Run a specific notebook
run_notebook:
	jupyter notebook notebooks/00_project_initialization.ipynb
