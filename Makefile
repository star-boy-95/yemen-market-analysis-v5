.PHONY: setup test lint docs build clean

setup:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/

lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/

docs:
	cd docs && make html

build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete