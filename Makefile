.PHONY: init test clean

# Create virtual environment and install dependencies
init:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -e .
	@echo "Environment ready. Activate with: source .venv/bin/activate"

# Run tests with pytest
test:
	. .venv/bin/activate && pytest

# Run tests with coverage
test-cov:
	. .venv/bin/activate && pytest --cov=chronovisor --cov-report=term-missing

# Clean build artifacts
clean:
	rm -rf .venv
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
