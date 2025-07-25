.PHONY: help install test lint format clean

help:
    @echo "Available targets:"
    @echo "  install   Install dependencies"
    @echo "  test      Run all tests"
    @echo "  lint      Run flake8 linter"
    @echo "  format    Run black code formatter"
    @echo "  clean     Remove Python cache and build artifacts"

install:
    pip install -r requirements.txt

test:
    pytest tests/

lint:
    flake8 src/ tests/

format:
    black src/ tests/

clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    rm -rf .pytest_cache .mypy_cache .coverage htmlcov dist build
