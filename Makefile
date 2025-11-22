# Run the FastAPI server
start:
	python -m src.api.main

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Lint code
lint:
	flake8 src tests --max-line-length=100
	black src tests --check

# Format code
format:
	black src tests

# Type check
typecheck:
	mypy src --ignore-missing-imports

# Clean up cache and build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov dist build

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Create sample data
sample-data:
	python scripts/generate_sample_data.py

# Help
help:
	@echo "Available commands:"
	@echo "  make start        - Run the FastAPI server"
	@echo "  make install      - Install dependencies"
	@echo "  make install-dev  - Install dev dependencies"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Lint code"
	@echo "  make format       - Format code"
	@echo "  make typecheck    - Type check code"
	@echo "  make clean        - Clean up cache files"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start Docker containers"
	@echo "  make docker-down  - Stop Docker containers"
	@echo "  make sample-data  - Generate sample data"

.PHONY: start install install-dev test test-cov lint format typecheck clean docker-build docker-up docker-down docker-logs sample-data help
