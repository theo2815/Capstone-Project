.PHONY: run dev test lint format migrate docker-up docker-down clean

# Run the API server (production mode)
run:
	uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000 --workers 2

# Run the API server (development mode with reload)
dev:
	uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000 --reload

# Run tests
test:
	pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=html

# Lint and format check
lint:
	ruff check src/ tests/
	mypy src/

# Auto-format code
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Run database migrations
migrate:
	alembic upgrade head

# Create a new migration
migration:
	alembic revision --autogenerate -m "$(msg)"

# Start all services with Docker
docker-up:
	docker compose up --build -d

# Stop all Docker services
docker-down:
	docker compose down

# Start with GPU support
docker-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

# Start Celery worker locally
celery:
	celery -A src.workers.celery_app worker --loglevel=info --concurrency=2

# Install dependencies for development
install:
	pip install -e ".[dev]"

# Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
