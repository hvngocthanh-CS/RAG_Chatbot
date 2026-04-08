.PHONY: help install setup-gpu start-ollama start-backend test health

# ===========================================
# RAG Chatbot Makefile
# ===========================================

help:
	@echo "RAG Chatbot Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install           - Install all dependencies"
	@echo "  make setup-gpu         - Setup GPU (install PyTorch CUDA)"
	@echo ""
	@echo "Development:"
	@echo "  make start-ollama      - Start Ollama + Backend"
	@echo "  make start-backend     - Start FastAPI backend only"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run all tests"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-integration  - Run integration tests only"
	@echo "  make health            - Check health endpoints"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build      - Build Docker image"
	@echo "  make docker-up         - Start all services"
	@echo "  make docker-down       - Stop all services"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed"

# Setup GPU (PyTorch CUDA 12.6)
setup-gpu:
	@echo "Installing PyTorch CUDA 12.6..."
	pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
	@echo "GPU setup complete"

# Start Ollama + Backend
start-ollama:
	@echo "Starting Ollama + Backend..."
	@if [ "$$(uname)" = "Darwin" ] || [ "$$(uname)" = "Linux" ]; then \
		echo "Use: bash scripts/start_ollama.bat"; \
	else \
		cmd /c scripts\\start_ollama.bat; \
	fi

# Start FastAPI backend
start-backend:
	@echo "Starting RAG backend..."
	uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# Test health endpoints
health:
	@echo "Testing Ollama health..."
	@curl -s http://localhost:11434 || echo "Ollama not responding"
	@echo ""
	@echo "Testing RAG API health..."
	@curl -s http://localhost:8000/health || echo "API not responding"

# Run all tests
test:
	@echo "Running all tests..."
	pytest tests/ -v --tb=short

# Run unit tests only
test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v --tb=short

# Run integration tests only
test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v --tb=short

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -f docker/Dockerfile -t rag-chatbot:latest .

docker-up:
	@echo "Starting Docker Compose services..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo "Services started"

docker-down:
	@echo "Stopping Docker Compose services..."
	docker-compose -f docker/docker-compose.yml down
	@echo "Services stopped"

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache build dist *.egg-info
	@echo "Cleanup complete"

# Quick setup for new environment
quick-setup: install setup-gpu
	@echo ""
	@echo "Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Update .env with your settings"
	@echo "2. Run: make start-backend"
	@echo "3. Test: make health"
