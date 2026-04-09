.PHONY: help install setup-gpu start-ollama start-backend test test-unit health evaluate evaluate-quick

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
	@echo "  make test              - Run all unit tests"
	@echo "  make test-unit         - Alias for make test"
	@echo "  make health            - Check health endpoints"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate          - Full RAG evaluation (retrieval + RAGAS)"
	@echo "  make evaluate-quick    - Quick smoke test (5 cases, no RAGAS)"
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
	@echo "Running unit tests..."
	pytest tests/unit/ -v --tb=short

# Alias
test-unit: test

# Full RAG evaluation (retrieval metrics + RAGAS)
evaluate:
	@echo "Running full RAG evaluation..."
	python -m evaluation.run_evaluation

# Quick smoke test — 5 cases, retrieval metrics only (no LLM judge)
evaluate-quick:
	@echo "Running quick RAG smoke test..."
	python -m evaluation.run_evaluation --limit 5 --no-ragas

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
