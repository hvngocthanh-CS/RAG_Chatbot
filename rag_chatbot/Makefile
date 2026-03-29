.PHONY: help install setup-gpu start-ollama start-backend test health

# ===========================================
# RAG Chatbot Makefile
# Production-ready setup with Ollama
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
	@echo "  make test              - Run tests"
	@echo "  make health            - Check health endpoints"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Setup GPU (PyTorch CUDA 12.6)
setup-gpu:
	@echo "Installing PyTorch CUDA 12.6..."
	pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
	@echo "✓ GPU setup complete"
	@python check_gpu.py

# Start vLLM server
start-vllm:
	@echo "Starting vLLM server..."
	@if [ "$$(uname)" = "Darwin" ] || [ "$$(uname)" = "Linux" ]; then \
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
	python scripts/run_server.py

# Test health endpoints
health:
	@echo "Testing Ollama health..."
	@curl -s http://localhost:11434 || echo "Ollama not responding"
	@echo ""
	@echo "Testing RAG API health..."
	@curl -s http://localhost:8000/health || echo "API not responding"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -f docker/Dockerfile -t rag-chatbot:latest .

docker-up:
	@echo "Starting Docker Compose services..."
	docker-compose -f docker/docker-compose.vllm.yml up -d
	@echo "✓ Services started"
	@docker-compose -f docker/docker-compose.vllm.yml ps

docker-down:
	@echo "Stopping Docker Compose services..."
	docker-compose -f docker/docker-compose.vllm.yml down
	@echo "✓ Services stopped"

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache build dist *.egg-info
	@echo "✓ Cleanup complete"

# Quick setup for new environment
quick-setup: install setup-gpu
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Update .env with your settings"
	@echo "2. Run: make start-all"
	@echo "3. Test: make health"
	@echo "4. Monitor: make monitor"
