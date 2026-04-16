#!/usr/bin/env python
"""
One-time setup: cache HuggingFace models needed by the backend.

Scope — what this script DOES:
  • Downloads BAAI/bge-base-en-v1.5 (embedding) to ./models/
  • Downloads BAAI/bge-reranker-base (reranker) to ./models/

Scope — what this script does NOT do:
  • Does NOT pull the Ollama LLM (llama3.1:8b etc.).
    Ollama manages its own model cache at ~/.ollama/models/ on the host OS.
    Pull separately with:  ollama pull llama3.1:8b

Why two separate systems?
  HuggingFace transformers and Ollama are independent runtimes with
  independent caches. HF models load inside the Python process; Ollama
  models run in the Ollama daemon on a different port (11434). They
  can't share cache — each has to be populated once.

Usage:
    python setup_embedding_models.py

Typical workflow (run once):
    1. python setup_embedding_models.py     # ~5-10 min, HF models
    2. ollama pull llama3.1:8b              # ~5 min, Ollama LLM
    3. uvicorn backend.api.main:app ...     # instant startup thereafter
"""
import os
import sys

# Send HF downloads to the project-local ./models folder (portable).
os.environ["HF_HOME"] = "./models"
os.environ["TRANSFORMERS_CACHE"] = "./models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

print("=" * 70)
print("  HUGGINGFACE MODELS SETUP (embedding + reranker)")
print("=" * 70)
print("\nCaching HF models to ./models/ ...\n")

# Embedding model
print("[1/2] BAAI/bge-base-en-v1.5 (embedding, ~500MB)")
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("BAAI/bge-base-en-v1.5")
    print("      OK\n")
except Exception as e:
    print(f"      FAILED: {e}")
    print("      Try: pip install sentence-transformers torch")
    sys.exit(1)

# Reranker model (optional — backend runs without it if USE_RERANKER=false)
print("[2/2] BAAI/bge-reranker-base (reranker, ~250MB)")
try:
    from sentence_transformers import CrossEncoder
    CrossEncoder("BAAI/bge-reranker-base")
    print("      OK\n")
except Exception as e:
    print(f"      WARNING: reranker download failed ({e})")
    print("      Reranker is optional — set USE_RERANKER=false in .env to skip.\n")

print("=" * 70)
print("HF setup complete.")
print("=" * 70)
print("\nNext steps:")
print("  1. Pull Ollama LLM:   ollama pull llama3.1:8b")
print("  2. Start backend:     uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload")
