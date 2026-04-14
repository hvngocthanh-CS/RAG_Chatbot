#!/usr/bin/env python
"""
Cache embedding & reranker models locally BEFORE starting backend.

This script downloads BAAI/bge-base-en-v1.5 and BAAI/bge-reranker-base
to ./models folder. Run this ONCE before the first backend startup.

⚠️  REQUIRED: Backend will not start without cached models.

Usage:
    python setup_embedding_models.py

Typical workflow:
    1. python setup_embedding_models.py   # ~5-10 minutes, one-time setup
    2. uvicorn backend.api.main:app ...   # ⚡ Instant startup every time
"""
import os
import sys

# Set cache paths
os.environ["HF_HOME"] = "./models"
os.environ["TRANSFORMERS_CACHE"] = "./models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

print("=" * 70)
print("  EMBEDDING MODELS SETUP")
print("=" * 70)
print("\n📦 Caching models to ./models directory...\n")

# Download embedding model
print("1️⃣  Downloading BAAI/bge-base-en-v1.5 (embedding model)...")
print("   Size: ~500MB | Time: ~2-5 minutes\n")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print("   ✅ Embedding model cached successfully\n")
except Exception as e:
    print(f"   ❌ FAILED: {e}\n")
    print("   Troubleshooting:")
    print("   - Check internet connection")
    print("   - Try: pip install sentence-transformers torch")
    print("   - Increase timeout: HF_HUB_DOWNLOAD_TIMEOUT=600")
    sys.exit(1)

# Download reranker model
print("2️⃣  Downloading BAAI/bge-reranker-base (reranker model)...")
print("   Size: ~250MB | Time: ~1-2 minutes\n")
try:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("BAAI/bge-reranker-base")
    print("   ✅ Reranker model cached successfully\n")
except Exception as e:
    print(f"   ⚠️  WARNING: Reranker download failed (optional)\n")
    print(f"   Error: {e}")
    print("   Note: Reranker is OPTIONAL for RAG pipeline\n")

print("=" * 70)
print("✅ Setup Complete!")
print("=" * 70)
print("\n📂 Models cached in: ./models")
print("\n🚀 Next step - Start backend:")
print("   uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload")
print("\n💡 Backend will use cached models. No network delays on startup!\n")
