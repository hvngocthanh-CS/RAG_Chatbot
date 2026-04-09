# RAG Chatbot - Ollama Local Setup

Production-ready RAG (Retrieval-Augmented Generation) chatbot system for Windows using **Ollama** for local LLM inference.

Thêm data vào Qdrant
python scripts/ingest_documents.py C:\Users\thanhhvn\Project\RAG_Chatbot\data

Cách 1: Dev local (code xong test nhanh, có hot-reload)
                                                                                                                                                                                  
  # Terminal 1 - Qdrant                                     
  docker run --name qdrant -p 6333:6333 qdrant/qdrant

  # Terminal 2 - Ollama
  ollama serve

  # Terminal 3 - Server (có --reload)
  uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

  Cách 2: Docker Compose (deploy, demo, hoặc không muốn cài gì trên máy)

  # Bước 1 - Ollama vẫn chạy trên host (cần GPU trực tiếp)
  ollama serve

  # Bước 2 - Qdrant + Redis + Backend chạy bằng Docker
  docker-compose -f docker/docker-compose.yml up -d

  Backend trong Docker sẽ tự kết nối tới Ollama trên host qua host.docker.internal:11434.


Chạy evaluation (xem `evaluation/README.md` để biết chi tiết)

  pip install ragas datasets langchain-community pyyaml
  make evaluate-quick                              # smoke test (5 cases, không RAGAS)
  make evaluate                                    # full run (retrieval + RAGAS)
  python -m evaluation.run_evaluation --categories factual_easy multi_hop