# RAG Evaluation

Đánh giá RAG pipeline trên benchmark gồm 50 câu hỏi chuẩn.

## Cấu trúc

```
evaluation/
├── datasets/techviet_qa_v2.json   # 50 test cases (question + expected answer + source docs)
├── metrics.py                      # Hit@k, Recall@k, MRR + RAGAS wrapper
├── run_evaluation.py               # Script chính
├── reports/                        # Report JSON được lưu ở đây
└── README.md
```

## Metrics

### Retrieval (rẻ, deterministic)
- **Hit@k** — top-k có chứa ít nhất 1 doc đúng không?
- **Recall@k** — tỷ lệ doc đúng được tìm thấy trong top-k
- **MRR** — mean reciprocal rank; doc đúng càng lên top càng cao
- **Refusal Accuracy** — câu không có trong corpus có bị retriever trả rỗng không?

### Generation (RAGAS, cần LLM judge, chậm)
- **Faithfulness** — câu trả lời có bám vào context không?
- **Answer Relevancy** — câu trả lời có đúng câu hỏi không?
- **Context Precision** — chunks có relevant và đúng thứ tự không?
- **Context Recall** — đã retrieve đủ context cần thiết chưa?

## Cách chạy

**Chuẩn bị:**
```bash
docker run --name qdrant -p 6333:6333 qdrant/qdrant     # Qdrant
ollama serve                                             # Ollama
ollama pull llama3.2                                     # pull model
python scripts/ingest_documents.py ../data/techviet_docs # ingest corpus
pip install ragas datasets langchain-community           # deps cho RAGAS
```

**Chạy:**
```bash
make evaluate-quick              # 5 cases, bỏ RAGAS — smoke test (~1 phút)
make evaluate                    # full 50 cases + RAGAS (~10-20 phút tùy LLM)

# Hoặc gọi trực tiếp
python -m evaluation.run_evaluation --limit 10
python -m evaluation.run_evaluation --top-k 10 --no-ragas
```

**Kết quả** được lưu ở `evaluation/reports/report_<timestamp>.json`.

## Unit test (không cần infra)

```bash
pytest tests/unit/test_metrics.py -v
```

## Ngưỡng tham khảo

| Metric | Acceptable | Good |
|---|---|---|
| Hit@6 | ≥ 0.80 | ≥ 0.90 |
| Recall@6 | ≥ 0.70 | ≥ 0.85 |
| MRR | ≥ 0.60 | ≥ 0.75 |
| Faithfulness | ≥ 0.75 | ≥ 0.85 |
| Answer Relevancy | ≥ 0.75 | ≥ 0.85 |
| Refusal Accuracy | ≥ 0.80 | ≥ 0.90 |
