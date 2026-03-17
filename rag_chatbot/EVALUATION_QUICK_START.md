# RAG Evaluation - Hướng Dẫn Nhanh

## Chuẩn bị (Lần đầu)

```bash
# 1. Upload document vào database
python scripts/ingest_documents.py paper.pdf

# 2. Chạy evaluation
python scripts/evaluate_rag.py --limit 5    # Test 5 câu
python scripts/evaluate_rag.py              # Chạy tất cả 26 câu
```

## Kết quả

- **Console:** Hiển thị progress + summary metrics
- **Logs:** `logs/evaluation_*.json` - Chi tiết từng test case

## Metrics Giải Thích

| Metric | Ý Nghĩa | Mục tiêu |
|--------|---------|---------|
| **Relevancy** | Câu trả lời liên quan đến câu hỏi | ≥ 0.7 |
| **Faithfulness** | Không hallucinate, gắn bó với context | ≥ 0.8 |
| **Citations** | Có source references đúng | ≥ 0.8 |
| **Refusal** | Từ chối câu hỏi ngoài scope đúng | ≥ 0.9 |
| **Precision** | Chunks retrieved có liên quan | ≥ 0.7 |
| **Recall** | Lấy được tất cả chunks cần thiết | ≥ 0.7 |

## Cấu hình

- **Embedding:** BGE-large-en-v1.5 (1024 dim)
- **LLM:** Ollama llama3.2
- **Vector DB:** ChromaDB
- **Chunk size:** 500 tokens
- **Top-K retrieval:** 5 chunks

## Troubleshooting

**Lỗi "No chunks retrieved"**  
→ Paper.pdf chưa upload, chạy `python scripts/ingest_documents.py paper.pdf`

**ChromaDB error**  
→ Xóa thư mục `data/chroma_db` và upload lại

**LLM timeout**  
→ Kiểm tra Ollama đang chạy: `curl http://localhost:11434/api/tags`
