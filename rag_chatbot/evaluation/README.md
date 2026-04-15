# RAG Evaluation

Đánh giá RAG pipeline trên 2 benchmark: **single-turn** (50 câu) và **multi-turn** (15 hội thoại / 58 lượt).

## Cấu trúc

```
evaluation/
├── datasets/
│   ├── techviet_qa_v2.json            # 50 single-turn test cases
│   └── techviet_multiturn_v1.json     # 15 multi-turn conversations (58 turns)
├── metrics.py                          # Hit@k, Recall@k, MRR + RAGAS wrapper (dùng chung)
├── run_evaluation.py                   # Script single-turn
├── run_multiturn_evaluation.py         # Script multi-turn
├── reports/                            # Report JSON
│   ├── report_YYYYMMDD_HHMMSS.json            # single-turn
│   └── multiturn_report_YYYYMMDD_HHMMSS.json  # multi-turn
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
python -m evaluation.run_evaluation --limit 5 --no-ragas   # smoke test (~1 phút)
python -m evaluation.run_evaluation                         # full 50 cases + RAGAS (~10-20 phút)
python -m evaluation.run_evaluation --limit 10              # custom limit
python -m evaluation.run_evaluation --top-k 10 --no-ragas   # custom top-k
```

**Kết quả** được lưu ở `evaluation/reports/report_<timestamp>.json`.

## Multi-turn Evaluation

Đánh giá khả năng hội thoại nhiều lượt của RAG: coreference, follow-up drill-down, topic shift, multi-hop reasoning, comparison, temporal, clarification, refusal trên follow-up ngoài corpus.

### Dataset (`techviet_multiturn_v1.json`)

15 hội thoại / 58 lượt, chia 8 category:

| Category | Conversations | Mô tả |
|---|---|---|
| `coreference_resolution` | 3 | Đại từ / demonstratives ("it", "that") tham chiếu turn trước |
| `follow_up_drilldown` | 3 | Đi sâu dần vào policy/product qua nhiều turn |
| `topic_shift` | 2 | Đổi chủ đề đột ngột — test rewriter có giữ context cũ không |
| `multi_hop_reasoning` | 2 | Tổng hợp 2-3 docs qua nhiều turn (OKR ↔ Projects ↔ Incident) |
| `comparison_across_turns` | 2 | Xây dựng so sánh qua các lượt |
| `temporal_reasoning` | 1 | Thứ tự sự kiện giữa migration / incident / roadmap |
| `clarification` | 1 | Câu mơ hồ → user làm rõ ở turn sau |
| `negation_refusal` | 1 | Follow-up hỏi info ngoài corpus + khả năng recover |

### Metrics bổ sung

Ngoài overall Hit@k / Recall@k / MRR / Refusal / RAGAS, multi-turn còn breakdown thêm:

- **By turn position** — `first` vs `follow_up`. Đo trực tiếp chất lượng query rewriting + coreference. Nếu `follow_up` thấp hơn `first` nhiều → rewriter yếu.
- **By category** — để xác định loại suy luận nào RAG còn yếu (multi-hop? topic-shift?).

### Cách chạy

```bash
# smoke test 2 conversations, bỏ RAGAS
python -m evaluation.run_multiturn_evaluation --limit 2 --no-ragas

# full (15 conversations, có RAGAS) — chậm, ~20-40 phút
python -m evaluation.run_multiturn_evaluation

# custom top-k
python -m evaluation.run_multiturn_evaluation --top-k 10 --no-ragas
```

Cách hoạt động: mỗi turn được chạy qua pipeline với `conversation_history` tích luỹ từ các turn trước (role=user/assistant). Điều này kích hoạt `QueryRewriterService` và đưa history vào LLM prompt — giống hệt production flow.

Report lưu ở `evaluation/reports/multiturn_report_<timestamp>.json` với 3 block chính:
`retrieval_metrics_overall`, `retrieval_metrics_by_turn_position`, `retrieval_metrics_by_category`.

### Ngưỡng tham khảo (multi-turn)

| Metric | Acceptable | Good |
|---|---|---|
| Hit@6 (overall) | ≥ 0.75 | ≥ 0.88 |
| Hit@6 (follow_up turns) | ≥ 0.70 | ≥ 0.85 |
| Gap `first` − `follow_up` | ≤ 0.15 | ≤ 0.05 |
| Faithfulness | ≥ 0.75 | ≥ 0.85 |
| Refusal Accuracy | ≥ 0.80 | ≥ 0.90 |

Gap lớn giữa `first` và `follow_up` là tín hiệu query rewriter / coreference còn yếu — ưu tiên fix trước khi release.

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
