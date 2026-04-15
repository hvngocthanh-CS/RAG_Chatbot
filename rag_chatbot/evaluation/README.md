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

## Quy tắc đánh giá — sample nào chạy metric nào

Dataset có 2 loại test case:
- **Answerable** (`has_answer: true`) — câu có đáp án trong corpus. Có `expected_answer` và `source_documents`.
- **Unanswerable** (`has_answer: false`) — câu ngoài corpus. `expected_answer=""`, `source_documents=[]`. Mục đích: test RAG có biết từ chối không.

Mỗi loại được chấm bằng **metric phù hợp** — **không sample nào bị bỏ qua hoàn toàn**:

| Metric | Chạy trên sample có | Lý do |
|---|---|---|
| Hit@k, Recall@k, MRR | `has_answer=true` | Cần `expected_docs` để so. Câu refusal không có docs đúng → metric vô nghĩa. |
| `retrieval_refusal_accuracy` | `has_answer=false` | Check retriever có trả empty cho câu ngoài scope không. |
| `generation_refusal_accuracy` | `has_answer=false` | Check LLM có nói "Not found in ..." đúng không. |
| RAGAS — Faithfulness, Relevancy, Context Precision/Recall | `has_answer=true` **AND** có ≥1 chunk retrieved | RAGAS cần có `reference` + `contexts` + `answer` có nội dung. Refusal không có claims để verify → chạy lên sẽ bẩn kết quả. |

**Ví dụ với dataset 50 câu (45 answerable + 5 unanswerable):**
- Retrieval IR metrics tính trên 45 câu.
- Refusal metrics tính trên 5 câu.
- RAGAS tính trên 45 câu (miễn là có chunks retrieved).
- Tổng: **mọi sample đều được chấm** — chỉ bằng metric đúng cho loại của nó.

**Vì sao không chạy RAGAS trên câu refusal:**
- Faithfulness: answer dạng "Not found in [...]" không có claims → judge không verify được → NaN.
- Context Recall: `reference=""` → chia cho 0 → NaN.
- Answer Relevancy: answer là refusal generic → điểm thấp giả tạo, không phản ánh đúng chất lượng.
→ Chạy RAGAS lên refusal sẽ **kéo điểm trung bình xuống sai lệch**.

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

## Debug RAGAS failures

RAGAS chấm bằng LLM judge, nên có thể bị **timeout** hoặc **OutputParserException** với một số sample → metric trả về NaN. Report JSON lưu 2 field để bạn biết sample nào được chấm và sample nào bị bỏ qua:

```json
"ragas_coverage": {
  "faithfulness":      {"scored": 48, "failed": 2, "total": 50},
  "answer_relevancy":  {"scored": 50, "failed": 0, "total": 50},
  "context_precision": {"scored": 49, "failed": 1, "total": 50},
  "context_recall":    {"scored": 47, "failed": 3, "total": 50}
},
"ragas_per_sample": [
  {
    "id": "multi_hop_003",
    "status": "partial",                          // "ok" | "partial" | "failed"
    "failed_metrics": ["faithfulness", "context_recall"],
    "scores": {"faithfulness": null, "answer_relevancy": 0.83, ...}
  }
]
```

- `coverage` — mỗi metric chấm được bao nhiêu sample. Mean trong `ragas_metrics` chỉ tính trên phần scored.
- `per_sample` — status từng sample: `ok` (mọi metric scored), `partial` (có metric NaN), `failed` (toàn bộ NaN).

Console cũng in coverage + danh sách ID fail ngay sau khi RAGAS xong.

### Ngưỡng cảnh báo

| Fail rate | Ý nghĩa | Hành động |
|---|---|---|
| 0% | Kết quả đáng tin hoàn toàn | Không cần làm gì |
| 1-10% | Acceptable — mean tính trên scored samples vẫn đại diện | Theo dõi, không cần đổi model |
| >10% | Judge model yếu hoặc data có pattern gây parse error | Đổi sang judge mạnh hơn (VD `llama3.1:8b` thay `llama3.2:3b`), hoặc rút ngắn answer dài |

### Nguyên nhân fail thường gặp

| Pattern | Nguyên nhân | Cách xử lý |
|---|---|---|
| `TimeoutError` | Answer quá dài (>1500 chars) → judge không parse kịp trong 300s | Rút ngắn answer bằng cách giảm `LLM_MAX_TOKENS`, hoặc giảm format cứng trong prompt |
| `OutputParserException: Invalid json` | Judge LLM (thường model nhỏ) không trả JSON hợp lệ | Đổi sang judge model mạnh hơn |
| Fail nhiều ở `context_recall` | Reference hoặc contexts quá noisy / dài | Check dataset expected_answer có quá dài không |
| Fail nhiều ở `faithfulness` | Answer có nhiều câu không cite → judge trích claim khó | Prompt bắt cite kỹ hơn, hoặc expected_answer viết gọn hơn |

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
