# RAG Evaluation Framework - Professional Guide

## Overview

RAG (Retrieval-Augmented Generation) evaluation framework đánh giá **3 thành phần riêng biệt** với **16 metrics** tổng cộng, theo tiêu chuẩn industry-standard.

```
┌─────────────────────────────────────────────────────┐
│           RAG PIPELINE                              │
│  Question → [Retrieval] → [Generation] → Answer     │
└─────────────────────────────────────────────────────┘
         ↓              ↓                ↓
    RETRIEVAL       GENERATION      END-TO-END
    METRICS        METRICS         METRICS
    (5 metrics)    (5 metrics)     (6 metrics)
```

---

## 1️⃣  RETRIEVAL METRICS (Document Retrieval Quality)

**Mục đích:** Đánh giá việc lấy documents liên quan từ vector database

### Metrics

| Metric | Công thức | Ý nghĩa | Target |
|--------|-----------|---------|--------|
| **Precision@5** | `relevant_docs / 5` | Trong top 5, bao nhiêu liên quan? | ≥ 0.70 |
| **Recall@5** | `retrieved_relevant / total_relevant` | Lấy được bao nhiêu docs cần thiết? | ≥ 0.70 |
| **F1@5** | `2 * (P*R)/(P+R)` | Balance giữa Precision & Recall | ≥ 0.70 |
| **MRR** | `1 / rank_first_relevant` | Avg rank of first relevant doc | ≥ 0.80 |
| **NDCG@5** | `DCG / IDCG` | Position-weighted ranking quality | ≥ 0.75 |

### Ví dụ

```
Question: "What is the title?"
Retrieved 5 docs:
  1. Document Title + Abstract (RELEVANT) → Rank 1
  2. Section about history
  3. Methods section (RELEVANT)
  4. Bibliography
  5. Author info (RELEVANT)

→ Precision@5 = 3/5 = 0.60
→ Recall@5 = 3/3 = 1.00 (retrieved all 3 relevant docs)
→ NDCG@5 = High (relevant at positions 1, 3, 5)
→ MRR = 1/1 = 1.00 (first relevant at position 1)
```

---

## 2️⃣  GENERATION METRICS (LLM Answer Quality)

**Mục đích:** Đánh giá câu trả lời mà LLM sinh ra

### Metrics

| Metric | Ý nghĩa | Cách tính | Target |
|--------|---------|----------|--------|
| **BLEU** | N-gram overlap with reference | Word-level similarity | ≥ 0.50 |
| **ROUGE-L** | Longest common subsequence | Recall-oriented similarity | ≥ 0.50 |
| **Semantic Similarity** | Meaning/semantic overlap | TF-IDF cosine | ≥ 0.60 |
| **Answer Relevancy** | Câu trả lời liên quan câu hỏi? | Q-A semantic similarity | ≥ 0.70 |
| **Faithfulness** | Không hallucinate, gắn bó context? | Sentence-level grounding | ≥ 0.80 |

### Ví dụ

```
Question: "What architecture did the paper propose?"
Reference: "The paper proposed the Transformer architecture using multi-head attention"
Answer: "The paper presented the Transformer model with multi-head attention mechanisms"

BLEU: 0.65 (good n-gram overlap)
ROUGE-L: 0.72 (good sequence overlap)
Faithfulness: 0.95 (all claims grounded in context)
```

---

## 3️⃣  END-TO-END METRICS (Complete Pipeline)

**Mục đích:** Đánh giá cả RAG pipeline hoạt động tốt không

### Metrics

| Metric | Ý nghĩa | Cách tính | Target |
|--------|---------|----------|--------|
| **Context Precision** | % chunks retrieved có ích | `useful_chunks / total_chunks` | ≥ 0.75 |
| **Context Recall** | Bao nhiêu chunks cần thiết lấy được | `retrieved_useful / all_useful` | ≥ 0.75 |
| **Context F1** | Balance của Precision & Recall | Harmonic mean | ≥ 0.75 |
| **Hallucination Detection** | Có info sai không có trong doc? | Semantic deviation check | = 0.0% (none) |
| **Citation Accuracy** | Source references có đúng? | Format + content check | ≥ 0.80 |
| **Answer Correctness** | Câu trả lời có đúng không? | vs ground_truth or vs context | ≥ 0.75 |

### Ví dụ

```
Question: "How many parameters does the model have?"
Context: "The original Transformer has 100M parameters"
Answer: "The model has 100 million parameters [Source 1: paper.pdf]"

→ Context Precision: 1.0 (all retrieved chunks useful)
→ Hallucination: Detected=False ✓ (answer grounded)
→ Citation Accuracy: 1.0 (correct source)
→ Answer Correctness: 1.0 (factually correct)

---

But if answer was: "The model has 200M parameters..."
→ Answer Correctness: 0.0 (factually wrong)
→ Hallucination: Detected=True ✗ (number not in context)
```

---

## Quick Start

```bash
# 1. Upload document
python scripts/ingest_documents.py paper.pdf

# 2. Run evaluation (tất cả test cases)
python scripts/evaluate_rag.py

# 3. Run evaluation nhanh (5 test cases)
python scripts/evaluate_rag.py --limit 5

# 4. Run + Save report
python scripts/evaluate_rag.py --save-report
```

---

## Interpreting Results

### Console Output

```
================================================================================
PROFESSIONAL RAG EVALUATION (16 Metrics - Industry Standard)
================================================================================

1/5 test_001 | factual_retrieval
  ✓ Overall Score: 0.82

[Table with all metrics]

1️⃣  RETRIEVAL METRICS (Document Retrieval Quality):
   Precision@5:  0.75  ✓
   Recall@5:     0.70  ✓
   NDCG@5:       0.78  ✓

2️⃣  GENERATION METRICS (LLM Answer Quality):
   BLEU:         0.62  ✓
   Faithfulness: 0.85  ✓

3️⃣  END-TO-END METRICS (Complete Pipeline):
   Context Prec:     0.80
   Hallucinations:   0/5 ✓ NONE

✅ Passed: 5/5
```

### Benchmark Scores

| Rating | Retrieval | Generation | E2E | Overall |
|--------|-----------|------------|-----|---------|
| 🟢 Excellent | > 0.80 | > 0.80 | > 0.80 | > 0.80 |
| 🟡 Good | 0.70-0.80 | 0.70-0.80 | 0.70-0.80 | 0.70-0.80 |
| 🔴 Needs Work | < 0.70 | < 0.70 | < 0.70 | < 0.70 |

---

## For Interview Preparation

### Key Points to Mention

1. **3-Dimensional Evaluation**
   - "We evaluate across retrieval, generation, and end-to-end"
   - "Retrieval quality doesn't guarantee good answers"

2. **Standard Industry Metrics**
   - Reference: NIST IR standards, RAGAS paper, DeepEval
   - "16 metrics give comprehensive view of RAG system"

3. **What Each Dimension Measures**
   - Retrieval: "Can we find the right documents?"
   - Generation: "Can LLM generate good answers?"
   - E2E: "Does the whole pipeline work together?"

4. **Real Example**
   ```
   Q: "What is the output dimensionality?"
   
   Without good retrieval: ⚠ Docs don't contain answer
   → Zero retrieval precision
   
   With good retrieval + bad generation: ⚠ A bad answer
   → High precision, low faithfulness
   
   With good E2E: ✓ Correct cited answer
   → All metrics high, zero hallucination
   ```

### Interview Questions You Can Answer

- **"How do you evaluate a RAG system?"**
  - Answer: Retrieval + Generation + End-to-End metrics
  
- **"What's the difference between BLEU and ROUGE?"**
  - BLEU: N-gram (word-level) overlap
  - ROUGE: Longest common subsequence (sentence-level)

- **"How do you detect hallucination?"**
  - Semantic deviation analysis
  - Unknown entities not in context
  - Low text-context similarity

- **"What's context precision vs recall?"**
  - Precision: Are retrieved chunks useful?
  - Recall: Did we get all needed chunks?

---

## Files Structure

```
backend/evaluation/
├── comprehensive_rag_metrics.py  (Main metrics implementation)
│   ├── RetrievalMetrics (5 metrics)
│   ├── GenerationMetrics (5 metrics)
│   ├── EndToEndMetrics (6 metrics)
│   └── RAGEvaluationResult (16 metrics total)
└── rag_metrics.py (Legacy - keep for compatibility)

scripts/
├── evaluate_rag.py (Main evaluation script - UPDATED)
└── ingest_documents.py (Document upload)

data/
└── eval_dataset.json (26 test cases)

logs/
└── evaluation_*.json (Results)
```

---

## For Production Use

```python
# Use in your code
from backend.evaluation.comprehensive_rag_metrics import RAGEvaluator

evaluator = RAGEvaluator()

result = await evaluator.evaluate_question(
    question="What is X?",
    retrieval_service=retrieval_svc,
    llm_service=llm_svc
)

print(f"Retrieval Score: {result.precision_at_5:.2f}")
print(f"Generation Score: {result.faithfulness:.2f}")
print(f"Overall Score: {result.overall_score:.2f}")

# Export to JSON
report = result.to_dict()
```

---

## References

- **NIST IR Evaluation**: Precision, Recall, DCG-based metrics
- **RAGAS Paper**: Reference-free RAG evaluation
- **DeepEval**: LLM-as-a-judge metrics
- **Industry Best Practices**: Used by OpenAI, Anthropic, LangChain

---

**Last Updated:** March 17, 2026
**Framework:** Production-Grade RAG Evaluation
**Metrics:** 16 (5 Retrieval + 5 Generation + 6 End-to-End)
