#!/usr/bin/env python3
"""RAG Evaluation - main script.

Runs RAG pipeline over eval dataset and computes:
  1. Retrieval metrics (Hit@k, Recall@k, MRR, Refusal)
  2. Generation metrics (RAGAS: faithfulness, relevancy, context precision/recall)

Usage:
    python -m evaluation.run_evaluation              # full run
    python -m evaluation.run_evaluation --limit 5    # smoke test
    python -m evaluation.run_evaluation --no-ragas   # skip RAGAS (fast)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Make `backend.*` and `evaluation.*` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.services import initialize_services, get_service
from backend.services.retrieval import RetrievalService
from backend.services.llm import LLMService
from evaluation.metrics import compute_retrieval_metrics, RAGASEvaluator

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "evaluation/datasets/techviet_qa_v2.json"
REPORT_DIR = "evaluation/reports"
REFUSAL_KEYWORDS = (
    "not found", "not mentioned", "no information", "cannot find",
    "don't know", "do not know", "not available",
    "không tìm thấy", "không có thông tin",
)

def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into context for LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("filename", "Unknown")
        page = meta.get("page_number", "N/A")
        parts.append(f"[Source {i}: {source}, Page {page}]\n{chunk['content']}")
    return "\n---\n".join(parts)


async def run_pipeline(
    test_cases: List[Dict[str, Any]],
    retrieval_service: RetrievalService,
    llm_service: LLMService,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run each test case through RAG pipeline."""
    samples = []

    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] {tc['id']} ({tc.get('category', '?')})")
        sample = {
            "test_id": tc["id"],
            "category": tc.get("category", "unknown"),
            "question": tc["question"],
            "expected_answer": tc.get("expected_answer", ""),
            "expected_docs": tc.get("source_documents", []) or [],
            "has_answer": tc.get("has_answer", True),
            "retrieved_chunks": [],
            "answer": "",
            "error": None,
        }

        try:
            chunks = await retrieval_service.retrieve(query=tc["question"], top_k=top_k)
            sample["retrieved_chunks"] = chunks or []

            if chunks:
                sample["answer"] = await llm_service.generate(
                    question=tc["question"], context=format_context(chunks)
                )
            else:
                sample["answer"] = "No relevant documents found."
            print(f"      {len(chunks)} chunks retrieved")
        except Exception as e:
            print(f"      ERROR: {e}")
            sample["error"] = str(e)

        samples.append(sample)

    return samples


def is_refusal(answer: str) -> bool:
    """Check if answer is a refusal."""
    return any(kw in (answer or "").lower() for kw in REFUSAL_KEYWORDS)

def print_retrieval(metrics: Dict[str, Any]):
    print(f"\n{'='*60}\n  RETRIEVAL METRICS (k={metrics['k']})\n{'='*60}")
    print(f"  Hit@k            {metrics['hit_at_k']:.4f}")
    print(f"  Recall@k         {metrics['recall_at_k']:.4f}")
    print(f"  MRR              {metrics['mrr']:.4f}")
    print(f"  Refusal Accuracy {metrics['refusal_accuracy']:.4f} "
          f"({metrics['num_unanswerable']} unanswerable)")


def print_ragas(summary: Dict[str, float]):
    print(f"\n{'='*60}\n  RAGAS METRICS\n{'='*60}")
    for name, score in summary.items():
        print(f"  {name:20s} {score:.4f}")

async def main():
    parser = argparse.ArgumentParser(description="RAG evaluation")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help=f"Dataset path (default: {DEFAULT_DATASET})")
    parser.add_argument("--limit", type=int, help="Limit test cases")
    parser.add_argument("--top-k", type=int, default=6, dest="top_k", help="Top-k chunks (default: 6)")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS (retrieval only)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.exists(args.dataset):
        print(f"ERROR: dataset not found: {args.dataset}")
        return
    
    with open(args.dataset, encoding="utf-8") as f:
        test_cases = json.load(f)["test_cases"]
    if args.limit:
        test_cases = test_cases[:args.limit]

    print(f"\n{'='*60}")
    print("  RAG EVALUATION")
    print(f"  Dataset:   {args.dataset}")
    print(f"  Cases:     {len(test_cases)}")
    print(f"  Top-k:     {args.top_k}")
    print(f"  LLM:       {settings.OLLAMA_MODEL}")
    print(f"  Embedding: {settings.EMBEDDING_MODEL}")
    print(f"{'='*60}\n")

    print("1. Initializing services...")
    await initialize_services()
    retrieval_service = get_service("retrieval")
    llm_service = get_service("llm")
    print("   Done.\n")

    print("2. Running RAG pipeline...")
    samples = await run_pipeline(test_cases, retrieval_service, llm_service, args.top_k)
    ok = [s for s in samples if s["error"] is None]
    print(f"   Done. {len(ok)}/{len(samples)} samples collected.\n")

    print("3. Computing retrieval metrics...")
    retrieval_metrics = compute_retrieval_metrics(ok, k=args.top_k).to_dict()
    print_retrieval(retrieval_metrics)

    unanswerable = [s for s in ok if not s["has_answer"]]
    gen_refusal = None
    if unanswerable:
        correct = sum(1 for s in unanswerable if is_refusal(s["answer"]))
        gen_refusal = round(correct / len(unanswerable), 4)
        print(f"\n  Gen refusal accuracy: {gen_refusal:.4f} ({correct}/{len(unanswerable)})")

    ragas_summary = None
    if not args.no_ragas:
        print("\n4. Running RAGAS (slow - LLM judge)...")
        ragas_inputs = [
            {
                "user_input": s["question"],
                "response": s["answer"],
                "retrieved_contexts": [c["content"] for c in s["retrieved_chunks"]],
                "reference": s["expected_answer"],
            }
            for s in ok if s["has_answer"] and s["retrieved_chunks"]
        ]
        if ragas_inputs:
            evaluator = RAGASEvaluator()
            result = await evaluator.evaluate(ragas_inputs)
            ragas_summary = result["summary"]
            print_ragas(ragas_summary)

    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"report_{ts}.json")
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "dataset": args.dataset,
                "top_k": args.top_k,
                "model": settings.OLLAMA_MODEL,
                "embedding": settings.EMBEDDING_MODEL,
            },
            "num_cases": len(samples),
            "retrieval_metrics": retrieval_metrics,
            "generation_refusal_accuracy": gen_refusal,
            "ragas_metrics": ragas_summary,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Report saved: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
