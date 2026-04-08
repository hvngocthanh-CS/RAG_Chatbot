#!/usr/bin/env python3
"""
RAG Evaluation Script.

Usage:
    python scripts/evaluate_rag.py                  # Full evaluation
    python scripts/evaluate_rag.py --limit 3        # Quick test (3 questions)
    python scripts/evaluate_rag.py --save-report    # Save JSON report
"""

import asyncio
import json
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.retrieval import RetrievalService
from backend.services.llm import LLMService
from backend.services import initialize_services
from backend.config import settings
from backend.evaluation.metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    EndToEndMetrics,
    RAGEvaluationResult,
)


def format_context(chunks: List[Dict]) -> str:
    """Format chunks into context string (same as chat endpoint)."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("filename", "Unknown")
        page = meta.get("page_number", "N/A")
        parts.append(f"[Source {i}: {source}, Page {page}]\n{chunk['content']}")
    return "\n---\n".join(parts)


async def evaluate_one(
    question: str,
    expected_answer: str,
    expected_pages: List[int],
    has_answer: bool,
    retrieval_service: RetrievalService,
    llm_service: LLMService,
) -> RAGEvaluationResult:
    """Run one question through the full pipeline and evaluate."""
    chunks = await retrieval_service.retrieve(query=question, top_k=6)
    context = format_context(chunks) if chunks else ""

    if chunks:
        answer = await llm_service.generate(question=question, context=context)
    else:
        answer = "No relevant documents found."

    return RAGEvaluationResult(
        hit_rate=RetrievalMetrics.hit_rate(chunks, expected_pages),
        mrr=RetrievalMetrics.mrr(chunks, expected_pages),
        precision_at_k=RetrievalMetrics.precision_at_k(chunks, expected_pages, k=6),
        ndcg_at_k=RetrievalMetrics.ndcg_at_k(chunks, expected_pages, k=6),
        faithfulness=GenerationMetrics.faithfulness(answer, context),
        rouge_l=GenerationMetrics.rouge_l(expected_answer, answer),
        answer_correctness=GenerationMetrics.answer_correctness(answer, expected_answer),
        hallucination_detected=EndToEndMetrics.hallucination_check(answer, context),
        citation_accuracy=EndToEndMetrics.citation_accuracy(answer, chunks),
        no_answer_correct=EndToEndMetrics.no_answer_correctness(answer, has_answer),
        question=question,
        answer=answer,
        num_chunks=len(chunks),
    )


async def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation")
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report")
    parser.add_argument("--dataset", default="data/eval_dataset.json", help="Eval dataset path")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"ERROR: {args.dataset} not found")
        return

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    test_cases = dataset["test_cases"]
    if args.limit:
        test_cases = test_cases[: args.limit]

    print(f"\n{'=' * 70}")
    print(f"  RAG EVALUATION — {len(test_cases)} test cases")
    print(f"{'=' * 70}\n")

    print("Initializing services...")
    await initialize_services()

    retrieval_service = RetrievalService()
    llm_service = LLMService()
    await llm_service.initialize()
    print("Ready.\n")

    results: List[Dict[str, Any]] = []

    for i, tc in enumerate(test_cases, 1):
        test_id = tc["id"]
        category = tc["category"]
        print(f"[{i}/{len(test_cases)}] {test_id} ({category})")

        try:
            result = await evaluate_one(
                question=tc["question"],
                expected_answer=tc["expected_answer"],
                expected_pages=tc.get("expected_pages", []),
                has_answer=tc.get("has_answer", True),
                retrieval_service=retrieval_service,
                llm_service=llm_service,
            )

            print(f"  Hit={result.hit_rate:.0f}  MRR={result.mrr:.2f}  "
                  f"Faith={result.faithfulness:.2f}  ROUGE={result.rouge_l:.2f}  "
                  f"Score={result.overall_score:.2f}")

            results.append({
                "test_id": test_id,
                "category": category,
                "question": tc["question"],
                "metrics": result.to_dict(),
                "overall_score": result.overall_score,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"test_id": test_id, "error": str(e)})

    # Summary
    valid = [r for r in results if "metrics" in r]
    if not valid:
        print("\nNo successful evaluations.")
        return

    avg_overall = sum(r["overall_score"] for r in valid) / len(valid)
    print(f"\n{'=' * 70}")
    print(f"  OVERALL SCORE: {avg_overall:.3f}")
    print(f"  Passed: {len(valid)}/{len(test_cases)}")
    print(f"{'=' * 70}\n")

    if args.save_report:
        os.makedirs("data", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"data/eval_report_{ts}.json"

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "model": settings.OLLAMA_MODEL,
                "embedding": settings.EMBEDDING_MODEL,
                "reranker": settings.RERANKER_MODEL if settings.USE_RERANKER else "disabled",
            },
            "summary": {"overall_score": avg_overall, "total_cases": len(test_cases), "passed": len(valid)},
            "results": results,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Report saved: {path}\n")


if __name__ == "__main__":
    asyncio.run(main())
