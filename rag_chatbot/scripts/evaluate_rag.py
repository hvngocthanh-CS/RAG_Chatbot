#!/usr/bin/env python3
"""
RAG Evaluation Script.

Evaluates the RAG pipeline across 3 dimensions:

  1. RETRIEVAL — Did we find the right chunks?
     - Hit Rate: at least one correct chunk in top-k?
     - MRR: how fast does the first relevant chunk appear?
     - Precision@k: fraction of top-k chunks that are relevant
     - NDCG@k: position-weighted precision

  2. GENERATION — Is the answer good?
     - Faithfulness: answer grounded in context?
     - ROUGE-L: overlap with expected answer
     - Answer Correctness: word-level F1 vs expected answer

  3. END-TO-END — Does the full pipeline work?
     - Hallucination: fabricated numbers detected?
     - Citation Accuracy: [Source N] references valid?
     - No-Answer Correctness: refuses when should, answers when should?

Usage:
    python scripts/evaluate_rag.py                  # Full evaluation
    python scripts/evaluate_rag.py --limit 3        # Quick test (3 questions)
    python scripts/evaluate_rag.py --save-report    # Save JSON report

Requires:
    - Server running (Qdrant + Ollama)
    - Document already uploaded
    - data/eval_dataset.json (evaluation questions)
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
from backend.evaluation.comprehensive_rag_metrics import (
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

    # Step 1: Retrieve
    chunks = await retrieval_service.retrieve(query=question, top_k=6)
    context = format_context(chunks) if chunks else ""

    # Step 2: Generate
    if chunks:
        answer = await llm_service.generate(question=question, context=context)
    else:
        answer = "No relevant documents found."

    # Step 3: Evaluate
    return RAGEvaluationResult(
        # Retrieval metrics
        hit_rate=RetrievalMetrics.hit_rate(chunks, expected_pages),
        mrr=RetrievalMetrics.mrr(chunks, expected_pages),
        precision_at_k=RetrievalMetrics.precision_at_k(chunks, expected_pages, k=6),
        ndcg_at_k=RetrievalMetrics.ndcg_at_k(chunks, expected_pages, k=6),
        # Generation metrics
        faithfulness=GenerationMetrics.faithfulness(answer, context),
        rouge_l=GenerationMetrics.rouge_l(expected_answer, answer),
        answer_correctness=GenerationMetrics.answer_correctness(answer, expected_answer),
        # End-to-end metrics
        hallucination_detected=EndToEndMetrics.hallucination_check(answer, context),
        citation_accuracy=EndToEndMetrics.citation_accuracy(answer, chunks),
        no_answer_correct=EndToEndMetrics.no_answer_correctness(answer, has_answer),
        # Metadata
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

    # Load dataset
    if not os.path.exists(args.dataset):
        print(f"ERROR: {args.dataset} not found")
        print("Create it first or upload a document and generate test questions.")
        return

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    test_cases = dataset["test_cases"]
    if args.limit:
        test_cases = test_cases[: args.limit]

    print(f"\n{'=' * 70}")
    print(f"  RAG EVALUATION — {len(test_cases)} test cases")
    print(f"  Document: {dataset.get('document', 'unknown')}")
    print(f"{'=' * 70}\n")

    # Initialize services
    print("Initializing services...")
    await initialize_services()

    retrieval_service = RetrievalService()
    llm_service = LLMService()
    await llm_service.initialize()
    print("Ready.\n")

    # Run evaluations
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
                "answer_preview": result.answer[:100] + "..." if len(result.answer) > 100 else result.answer,
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

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}\n")

    # Retrieval summary
    avg_hit = sum(r["metrics"]["retrieval"]["hit_rate"] for r in valid) / len(valid)
    avg_mrr = sum(r["metrics"]["retrieval"]["mrr"] for r in valid) / len(valid)
    avg_prec = sum(r["metrics"]["retrieval"]["precision@k"] for r in valid) / len(valid)
    avg_ndcg = sum(r["metrics"]["retrieval"]["ndcg@k"] for r in valid) / len(valid)

    print("1. RETRIEVAL (did we find the right chunks?)")
    print(f"   Hit Rate:      {avg_hit:.3f}  {'GOOD' if avg_hit >= 0.8 else 'NEEDS WORK'}")
    print(f"   MRR:           {avg_mrr:.3f}  {'GOOD' if avg_mrr >= 0.5 else 'NEEDS WORK'}")
    print(f"   Precision@k:   {avg_prec:.3f}")
    print(f"   NDCG@k:        {avg_ndcg:.3f}")

    # Generation summary
    avg_faith = sum(r["metrics"]["generation"]["faithfulness"] for r in valid) / len(valid)
    avg_rouge = sum(r["metrics"]["generation"]["rouge_l"] for r in valid) / len(valid)
    avg_correct = sum(r["metrics"]["generation"]["answer_correctness"] for r in valid) / len(valid)

    print(f"\n2. GENERATION (is the answer good?)")
    print(f"   Faithfulness:  {avg_faith:.3f}  {'GOOD' if avg_faith >= 0.7 else 'NEEDS WORK'}")
    print(f"   ROUGE-L:       {avg_rouge:.3f}")
    print(f"   Correctness:   {avg_correct:.3f}")

    # E2E summary
    hall_count = sum(1 for r in valid if r["metrics"]["end_to_end"]["hallucination_detected"])
    avg_cite = sum(r["metrics"]["end_to_end"]["citation_accuracy"] for r in valid) / len(valid)
    avg_noans = sum(r["metrics"]["end_to_end"]["no_answer_correct"] for r in valid) / len(valid)

    print(f"\n3. END-TO-END (full pipeline)")
    print(f"   Hallucinations: {hall_count}/{len(valid)}  {'GOOD' if hall_count == 0 else 'WARNING'}")
    print(f"   Citation Acc:   {avg_cite:.3f}")
    print(f"   No-Answer:      {avg_noans:.3f}")

    # Overall
    avg_overall = sum(r["overall_score"] for r in valid) / len(valid)
    print(f"\n{'=' * 70}")
    print(f"  OVERALL SCORE: {avg_overall:.3f}")
    if avg_overall >= 0.7:
        print("  Status: GOOD")
    elif avg_overall >= 0.5:
        print("  Status: ACCEPTABLE — room for improvement")
    else:
        print("  Status: NEEDS WORK")
    print(f"  Passed: {len(valid)}/{len(test_cases)}")
    print(f"{'=' * 70}\n")

    # Save report
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
                "hybrid_search": settings.USE_HYBRID_SEARCH,
            },
            "summary": {
                "overall_score": avg_overall,
                "hit_rate": avg_hit,
                "mrr": avg_mrr,
                "faithfulness": avg_faith,
                "hallucinations": hall_count,
                "total_cases": len(test_cases),
                "passed": len(valid),
            },
            "results": results,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Report saved: {path}\n")


if __name__ == "__main__":
    asyncio.run(main())
