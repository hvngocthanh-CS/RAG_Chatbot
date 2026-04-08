#!/usr/bin/env python3
"""
RAG Evaluation Script using RAGAS.

Evaluates the RAG pipeline with industry-standard RAGAS metrics:
  - Faithfulness:      Is the answer grounded in the retrieved context?
  - Answer Relevancy:  Does the answer address the question?
  - Context Precision:  Are retrieved chunks relevant?
  - Context Recall:     Did we retrieve everything needed?

Usage:
    python scripts/evaluate_rag.py                  # Full evaluation
    python scripts/evaluate_rag.py --limit 3        # Quick test (3 questions)
    python scripts/evaluate_rag.py --save-report    # Save JSON report

Requires:
    - Qdrant running (docker)
    - Ollama running with a model pulled
    - Documents uploaded to the system
    - data/eval_dataset.json
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
from backend.evaluation.metrics import RAGASEvaluator, RAGEvaluationResult


def format_context(chunks: List[Dict]) -> str:
    """Format chunks into context string (same as chat endpoint)."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("filename", "Unknown")
        page = meta.get("page_number", "N/A")
        parts.append(f"[Source {i}: {source}, Page {page}]\n{chunk['content']}")
    return "\n---\n".join(parts)


async def run_pipeline(
    test_cases: List[Dict],
    retrieval_service: RetrievalService,
    llm_service: LLMService,
) -> List[Dict[str, Any]]:
    """Run each test case through the RAG pipeline and collect samples."""
    samples = []

    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] {tc['id']} ({tc['category']})")

        try:
            # Retrieve
            chunks = await retrieval_service.retrieve(query=tc["question"], top_k=6)
            contexts = [chunk["content"] for chunk in chunks] if chunks else []
            context_str = format_context(chunks) if chunks else ""

            # Generate
            if chunks:
                answer = await llm_service.generate(
                    question=tc["question"], context=context_str,
                )
            else:
                answer = "No relevant documents found."

            samples.append({
                "user_input": tc["question"],
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": tc.get("expected_answer", ""),
                # Extra metadata for reporting
                "_test_id": tc["id"],
                "_category": tc["category"],
                "_num_chunks": len(chunks),
            })

            print(f"         Retrieved {len(chunks)} chunks, answer: {answer[:80]}...")

        except Exception as e:
            print(f"         ERROR: {e}")
            samples.append({
                "user_input": tc["question"],
                "response": f"Error: {e}",
                "retrieved_contexts": [],
                "reference": tc.get("expected_answer", ""),
                "_test_id": tc["id"],
                "_category": tc["category"],
                "_num_chunks": 0,
                "_error": str(e),
            })

    return samples


async def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation with RAGAS")
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report")
    parser.add_argument("--dataset", default="data/eval_dataset.json", help="Eval dataset path")
    parser.add_argument("--metrics", nargs="*",
                        choices=["faithfulness", "answer_relevancy",
                                 "context_precision", "context_recall"],
                        help="Select specific metrics (default: all)")
    args = parser.parse_args()

    # Load dataset
    if not os.path.exists(args.dataset):
        print(f"ERROR: {args.dataset} not found")
        return

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    test_cases = dataset["test_cases"]
    if args.limit:
        test_cases = test_cases[: args.limit]

    print(f"\n{'=' * 70}")
    print(f"  RAG EVALUATION (RAGAS) — {len(test_cases)} test cases")
    print(f"  Model: {settings.OLLAMA_MODEL}")
    print(f"  Embedding: {settings.EMBEDDING_MODEL}")
    print(f"{'=' * 70}\n")

    # Initialize services
    print("1. Initializing services...")
    await initialize_services()

    retrieval_service = RetrievalService()
    llm_service = LLMService()
    await llm_service.initialize()
    print("   Done.\n")

    # Run RAG pipeline on all test cases
    print("2. Running RAG pipeline...")
    samples = await run_pipeline(test_cases, retrieval_service, llm_service)
    print(f"   Done. {len(samples)} samples collected.\n")

    # Filter out errored samples
    valid_samples = [s for s in samples if "_error" not in s]
    if not valid_samples:
        print("No valid samples to evaluate.")
        return

    # Run RAGAS evaluation
    print("3. Running RAGAS evaluation (this may take a few minutes)...")
    evaluator = RAGASEvaluator()
    results = await evaluator.evaluate(valid_samples, metrics=args.metrics)
    print("   Done.\n")

    # Print results
    print(f"{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}\n")

    summary = results["summary"]
    for metric, score in summary.items():
        status = "GOOD" if score >= 0.7 else ("OK" if score >= 0.5 else "NEEDS WORK")
        print(f"  {metric:25s}  {score:.4f}  {status}")

    # Overall
    if summary:
        overall = sum(summary.values()) / len(summary)
        print(f"\n  {'OVERALL':25s}  {overall:.4f}")

    # Per-sample details
    print(f"\n{'=' * 70}")
    print("  PER-SAMPLE DETAILS")
    print(f"{'=' * 70}\n")

    for i, detail in enumerate(results["per_sample"], 1):
        test_id = samples[i - 1].get("_test_id", f"sample_{i}")
        print(f"  [{i}] {test_id}")
        print(f"      Q: {detail['question'][:70]}...")
        for key, val in detail.items():
            if key not in ("question", "answer_preview") and val is not None:
                print(f"      {key}: {val}")
        print()

    print(f"  Evaluated: {len(valid_samples)}/{len(test_cases)} samples")
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
            "summary": summary,
            "per_sample": results["per_sample"],
            "num_samples": results["num_samples"],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Report saved: {path}\n")


if __name__ == "__main__":
    asyncio.run(main())
