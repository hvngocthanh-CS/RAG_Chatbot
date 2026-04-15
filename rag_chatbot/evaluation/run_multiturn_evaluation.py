#!/usr/bin/env python3
"""Multi-turn RAG Evaluation.

Runs RAG pipeline over multi-turn conversation dataset and computes:
  1. Retrieval metrics per turn and aggregated (Hit@k, Recall@k, MRR, Refusal)
  2. Breakdown by turn position (turn 1 vs follow-up turns) and category
  3. Generation metrics (RAGAS) on all turns

Usage:
    python -m evaluation.run_multiturn_evaluation
    python -m evaluation.run_multiturn_evaluation --limit 3
    python -m evaluation.run_multiturn_evaluation --no-ragas
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.services import initialize_services, get_service
from backend.services.retrieval import RetrievalService
from backend.services.llm import LLMService
from evaluation.metrics import compute_retrieval_metrics, RAGASEvaluator

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "evaluation/datasets/techviet_multiturn_v1.json"
REPORT_DIR = "evaluation/reports"
REFUSAL_KEYWORDS = (
    "not found", "not mentioned", "no information", "cannot find",
    "don't know", "do not know", "not available",
    "không tìm thấy", "không có thông tin",
)


def format_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("filename", "Unknown")
        page = meta.get("page_number", "N/A")
        parts.append(f"[Source {i}: {source}, Page {page}]\n{chunk['content']}")
    return "\n---\n".join(parts)


def is_refusal(answer: str) -> bool:
    return any(kw in (answer or "").lower() for kw in REFUSAL_KEYWORDS)


async def run_conversation(
    conv: Dict[str, Any],
    retrieval_service: RetrievalService,
    llm_service: LLMService,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run a conversation through the RAG pipeline, carrying history between turns."""
    history: List[Dict[str, str]] = []
    turn_samples: List[Dict[str, Any]] = []

    for turn in conv["turns"]:
        sample = {
            "conversation_id": conv["conversation_id"],
            "category": conv.get("category", "unknown"),
            "difficulty": conv.get("difficulty", "unknown"),
            "turn_id": turn["turn_id"],
            "turn_position": "first" if turn["turn_id"] == 1 else "follow_up",
            "question": turn["user"],
            "expected_answer": turn.get("expected_answer", ""),
            "expected_docs": turn.get("source_documents", []) or [],
            "has_answer": turn.get("has_answer", True),
            "tests": turn.get("tests", []),
            "retrieved_chunks": [],
            "answer": "",
            "error": None,
        }

        try:
            chunks = await retrieval_service.retrieve(
                query=turn["user"],
                top_k=top_k,
                conversation_history=history if history else None,
            )
            sample["retrieved_chunks"] = chunks or []

            if chunks:
                sample["answer"] = await llm_service.generate(
                    question=turn["user"],
                    context=format_context(chunks),
                    conversation_history=history if history else None,
                )
            else:
                sample["answer"] = "No relevant documents found."

            print(f"    turn {turn['turn_id']}: {len(chunks)} chunks")
        except Exception as e:
            print(f"    turn {turn['turn_id']}: ERROR {e}")
            sample["error"] = str(e)

        turn_samples.append(sample)

        # Update history with this turn (user + assistant) for next turn
        history.append({"role": "user", "content": turn["user"]})
        history.append({"role": "assistant", "content": sample["answer"]})

    return turn_samples


def breakdown_metrics(samples: List[Dict[str, Any]], k: int, key: str) -> Dict[str, Any]:
    """Compute retrieval metrics grouped by a sample key (e.g. 'category', 'turn_position')."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        groups[s.get(key, "unknown")].append(s)
    return {g: compute_retrieval_metrics(items, k=k).to_dict() for g, items in groups.items()}


def print_block(title: str, metrics: Dict[str, Any]):
    print(f"\n  {title}")
    print(f"  {'-' * len(title)}")
    print(f"    Hit@k            {metrics['hit_at_k']:.4f}")
    print(f"    Recall@k         {metrics['recall_at_k']:.4f}")
    print(f"    MRR              {metrics['mrr']:.4f}")
    print(f"    Refusal Accuracy {metrics['refusal_accuracy']:.4f} "
          f"({metrics['num_unanswerable']} unanswerable)")


def print_ragas(summary: Dict[str, float]):
    print(f"\n{'='*60}\n  RAGAS METRICS (multi-turn)\n{'='*60}")
    for name, score in summary.items():
        print(f"  {name:20s} {score:.4f}")


async def main():
    parser = argparse.ArgumentParser(description="Multi-turn RAG evaluation")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, help="Limit number of conversations")
    parser.add_argument("--top-k", type=int, default=6, dest="top_k")
    parser.add_argument("--no-ragas", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.exists(args.dataset):
        print(f"ERROR: dataset not found: {args.dataset}")
        return

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    conversations = data["conversations"]
    if args.limit:
        conversations = conversations[:args.limit]

    total_turns = sum(len(c["turns"]) for c in conversations)

    print(f"\n{'='*60}")
    print("  MULTI-TURN RAG EVALUATION")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Conversations:{len(conversations)}")
    print(f"  Total turns:  {total_turns}")
    print(f"  Top-k:        {args.top_k}")
    print(f"  LLM:          {settings.OLLAMA_MODEL}")
    print(f"  Embedding:    {settings.EMBEDDING_MODEL}")
    print(f"{'='*60}\n")

    print("1. Initializing services...")
    await initialize_services()
    retrieval_service = get_service("retrieval")
    llm_service = get_service("llm")
    print("   Done.\n")

    print("2. Running conversations...")
    all_samples: List[Dict[str, Any]] = []
    for i, conv in enumerate(conversations, 1):
        print(f"  [{i}/{len(conversations)}] {conv['conversation_id']} "
              f"({conv.get('category', '?')}, {len(conv['turns'])} turns)")
        turn_samples = await run_conversation(conv, retrieval_service, llm_service, args.top_k)
        all_samples.extend(turn_samples)

    ok = [s for s in all_samples if s["error"] is None]
    print(f"\n   Done. {len(ok)}/{len(all_samples)} turn-samples collected.\n")

    print("3. Computing retrieval metrics...")
    overall = compute_retrieval_metrics(ok, k=args.top_k).to_dict()
    print(f"\n{'='*60}\n  RETRIEVAL METRICS - OVERALL (k={overall['k']})\n{'='*60}")
    print_block("All turns", overall)

    by_position = breakdown_metrics(ok, args.top_k, "turn_position")
    print(f"\n{'='*60}\n  BREAKDOWN BY TURN POSITION\n{'='*60}")
    for pos, m in by_position.items():
        print_block(f"{pos} (n={m['num_answerable'] + m['num_unanswerable']})", m)

    by_category = breakdown_metrics(ok, args.top_k, "category")
    print(f"\n{'='*60}\n  BREAKDOWN BY CATEGORY\n{'='*60}")
    for cat, m in by_category.items():
        print_block(f"{cat} (n={m['num_answerable'] + m['num_unanswerable']})", m)

    # Generation refusal
    unanswerable = [s for s in ok if not s["has_answer"]]
    gen_refusal = None
    if unanswerable:
        correct = sum(1 for s in unanswerable if is_refusal(s["answer"]))
        gen_refusal = round(correct / len(unanswerable), 4)
        print(f"\n  Generation refusal accuracy: {gen_refusal:.4f} "
              f"({correct}/{len(unanswerable)} unanswerable turns)")

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
    report_path = os.path.join(REPORT_DIR, f"multiturn_report_{ts}.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "dataset": args.dataset,
                "top_k": args.top_k,
                "model": settings.OLLAMA_MODEL,
                "embedding": settings.EMBEDDING_MODEL,
            },
            "num_conversations": len(conversations),
            "num_turns": len(all_samples),
            "retrieval_metrics_overall": overall,
            "retrieval_metrics_by_turn_position": by_position,
            "retrieval_metrics_by_category": by_category,
            "generation_refusal_accuracy": gen_refusal,
            "ragas_metrics": ragas_summary,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Report saved: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
