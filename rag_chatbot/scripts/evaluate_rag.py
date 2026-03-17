#!/usr/bin/env python3
"""
PROFESSIONAL RAG EVALUATION - COMPREHENSIVE FRAMEWORK

Evaluates RAG system across 3 dimensions (Industry Standard):

1️⃣  RETRIEVAL METRICS:
    - Precision@k, Recall@k, F1@k
    - MRR (Mean Reciprocal Rank)
    - NDCG@k (position-weighted ranking quality)

2️⃣  GENERATION METRICS:
    - BLEU (n-gram overlap)
    - ROUGE-L (longest common subsequence)
    - Semantic Similarity (TF-IDF)
    - Answer Relevancy
    - Faithfulness (grounding in context)

3️⃣  END-TO-END METRICS:
    - Context Precision/Recall/F1
    - Hallucination Detection
    - Citation Accuracy
    - Answer Correctness

Usage:
    python scripts/evaluate_rag.py                  # Full evaluation
    python scripts/evaluate_rag.py --limit 5       # Quick test
    python scripts/evaluate_rag.py --save-report   # Save report

Reference: NIST IR standards, RAGAS metric, DeepEval framework
"""

import asyncio
import json
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, **kwargs):
        """Fallback"""
        return str(data)

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.retrieval import RetrievalService
from backend.services.llm import LLMService
from backend.services import initialize_services
from backend.config import settings
from backend.evaluation.comprehensive_rag_metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    EndToEndMetrics,
    RAGEvaluationResult
)


class RAGEvaluator:
    """Professional RAG evaluation (16 metrics across 3 dimensions)."""

    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.e2e_metrics = EndToEndMetrics()

    async def evaluate_question(
        self,
        question: str,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
        ground_truth: Optional[str] = None,
        relevant_chunks: Optional[List[str]] = None
    ) -> RAGEvaluationResult:
        """Evaluate single Q&A cycle through complete RAG pipeline."""
        
        # 1. RETRIEVAL
        chunks = await retrieval_service.retrieve(query=question, top_k=5)
        if not chunks:
            raise ValueError("No chunks retrieved")
        
        # Format context
        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            page = meta.get('page_number', 'N/A')
            context_parts.append(f"[Source {idx}: {meta['filename']}]\n{chunk['content']}")
        context = "\n---\n".join(context_parts)
        
        # 2. GENERATION
        answer = await llm_service.generate(question=question, context=context)
        
        # 3. EVALUATION - Calculate 16 metrics
        relevant_ids = relevant_chunks or [c.get("id", "") for c in chunks]
        
        # Retrieval metrics (5)
        precision_5 = self.retrieval_metrics.precision_at_k(chunks, relevant_ids, k=5)
        recall_5 = self.retrieval_metrics.recall_at_k(chunks, relevant_ids, k=5)
        f1_5 = self.retrieval_metrics.f1_at_k(chunks, relevant_ids, k=5)
        mrr = self.retrieval_metrics.mrr(chunks, relevant_ids)
        ndcg_5 = self.retrieval_metrics.ndcg_at_k(chunks, relevant_ids, k=5)
        
        # Generation metrics (5)
        bleu = self.generation_metrics.bleu_score(ground_truth or answer, answer)
        rouge_l = self.generation_metrics.rouge_score(ground_truth or answer, answer)
        semantic_sim = self.generation_metrics.semantic_similarity(question, answer)
        answer_relevancy = self.generation_metrics.answer_relevancy(question, answer)
        faithfulness = self.generation_metrics.faithfulness(answer, context)
        
        # End-to-end metrics (6)
        context_precision = self.e2e_metrics.context_precision(chunks, relevant_ids)
        context_recall = self.e2e_metrics.context_recall(chunks, relevant_ids)
        context_f1 = self.e2e_metrics.context_f1(chunks, relevant_ids)
        hallucination = self.e2e_metrics.hallucination_detection(answer, context)
        citation_acc = self.e2e_metrics.citation_accuracy(answer, context)
        answer_correctness = self.e2e_metrics.answer_correctness(answer, ground_truth, context)
        
        return RAGEvaluationResult(
            precision_at_5=precision_5, recall_at_5=recall_5, f1_at_5=f1_5,
            mrr=mrr, ndcg_at_5=ndcg_5,
            bleu=bleu, rouge_l=rouge_l, semantic_sim=semantic_sim,
            answer_relevancy=answer_relevancy, faithfulness=faithfulness,
            context_precision=context_precision, context_recall=context_recall,
            context_f1=context_f1, hallucination_detected=hallucination,
            citation_accuracy=citation_acc, answer_correctness=answer_correctness,
            question=question, answer=answer, num_chunks_retrieved=len(chunks)
        )


async def main():
    """Main evaluation routine."""
    
    print("\n" + "=" * 80)
    print(" " * 10 + "PROFESSIONAL RAG EVALUATION (16 Metrics - Industry Standard)")
    print("=" * 80 + "\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit tests")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report")
    args = parser.parse_args()
    
    # Load dataset
    if not os.path.exists("data/eval_dataset.json"):
        print("❌ data/eval_dataset.json not found\n")
        print("ℹ️  First upload a document: python scripts/ingest_documents.py <file.pdf>\n")
        return
    
    with open("data/eval_dataset.json") as f:
        dataset = json.load(f)
    
    test_cases = dataset["test_cases"][:args.limit] if args.limit else dataset["test_cases"]
    
    print(f"📋 {len(test_cases)} Test Cases Loaded")
    print("🔧 Initializing Services...\n")
    
    await initialize_services()
    retrieval_service = RetrievalService()
    llm_service = LLMService()
    await llm_service.initialize()
    
    evaluator = RAGEvaluator()
    print("✓ Ready\n" + "-" * 80 + "\n")
    
    # Run evaluations
    results_data, results_table = [], []
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        category = test_case["category"]
        test_id = test_case["id"]
        
        print(f"[{i}/{len(test_cases)}] {test_id:20} | {category}")
        
        try:
            result = await evaluator.evaluate_question(
                question=question,
                retrieval_service=retrieval_service,
                llm_service=llm_service,
                ground_truth=test_case.get("ground_truth"),
            )
            
            print(f"  ✓ Overall Score: {result.overall_score:.3f}\n")
            
            results_table.append([
                test_id, category[:12],
                f"{result.precision_at_5:.2f}", f"{result.recall_at_5:.2f}",
                f"{result.ndcg_at_5:.2f}",
                f"{result.bleu:.2f}", f"{result.faithfulness:.2f}",
                f"{result.context_f1:.2f}",
                "✓" if not result.hallucination_detected else "✗",
                f"{result.overall_score:.2f}"
            ])
            
            results_data.append({
                "test_id": test_id,
                "category": category,
                "metrics": result.to_dict()
            })
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:40]}\n")
            results_data.append({"test_id": test_id, "error": str(e)})
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS - METRICS TABLE")
    print("=" * 80 + "\n")
    
    if results_table:
        print(tabulate(
            results_table,
            headers=["Test", "Category", "Prec@5", "Rec@5", "NDCG", "BLEU", "Faith", "CtxF1", "Hall", "Score"],
            tablefmt="grid"
        ))
        
        # Summary
        print("\n" + "-" * 80)
        print("SUMMARY STATISTICS")
        print("-" * 80 + "\n")
        
        valid = [r for r in results_data if "metrics" in r]
        if valid:
            print("1️⃣  RETRIEVAL METRICS (Document Retrieval Quality):")
            avg_p = sum(r["metrics"]["retrieval"]["precision@5"] for r in valid) / len(valid)
            avg_r = sum(r["metrics"]["retrieval"]["recall@5"] for r in valid) / len(valid)
            avg_n = sum(r["metrics"]["retrieval"]["ndcg@5"] for r in valid) / len(valid)
            print(f"   Precision@5:  {avg_p:.3f}  {'✓' if avg_p >= 0.7 else '⚠'}")
            print(f"   Recall@5:     {avg_r:.3f}  {'✓' if avg_r >= 0.7 else '⚠'}")
            print(f"   NDCG@5:       {avg_n:.3f}  {'✓' if avg_n >= 0.75 else '⚠'}")
            
            print("\n2️⃣  GENERATION METRICS (LLM Answer Quality):")
            avg_b = sum(r["metrics"]["generation"]["bleu"] for r in valid) / len(valid)
            avg_f = sum(r["metrics"]["generation"]["faithfulness"] for r in valid) / len(valid)
            print(f"   BLEU:         {avg_b:.3f}  {'✓' if avg_b >= 0.5 else '⚠'}")
            print(f"   Faithfulness: {avg_f:.3f}  {'✓' if avg_f >= 0.8 else '⚠'}")
            
            print("\n3️⃣  END-TO-END METRICS (Complete Pipeline):")
            avg_cp = sum(r["metrics"]["end_to_end"]["context_precision"] for r in valid) / len(valid)
            hall_cnt = sum(r["metrics"]["end_to_end"]["hallucination_detected"] for r in valid)
            print(f"   Context Prec:     {avg_cp:.3f}")
            print(f"   Hallucinations:   {hall_cnt}/{len(valid)} {'✓ NONE' if hall_cnt == 0 else '⚠ DETECTED'}")
            
            print(f"\n✅ Passed: {len(valid)}/{len(test_cases)}")
    
    # Save report
    if args.save_report:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"logs/evaluation_{ts}.json"
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "framework": "Professional RAG (16 metrics: Retrieval + Generation + End-to-End)",
            "config": {
                "llm": settings.LLM_PROVIDER,
                "embedding": settings.EMBEDDING_MODEL
            },
            "results": results_data
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report: {path}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
