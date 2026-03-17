"""
Comprehensive RAG Evaluation Module.

Industry-standard metrics for evaluating RAG systems:
- Retrieval: Precision@k, Recall@k, F1@k, MRR, NDCG
- Generation: BLEU, ROUGE, Semantic Similarity, Answer Relevancy, Faithfulness
- End-to-End: Context Precision/Recall/F1, Citation Accuracy, Hallucination Detection, Answer Correctness

Usage:
    from backend.evaluation import ComprehensiveRAGMetrics
    
    metrics = ComprehensiveRAGMetrics()
    result = metrics.evaluate_retrieval(retrieved_chunks, ground_truth)
"""

from .comprehensive_rag_metrics import (
    ComprehensiveRAGMetrics,
    RAGEvaluationResult,
    RetrievalMetrics,
    GenerationMetrics,
    EndToEndMetrics,
)

__all__ = [
    "ComprehensiveRAGMetrics",
    "RAGEvaluationResult",
    "RetrievalMetrics",
    "GenerationMetrics",
    "EndToEndMetrics",
]
