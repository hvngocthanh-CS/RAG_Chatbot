"""
RAG evaluation metrics.

Retrieval: Hit@k, Recall@k, MRR (fast, deterministic)
Generation: RAGAS - Faithfulness, Relevancy, Context Precision/Recall (slow, LLM-judged)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from backend.config import settings


# ======================================================================
# Retrieval metrics (IR) — pure functions
# ======================================================================

@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics."""
    hit_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    refusal_accuracy: float = 0.0
    k: int = 0
    num_answerable: int = 0
    num_unanswerable: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def hit_at_k(retrieved_docs: List[str], relevant: set) -> int:
    """1 if any relevant doc found, else 0."""
    return int(any(d in relevant for d in retrieved_docs))


def recall_at_k(retrieved_docs: List[str], relevant: set) -> float:
    """Fraction of relevant docs recovered."""
    return len(set(retrieved_docs) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved_docs: List[str], relevant: set) -> float:
    """1/rank of first relevant doc (0 if not found)."""
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def _get_doc(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata") or {}
    return meta.get("filename") or meta.get("source") or ""


def compute_retrieval_metrics(runs: List[Dict[str, Any]], k: int) -> RetrievalMetrics:
    """Compute aggregated retrieval metrics over test cases.
    
    Args:
        runs: list of dicts with has_answer, expected_docs, retrieved_chunks
        k: cutoff for @k metrics
    """
    hit_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0
    n_answerable = 0

    refusal_correct = 0
    n_unanswerable = 0

    for run in runs:
        chunks = run.get("retrieved_chunks", [])[:k]
        retrieved_docs = [_get_doc(c) for c in chunks]
        relevant = set(run.get("expected_docs") or [])

        if run.get("has_answer", True):
            hit_sum += hit_at_k(retrieved_docs, relevant)
            recall_sum += recall_at_k(retrieved_docs, relevant)
            mrr_sum += reciprocal_rank(retrieved_docs, relevant)
            n_answerable += 1
        else:
            n_unanswerable += 1
            if not chunks:
                refusal_correct += 1

    def avg(total: float, n: int) -> float:
        return total / n if n > 0 else 0.0

    return RetrievalMetrics(
        hit_at_k=avg(hit_sum, n_answerable),
        recall_at_k=avg(recall_sum, n_answerable),
        mrr=avg(mrr_sum, n_answerable),
        refusal_accuracy=avg(refusal_correct, n_unanswerable),
        k=k,
        num_answerable=n_answerable,
        num_unanswerable=n_unanswerable,
    )


# ======================================================================
# Generation metrics — RAGAS wrapper
# ======================================================================

class RAGASEvaluator:
    """
    Evaluates answer quality with RAGAS:
      - Faithfulness       : answer grounded in context?
      - Answer Relevancy   : answer addresses the question?
      - Context Precision  : retrieved chunks relevant & well-ranked?
      - Context Recall     : all needed context retrieved?

    Uses Ollama as the judge LLM (no OpenAI key needed).
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or settings.OLLAMA_MODEL
        self._llm = None
        self._embeddings = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        from ragas.llms import LangchainLLMWrapper
        from langchain_ollama import ChatOllama

        base_url = settings.OLLAMA_BASE_URL.replace("/v1", "")
        ollama_llm = ChatOllama(model=self._model_name, base_url=base_url)
        self._llm = LangchainLLMWrapper(ollama_llm)
        return self._llm

    def _get_embeddings(self):
        if self._embeddings is not None:
            return self._embeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_huggingface import HuggingFaceEmbeddings
        
        hf_embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self._embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        return self._embeddings

    async def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate samples with RAGAS.
        
        Each sample needs: user_input, response, retrieved_contexts, reference
        """
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.run_config import RunConfig
        from ragas.metrics import Faithfulness, ResponseRelevancy
        
        # Context metrics (handle version differences)
        context_precision_cls = context_recall_cls = None
        try:
            from ragas.metrics import LLMContextPrecisionWithoutReference as CPrecision
            from ragas.metrics import LLMContextRecallWithoutReference as CRecall
            context_precision_cls, context_recall_cls = CPrecision, CRecall
        except ImportError:
            try:
                from ragas.metrics import ContextPrecision, ContextRecall
                context_precision_cls, context_recall_cls = ContextPrecision, ContextRecall
            except ImportError:
                pass

        llm = self._get_llm()
        embeddings = self._get_embeddings()
        
        metrics = [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=embeddings)]
        if context_precision_cls:
            metrics.append(context_precision_cls(llm=llm))
        if context_recall_cls:
            metrics.append(context_recall_cls(llm=llm))

        dataset = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=s["user_input"],
                response=s["response"],
                retrieved_contexts=s["retrieved_contexts"],
                reference=s.get("reference", ""),
            ) for s in samples
        ])

        result = evaluate(dataset=dataset, metrics=metrics, run_config=RunConfig(timeout=300, max_retries=3))
        
        df = result.to_pandas()
        skip = {"user_input", "response", "retrieved_contexts", "reference"}
        summary = {col: round(float(df[col].dropna().mean()), 4) 
                   for col in df.columns if col not in skip and df[col].dropna().size > 0}
        return {"summary": summary, "num_samples": len(samples)}
