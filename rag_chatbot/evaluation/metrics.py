"""
RAG evaluation metrics.

Two groups:
  1. Retrieval metrics (IR)  — Hit@k, Recall@k, MRR. Pure functions, fast.
  2. Generation metrics       — RAGAS (faithfulness, relevancy, context precision/recall).
                                Slow, LLM-judged.

Also provides refusal_accuracy() for unanswerable questions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Set

from backend.config import settings


# ======================================================================
# Retrieval metrics (IR) — pure functions
# ======================================================================

@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics for a dataset run."""
    hit_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    refusal_accuracy: float = 0.0
    k: int = 0
    num_answerable: int = 0
    num_unanswerable: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self).items()}


def hit_at_k(retrieved_docs: List[str], relevant: Set[str]) -> int:
    """1 if any relevant doc appears in the retrieved list, else 0."""
    return int(any(d in relevant for d in retrieved_docs))


def recall_at_k(retrieved_docs: List[str], relevant: Set[str]) -> float:
    """Fraction of relevant docs that were recovered."""
    if not relevant:
        return 0.0
    return len(set(retrieved_docs) & relevant) / len(relevant)


def reciprocal_rank(retrieved_docs: List[str], relevant: Set[str]) -> float:
    """1 / rank of the first relevant doc (0 if not found)."""
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def _get_doc(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata") or {}
    return meta.get("filename") or meta.get("source") or ""


def compute_retrieval_metrics(
    runs: List[Dict[str, Any]],
    k: int,
) -> RetrievalMetrics:
    """
    Compute aggregated retrieval metrics over all test cases.

    Args:
        runs: list of dicts with keys: has_answer, expected_docs, retrieved_chunks
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
        relevant: Set[str] = set(run.get("expected_docs") or [])

        if run.get("has_answer", True):
            hit_sum += hit_at_k(retrieved_docs, relevant)
            recall_sum += recall_at_k(retrieved_docs, relevant)
            mrr_sum += reciprocal_rank(retrieved_docs, relevant)
            n_answerable += 1
        else:
            # Unanswerable: retriever should return nothing
            n_unanswerable += 1
            if len(chunks) == 0:
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
        return self._embeddings

    async def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Each sample must have:
          - user_input: str (the question)
          - response: str (the generated answer)
          - retrieved_contexts: List[str]
          - reference: str (expected answer)
        """
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.run_config import RunConfig
        from ragas.metrics import Faithfulness, ResponseRelevancy
        
        # Try importing context metrics (name varies by RAGAS version)
        try:
            from ragas.metrics import LLMContextPrecisionWithoutReference, LLMContextRecallWithoutReference
            context_precision_cls = LLMContextPrecisionWithoutReference
            context_recall_cls = LLMContextRecallWithoutReference
        except ImportError:
            try:
                from ragas.metrics import ContextPrecision, ContextRecall
                context_precision_cls = ContextPrecision
                context_recall_cls = ContextRecall
            except ImportError:
                context_precision_cls = None
                context_recall_cls = None

        llm = self._get_llm()
        embeddings = self._get_embeddings()
        metric_objects = [
            Faithfulness(llm=llm),
            ResponseRelevancy(llm=llm, embeddings=embeddings),
        ]
        if context_precision_cls:
            metric_objects.append(context_precision_cls(llm=llm))
        if context_recall_cls:
            metric_objects.append(context_recall_cls(llm=llm))

        ragas_samples = [
            SingleTurnSample(
                user_input=s["user_input"],
                response=s["response"],
                retrieved_contexts=s["retrieved_contexts"],
                reference=s.get("reference", ""),
            )
            for s in samples
        ]

        result = evaluate(
            dataset=EvaluationDataset(samples=ragas_samples),
            metrics=metric_objects,
            run_config=RunConfig(timeout=300, max_retries=3),
        )

        # Aggregate scores from the result DataFrame
        df = result.to_pandas()
        skip = {"user_input", "response", "retrieved_contexts", "reference"}
        summary = {
            col: round(float(df[col].dropna().mean()), 4)
            for col in df.columns
            if col not in skip and df[col].dropna().size > 0
        }
        return {"summary": summary, "num_samples": len(samples)}
