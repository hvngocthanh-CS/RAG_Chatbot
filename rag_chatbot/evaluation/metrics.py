"""
RAG evaluation metrics.

Retrieval: Hit@k, Recall@k, MRR (fast, deterministic)
Generation: RAGAS - Faithfulness, Relevancy, Context Precision/Recall (slow, LLM-judged)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from backend.config import settings
# RAGAS imports (optional - only needed for generation metrics)
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.run_config import RunConfig
    from ragas.metrics import Faithfulness, ResponseRelevancy
    
    # Context metrics (name varies by RAGAS version)
    try:
        from ragas.metrics import LLMContextPrecisionWithoutReference as ContextPrecision
        from ragas.metrics import LLMContextRecallWithoutReference as ContextRecall
    except ImportError:
        from ragas.metrics import ContextPrecision, ContextRecall
    
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    pass

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
    """Evaluates answer quality with RAGAS (Faithfulness, Relevancy, Context Precision/Recall)."""

    def __init__(self, model_name: Optional[str] = None):
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS not installed. Run: pip install ragas langchain-ollama langchain-huggingface")
        # Judge model resolution order:
        # 1) explicit arg, 2) RAGAS_JUDGE_MODEL setting, 3) OLLAMA_MODEL fallback
        self._model_name = (
            model_name
            or getattr(settings, "RAGAS_JUDGE_MODEL", "")
            or settings.OLLAMA_MODEL
        )
        self._llm = None
        self._embeddings = None

    def _get_llm(self):
        if self._llm is None:
            base_url = settings.OLLAMA_BASE_URL.replace("/v1", "")
            self._llm = LangchainLLMWrapper(ChatOllama(model=self._model_name, base_url=base_url))
        return self._llm

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
            )
        return self._embeddings

    async def evaluate(
        self,
        samples: List[Dict[str, Any]],
        sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate samples with RAGAS. Each sample needs: user_input, response, retrieved_contexts, reference.

        Args:
            samples: list of dicts for RAGAS
            sample_ids: optional list of stable IDs aligned with `samples`
                        (used for per-sample success/failure reporting)
        """
        llm = self._get_llm()
        embeddings = self._get_embeddings()

        metrics = [
            Faithfulness(llm=llm),
            ResponseRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ]

        dataset = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=s["user_input"],
                response=s["response"],
                retrieved_contexts=s["retrieved_contexts"],
                reference=s.get("reference", ""),
            ) for s in samples
        ])

        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=RunConfig(
                timeout=getattr(settings, "RAGAS_TIMEOUT_SECONDS", 600),
                max_retries=getattr(settings, "RAGAS_MAX_RETRIES", 3),
            ),
        )

        df = result.to_pandas()
        skip = {"user_input", "response", "retrieved_contexts", "reference"}
        metric_cols = [c for c in df.columns if c not in skip]

        # Aggregate: mean over non-null, and per-metric success/failure counts
        summary: Dict[str, float] = {}
        coverage: Dict[str, Dict[str, int]] = {}
        for col in metric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                summary[col] = round(float(values.mean()), 4)
            coverage[col] = {
                "scored": int(df[col].notna().sum()),
                "failed": int(df[col].isna().sum()),
                "total": int(len(df)),
            }

        # Per-sample results — one row per input sample with its metric scores
        # and a status ("ok" / "partial" / "failed").
        ids = sample_ids or [f"sample_{i}" for i in range(len(samples))]
        per_sample: List[Dict[str, Any]] = []
        for i in range(len(df)):
            scores: Dict[str, Any] = {}
            failed_metrics: List[str] = []
            for col in metric_cols:
                v = df[col].iloc[i]
                if v is None or (isinstance(v, float) and v != v):  # NaN check
                    scores[col] = None
                    failed_metrics.append(col)
                else:
                    scores[col] = round(float(v), 4)
            if not failed_metrics:
                status = "ok"
            elif len(failed_metrics) == len(metric_cols):
                status = "failed"
            else:
                status = "partial"
            per_sample.append({
                "id": ids[i] if i < len(ids) else f"sample_{i}",
                "status": status,
                "failed_metrics": failed_metrics,
                "scores": scores,
            })

        return {
            "summary": summary,
            "num_samples": len(samples),
            "coverage": coverage,
            "per_sample": per_sample,
        }
