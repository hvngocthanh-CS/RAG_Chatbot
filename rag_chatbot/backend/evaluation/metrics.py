"""
RAG Evaluation using RAGAS framework.

Evaluates 4 dimensions of a RAG system:

1. Faithfulness     — Is the answer grounded in the retrieved context?
2. Answer Relevancy — Does the answer actually address the question?
3. Context Precision — Are the retrieved chunks relevant and well-ranked?
4. Context Recall    — Did we retrieve all the context needed to answer?

Uses Ollama as the evaluator LLM (no OpenAI API key needed).

Reference: https://docs.ragas.io/
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from backend.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# RAGAS Evaluator
# ============================================================

class RAGASEvaluator:
    """
    Evaluates RAG pipeline quality using RAGAS metrics with Ollama.

    Usage:
        evaluator = RAGASEvaluator()
        results = await evaluator.evaluate(samples)
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or settings.OLLAMA_MODEL
        self._llm = None
        self._embeddings = None

    def _get_llm(self):
        """Lazy-init RAGAS LLM wrapper backed by Ollama."""
        if self._llm is not None:
            return self._llm

        from ragas.llms import LangchainLLMWrapper
        from langchain_community.chat_models import ChatOllama

        base_url = settings.OLLAMA_BASE_URL.replace("/v1", "")

        ollama_llm = ChatOllama(
            model=self._model_name,
            base_url=base_url,
        )
        self._llm = LangchainLLMWrapper(ollama_llm)
        return self._llm

    def _get_embeddings(self):
        """Get RAGAS embedding wrapper backed by HuggingFace."""
        if self._embeddings is not None:
            return self._embeddings

        from ragas.embeddings import HuggingfaceEmbeddings

        self._embeddings = HuggingfaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
        )
        return self._embeddings

    async def evaluate(
        self,
        samples: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run RAGAS evaluation on a list of samples.

        Each sample dict must have:
          - user_input: str              (the question)
          - response: str                (the generated answer)
          - retrieved_contexts: List[str] (the retrieved chunks)
          - reference: str               (expected answer, for recall)

        Args:
            samples: List of evaluation samples.
            metrics: Which metrics to run. Default: all four.
                     Options: "faithfulness", "answer_relevancy",
                              "context_precision", "context_recall"

        Returns:
            Dict with summary scores and per-sample details.
        """
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.run_config import RunConfig

        metric_objects = self._build_metrics(metrics)

        ragas_samples = []
        for s in samples:
            ragas_samples.append(SingleTurnSample(
                user_input=s["user_input"],
                response=s["response"],
                retrieved_contexts=s["retrieved_contexts"],
                reference=s.get("reference", ""),
            ))

        dataset = EvaluationDataset(samples=ragas_samples)

        run_config = RunConfig(
            timeout=300,
            max_retries=3,
        )

        logger.info("Running RAGAS evaluation on %d samples...", len(samples))

        result = evaluate(
            dataset=dataset,
            metrics=metric_objects,
            run_config=run_config,
        )

        return self._format_results(result, samples)

    def _build_metrics(self, metric_names: Optional[List[str]] = None):
        """Build RAGAS metric instances with Ollama LLM."""
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithoutReference,
            LLMContextRecallWithoutReference,
        )

        llm = self._get_llm()
        embeddings = self._get_embeddings()

        all_metrics = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings),
            "context_precision": LLMContextPrecisionWithoutReference(llm=llm),
            "context_recall": LLMContextRecallWithoutReference(llm=llm),
        }

        if metric_names:
            return [all_metrics[m] for m in metric_names if m in all_metrics]
        return list(all_metrics.values())

    @staticmethod
    def _format_results(
        ragas_result, samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format RAGAS result into a clean dict."""
        df = ragas_result.to_pandas()

        skip_cols = {"user_input", "response", "retrieved_contexts", "reference"}

        # Per-sample details
        per_sample = []
        for i, row in df.iterrows():
            detail = {
                "question": samples[i]["user_input"],
                "answer_preview": samples[i]["response"][:150],
            }
            for col in df.columns:
                if col not in skip_cols:
                    val = row[col]
                    detail[col] = round(float(val), 4) if val is not None else None
            per_sample.append(detail)

        # Aggregate scores
        summary = {}
        for col in df.columns:
            if col not in skip_cols:
                values = df[col].dropna()
                if len(values) > 0:
                    summary[col] = round(float(values.mean()), 4)

        return {
            "summary": summary,
            "per_sample": per_sample,
            "num_samples": len(samples),
        }


# ============================================================
# Evaluation Result dataclass
# ============================================================

@dataclass
class RAGEvaluationResult:
    """Evaluation result for one question."""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    question: str = ""
    answer: str = ""
    num_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
        }

    @property
    def overall_score(self) -> float:
        """
        Weighted overall score.

        - Faithfulness 30%: answer must be grounded in context
        - Answer Relevancy 30%: answer must address the question
        - Context Precision 20%: retrieved chunks should be relevant
        - Context Recall 20%: must retrieve all needed context
        """
        return round(
            0.3 * self.faithfulness
            + 0.3 * self.answer_relevancy
            + 0.2 * self.context_precision
            + 0.2 * self.context_recall,
            4,
        )
