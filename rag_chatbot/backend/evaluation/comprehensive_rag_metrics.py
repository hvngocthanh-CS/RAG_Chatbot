"""
RAG Evaluation Framework — Industry Standard Metrics.

Evaluates 3 aspects of a RAG system:

1. RETRIEVAL — Did we find the right chunks?
   - Hit Rate: Did ANY of the top-k chunks come from the correct page/section?
   - MRR: How quickly does the first relevant chunk appear?
   - Context Precision@k: What fraction of retrieved chunks are relevant?

2. GENERATION — Is the answer good?
   - Faithfulness: Is every claim in the answer supported by the context?
   - ROUGE-L: Overlap with reference answer (longest common subsequence)
   - Answer Correctness: Semantic similarity to expected answer

3. END-TO-END — Does the full pipeline work?
   - Hallucination Rate: % of test cases where answer contains fabricated info
   - Citation Accuracy: Do [Source N] references point to real sources?

What you need to run evaluation:
  - An eval dataset: list of {question, expected_answer, expected_pages}
  - A running RAG system (Qdrant + Ollama + server)

Reference frameworks: RAGAS, DeepEval, ARES
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


# ============================================================
# 1. RETRIEVAL METRICS
# ============================================================

class RetrievalMetrics:
    """
    Evaluate retrieval quality: did we find the right documents?

    These metrics require "ground truth" — you must know which chunks
    (or at minimum which pages) contain the answer.
    """

    @staticmethod
    def hit_rate(retrieved_chunks: List[Dict], expected_pages: List[int]) -> float:
        """
        Hit Rate (= Recall@k binary): Is the answer somewhere in the retrieved chunks?

        Simplest and most important retrieval metric.
        1.0 = at least one chunk from the correct page was retrieved.
        0.0 = none of the retrieved chunks contain the answer.

        This is the metric to optimize first. If hit_rate is low,
        nothing else matters — the LLM can't answer without context.
        """
        if not expected_pages:
            return 1.0
        retrieved_pages = {
            c.get("metadata", {}).get("page_number")
            for c in retrieved_chunks
        }
        return 1.0 if retrieved_pages & set(expected_pages) else 0.0

    @staticmethod
    def mrr(retrieved_chunks: List[Dict], expected_pages: List[int]) -> float:
        """
        Mean Reciprocal Rank: How quickly does the first relevant chunk appear?

        MRR = 1/rank of the first relevant result.
        - Relevant at position 1 → MRR = 1.0 (perfect)
        - Relevant at position 3 → MRR = 0.33
        - Not found → MRR = 0.0

        Higher is better. Measures ranking quality, not just recall.
        """
        expected_set = set(expected_pages)
        for rank, chunk in enumerate(retrieved_chunks, 1):
            page = chunk.get("metadata", {}).get("page_number")
            if page in expected_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def precision_at_k(
        retrieved_chunks: List[Dict], expected_pages: List[int], k: int = 5
    ) -> float:
        """
        Precision@k: What fraction of top-k retrieved chunks are relevant?

        Retrieved 6 chunks, 4 are from correct pages → precision@6 = 0.67

        Low precision means the LLM gets noisy context, which can
        confuse it and reduce answer quality.
        """
        if k == 0 or not expected_pages:
            return 0.0
        expected_set = set(expected_pages)
        relevant = sum(
            1 for c in retrieved_chunks[:k]
            if c.get("metadata", {}).get("page_number") in expected_set
        )
        return relevant / k

    @staticmethod
    def ndcg_at_k(
        retrieved_chunks: List[Dict], expected_pages: List[int], k: int = 5
    ) -> float:
        """
        NDCG@k (Normalized Discounted Cumulative Gain).

        Like precision, but position-weighted: a relevant chunk at position 1
        contributes more than one at position 5.

        DCG = sum(rel_i / log2(i+1))
        NDCG = DCG / ideal_DCG
        """
        expected_set = set(expected_pages)

        dcg = 0.0
        for i, chunk in enumerate(retrieved_chunks[:k], 1):
            page = chunk.get("metadata", {}).get("page_number")
            rel = 1.0 if page in expected_set else 0.0
            dcg += rel / math.log2(i + 1)

        # Ideal DCG: all relevant chunks ranked first
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(expected_pages), k) + 1))

        return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# 2. GENERATION METRICS
# ============================================================

class GenerationMetrics:
    """
    Evaluate answer quality: is the generated answer correct and faithful?

    Two types:
    - Reference-based (need expected_answer): ROUGE-L, answer correctness
    - Reference-free (only need context): faithfulness
    """

    @staticmethod
    def rouge_l(reference: str, hypothesis: str) -> float:
        """
        ROUGE-L: Longest Common Subsequence between reference and generated answer.

        Measures overlap at the word level. Better than BLEU for long answers
        because it captures sentence-level structure, not just n-grams.

        Range [0, 1]. Typical good score: > 0.4
        """
        if not reference or not hypothesis:
            return 0.0
        try:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)
            return scores["rougeL"].fmeasure
        except Exception as e:
            logger.warning("ROUGE-L error: %s", e)
            return 0.0

    @staticmethod
    def faithfulness(answer: str, context: str) -> float:
        """
        Faithfulness: Is the answer grounded in the provided context?

        Splits the answer into claims (sentences), then checks if each
        claim has supporting evidence in the context using word overlap.

        Score = supported_claims / total_claims
        Range [0, 1]. Target: > 0.8

        Note: This is a simple word-overlap heuristic. Production systems
        use LLM-as-a-judge (e.g., RAGAS) for more accurate faithfulness
        scoring, but that requires an additional LLM call per evaluation.
        """
        if not answer or not context:
            return 0.0

        sentences = [s.strip() for s in re.split(r"[.!?\n]+", answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0

        context_lower = context.lower()
        context_words = set(context_lower.split())

        supported = 0
        for sentence in sentences:
            sent_words = set(sentence.lower().split())
            # Remove common stop words for more meaningful overlap
            meaningful = sent_words - {
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "shall", "can",
                "to", "of", "in", "for", "on", "with", "at", "by", "from",
                "as", "into", "through", "during", "before", "after", "and",
                "but", "or", "nor", "not", "no", "so", "if", "than", "that",
                "this", "it", "its", "they", "their", "them",
            }
            if not meaningful:
                supported += 1
                continue
            overlap = len(meaningful & context_words) / len(meaningful)
            if overlap >= 0.5:
                supported += 1

        return supported / len(sentences)

    @staticmethod
    def answer_correctness(answer: str, expected_answer: str) -> float:
        """
        Answer Correctness: Does the answer convey the same information
        as the expected answer?

        Uses word-level F1 score between answer and expected answer.
        This captures whether the key facts are present regardless of
        exact wording.

        Range [0, 1]. Target: > 0.5
        """
        if not answer or not expected_answer:
            return 0.0

        answer_tokens = set(answer.lower().split())
        expected_tokens = set(expected_answer.lower().split())

        if not answer_tokens or not expected_tokens:
            return 0.0

        common = answer_tokens & expected_tokens
        if not common:
            return 0.0

        precision = len(common) / len(answer_tokens)
        recall = len(common) / len(expected_tokens)
        return 2 * precision * recall / (precision + recall)


# ============================================================
# 3. END-TO-END METRICS
# ============================================================

class EndToEndMetrics:
    """
    End-to-end metrics that evaluate the complete pipeline.
    """

    @staticmethod
    def hallucination_check(answer: str, context: str) -> bool:
        """
        Hallucination Detection: Does the answer contain fabricated numbers?

        Checks if numbers in the answer exist in the context.
        If >30% of numbers are new (not in context), flags as hallucination.

        Returns True if hallucination detected, False if grounded.

        Note: This only catches numerical hallucinations. For full
        hallucination detection, use LLM-as-a-judge.
        """
        answer_numbers = set(re.findall(r"\d+", answer))
        if not answer_numbers:
            return False

        context_numbers = set(re.findall(r"\d+", context))
        new_numbers = answer_numbers - context_numbers

        return len(new_numbers) > len(answer_numbers) * 0.3

    @staticmethod
    def citation_accuracy(answer: str, retrieved_chunks: List[Dict]) -> float:
        """
        Citation Accuracy: Do the [Source N] references in the answer
        correspond to actual retrieved sources?

        Checks each [Source N: filename] citation against the retrieved
        chunk filenames.

        Range [0, 1]. 1.0 = all citations are valid.
        """
        citations = re.findall(r"\[Source (\d+):[^\]]*\]", answer)
        if not citations:
            return 0.0

        valid = 0
        for source_num in citations:
            idx = int(source_num) - 1
            if 0 <= idx < len(retrieved_chunks):
                valid += 1

        return valid / len(citations)

    @staticmethod
    def no_answer_correctness(answer: str, has_answer_in_context: bool) -> float:
        """
        No-Answer Correctness: When the context doesn't contain the answer,
        does the system correctly say "not found" instead of hallucinating?

        Returns:
            1.0 if behavior is correct (answers when should, refuses when should)
            0.0 if behavior is wrong (hallucinating or refusing valid answers)
        """
        refusal_patterns = [
            r"not found", r"not mentioned", r"no information",
            r"not specified", r"does not (contain|mention|include)",
            r"cannot (find|determine)", r"no relevant",
        ]
        is_refusal = any(
            re.search(p, answer, re.IGNORECASE) for p in refusal_patterns
        )

        if has_answer_in_context and not is_refusal:
            return 1.0  # Correctly answered
        if not has_answer_in_context and is_refusal:
            return 1.0  # Correctly refused
        return 0.0


# ============================================================
# EVALUATION RESULT
# ============================================================

@dataclass
class RAGEvaluationResult:
    """Complete evaluation result for one question."""

    # Retrieval
    hit_rate: float
    mrr: float
    precision_at_k: float
    ndcg_at_k: float

    # Generation
    faithfulness: float
    rouge_l: float
    answer_correctness: float

    # End-to-end
    hallucination_detected: bool
    citation_accuracy: float
    no_answer_correct: float

    # Metadata
    question: str = ""
    answer: str = ""
    num_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval": {
                "hit_rate": round(self.hit_rate, 3),
                "mrr": round(self.mrr, 3),
                "precision@k": round(self.precision_at_k, 3),
                "ndcg@k": round(self.ndcg_at_k, 3),
            },
            "generation": {
                "faithfulness": round(self.faithfulness, 3),
                "rouge_l": round(self.rouge_l, 3),
                "answer_correctness": round(self.answer_correctness, 3),
            },
            "end_to_end": {
                "hallucination_detected": self.hallucination_detected,
                "citation_accuracy": round(self.citation_accuracy, 3),
                "no_answer_correct": round(self.no_answer_correct, 3),
            },
        }

    @property
    def overall_score(self) -> float:
        """
        Weighted overall score.

        Weights reflect importance in production:
        - Retrieval 40%: if you don't find the right chunks, nothing works
        - Generation 40%: faithfulness is critical for trust
        - End-to-end 20%: hallucination and citation are safety nets
        """
        retrieval = np.mean([self.hit_rate, self.mrr, self.ndcg_at_k])
        generation = np.mean([self.faithfulness, self.answer_correctness])
        e2e = np.mean([
            0.0 if self.hallucination_detected else 1.0,
            self.citation_accuracy,
            self.no_answer_correct,
        ])
        return round(0.4 * retrieval + 0.4 * generation + 0.2 * e2e, 3)
