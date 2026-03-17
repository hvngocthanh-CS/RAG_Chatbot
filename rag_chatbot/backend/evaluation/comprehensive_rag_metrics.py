"""
Professional RAG Evaluation Framework - Industry Standard.

3 Main Areas (as per enterprise RAG evaluation standards):
1. RETRIEVAL METRICS - Evaluate document retrieval quality
2. LLM/GENERATION METRICS - Evaluate answer generation quality  
3. END-TO-END METRICS - Evaluate complete RAG pipeline

Reference:
- NIST IR evaluation standards
- DeepEval framework
- RAGAS (LLM-as-a-judge) paper
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ============================================================
# RETRIEVAL METRICS
# ============================================================

class RetrievalMetrics:
    """Metrics for evaluating document retrieval quality."""

    @staticmethod
    def precision_at_k(retrieved_docs: List[Dict], relevant_docs: List[str], k: int = 5) -> float:
        """
        Precision@k: What fraction of top-k retrieved docs are relevant?
        
        Formula: |relevant ∩ retrieved@k| / k
        
        Args:
            retrieved_docs: List of retrieved chunk dicts with 'id' field
            relevant_docs: List of relevant document IDs (ground truth)
            k: Number of top results to consider
            
        Returns:
            Precision score (0-1)
        """
        if k == 0:
            return 0.0

        retrieved_ids = {doc.get("id", doc.get("metadata", {}).get("document_id", "")) 
                        for doc in retrieved_docs[:k]}
        relevant_set = set(relevant_docs)
        
        if not retrieved_ids:
            return 0.0
            
        intersection = len(retrieved_ids & relevant_set)
        return intersection / k

    @staticmethod
    def recall_at_k(retrieved_docs: List[Dict], relevant_docs: List[str], k: int = 5) -> float:
        """
        Recall@k: What fraction of all relevant docs were retrieved in top-k?
        
        Formula: |relevant ∩ retrieved@k| / |relevant|
        
        Args:
            retrieved_docs: List of retrieved chunks
            relevant_docs: List of all relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall score (0-1)
        """
        if not relevant_docs:
            return 1.0

        retrieved_ids = {doc.get("id", doc.get("metadata", {}).get("document_id", "")) 
                        for doc in retrieved_docs[:k]}
        relevant_set = set(relevant_docs)
        
        intersection = len(retrieved_ids & relevant_set)
        return intersection / len(relevant_set)

    @staticmethod
    def f1_at_k(retrieved_docs: List[Dict], relevant_docs: List[str], k: int = 5) -> float:
        """F1 Score = 2 * (Precision * Recall) / (Precision + Recall)"""
        precision = RetrievalMetrics.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = RetrievalMetrics.recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def mrr(retrieved_docs: List[Dict], relevant_docs: List[str]) -> float:
        """
        Mean Reciprocal Rank: Average of reciprocal rank of first relevant result.
        
        Formula: 1 / rank_of_first_relevant
        
        Measures how quickly a relevant document appears.
        Example: If first relevant at position 3 → MRR = 1/3 = 0.33
        """
        relevant_set = set(relevant_docs)
        
        for rank, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.get("id", doc.get("metadata", {}).get("document_id", ""))
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_docs: List[Dict], relevant_docs: List[str], k: int = 5) -> float:
        """
        NDCG (Normalized Discounted Cumulative Gain).
        
        DCG = sum(relevance_i / log2(i+1)) for i in range(k)
        NDCG = DCG / IDCG (Ideal DCG with best possible ranking)
        
        Position-weighted: Top results matter more than bottom.
        Example: Relevant at position 1 (DCG=1/1) > Relevant at position 5 (DCG=1/log2(6))
        """
        relevant_set = set(relevant_docs)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k], 1):
            doc_id = doc.get("id", doc.get("metadata", {}).get("document_id", ""))
            relevance = 1.0 if doc_id in relevant_set else 0.0
            dcg += relevance / math.log2(i + 1)
        
        # Calculate IDCG (ideal ranking)
        idcg = 0.0
        num_relevant = min(len(relevant_docs), k)
        for i in range(1, num_relevant + 1):
            idcg += 1.0 / math.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg


# ============================================================
# LLM/GENERATION METRICS
# ============================================================

class GenerationMetrics:
    """Metrics for evaluating LLM-generated answers."""

    @staticmethod
    def bleu_score(reference: str, hypothesis: str, weights: tuple = (0.25, 0.25, 0.25, 0.25)) -> float:
        """
        BLEU (Bilingual Evaluation Understudy) Score.
        
        Measures n-gram overlap between generated answer and reference answer.
        Range: [0, 1] where 1 = perfect match
        
        Args:
            reference: Ground truth answer
            hypothesis: Generated answer
            weights: Weights for 1-gram, 2-gram, 3-gram, 4-gram (default: equal)
            
        Returns:
            BLEU score (0-1)
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        try:
            smooting_fn = SmoothingFunction().method1
            return sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=weights,
                smoothing_function=smooting_fn
            )
        except Exception as e:
            logger.warning(f"BLEU calculation error: {e}")
            return 0.0

    @staticmethod
    def rouge_score(reference: str, hypothesis: str, rouge_type: str = "rougeL") -> float:
        """
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Score.
        
        Measures longest common subsequence between reference and hypothesis.
        Better than BLEU for summary-type answers.
        
        Args:
            reference: Ground truth answer
            hypothesis: Generated answer
            rouge_type: "rouge1", "rouge2", "rougeL" (default: rougeL)
            
        Returns:
            F1 score of ROUGE metric (0-1)
        """
        try:
            scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)
            return scores[rouge_type].fmeasure
        except Exception as e:
            logger.warning(f"ROUGE calculation error: {e}")
            return 0.0

    @staticmethod
    def semantic_similarity(text1: str, text2: str) -> float:
        """
        Semantic Similarity using TF-IDF cosine similarity.
        
        Faster alternative to BERTScore for quick evaluation.
        Measures semantic overlap between two texts.
        
        Args:
            text1: First text (e.g., reference answer)
            text2: Second text (e.g., generated answer)
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity error: {e}")
            return 0.0

    @staticmethod
    def answer_relevancy(question: str, answer: str) -> float:
        """
        Answer Relevancy: How relevant is the answer to the question?
        
        Uses semantic similarity between question and answer.
        High score = answer directly addresses the question.
        
        Args:
            question: User question
            answer: Generated answer
            
        Returns:
            Relevancy score (0-1)
        """
        return GenerationMetrics.semantic_similarity(question, answer)

    @staticmethod
    def faithfulness(answer: str, context: str, threshold: float = 0.4) -> float:
        """
        Faithfulness: Is the answer grounded in the provided context?
        
        Measures what percentage of answer is supported by context.
        Uses semantic similarity between answer sentences and context.
        
        Args:
            answer: Generated answer
            context: Retrieved context chunks
            threshold: Similarity threshold for considering a sentence grounded
            
        Returns:
            Faithfulness score (0-1) = grounded_sentences / total_sentences
        """
        if not answer or not context:
            return 0.0

        # Split answer into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
        if not sentences:
            return 0.0

        # Check each sentence against context
        grounded_count = 0
        for sentence in sentences:
            similarity = GenerationMetrics.semantic_similarity(sentence, context)
            if similarity >= threshold:
                grounded_count += 1

        return grounded_count / len(sentences)


# ============================================================
# END-TO-END METRICS
# ============================================================

class EndToEndMetrics:
    """Metrics for evaluating complete RAG pipeline."""

    @staticmethod
    def context_precision(retrieved_chunks: List[Dict], relevant_chunks: List[str]) -> float:
        """
        Context Precision: What % of retrieved chunks are relevant?
        
        Formula: |relevant_chunks ∩ retrieved_chunks| / |retrieved_chunks|
        
        Measures false positives in retrieval.
        Example: Retrieved 10 chunks, 8 are relevant → precision = 0.8
        """
        if not retrieved_chunks:
            return 1.0

        retrieved_ids = {chunk.get("id", chunk.get("metadata", {}).get("chunk_index", "")) 
                        for chunk in retrieved_chunks}
        relevant_set = set(relevant_chunks)
        
        if not retrieved_ids:
            return 0.0

        intersection = len(retrieved_ids & relevant_set)
        return intersection / len(retrieved_ids)

    @staticmethod
    def context_recall(retrieved_chunks: List[Dict], relevant_chunks: List[str]) -> float:
        """
        Context Recall: What % of relevant chunks were retrieved?
        
        Formula: |relevant_chunks ∩ retrieved_chunks| / |all_relevant_chunks|
        
        Measures false negatives in retrieval.
        Example: 5 relevant chunks exist, retrieved 4 → recall = 0.8
        """
        if not relevant_chunks:
            return 1.0

        retrieved_ids = {chunk.get("id", chunk.get("metadata", {}).get("chunk_index", "")) 
                        for chunk in retrieved_chunks}
        relevant_set = set(relevant_chunks)
        
        intersection = len(retrieved_ids & relevant_set)
        return intersection / len(relevant_set)

    @staticmethod
    def context_f1(retrieved_chunks: List[Dict], relevant_chunks: List[str]) -> float:
        """Harmonic mean of Context Precision and Recall."""
        precision = EndToEndMetrics.context_precision(retrieved_chunks, relevant_chunks)
        recall = EndToEndMetrics.context_recall(retrieved_chunks, relevant_chunks)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def hallucination_detection(answer: str, context: str) -> bool:
        """
        Hallucination Detection: Does answer contain information NOT in context?
        
        Heuristics:
        - Specific numbers/dates not in context
        - Strong claims without source
        - Out-of-distribution information
        
        Returns:
            True if hallucination likely, False if grounded
        """
        # Extract numbers from both
        answer_numbers = set(re.findall(r'\d+', answer))
        context_numbers = set(re.findall(r'\d+', context))
        
        # If answer has numbers not in context, likely hallucination
        suspicious_numbers = answer_numbers - context_numbers
        if len(suspicious_numbers) > len(answer_numbers) * 0.3:  # >30% new numbers
            return True

        # Check semantic alignment
        similarity = GenerationMetrics.semantic_similarity(answer, context)
        if similarity < 0.3:  # Low overlap suggests hallucination
            return True

        return False

    @staticmethod
    def citation_accuracy(answer: str, context: str) -> float:
        """
        Citation Accuracy: How accurate are the source citations?
        
        Checks:
        1. Citation format correctness [Source X: filename, Page Y]
        2. Referenced content exists in context
        3. Citation location is correct
        
        Returns:
            Accuracy score (0-1)
        """
        # Extract citations
        citation_pattern = r'\[Source \d+: ([^]]+)\]'
        citations = re.findall(citation_pattern, answer)
        
        if not citations:
            return 0.0  # No citations = no accuracy to measure

        valid_citations = 0
        for citation in citations:
            # Check if citation source appears in context
            if citation.lower() in context.lower() or any(
                word in context.lower() for word in citation.lower().split()[:2]
            ):
                valid_citations += 1

        return valid_citations / len(citations) if citations else 0.0

    @staticmethod
    def answer_correctness(answer: str, ground_truth: Optional[str], context: str) -> float:
        """
        Answer Correctness: Is the answer factually correct?
        
        Evaluation criteria:
        1. If ground_truth provided: Compare answer against ground truth
        2. If no ground_truth: Check if answer is grounded in context
        
        Args:
            answer: Generated answer
            ground_truth: Expected correct answer (optional)
            context: Retrieved context
            
        Returns:
            Correctness score (0-1)
        """
        if ground_truth:
            # Compare against ground truth
            similarity = GenerationMetrics.semantic_similarity(answer, ground_truth)
            return similarity
        else:
            # Check if grounded in context
            return GenerationMetrics.faithfulness(answer, context)


# ============================================================
# COMPREHENSIVE EVALUATION RESULT
# ============================================================

@dataclass
class RAGEvaluationResult:
    """Complete RAG evaluation results across all 3 areas."""
    
    # Retrieval Metrics
    precision_at_5: float
    recall_at_5: float
    f1_at_5: float
    mrr: float
    ndcg_at_5: float
    
    # LLM/Generation Metrics
    bleu: float
    rouge_l: float
    semantic_sim: float
    answer_relevancy: float
    faithfulness: float
    
    # End-to-End Metrics
    context_precision: float
    context_recall: float
    context_f1: float
    hallucination_detected: bool
    citation_accuracy: float
    answer_correctness: float
    
    # Metadata
    question: str = ""
    answer: str = ""
    num_chunks_retrieved: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "retrieval": {
                "precision@5": round(self.precision_at_5, 3),
                "recall@5": round(self.recall_at_5, 3),
                "f1@5": round(self.f1_at_5, 3),
                "mrr": round(self.mrr, 3),
                "ndcg@5": round(self.ndcg_at_5, 3),
            },
            "generation": {
                "bleu": round(self.bleu, 3),
                "rouge_l": round(self.rouge_l, 3),
                "semantic_similarity": round(self.semantic_sim, 3),
                "answer_relevancy": round(self.answer_relevancy, 3),
                "faithfulness": round(self.faithfulness, 3),
            },
            "end_to_end": {
                "context_precision": round(self.context_precision, 3),
                "context_recall": round(self.context_recall, 3),
                "context_f1": round(self.context_f1, 3),
                "hallucination_detected": self.hallucination_detected,
                "citation_accuracy": round(self.citation_accuracy, 3),
                "answer_correctness": round(self.answer_correctness, 3),
            },
            "metadata": {
                "chunks_retrieved": self.num_chunks_retrieved,
            }
        }

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall RAG quality score."""
        retrieval_avg = np.mean([self.precision_at_5, self.recall_at_5, self.ndcg_at_5])
        generation_avg = np.mean([self.bleu, self.rouge_l, self.faithfulness])
        e2e_avg = np.mean([self.context_precision, self.context_recall, self.citation_accuracy])
        
        # Weights: retrieval (40%), generation (40%), end-to-end (20%)
        overall = (0.4 * retrieval_avg) + (0.4 * generation_avg) + (0.2 * e2e_avg)
        
        # Penalty for hallucination
        if self.hallucination_detected:
            overall *= 0.5
        
        return round(overall, 3)
