"""
Tests for comprehensive RAG evaluation metrics.
"""
import pytest
import numpy as np
from backend.evaluation import RetrievalMetrics, GenerationMetrics


class TestRetrievalMetrics:
    """Tests for retrieval quality metrics."""

    def setup_method(self):
        self.metrics = RetrievalMetrics()

    def test_precision_at_k(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc5"}

        precision = self.metrics.precision_at_k(retrieved, relevant, k=5)
        assert precision == 3 / 5

    def test_precision_empty_retrieved(self):
        precision = self.metrics.precision_at_k([], {"doc1"}, k=5)
        assert precision == 0.0

    def test_recall_at_k(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc3", "doc4", "doc5"}

        recall = self.metrics.recall_at_k(retrieved, relevant, k=3)
        assert recall == 2 / 4

    def test_recall_empty_relevant(self):
        recall = self.metrics.recall_at_k(["doc1"], set(), k=1)
        assert recall == 0.0

    def test_mrr(self):
        retrieved = ["doc2", "doc3", "doc1"]
        relevant = {"doc1"}

        mrr = self.metrics.mrr(retrieved, relevant)
        assert abs(mrr - 1 / 3) < 0.001

    def test_mrr_first_position(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}

        mrr = self.metrics.mrr(retrieved, relevant)
        assert mrr == 1.0

    def test_mrr_no_relevant(self):
        mrr = self.metrics.mrr(["doc1", "doc2"], {"doc3"})
        assert mrr == 0.0

    def test_ndcg_perfect_ranking(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc2"}

        ndcg = self.metrics.ndcg_at_k(retrieved, relevant, k=4)
        assert ndcg == 1.0

    def test_ndcg_imperfect_ranking(self):
        retrieved = ["doc3", "doc1", "doc4", "doc2"]
        relevant = {"doc1", "doc2"}

        ndcg = self.metrics.ndcg_at_k(retrieved, relevant, k=4)
        assert 0.0 < ndcg < 1.0
