"""
Tests for comprehensive RAG evaluation metrics.
"""
import pytest
import numpy as np
from backend.evaluation import RetrievalMetrics, GenerationMetrics, ComprehensiveRAGMetrics


class TestRetrievalMetrics:
    """Tests for retrieval quality metrics."""

    def setup_method(self):
        self.metrics = RetrievalMetrics()

    def test_precision_at_k(self):
        """Test precision@k calculation."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc5"}
        
        precision = self.metrics.precision_at_k(retrieved, relevant, k=5)
        
        # 3 out of 5 are relevant
        assert precision == 3 / 5

    def test_precision_empty_retrieved(self):
        """Test precision with no retrieved documents."""
        precision = self.metrics.precision_at_k([], {"doc1"}, k=5)
        assert precision == 0.0

    def test_recall_at_k(self):
        """Test recall@k calculation."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc3", "doc4", "doc5"}
        
        recall = self.metrics.recall_at_k(retrieved, relevant, k=3)
        
        # Found 2 out of 4 relevant
        assert recall == 2 / 4

    def test_recall_empty_relevant(self):
        """Test recall with no relevant documents."""
        recall = self.metrics.recall_at_k(["doc1"], set(), k=1)
        assert recall == 0.0

    def test_mrr(self):
        """Test Mean Reciprocal Rank."""
        retrieved = ["doc2", "doc3", "doc1"]  # doc1 is at position 3
        relevant = {"doc1"}
        
        mrr = self.metrics.mrr(retrieved, relevant)
        
        # First relevant at position 3 -> 1/3
        assert abs(mrr - 1 / 3) < 0.001

    def test_mrr_first_position(self):
        """Test MRR when first result is relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        
        mrr = self.metrics.mrr(retrieved, relevant)
        assert mrr == 1.0

    def test_mrr_no_relevant(self):
        """Test MRR when no relevant documents found."""
        mrr = self.metrics.mrr(["doc1", "doc2"], {"doc3"})
        assert mrr == 0.0

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        # All relevant docs at top
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc2"}
        
        ndcg = self.metrics.ndcg_at_k(retrieved, relevant, k=4)
        
        # Perfect ranking should give NDCG = 1.0
        assert ndcg == 1.0

    def test_ndcg_imperfect_ranking(self):
        """Test NDCG with imperfect ranking."""
        # Relevant docs not at top
        retrieved = ["doc3", "doc1", "doc4", "doc2"]
        relevant = {"doc1", "doc2"}
        
        ndcg = self.metrics.ndcg_at_k(retrieved, relevant, k=4)
        
        # NDCG should be less than 1.0 for imperfect ranking
        assert 0.0 < ndcg < 1.0
        
        # Check types
        assert isinstance(metrics["precision"], float)
        assert isinstance(metrics["recall"], float)
        assert isinstance(metrics["mrr"], float)
        assert isinstance(metrics["ndcg"], float)
        
        # Check bounds
        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{name} out of bounds: {value}"
