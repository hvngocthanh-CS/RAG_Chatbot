"""
Unit tests for retrieval metrics (Hit@k, Recall@k, MRR).

Generation metrics (RAGAS) are not unit-tested — they require a real LLM
and are covered by running the full pipeline via run_evaluation.py.
"""
from evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    compute_retrieval_metrics,
)


# ======================================================================
# hit_at_k
# ======================================================================

class TestHitAtK:
    def test_hit_found(self):
        assert hit_at_k(["a.pdf", "b.pdf"], {"b.pdf"}) == 1

    def test_hit_not_found(self):
        assert hit_at_k(["a.pdf", "b.pdf"], {"c.pdf"}) == 0

    def test_hit_empty_retrieved(self):
        assert hit_at_k([], {"a.pdf"}) == 0


# ======================================================================
# recall_at_k
# ======================================================================

class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k(["a.pdf", "b.pdf"], {"a.pdf", "b.pdf"}) == 1.0

    def test_half_recall(self):
        assert recall_at_k(["a.pdf"], {"a.pdf", "b.pdf"}) == 0.5

    def test_no_relevant_returns_zero(self):
        assert recall_at_k(["a.pdf"], set()) == 0.0


# ======================================================================
# reciprocal_rank (MRR per-sample)
# ======================================================================

class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a.pdf", "b.pdf"], {"a.pdf"}) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["x.pdf", "a.pdf"], {"a.pdf"}) == 0.5

    def test_third_position(self):
        assert round(reciprocal_rank(["x.pdf", "y.pdf", "a.pdf"], {"a.pdf"}), 4) == 0.3333

    def test_not_found(self):
        assert reciprocal_rank(["x.pdf"], {"a.pdf"}) == 0.0


# ======================================================================
# compute_retrieval_metrics (aggregate)
# ======================================================================

class TestComputeRetrievalMetrics:
    def _chunk(self, filename):
        return {"content": "...", "metadata": {"filename": filename}}

    def test_perfect_run(self):
        runs = [{
            "has_answer": True,
            "expected_docs": ["a.pdf"],
            "retrieved_chunks": [self._chunk("a.pdf")],
        }]
        m = compute_retrieval_metrics(runs, k=5)
        assert m.hit_at_k == 1.0
        assert m.recall_at_k == 1.0
        assert m.mrr == 1.0
        assert m.num_answerable == 1

    def test_hit_at_rank_2(self):
        runs = [{
            "has_answer": True,
            "expected_docs": ["a.pdf"],
            "retrieved_chunks": [self._chunk("x.pdf"), self._chunk("a.pdf")],
        }]
        m = compute_retrieval_metrics(runs, k=5)
        assert m.hit_at_k == 1.0
        assert m.mrr == 0.5

    def test_unanswerable_refusal(self):
        runs = [
            # Unanswerable, retriever returned nothing → refusal correct
            {"has_answer": False, "expected_docs": [], "retrieved_chunks": []},
            # Unanswerable, retriever leaked chunks → refusal incorrect
            {"has_answer": False, "expected_docs": [],
             "retrieved_chunks": [self._chunk("noise.pdf")]},
        ]
        m = compute_retrieval_metrics(runs, k=5)
        assert m.num_unanswerable == 2
        assert m.refusal_accuracy == 0.5  # 1 of 2 correctly refused

    def test_k_cutoff(self):
        """With k=1, a hit at rank 2 should not count."""
        runs = [{
            "has_answer": True,
            "expected_docs": ["a.pdf"],
            "retrieved_chunks": [self._chunk("x.pdf"), self._chunk("a.pdf")],
        }]
        m = compute_retrieval_metrics(runs, k=1)
        assert m.hit_at_k == 0.0
        assert m.mrr == 0.0

    def test_mixed_dataset(self):
        runs = [
            # hit at rank 1, full recall
            {"has_answer": True, "expected_docs": ["a.pdf"],
             "retrieved_chunks": [self._chunk("a.pdf"), self._chunk("x.pdf")]},
            # miss entirely
            {"has_answer": True, "expected_docs": ["b.pdf"],
             "retrieved_chunks": [self._chunk("x.pdf"), self._chunk("y.pdf")]},
        ]
        m = compute_retrieval_metrics(runs, k=5)
        assert m.hit_at_k == 0.5   # 1 hit / 2 answerable
        assert m.recall_at_k == 0.5
        assert m.mrr == 0.5        # (1.0 + 0.0) / 2
