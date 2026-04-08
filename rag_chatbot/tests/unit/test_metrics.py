"""
Tests for RAG evaluation metrics (RAGAS-based).
"""
import pytest
from backend.evaluation.metrics import RAGEvaluationResult, RAGASEvaluator


class TestRAGEvaluationResult:
    """Tests for the evaluation result dataclass."""

    def test_overall_score_perfect(self):
        result = RAGEvaluationResult(
            faithfulness=1.0,
            answer_relevancy=1.0,
            context_precision=1.0,
            context_recall=1.0,
        )
        assert result.overall_score == 1.0

    def test_overall_score_zero(self):
        result = RAGEvaluationResult()
        assert result.overall_score == 0.0

    def test_overall_score_weighted(self):
        result = RAGEvaluationResult(
            faithfulness=0.8,
            answer_relevancy=0.6,
            context_precision=0.7,
            context_recall=0.5,
        )
        # 0.3*0.8 + 0.3*0.6 + 0.2*0.7 + 0.2*0.5 = 0.24 + 0.18 + 0.14 + 0.10 = 0.66
        assert result.overall_score == 0.66

    def test_to_dict(self):
        result = RAGEvaluationResult(
            faithfulness=0.85,
            answer_relevancy=0.72,
            context_precision=0.90,
            context_recall=0.65,
        )
        d = result.to_dict()
        assert d["faithfulness"] == 0.85
        assert d["answer_relevancy"] == 0.72
        assert d["context_precision"] == 0.9
        assert d["context_recall"] == 0.65

    def test_to_dict_keys(self):
        result = RAGEvaluationResult()
        d = result.to_dict()
        expected_keys = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
        assert set(d.keys()) == expected_keys

    def test_score_bounds(self):
        result = RAGEvaluationResult(
            faithfulness=1.0,
            answer_relevancy=1.0,
            context_precision=1.0,
            context_recall=1.0,
        )
        assert 0.0 <= result.overall_score <= 1.0


class TestRAGASEvaluator:
    """Tests for the RAGAS evaluator class (unit tests, no LLM calls)."""

    def test_init_default_model(self):
        evaluator = RAGASEvaluator()
        assert evaluator._model_name is not None

    def test_init_custom_model(self):
        evaluator = RAGASEvaluator(model_name="llama3.2")
        assert evaluator._model_name == "llama3.2"

    def test_format_results(self):
        """Test result formatting with a mock pandas DataFrame."""
        import pandas as pd

        mock_df = pd.DataFrame({
            "user_input": ["What is X?"],
            "response": ["X is Y."],
            "retrieved_contexts": [["Context about X"]],
            "faithfulness": [0.85],
            "answer_relevancy": [0.72],
        })

        # Mock ragas result object
        class MockResult:
            def to_pandas(self):
                return mock_df

        samples = [{"user_input": "What is X?", "response": "X is Y."}]
        result = RAGASEvaluator._format_results(MockResult(), samples)

        assert result["num_samples"] == 1
        assert result["summary"]["faithfulness"] == 0.85
        assert result["summary"]["answer_relevancy"] == 0.72
        assert len(result["per_sample"]) == 1
