import pytest
from src.evaluate.metrics import recall_at_k, mrr, ndcg_at_k, evaluate_retrieval


RETRIEVED = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
RELEVANT = {"doc_a", "doc_c"}


class TestRecall:
    def test_recall_at_1(self):
        assert recall_at_k(RETRIEVED, RELEVANT, 1) == 1 / 2

    def test_recall_at_3(self):
        assert recall_at_k(RETRIEVED, RELEVANT, 3) == 2 / 2

    def test_recall_at_5(self):
        assert recall_at_k(RETRIEVED, RELEVANT, 5) == 2 / 2

    def test_no_relevant(self):
        assert recall_at_k(RETRIEVED, {"doc_x"}, 5) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(RETRIEVED, set(), 3) == 0.0


class TestMRR:
    def test_first_is_relevant(self):
        assert mrr(["doc_a", "doc_b"], RELEVANT) == 1.0

    def test_second_is_relevant(self):
        assert mrr(["doc_x", "doc_a"], RELEVANT) == 1 / 2

    def test_none_relevant(self):
        assert mrr(["doc_x", "doc_y"], RELEVANT) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["doc_a", "doc_c"], RELEVANT, 2) == 1.0

    def test_worst_ranking(self):
        assert ndcg_at_k(["doc_x", "doc_y"], RELEVANT, 2) == 0.0


class TestEvaluateRetrieval:
    def test_returns_all_metrics(self):
        metrics = evaluate_retrieval(
            [{"doc_id": "doc_a"}, {"doc_id": "doc_b"}],
            {"doc_a"},
            ks=[1, 3],
        )
        assert "Recall@1" in metrics
        assert "MRR" in metrics
        assert "NDCG@3" in metrics
