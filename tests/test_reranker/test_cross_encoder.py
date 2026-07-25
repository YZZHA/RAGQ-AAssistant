import pytest
from src.reranker.cross_encoder import dynamic_top_k


MOCK_CHUNKS_FLAT = [
    {"chunk_id": "a", "rerank_score": 0.5},
    {"chunk_id": "b", "rerank_score": 0.48},
    {"chunk_id": "c", "rerank_score": 0.45},
    {"chunk_id": "d", "rerank_score": 0.43},
    {"chunk_id": "e", "rerank_score": 0.40},
]

MOCK_CHUNKS_GAP = [
    {"chunk_id": "a", "rerank_score": 0.95},
    {"chunk_id": "b", "rerank_score": 0.92},
    {"chunk_id": "c", "rerank_score": 0.30},
    {"chunk_id": "d", "rerank_score": 0.28},
    {"chunk_id": "e", "rerank_score": 0.25},
]

MOCK_CHUNKS_LOW = [
    {"chunk_id": "a", "rerank_score": 0.05},
    {"chunk_id": "b", "rerank_score": 0.04},
    {"chunk_id": "c", "rerank_score": 0.03},
]


class TestDynamicTopK:
    def test_empty_input(self):
        assert dynamic_top_k([]) == []

    def test_fewer_than_min_k(self):
        result = dynamic_top_k(MOCK_CHUNKS_LOW, min_k=3, max_k=5)
        assert len(result) == 3

    def test_gap_triggers_early_cut(self):
        result = dynamic_top_k(MOCK_CHUNKS_GAP, min_k=2, max_k=5, gap_threshold=0.5)
        assert len(result) <= 3

    def test_flat_distribution_takes_more(self):
        result = dynamic_top_k(MOCK_CHUNKS_FLAT, min_k=2, max_k=5, gap_threshold=0.5)
        assert len(result) >= 3

    def test_each_result_retains_keys(self):
        result = dynamic_top_k(MOCK_CHUNKS_FLAT, min_k=2, max_k=3)
        for r in result:
            assert "chunk_id" in r
            assert "rerank_score" in r

    def test_no_rerank_score_uses_defaults(self):
        chunks = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        result = dynamic_top_k(chunks, min_k=2, max_k=5)
        assert len(result) == 2
