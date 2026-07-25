import pytest
from src.retrieval.fusion import rrf_fusion, weighted_fusion


BM25 = [
    {"chunk_id": "a", "content": "A", "score": 2.0, "method": "bm25"},
    {"chunk_id": "b", "content": "B", "score": 1.5, "method": "bm25"},
    {"chunk_id": "c", "content": "C", "score": 1.0, "method": "bm25"},
]

EMB = [
    {"chunk_id": "b", "content": "B", "score": 0.9, "method": "embedding"},
    {"chunk_id": "d", "content": "D", "score": 0.8, "method": "embedding"},
    {"chunk_id": "e", "content": "E", "score": 0.7, "method": "embedding"},
]


class TestRRFFusion:
    def test_rrf_returns_merged_list(self):
        result = rrf_fusion(BM25, EMB, top_k=5)
        assert len(result) == 5
        assert result[0]["method"] == "rrf"

    def test_rrf_both_empty(self):
        assert rrf_fusion([], []) == []

    def test_rrf_one_empty(self):
        result = rrf_fusion(BM25, [], top_k=3)
        assert len(result) == 3

    def test_rrf_duplicates_merged(self):
        result = rrf_fusion(BM25, EMB, top_k=5)
        cids = [r["chunk_id"] for r in result]
        assert len(cids) == len(set(cids))

    def test_rrf_has_rrf_score(self):
        result = rrf_fusion(BM25, EMB, top_k=2)
        for r in result:
            assert "rrf_score" in r


class TestWeightedFusion:
    def test_weighted_returns_merged_list(self):
        result = weighted_fusion(BM25, EMB, top_k=5)
        assert len(result) == 5

    def test_weighted_both_empty(self):
        assert weighted_fusion([], []) == []

    def test_weighted_has_fusion_score(self):
        result = weighted_fusion(BM25, EMB, top_k=2)
        for r in result:
            assert "fusion_score" in r
