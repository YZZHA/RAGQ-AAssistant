import pytest
from src.retrieval.bm25 import BM25Index


SAMPLE_CHUNKS = [
    {"chunk_id": "doc1_p_001", "doc_id": "doc1", "content": "跨境投资登记系统用于外商投资项目在线登记"},
    {"chunk_id": "doc1_p_002", "doc_id": "doc1", "content": "系统支持项目信息录入和材料上传"},
    {"chunk_id": "doc2_p_001", "doc_id": "doc2", "content": "外商投资负面清单每年更新一次"},
    {"chunk_id": "doc2_p_002", "doc_id": "doc2", "content": "限制类行业需要额外审批流程"},
    {"chunk_id": "doc3_p_001", "doc_id": "doc3", "content": "企业所得税标准税率25%，高新企业15%"},
]


class TestBM25Build:
    def test_build_empty(self):
        idx = BM25Index()
        idx.build([])
        assert idx._doc_count == 0

    def test_build_with_chunks(self):
        idx = BM25Index()
        idx.build(SAMPLE_CHUNKS)
        assert idx._doc_count == 5
        assert idx._avg_dl > 0


class TestBM25Search:
    def test_search_returns_list(self):
        idx = BM25Index()
        idx.build(SAMPLE_CHUNKS)
        results = idx.search("外商投资登记", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_relevant_docs_first(self):
        idx = BM25Index()
        idx.build(SAMPLE_CHUNKS)
        results = idx.search("跨境投资登记系统", top_k=5)
        assert len(results) > 0
        top_doc_id = results[0]["doc_id"]
        top_ids = [r["doc_id"] for r in results]
        assert "doc1" in top_ids

    def test_search_unrelated_query(self):
        idx = BM25Index()
        idx.build(SAMPLE_CHUNKS)
        results = idx.search("天气很好今天吃饭", top_k=3)
        assert results == []

    def test_search_empty_query(self):
        idx = BM25Index()
        idx.build(SAMPLE_CHUNKS)
        results = idx.search("", top_k=3)
        assert results == []


class TestBM25Score:
    def test_each_result_has_score_and_method(self):
        idx = BM25Index()
        idx.build(SAMPLE_CHUNKS)
        results = idx.search("投资", top_k=5)
        for r in results:
            assert "score" in r
            assert r["method"] == "bm25"
            assert r["score"] > 0

    def test_scores_decreasing(self):
        idx = BM25Index()
        idx.build(SAMPLE_CHUNKS)
        results = idx.search("外商投资", top_k=5)
        scores = [r["score"] for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
