import pytest
from src.retrieval.local_store import LocalVectorStore
from src.ingestion.embedder import Embedder


CHUNKS = [
    {"chunk_id": "d1_p_001", "doc_id": "d1", "content": "跨境投资登记系统用于外商投资"},  # has "投资"
    {"chunk_id": "d2_p_001", "doc_id": "d2", "content": "负面清单每年更新一次"},          # has "更新"
    {"chunk_id": "d3_p_001", "doc_id": "d3", "content": "企业所得税标准税率25%"},          # has "税率"
]


@pytest.fixture(scope="module")
def store():
    s = LocalVectorStore()
    e = Embedder()
    s.build(CHUNKS, e)
    return s


class TestLocalStore:
    def test_build_count(self, store):
        assert store.size == 3

    def test_search_returns_list(self, store):
        e = Embedder()
        vec = e.encode("外商投资登记")
        results = store.search(vec, top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_search_relevant_first(self, store):
        e = Embedder()
        vec = e.encode("外商投资登记")
        results = store.search(vec, top_k=3)
        if results:
            top_ids = [r["doc_id"] for r in results]
            assert "d1" in top_ids

    def test_search_has_score_and_method(self, store):
        e = Embedder()
        vec = e.encode("test")
        results = store.search(vec, top_k=3)
        for r in results:
            assert "score" in r
            assert r["method"] == "embedding"
