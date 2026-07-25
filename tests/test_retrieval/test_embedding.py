import pytest
from unittest.mock import patch, MagicMock


class TestEmbeddingSearch:
    def test_search_returns_list_with_method(self):
        with patch("src.retrieval.embedding.Embedder") as MockEmbedder, \
             patch("src.retrieval.embedding.vector_store") as mock_store:

            mock_e = MagicMock()
            mock_e.encode.return_value = [0.1] * 384
            MockEmbedder.return_value = mock_e
            mock_store.search.return_value = [
                {"chunk_id": "c1", "content": "test", "score": 0.95},
            ]

            from src.retrieval.embedding import search_by_text
            results = search_by_text("测试查询", top_k=5)
            assert len(results) == 1
            assert results[0]["method"] == "embedding"

    def test_search_encodes_query(self):
        with patch("src.retrieval.embedding.Embedder") as MockEmbedder, \
             patch("src.retrieval.embedding.vector_store") as mock_store:

            mock_e = MagicMock()
            MockEmbedder.return_value = mock_e
            mock_e.encode.return_value = [0.2] * 384
            mock_store.search.return_value = []

            from src.retrieval.embedding import search_by_text
            search_by_text("你好", top_k=3)
            mock_e.encode.assert_called_once_with("你好")

    def test_search_empty_query(self):
        with patch("src.retrieval.embedding.Embedder") as MockEmbedder, \
             patch("src.retrieval.embedding.vector_store") as mock_store:

            mock_e = MagicMock()
            MockEmbedder.return_value = mock_e
            mock_e.encode.return_value = [0.0] * 384
            mock_store.search.return_value = []

            from src.retrieval.embedding import search_by_text
            results = search_by_text("")
            assert results == []
