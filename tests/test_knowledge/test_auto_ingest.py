import pytest
from unittest.mock import patch, MagicMock
from src.knowledge.auto_ingest import _qa_to_document


class TestQAToDocument:
    def test_creates_markdown(self):
        result = _qa_to_document({"question": "测试问题？", "answer": "测试答案。"})
        assert "# 自动生成知识" in result
        assert "测试问题" in result
        assert "测试答案" in result


class TestAutoIngest:
    def test_ingest_success(self):
        qa = {"question": "测试问题？", "answer": "测试答案。"}
        mock_bm25 = MagicMock()
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [0.1] * 384

        with patch("src.ingestion.chunker.chunk_document", return_value=[
            {"chunk_id": "auto_p_001", "doc_id": "auto_test", "content": "测试", "char_count": 2},
        ]), \
             patch("src.knowledge.auto_ingest._rebuild_indexes") as mock_rebuild:
            from src.knowledge.auto_ingest import auto_ingest
            result = auto_ingest(qa, mock_bm25, mock_store, mock_embedder)
            assert result is True
            mock_rebuild.assert_called_once()
