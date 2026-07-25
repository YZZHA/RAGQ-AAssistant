import pytest
from unittest.mock import patch, MagicMock


class TestRunIngestion:
    def test_empty_directory(self):
        with patch("src.ingestion.pipeline.load_directory", return_value=[]):
            from src.ingestion.pipeline import run_ingestion
            result = run_ingestion("empty_dir", create_collection=False)
            assert result["documents_found"] == 0
            assert result["documents_processed"] == 0
            assert result["vectors_inserted"] == 0

    def test_skip_create_collection(self):
        doc = ("test content", {"doc_id": "test", "filename": "test.md", "source_type": "MD"})
        with patch("src.ingestion.pipeline.load_directory", return_value=[doc]), \
             patch("src.ingestion.pipeline.chunk_document", return_value=[{
                 "chunk_id": "test_p_001", "doc_id": "test", "content": "test",
                 "parent_chunk_id": "", "char_count": 4, "is_parent": True,
             }]), \
             patch("src.ingestion.pipeline.Embedder") as MockEmbedder, \
             patch("src.ingestion.indexer.insert_chunks") as mock_insert:
            mock_embedder = MagicMock()
            mock_embedder.encode.return_value = [0.1] * 384
            MockEmbedder.return_value = mock_embedder
            mock_insert.return_value = {"insert_count": 1}
            from src.ingestion.pipeline import run_ingestion
            result = run_ingestion("test_dir", create_collection=False)
            assert result["documents_found"] == 1
            assert result["documents_processed"] == 1
            assert result["chunks_total"] == 1
            assert result["vectors_inserted"] == 1

    def test_create_collection_called(self):
        doc = ("content", {"doc_id": "doc1", "filename": "doc1.md", "source_type": "MD"})
        with patch("src.ingestion.pipeline.load_directory", return_value=[doc]), \
             patch("src.ingestion.indexer.create_collection") as mock_create, \
             patch("src.ingestion.indexer.insert_chunks") as mock_insert, \
             patch("src.ingestion.pipeline.chunk_document", return_value=[{
                 "chunk_id": "doc1_p_001", "doc_id": "doc1", "content": "x",
                 "parent_chunk_id": "", "char_count": 1, "is_parent": True,
             }]), \
             patch("src.ingestion.pipeline.Embedder") as MockEmbedder:
            mock_embedder = MagicMock()
            mock_embedder.encode.return_value = [0.1] * 384
            MockEmbedder.return_value = mock_embedder
            mock_insert.return_value = {"insert_count": 1}
            from src.ingestion.pipeline import run_ingestion
            result = run_ingestion("test_dir", create_collection=True, drop_existing=True)
            mock_create.assert_called_once()

    def test_document_processing_error_skipped(self):
        doc = ("content", {"doc_id": "broken", "filename": "broken.md", "source_type": "MD"})
        with patch("src.ingestion.pipeline.load_directory", return_value=[doc]), \
             patch("src.ingestion.pipeline.chunk_document", side_effect=ValueError("parse error")):
            from src.ingestion.pipeline import run_ingestion
            result = run_ingestion("test_dir", create_collection=False)
            assert result["documents_found"] == 1
            assert result["documents_processed"] == 0
            assert len(result["errors"]) == 1
