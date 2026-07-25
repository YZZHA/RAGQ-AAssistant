import pytest
from unittest.mock import patch, MagicMock


class TestIndexerSchema:
    def test_build_schema_returns_valid_schema(self):
        with patch("pymilvus.FieldSchema") as MockField, \
             patch("pymilvus.CollectionSchema") as MockSchema:
            MockField.side_effect = lambda **kw: MagicMock(**kw)
            MockSchema.return_value = "test_schema"
            from src.ingestion.indexer import _build_schema
            schema = _build_schema()
            assert schema == "test_schema"

    def test_dense_vector_dim_is_384_in_schema_call(self):
        call_args = {}

        def track_field(**kw):
            if kw.get("name") == "dense_vector":
                call_args.update(kw)
            return MagicMock(**kw)

        with patch("pymilvus.FieldSchema", side_effect=track_field), \
             patch("pymilvus.CollectionSchema"):
            from src.ingestion.indexer import _build_schema
            _build_schema()
            assert call_args.get("dim") == 384


class TestIndexerMocked:
    def test_create_collection_mocked(self):
        with patch("src.ingestion.indexer._connect") as mock_connect, \
             patch("pymilvus.Collection") as MockCollection, \
             patch("pymilvus.utility") as mock_util:
            mock_util.has_collection.return_value = False
            mock_instance = MagicMock()
            MockCollection.return_value = mock_instance
            from src.ingestion.indexer import create_collection
            name = create_collection("test_collection", drop_if_exists=False)
            assert name == "test_collection"
            mock_connect.assert_called_once()

    def test_insert_chunks_mocked(self):
        chunks = [{
            "chunk_id": "test_p_001",
            "doc_id": "test",
            "content": "test",
            "dense_vector": [0.1] * 384,
        }]
        with patch("src.ingestion.indexer._connect") as mock_connect, \
             patch("pymilvus.Collection") as MockCollection:
            mock_instance = MagicMock()
            MockCollection.return_value = mock_instance
            from src.ingestion.indexer import insert_chunks
            result = insert_chunks(chunks, "test_collection")
            assert result["insert_count"] == 1
            mock_connect.assert_called_once()

    def test_insert_empty_chunks_mocked(self):
        with patch("src.ingestion.indexer._connect"), \
             patch("pymilvus.Collection") as MockCollection:
            mock_instance = MagicMock()
            MockCollection.return_value = mock_instance
            from src.ingestion.indexer import insert_chunks
            result = insert_chunks([], "test_collection")
            assert result["insert_count"] == 0
