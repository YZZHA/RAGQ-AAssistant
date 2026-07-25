import pytest
import tempfile
from unittest.mock import patch


@pytest.fixture(scope="module")
def milvus_lite_setup():
    from milvus_lite import server_manager_instance
    import atexit

    data_dir = tempfile.mkdtemp(prefix="ml_test_")
    uri = server_manager_instance.start_and_get_uri(data_dir)
    atexit.register(lambda: server_manager_instance.release_server(data_dir))

    from pymilvus import connections, Collection, CollectionSchema, DataType, FieldSchema

    connections.connect(alias="_setup", uri=uri)

    schema = CollectionSchema([
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
    ], description="test")

    collection = Collection(name="test_kb", schema=schema, using="_setup")
    index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 16}}
    collection.create_index(field_name="dense_vector", index_params=index_params)

    import numpy as np
    entities = [
        [f"chunk_{i}" for i in range(5)],
        [f"doc_{i}" for i in range(5)],
        [f"Title {i}" for i in range(5)],
        [""] * 5,
        [f"这是第{i}号文档的内容" for i in range(5)],
        ["tenant_a"] * 3 + ["tenant_b"] * 2,
        [np.random.rand(384).tolist() for _ in range(5)],
    ]
    collection.insert(entities)
    import time; time.sleep(1)
    collection.load()
    connections.disconnect("_setup")

    yield uri

    server_manager_instance.release_server(data_dir)


class TestSearchWithData:
    def test_search_returns_list(self, milvus_lite_setup):
        from src.memory.long_term import search
        import numpy as np
        with patch("src.memory.long_term.get_milvus_uri", return_value=milvus_lite_setup):
            vec = np.random.rand(384).tolist()
            results = search(vec, tenant_id="tenant_a", top_k=3, collection_name="test_kb")
            assert isinstance(results, list)

    def test_search_tenant_filter_returns_only_matching(self, milvus_lite_setup):
        from src.memory.long_term import search
        import numpy as np
        with patch("src.memory.long_term.get_milvus_uri", return_value=milvus_lite_setup):
            vec = np.random.rand(384).tolist()
            results = search(vec, tenant_id="tenant_a", top_k=5, collection_name="test_kb")
            assert len(results) <= 3

    def test_search_returns_chunk_content(self, milvus_lite_setup):
        from src.memory.long_term import search
        import numpy as np
        with patch("src.memory.long_term.get_milvus_uri", return_value=milvus_lite_setup):
            vec = np.random.rand(384).tolist()
            results = search(vec, tenant_id="tenant_a", top_k=1, collection_name="test_kb")
            if results:
                assert "chunk_id" in results[0]
                assert "content" in results[0]
                assert "score" in results[0]


class TestSearchNoData:
    def test_search_non_existent_collection(self):
        from src.memory.long_term import search
        import numpy as np
        with patch("src.memory.long_term.get_milvus_uri", return_value="http://127.0.0.1:65535"):
            vec = np.random.rand(384).tolist()
            results = search(vec, tenant_id="default", collection_name="nonexistent_kb")
            assert results == []


class TestGetParentChunk:
    def test_get_parent_returns_dict(self, milvus_lite_setup):
        from src.memory.long_term import get_parent_chunk
        with patch("src.memory.long_term.get_milvus_uri", return_value=milvus_lite_setup):
            parent = get_parent_chunk("chunk_0", collection_name="test_kb")
            assert parent is not None
            assert parent["chunk_id"] == "chunk_0"
            assert parent["doc_id"] == "doc_0"

    def test_get_parent_non_existent(self):
        from src.memory.long_term import get_parent_chunk
        with patch("src.memory.long_term.get_milvus_uri", return_value="http://127.0.0.1:65535"):
            result = get_parent_chunk("nonexistent", collection_name="test_kb")
            assert result is None
