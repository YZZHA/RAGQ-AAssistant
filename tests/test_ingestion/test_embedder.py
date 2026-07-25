import pytest
from src.ingestion.embedder import Embedder


class TestEmbedderInit:
    def test_default_model_from_settings(self):
        e = Embedder()
        assert e.model_name == "paraphrase-multilingual-MiniLM-L12-v2"


class TestEmbedderDim:
    def test_dimension_is_384(self):
        e = Embedder()
        assert e.dim == 384


class TestEmbedderEncode:
    def test_encode_returns_list_of_float(self):
        e = Embedder()
        vec = e.encode("测试文本")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)
        assert len(vec) == 384

    def test_encode_normalized(self):
        e = Embedder()
        vec = e.encode("测试文本")
        magnitude = sum(v * v for v in vec) ** 0.5
        assert abs(magnitude - 1.0) < 0.01

    def test_encode_empty_string(self):
        e = Embedder()
        vec = e.encode("")
        assert isinstance(vec, list)
        assert len(vec) == 384


class TestEmbedderBatch:
    def test_encode_batch_returns_list(self):
        e = Embedder()
        texts = ["文本A", "文本B", "文本C"]
        vecs = e.encode_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_similar_texts_have_similar_embeddings(self):
        e = Embedder()
        v1 = e.encode("跨境投资登记")
        v2 = e.encode("跨境投资登记系统")
        v3 = e.encode("天气很好")
        import math
        sim12 = sum(a * b for a, b in zip(v1, v2))
        sim13 = sum(a * b for a, b in zip(v1, v3))
        assert sim12 > sim13
