import pytest
from src.core.config import settings


class TestSettingsBasic:
    def test_milvus_defaults(self):
        assert settings.milvus_host == "localhost"
        assert settings.milvus_port == 19530

    def test_redis_defaults(self):
        assert settings.redis_url == "redis://localhost:6379"

    def test_embedding_model(self):
        assert settings.embedding_model == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_llm_keys_loaded_from_env(self):
        assert settings.openai_api_key == ""

    def test_qwen_key_loaded_from_env(self):
        assert settings.qwen_api_key != ""


class TestSettingsTypes:
    def test_milvus_port_is_int(self):
        assert isinstance(settings.milvus_port, int)

    def test_langfuse_host_is_str(self):
        assert isinstance(settings.langfuse_host, str)


class TestSettingsEnvOverride:
    def test_can_override_via_env(self, monkeypatch):
        monkeypatch.setenv("MILVUS_HOST", "10.0.0.1")
        monkeypatch.setenv("EMBEDDING_MODEL", "text2vec-large-chinese")
        from pydantic_settings import BaseSettings

        class TempSettings(BaseSettings):
            milvus_host: str = "localhost"
            embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
            model_config = {"extra": "ignore"}

        s = TempSettings()
        assert s.milvus_host == "10.0.0.1"
        assert s.embedding_model == "text2vec-large-chinese"
