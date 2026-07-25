# input:  environment variables (.env file)
# output: global Settings singleton
# pos:    核心层 → 所有模块读取统一配置入口

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Milvus ---
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_lite_path: str = "data/milvus_lite"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379"
    redis_use_fake: bool = True

    # --- LLM ---
    openai_api_key: str = ""
    qwen_api_key: str = ""

    # --- Embedding ---
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # --- LangFuse ---
    langfuse_enabled: bool = False
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
