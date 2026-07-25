# input:  system prompt + user question + context chunks, model name
# output: LLM response text (OpenAI / Qwen)
# pos:    生成层 → LLM 客户端封装，统一接口屏蔽厂商差异

from typing import Optional, List, Dict

from openai import OpenAI

from src.core.config import settings
from src.core.logging import logger


BACKEND_QWEN = "qwen"
BACKEND_OPENAI = "openai"

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-max"

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"


class LLM:
    def __init__(self, backend: str = BACKEND_QWEN):
        self.backend = backend

    @property
    def _client(self) -> OpenAI:
        if self.backend == BACKEND_OPENAI:
            return OpenAI(api_key=settings.openai_api_key, base_url=OPENAI_BASE_URL)
        return OpenAI(api_key=settings.qwen_api_key, base_url=QWEN_BASE_URL)

    @property
    def _model(self) -> str:
        return OPENAI_MODEL if self.backend == BACKEND_OPENAI else QWEN_MODEL

    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        api_key = settings.qwen_api_key if self.backend == BACKEND_QWEN else settings.openai_api_key
        if not api_key:
            raise ValueError(f"{self.backend} API Key 未设置")

        client = self._client
        resp = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def generate_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        api_key = settings.qwen_api_key if self.backend == BACKEND_QWEN else settings.openai_api_key
        if not api_key:
            raise ValueError(f"{self.backend} API Key 未设置")

        client = self._client
        stream = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
