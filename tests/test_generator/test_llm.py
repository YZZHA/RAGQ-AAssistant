import pytest
from unittest.mock import patch, MagicMock
from src.generator.llm import LLM


class TestLLMInit:
    def test_default_backend_is_qwen(self):
        llm = LLM()
        assert llm.backend == "qwen"

    def test_openai_backend(self):
        llm = LLM(backend="openai")
        assert llm.backend == "openai"


class TestLLMGenerate:
    def test_generate_returns_str(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "测试回答"
        messages = [{"role": "user", "content": "你好"}]

        with patch("src.generator.llm.settings.qwen_api_key", "test_key"), \
             patch.object(LLM, "_client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            llm = LLM()
            result = llm.generate(messages)
            assert isinstance(result, str)
            assert result == "测试回答"

    def test_generate_no_key_raises(self):
        with patch("src.generator.llm.settings.qwen_api_key", ""):
            llm = LLM()
            with pytest.raises(ValueError, match="API Key 未设置"):
                llm.generate([{"role": "user", "content": "hi"}])


class TestLLMStream:
    def test_generate_stream_yields_tokens(self):
        class MockChunk:
            def __init__(self, content):
                class Delta:
                    def __init__(self, c):
                        self.content = c
                self.choices = [type("Choice", (), {"delta": Delta(content)})()]

        mock_stream = [MockChunk("A"), MockChunk("B"), MockChunk("C")]
        with patch("src.generator.llm.settings.qwen_api_key", "test_key"), \
             patch.object(LLM, "_client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_stream
            llm = LLM()
            tokens = list(llm.generate_stream([{"role": "user", "content": "hi"}]))
            assert tokens == ["A", "B", "C"]
