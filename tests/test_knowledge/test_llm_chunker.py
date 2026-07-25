import json
from unittest.mock import patch
from src.knowledge.llm_chunker import llm_chunk_document, _parse_json_to_chunks


class TestParseJson:
    def test_plain_json(self):
        raw = '[{"heading_chain": ["标题"], "content": "内容"}]'
        result = _parse_json_to_chunks(raw, "doc")
        assert result is not None
        assert result[0]["content"] == "内容"

    def test_code_block_json(self):
        raw = '```json\n[{"heading_chain": [], "content": "内容"}]\n```'
        result = _parse_json_to_chunks(raw, "doc")
        assert result is not None

    def test_invalid_returns_none(self):
        assert _parse_json_to_chunks("not json", "doc") is None

    def test_missing_content_returns_none(self):
        assert _parse_json_to_chunks('[{"heading_chain": []}]', "doc") is None


class TestLlmChunk:
    def test_cache_hit(self):
        with patch("src.knowledge.llm_chunker._read_cache") as mock_read:
            mock_read.return_value = [
                {"chunk_id": "doc_llm_001", "content": "cached", "heading_chain": []}
            ]
            result = llm_chunk_document("test", "doc")
            assert result is not None
            assert result[0]["content"] == "cached"

    def test_cache_miss_calls_llm(self):
        """当缓存未命中且 API Key 存在时，调用 LLM"""
        with patch("src.knowledge.llm_chunker._read_cache", return_value=None), \
             patch("src.knowledge.llm_chunker.settings.qwen_api_key", "test_key"), \
             patch("src.knowledge.llm_chunker.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_resp = mock_client.chat.completions.create.return_value
            mock_resp.choices[0].message.content = '[{"heading_chain": ["H1"], "content": "LLM chunk"}]'

            result = llm_chunk_document("文档内容", "doc", force_refresh=True)
            assert result is not None
            assert result[0]["content"] == "LLM chunk"
            assert result[0]["chunk_id"].startswith("doc_llm_")

    def test_no_key_returns_none(self):
        with patch("src.knowledge.llm_chunker.settings.qwen_api_key", ""):
            result = llm_chunk_document("test", "doc")
            assert result is None

    def test_llm_failure_returns_none(self):
        with patch("src.knowledge.llm_chunker._read_cache", return_value=None), \
             patch("src.knowledge.llm_chunker.settings.qwen_api_key", "test_key"), \
             patch("src.knowledge.llm_chunker.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create.side_effect = Exception("API error")
            result = llm_chunk_document("test", "doc")
            assert result is None
