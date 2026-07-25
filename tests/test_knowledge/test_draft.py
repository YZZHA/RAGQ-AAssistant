import pytest
from unittest.mock import patch
from src.knowledge.draft import generate_draft, _parse_json


class TestParseJson:
    def test_plain_json(self):
        result = _parse_json('{"question": "Q?", "answer": "A."}')
        assert result["question"] == "Q?"

    def test_json_in_code_block(self):
        text = '```json\n{"question": "Q?", "answer": "A."}\n```'
        result = _parse_json(text)
        assert result["question"] == "Q?"

    def test_invalid_returns_none(self):
        assert _parse_json("not json") is None


class TestGenerateDraft:
    def test_generate_success(self):
        mock_response = '{"question": "标准问题", "answer": "标准答案"}'
        with patch("src.knowledge.draft.settings.qwen_api_key", "test_key"), \
             patch("src.knowledge.draft.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_resp = mock_client.chat.completions.create.return_value
            mock_resp.choices[0].message.content = mock_response
            result = generate_draft("用户问题", "原始回答")
            assert result is not None
            assert result["question"] == "标准问题"
            assert result["answer"] == "标准答案"

    def test_no_key_returns_none(self):
        with patch("src.knowledge.draft.settings.qwen_api_key", ""):
            result = generate_draft("问题", "回答")
            assert result is None
