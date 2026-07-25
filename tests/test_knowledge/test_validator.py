import pytest
from unittest.mock import patch
from src.knowledge.validator import validate


class TestValidate:
    def test_validate_passed(self):
        with patch("src.knowledge.validator.settings.qwen_api_key", "test_key"), \
             patch("src.knowledge.validator.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_resp = mock_client.chat.completions.create.return_value
            mock_resp.choices[0].message.content = '{"passed": true, "reason": "通过"}'
            result = validate("问题", "答案")
            assert result["passed"] is True

    def test_validate_rejected(self):
        with patch("src.knowledge.validator.settings.qwen_api_key", "test_key"), \
             patch("src.knowledge.validator.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_resp = mock_client.chat.completions.create.return_value
            mock_resp.choices[0].message.content = '{"passed": false, "reason": "存在幻觉"}'
            result = validate("问题", "答案")
            assert result["passed"] is False

    def test_no_key_default_pass(self):
        with patch("src.knowledge.validator.settings.qwen_api_key", ""):
            result = validate("问题", "答案")
            assert result["passed"] is True
