import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.rewrite import _rule_fallback, rewrite


class TestRuleFallback:
    def test_remove_demonstrative(self):
        assert "产品" in _rule_fallback("这个产品")

    def test_remove_question_prefix(self):
        assert "价格" in _rule_fallback("我想问一下价格")
        assert "价格" in _rule_fallback("请问价格")

    def test_trim_whitespace(self):
        assert _rule_fallback("  测试  ") == "测试"

    def test_empty_input(self):
        assert _rule_fallback("") == "未识别的问题"


class TestRewriteLLM:
    def test_rewrite_without_history(self):
        with patch("src.retrieval.rewrite._call_llm", return_value="A产品和B产品的价格分别是多少"):
            result = rewrite("A和B多少钱")
            assert "A" in result
            assert "B" in result

    def test_rewrite_call_llm_fallback(self):
        with patch("src.retrieval.rewrite._call_llm", side_effect=Exception("API error")):
            result = rewrite("这个产品价格")
            assert result is not None

    def test_rewrite_prompt_builds_with_history(self):
        from src.retrieval.rewrite import _build_prompt
        history = [{"role": "user", "content": "A产品功能是什么？"}, {"role": "assistant", "content": "A产品用于跨境投资登记"}]
        prompt = _build_prompt("它支持哪些格式？", history)
        assert "对话历史" in prompt
        assert "A产品功能" in prompt
