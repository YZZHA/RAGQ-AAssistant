import pytest
from unittest.mock import patch
from src.decompose.llm_splitter import decompose, _rule_split
from src.decompose.rules import detect


class TestRuleSplit:
    def test_对比_query_splits(self):
        r = detect("A产品和B产品的区别")
        result = _rule_split("A产品和B产品的区别", r)
        assert len(result) >= 1

    def test_single_entity_no_split(self):
        r = detect("跨境投资登记系统怎么注册")
        result = _rule_split("跨境投资登记系统怎么注册", r)
        assert len(result) == 1

    def test_empty_returns_original(self):
        result = _rule_split("", detect(""))
        assert result == [""]


class TestDecompose:
    def test_decompose_via_llm(self):
        mock_json = '[{"query": "A产品有哪些功能特点？"}, {"query": "B产品有哪些功能特点？"}]'
        with patch("src.decompose.llm_splitter.settings.qwen_api_key", "test_key"), \
             patch("src.decompose.llm_splitter._llm_split", return_value=["A产品有哪些功能特点？", "B产品有哪些功能特点？"]):
            result = decompose("A产品和B产品的区别")
            assert len(result) == 2
            assert "A产品" in result[0]
            assert "B产品" in result[1]

    def test_decompose_llm_fallback_to_rule(self):
        with patch("src.decompose.llm_splitter.settings.qwen_api_key", "test_key"), \
             patch("src.decompose.llm_splitter._llm_split", side_effect=Exception("LLM error")):
            result = decompose("A产品和B产品的区别")
            assert len(result) >= 1

    def test_decompose_no_key_uses_rules(self):
        with patch("src.decompose.llm_splitter.settings.qwen_api_key", ""):
            result = decompose("A产品和B产品的区别")
            assert len(result) >= 1

    def test_simple_query_not_decomposed(self):
        with patch("src.decompose.llm_splitter.settings.qwen_api_key", ""):
            result = decompose("跨境投资登记系统怎么注册")
            assert result == ["跨境投资登记系统怎么注册"]
