import pytest
from src.decompose.rules import detect


class TestDetectKeywords:
    def test_对比_triggers(self):
        r = detect("A产品和B产品的区别")
        assert r["should_decompose"] is True
        assert "区别" in r["keywords"]

    def test_multi_entity_triggers(self):
        r = detect("A产品、B产品和C产品的价格对比")
        assert r["should_decompose"] is True
        assert r["entity_count"] >= 2

    def test_单实体_no_decompose(self):
        r = detect("跨境投资登记系统怎么注册")
        assert r["should_decompose"] is False


class TestDetectOutput:
    def test_returns_expected_keys(self):
        r = detect("测试")
        assert "should_decompose" in r
        assert "reason" in r
        assert "keywords" in r
        assert "entities" in r
        assert "entity_count" in r

    def test_empty_query(self):
        r = detect("")
        assert r["should_decompose"] is False
        assert r["entity_count"] == 0
