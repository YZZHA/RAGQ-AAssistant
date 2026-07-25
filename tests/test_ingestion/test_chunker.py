import pytest
from unittest.mock import patch
from src.ingestion.chunker import chunk_document


# 所有 chunker 测试都模拟 LLM 切分失败 → 降级到正则
@pytest.fixture(autouse=True)
def _mock_llm_chunker():
    with patch("src.knowledge.llm_chunker.llm_chunk_document", return_value=None):
        yield


SAMPLE_TEXT = """# 测试文档

## 第一章

这是第一章的内容。包含一些测试文本。

## 第二章

这是第二章的内容。

### 2.1 小节

第二章节的第一部分。

### 2.2 小节

第二章节的第二部分。

## 第三章

这是第三章的内容，足够长的一段文本来测试切分。重复：这是第三章的内容，足够长的一段文本来测试切分。这是第三章的内容，足够长的一段文本来测试切分。这是第三章的内容，足够长的一段文本来测试切分。这是第三章的内容，足够长的一段文本来测试切分。
"""


class TestChunkBasic:
    def test_chunk_returns_list(self):
        result = chunk_document(SAMPLE_TEXT, "test_doc", use_llm=False)
        assert isinstance(result, list)

    def test_chunk_has_content(self):
        result = chunk_document(SAMPLE_TEXT, "test_doc")
        assert len(result) > 0
        for c in result:
            assert len(c["content"]) > 0

    def test_empty_text_returns_empty(self):
        assert chunk_document("", "test_doc") == []


class TestChunkStructure:
    def test_each_chunk_has_required_keys(self):
        result = chunk_document(SAMPLE_TEXT, "test_doc")
        required = {"chunk_id", "doc_id", "parent_chunk_id", "content", "char_count", "is_parent"}
        for c in result:
            assert required.issubset(c.keys())

    def test_chunk_id_format_parent(self):
        result = chunk_document(SAMPLE_TEXT, "test_doc")
        parents = [c for c in result if c["is_parent"]]
        for c in parents:
            assert c["chunk_id"].startswith("test_doc_p_")
            assert c["parent_chunk_id"] == ""

    def test_chunk_id_format_child(self):
        result = chunk_document(SAMPLE_TEXT, "test_doc")
        children = [c for c in result if not c["is_parent"]]
        for c in children:
            assert c["chunk_id"].startswith("test_doc_c_")
            assert c["parent_chunk_id"] != ""


class TestChunkLimit:
    def test_parent_chunk_under_max(self):
        result = chunk_document(SAMPLE_TEXT, "test_doc", parent_max=1200)
        parents = [c for c in result if c["is_parent"]]
        for c in parents:
            assert c["char_count"] <= 1200

    def test_child_chunk_under_max(self):
        result = chunk_document(SAMPLE_TEXT, "test_doc", child_max=400)
        children = [c for c in result if not c["is_parent"]]
        for c in children:
            assert c["char_count"] <= 400


class TestChunkRelationship:
    def test_child_points_to_parent(self):
        result = chunk_document(SAMPLE_TEXT, "test_doc", child_max=200)
        children = [c for c in result if not c["is_parent"]]
        parents = {c["chunk_id"] for c in result if c["is_parent"]}
        for c in children:
            assert c["parent_chunk_id"] in parents

    def test_parents_always_have_empty_parent(self):
        result = chunk_document(SAMPLE_TEXT, "test_doc")
        for c in result:
            if c["is_parent"]:
                assert c["parent_chunk_id"] == ""


def test_real_document():
    from pathlib import Path
    path = Path("data/raw/cross_border_registration.md")
    text = path.read_text(encoding="utf-8")
    result = chunk_document(text, "cross_border_registration")
    assert len(result) >= 3
    parents = [c for c in result if c["is_parent"]]
    assert any("项目登记" in c["content"] for c in parents)


# === Step 1: 标题层级 + heading_chain ===

H1_H3_TEXT = """# 系统概览

内容一

## 核心功能

内容二

### 项目登记

内容三

### 材料上传

内容四

## 操作流程

内容五"""


class TestHeadingChain:
    def test_chunk_has_heading_chain_key(self):
        result = chunk_document(H1_H3_TEXT, "test_doc")
        for c in result:
            assert "heading_chain" in c

    def test_h1_becomes_root(self):
        result = chunk_document(H1_H3_TEXT, "test_doc")
        assert result[0]["heading_chain"] == ["系统概览"]

    def test_h2_under_h1(self):
        result = chunk_document(H1_H3_TEXT, "test_doc")
        h2 = [c for c in result if c["heading_chain"] == ["系统概览", "核心功能"]]
        assert len(h2) >= 1

    def test_h3_under_h2(self):
        result = chunk_document(H1_H3_TEXT, "test_doc")
        h3 = [c for c in result if c["heading_chain"] == ["系统概览", "核心功能", "项目登记"]]
        assert len(h3) >= 1

    def test_no_headings_has_empty_chain(self):
        result = chunk_document("纯文本段落\n\n没有标题", "doc")
        for c in result:
            assert c["heading_chain"] == []

    def test_multiple_same_level_headings(self):
        result = chunk_document(H1_H3_TEXT, "test_doc")
        h2s = [c for c in result if len(c["heading_chain"]) == 2]
        assert len(h2s) >= 2  # "核心功能" 和 "操作流程"


# === Step 2: 过短合并 + 过长拆分 ===

class TestMergeShort:
    def test_short_child_merged_to_previous(self):
        text = "## 标题\n\nAAAA BBBB\n\nCC"
        result = chunk_document(text, "doc", child_min=10)
        for c in result:
            assert c["char_count"] >= 10 or c["is_parent"]


class TestSplitLong:
    def test_long_split_by_paragraph(self):
        text = "## 标题\n\n" + "A" * 600 + "\n\n" + "B" * 600
        result = chunk_document(text, "doc", parent_max=800)
        assert len(result) >= 2

    def test_long_split_by_comma(self):
        text = "## 标题\n\n" + ",".join(["A" * 300 for _ in range(5)])
        result = chunk_document(text, "doc", parent_max=800)
        assert len(result) >= 2


# === Step 2: Overlap ===

class TestOverlap:
    def test_overlap_added_between_chunks(self):
        text = "## A\n\n" + "X" * 200 + "\n\n## B\n\n" + "Y" * 200
        result = chunk_document(text, "doc", overlap=30)
        if len(result) >= 2:
            assert "X" in result[1]["content"][:40]
