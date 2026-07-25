import pytest
from src.generator.prompts import build_rag_messages


CHUNKS = [
    {"doc_title": "跨境投资登记系统", "doc_id": "doc1", "content": "跨境投资登记系统用于外商投资项目的在线登记。"},
    {"doc_title": "负面清单", "doc_id": "doc2", "content": "限制类行业需要额外审批流程。"},
]


class TestBuildRagMessages:
    def test_returns_list_of_dicts(self):
        msgs = build_rag_messages("测试问题", CHUNKS)
        assert isinstance(msgs, list)
        for m in msgs:
            assert "role" in m
            assert "content" in m

    def test_first_message_is_system(self):
        msgs = build_rag_messages("测试问题", CHUNKS)
        assert msgs[0]["role"] == "system"

    def test_last_message_is_user_with_context(self):
        msgs = build_rag_messages("测试问题", CHUNKS)
        assert msgs[-1]["role"] == "user"
        assert "参考文档" in msgs[-1]["content"]
        assert "跨境投资登记系统" in msgs[-1]["content"]
        assert "测试问题" in msgs[-1]["content"]

    def test_history_included_before_user(self):
        history = [{"role": "user", "content": "之前的提问"}]
        msgs = build_rag_messages("当前问题", CHUNKS, history)
        assert len(msgs) == 3
        assert msgs[1] == history[0]
        assert msgs[2]["role"] == "user"

    def test_no_chunks_still_works(self):
        msgs = build_rag_messages("问题", [])
        assert msgs[-1]["role"] == "user"
        assert "参考文档:" in msgs[-1]["content"]
