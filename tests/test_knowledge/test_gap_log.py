import json
from pathlib import Path
from src.knowledge.gap_log import GapLog


class TestGapLog:
    def test_append_and_read(self, tmp_path):
        p = tmp_path / "gaps.jsonl"
        log = GapLog(str(p))
        log.append({"question": "test?", "trigger": "score_threshold", "top_score": 0.1})
        records = log.read_all()
        assert len(records) == 1
        assert records[0]["question"] == "test?"

    def test_read_empty(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        log = GapLog(str(p))
        assert log.read_all() == []
        assert log.count() == 0

    def test_clear(self, tmp_path):
        p = tmp_path / "gaps.jsonl"
        log = GapLog(str(p))
        log.append({"question": "q"})
        log.clear()
        assert not p.exists()

    def test_append_adds_timestamp(self, tmp_path):
        p = tmp_path / "gaps.jsonl"
        log = GapLog(str(p))
        log.append({"question": "q"})
        r = log.read_all()[0]
        assert "created_at" in r
