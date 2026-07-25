from src.knowledge.gap_detector import GapDetector


class TestGapDetector:
    def test_high_score_no_gap(self):
        d = GapDetector()
        chunks = [{"rerank_score": 0.85}, {"rerank_score": 0.72}]
        result = d.check("测试问题", "回答", chunks)
        assert result is None

    def test_low_score_triggers_gap(self):
        d = GapDetector()
        chunks = [{"rerank_score": 0.12}]
        result = d.check("测试问题", "回答", chunks)
        assert result is not None
        assert result["trigger"] == "score_threshold"
        assert result["top_score"] == 0.12

    def test_user_reported_triggers_gap(self):
        d = GapDetector()
        result = d.check("测试问题", "回答", chunks=[], user_reported=True)
        assert result is not None
        assert result["trigger"] == "user_feedback"

    def test_empty_chunks_with_high_score(self):
        d = GapDetector()
        result = d.check("测试", "回答", chunks=[])
        assert result is not None
        assert result["trigger"] == "empty_retrieval"
