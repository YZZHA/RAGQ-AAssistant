# input:  retrieval chunks (scores) + user feedback signal
# output: gap record (if score < threshold or user reported)
# pos:    知识生产前置 → 判断当前问题是否超出知识库范围

from typing import List, Dict, Optional

from src.core.logging import logger
from src.knowledge.gap_log import GapLog


SCORE_THRESHOLD = 0.3


class GapDetector:
    def __init__(self):
        self.log = GapLog()

    def check(
        self,
        question: str,
        answer: str,
        chunks: List[Dict],
        user_reported: bool = False,
        session_id: str = "",
        tenant_id: str = "default",
    ) -> Optional[Dict]:
        top_score = max((c.get("rerank_score", 0) for c in chunks), default=0)

        trigger = None
        if user_reported:
            trigger = "user_feedback"
        elif not chunks:
            trigger = "empty_retrieval"
        elif top_score < SCORE_THRESHOLD:
            trigger = "score_threshold"

        if trigger is None:
            return None

        record = {
            "question": question,
            "answer": answer,
            "trigger": trigger,
            "top_score": round(top_score, 4),
            "session_id": session_id,
            "tenant_id": tenant_id,
            "chunk_count": len(chunks),
        }

        self.log.append(record)
        logger.info("知识缺口已记录 [%s]: '%s' (top_score=%.2f)", trigger, question[:50], top_score)
        return record
