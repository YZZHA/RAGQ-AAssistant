# input:  (none — package marker)
# output: knowledge module init
# pos:    知识生产模块 → 自动知识生产包导入

from src.knowledge.gap_log import GapLog
from src.knowledge.gap_detector import GapDetector
from src.knowledge.draft import generate_draft
from src.knowledge.validator import validate
from src.knowledge.auto_ingest import auto_ingest
from src.knowledge.llm_chunker import llm_chunk_document, rebuild_all_cache

__all__ = [
    "GapLog", "GapDetector", "generate_draft", "validate", "auto_ingest",
    "llm_chunk_document", "rebuild_all_cache",
]
