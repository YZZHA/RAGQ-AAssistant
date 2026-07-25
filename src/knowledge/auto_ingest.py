# input:  validated Q&A, BM25 instance, vector_store
# output: ingested Q&A persisted + indexes updated
# pos:    知识生产 → 将校验通过的 Q&A 摄入知识库，增量更新索引

import json
import time
from pathlib import Path
from typing import Dict

from src.core.logging import logger


INGESTED_LOG = Path("data/knowledge_gaps/ingested.jsonl")


def auto_ingest(
    qa: Dict,
    bm25,
    vector_store,
    embedder,
) -> bool:
    doc_text = _qa_to_document(qa)
    doc_id = f"auto_{int(time.time())}"

    from src.ingestion.chunker import chunk_document
    chunks = chunk_document(doc_text, doc_id)

    for c in chunks:
        c["doc_title"] = f"[Auto] {qa.get('question', '')[:40]}"
        c["source_type"] = "AUTO"
        c["tenant_id"] = "default"

    import numpy as np
    for c in chunks:
        vec = embedder.encode(c["content"])
        c["dense_vector"] = vec

    _rebuild_indexes(bm25, vector_store, chunks, embedder)

    _log_ingested(qa, doc_id, len(chunks))
    return True


def _qa_to_document(qa: Dict) -> str:
    question = qa.get("question", "")
    answer = qa.get("answer", "")
    return f"# 自动生成知识\n\n## {question}\n\n{answer}"


def _rebuild_indexes(bm25, vector_store, new_chunks, embedder):
    old_chunks = getattr(bm25, "_docs", [])
    all_chunks = old_chunks + new_chunks
    bm25.build(all_chunks)
    logger.info("BM25 重建完成: %d chunks", len(all_chunks))

    vector_store.build(all_chunks, embedder)
    logger.info("FAISS 重建完成: %d chunks", len(all_chunks))


def _log_ingested(qa: Dict, doc_id: str, chunk_count: int):
    INGESTED_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "doc_id": doc_id,
        "question": qa.get("question", ""),
        "answer": qa.get("answer", ""),
        "chunk_count": chunk_count,
        "created_at": time.time(),
    }
    with INGESTED_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("已入库: %s (%d chunks)", doc_id, chunk_count)
