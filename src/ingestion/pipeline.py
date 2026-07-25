# input:  source directory path, embedder, indexer
# output: ingestion status report
# pos:    摄入管道 → 全链路编排入口，协调 loader→chunker→embedder→indexer

from typing import List, Dict

from src.core.logging import logger
from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import Embedder


def run_ingestion(
    source_dir: str,
    collection_name: str = "fdi_knowledge_base",
    create_collection: bool = True,
    drop_existing: bool = False,
) -> Dict:
    result = {
        "source_dir": source_dir,
        "documents_found": 0,
        "documents_processed": 0,
        "chunks_total": 0,
        "vectors_inserted": 0,
        "errors": [],
    }

    docs = load_directory(source_dir)
    result["documents_found"] = len(docs)

    if not docs:
        logger.warning("未找到任何文档: %s", source_dir)
        return result

    if create_collection:
        from src.ingestion.indexer import create_collection as _create_collection
        _create_collection(collection_name, drop_if_exists=drop_existing)
        logger.info("Milvus collection 已准备: %s", collection_name)

    embedder = Embedder()

    all_entities = []
    for text, meta in docs:
        doc_id = meta["doc_id"]
        try:
            chunks = chunk_document(text, doc_id)
            if not chunks:
                continue

            for c in chunks:
                vec = embedder.encode(c["content"])
                c["dense_vector"] = vec
                c["doc_title"] = meta.get("filename", doc_id)
                c["source_type"] = meta.get("source_type", "MD")
                c["tenant_id"] = "default"
                c["updated_at"] = 0
                c["version"] = "1.0"

            all_entities.extend(chunks)
            result["documents_processed"] += 1
            result["chunks_total"] += len(chunks)
            logger.info("  ✓ %s → %d chunks (%d chars)", doc_id, len(chunks), len(text))
        except Exception as e:
            msg = f"{doc_id}: {e}"
            result["errors"].append(msg)
            logger.error("  ✗ %s", msg)

    if all_entities:
        from src.ingestion.indexer import insert_chunks
        insert_result = insert_chunks(all_entities, collection_name)
        result["vectors_inserted"] = insert_result["insert_count"]
    else:
        result["vectors_inserted"] = 0

    logger.info("摄入完成: %d 文档 → %d chunks → %d 向量已写入",
                result["documents_processed"], result["chunks_total"], result["vectors_inserted"])
    return result
