# input:  query vector, Milvus collection, tenant_id
# output: relevant document chunks from knowledge base
# pos:    记忆层 → 长时记忆，持久化知识库向量检索

from typing import List, Dict, Optional

from src.core.milvus_helper import get_milvus_uri
from src.ingestion.indexer import COLLECTION_NAME


def search(
    query_vector: List[float],
    tenant_id: str = "default",
    top_k: int = 10,
    collection_name: str = COLLECTION_NAME,
) -> List[Dict]:
    from pymilvus import connections, Collection, utility

    try:
        uri = get_milvus_uri()
        connections.connect(alias="milvus", uri=uri, timeout=5)
        if not utility.has_collection(collection_name, using="milvus"):
            connections.disconnect("milvus")
            return []
    except Exception:
        return []

    try:
        collection = Collection(name=collection_name, using="milvus")
        collection.load()

        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        expr = f'tenant_id == "{tenant_id}"'

        results = collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["chunk_id", "doc_id", "doc_title", "content", "parent_chunk_id"],
        )

        hits = []
        if results:
            for hit in results[0]:
                hits.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "doc_id": hit.entity.get("doc_id"),
                    "doc_title": hit.entity.get("doc_title", ""),
                    "content": hit.entity.get("content", ""),
                    "parent_chunk_id": hit.entity.get("parent_chunk_id", ""),
                    "score": hit.score,
                })
        return hits
    finally:
        try:
            connections.disconnect("milvus")
        except Exception:
            pass


def get_parent_chunk(
    parent_chunk_id: str,
    collection_name: str = COLLECTION_NAME,
) -> Optional[Dict]:
    from pymilvus import connections, Collection, utility

    try:
        uri = get_milvus_uri()
        connections.connect(alias="milvus", uri=uri, timeout=5)

        if not utility.has_collection(collection_name, using="milvus"):
            connections.disconnect("milvus")
            return None

        collection = Collection(name=collection_name, using="milvus")
        collection.load()

        results = collection.query(
            expr=f'chunk_id == "{parent_chunk_id}"',
            output_fields=["chunk_id", "doc_id", "doc_title", "content", "parent_chunk_id"],
        )

        if results:
            return dict(results[0])
        return None
    except Exception:
        return None
    finally:
        try:
            connections.disconnect("milvus")
        except Exception:
            pass
