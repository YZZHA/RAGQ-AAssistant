# input:  query text, FAISS vector store, embedder
# output: list of candidate chunks (from dense vector similarity)
# pos:    召回层 → 稠密向量检索，捕捉语义相似性，本地 FAISS 替代 Milvus

from typing import List, Dict, Optional

from src.ingestion.embedder import Embedder
from src.retrieval.local_store import LocalVectorStore


vector_store = LocalVectorStore()


def search_by_text(
    query: str,
    tenant_id: str = "default",
    top_k: int = 10,
    embedder: Optional[Embedder] = None,
) -> List[Dict]:
    try:
        if embedder is None:
            embedder = Embedder()

        query_vector = embedder.encode(query)
        results = vector_store.search(query_vector, top_k=top_k)

        for r in results:
            r["method"] = "embedding"

        return results
    except Exception as e:
        import logging
        logging.getLogger("rag_qa").warning("向量检索失败: %s", e)
        return []
