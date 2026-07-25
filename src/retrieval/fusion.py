# input:  BM25 results + Embedding results
# output: merged & deduplicated chunk list (RRF or weighted fusion)
# pos:    召回层 → 多路结果融合，统一排序后送精排

from typing import List, Dict


def rrf_fusion(
    bm25_results: List[Dict],
    embedding_results: List[Dict],
    k: int = 60,
    top_k: int = 10,
) -> List[Dict]:
    if not bm25_results and not embedding_results:
        return []

    score_map: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for rank, doc in enumerate(bm25_results):
        cid = doc["chunk_id"]
        score_map[cid] = score_map.get(cid, 0) + 1.0 / (k + rank + 1)
        doc_map[cid] = doc

    for rank, doc in enumerate(embedding_results):
        cid = doc["chunk_id"]
        score_map[cid] = score_map.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in doc_map:
            doc_map[cid] = doc

    sorted_docs = sorted(score_map.items(), key=lambda x: -x[1])
    results = []
    for cid, score in sorted_docs[:top_k]:
        doc = dict(doc_map[cid])
        doc["rrf_score"] = round(score, 4)
        doc["method"] = "rrf"
        results.append(doc)

    return results


def weighted_fusion(
    bm25_results: List[Dict],
    embedding_results: List[Dict],
    bm25_weight: float = 0.3,
    embedding_weight: float = 0.7,
    top_k: int = 10,
) -> List[Dict]:
    if not bm25_results and not embedding_results:
        return []

    score_map: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    max_bm25 = max((d.get("score", 0) for d in bm25_results), default=1)
    max_emb = max((d.get("score", 0) for d in embedding_results), default=1)

    for doc in bm25_results:
        cid = doc["chunk_id"]
        score_map[cid] = score_map.get(cid, 0) + bm25_weight * doc.get("score", 0) / max_bm25
        doc_map[cid] = doc

    for doc in embedding_results:
        cid = doc["chunk_id"]
        score_map[cid] = score_map.get(cid, 0) + embedding_weight * doc.get("score", 0) / max_emb
        if cid not in doc_map:
            doc_map[cid] = doc

    sorted_docs = sorted(score_map.items(), key=lambda x: -x[1])
    results = []
    for cid, score in sorted_docs[:top_k]:
        doc = dict(doc_map[cid])
        doc["fusion_score"] = round(score, 4)
        doc["method"] = "weighted"
        results.append(doc)

    return results
