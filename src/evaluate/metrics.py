# input:  ranked retrieval results + ground truth relevant doc IDs
# output: metric values (Recall@K, MRR, NDCG)
# pos:    评估层 → 核心指标计算函数

from typing import List, Set, Dict


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return hits / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k, 1):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += (2 ** rel - 1) / (i.bit_length())  # log2(i+1) approximated

    ideal_count = min(k, len(relevant))
    idcg = sum(1.0 / (i.bit_length()) for i in range(1, ideal_count + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    results: List[Dict],
    relevant_doc_ids: Set[str],
    ks: List[int] = None,
) -> Dict:
    if ks is None:
        ks = [1, 3, 5, 10]

    retrieved_ids = [r.get("doc_id", "") for r in results]

    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = round(recall_at_k(retrieved_ids, relevant_doc_ids, k), 4)

    metrics["MRR"] = round(mrr(retrieved_ids, relevant_doc_ids), 4)

    for k in ks:
        metrics[f"NDCG@{k}"] = round(ndcg_at_k(retrieved_ids, relevant_doc_ids, k), 4)

    return metrics
