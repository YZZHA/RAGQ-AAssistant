# input:  candidate chunks from retrieval, query text
# output: reranked chunks with relevance scores + dynamic top-k selection
# pos:    精排层 → Cross-Encoder token 级重排序，衔接召回与生成

from typing import List, Dict, Optional


class Reranker:
    _model = None

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    def _load(self):
        if Reranker._model is None:
            from sentence_transformers import CrossEncoder
            Reranker._model = CrossEncoder(self.model_name)
        return Reranker._model

    def rerank(self, query: str, chunks: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        if not chunks:
            return []

        model = self._load()
        pairs = [(query, c["content"]) for c in chunks]
        scores = model.predict(pairs)

        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            chunk["rerank_score"] = round(float(score), 4)

        sorted_chunks = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

        if top_k is not None:
            sorted_chunks = sorted_chunks[:top_k]

        return sorted_chunks


def dynamic_top_k(
    chunks: List[Dict],
    min_k: int = 2,
    max_k: int = 5,
    gap_threshold: float = 0.5,
) -> List[Dict]:
    if not chunks:
        return []

    scores = [c.get("rerank_score", 0) for c in chunks]

    if len(scores) <= min_k:
        return chunks[:min_k]

    for i in range(1, len(scores)):
        gap = scores[i - 1] - scores[i]
        if gap > gap_threshold and i >= min_k:
            return chunks[:min_k]

    if all(s < 0.1 for s in scores[:max_k]):
        return chunks[:min_k]

    top_scores = scores[:5] if len(scores) >= 5 else scores
    if max(top_scores) - min(top_scores) < 0.3:
        return chunks[:max_k]

    return chunks[:max_k]
