# input:  chunks with content text, embedder
# output: FAISS index for fast vector similarity search
# pos:    召回层 → 本地向量索引，替代 Milvus，开发期使用

from typing import List, Dict, Optional

import numpy as np


class LocalVectorStore:
    def __init__(self):
        self._chunks: List[Dict] = []
        self._index = None
        self._is_built = False

    def build(self, chunks: List[Dict], embedder) -> int:
        texts = [c["content"] for c in chunks]
        vecs = embedder.encode_batch(texts)
        dim = len(vecs[0]) if vecs else 384

        import faiss
        self._index = faiss.IndexFlatIP(dim)
        matrix = np.array(vecs, dtype=np.float32)
        if matrix.size > 0:
            faiss.normalize_L2(matrix)
            self._index.add(matrix)

        self._chunks = chunks
        self._is_built = True
        return len(chunks)

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict]:
        if not self._is_built or self._index.ntotal == 0:
            return []

        vec = np.array([query_vector], dtype=np.float32)
        import faiss
        faiss.normalize_L2(vec)

        scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = round(float(score), 4)
            chunk["method"] = "embedding"
            results.append(chunk)

        return results

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
