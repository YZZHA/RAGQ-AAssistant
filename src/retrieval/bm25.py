# input:  query text, chunk corpus, jieba tokenizer
# output: ranked list of relevant chunks (sparse BM25 scoring)
# pos:    召回层 → 稀疏检索，关键词精确匹配专有名词/产品型号

import math
from collections import Counter
from typing import List, Dict, Optional

import jieba


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[Dict] = []
        self._doc_count = 0
        self._avg_dl = 0.0
        self._term_doc_freq: Dict[str, int] = {}
        self._doc_terms: List[Counter] = []

    def build(self, chunks: List[Dict]):
        self._docs = chunks
        self._doc_count = len(chunks)
        total_len = 0
        term_docs: Dict[str, set] = {}

        self._doc_terms = []
        for chunk in chunks:
            text = chunk.get("content", "")
            tokens = list(jieba.cut(text))
            token_set = set(tokens)
            counter = Counter(tokens)

            self._doc_terms.append(counter)
            total_len += len(tokens)

            for token in token_set:
                if token.strip():
                    term_docs.setdefault(token, set()).add(len(self._doc_terms) - 1)

        self._avg_dl = total_len / self._doc_count if self._doc_count else 0
        self._term_doc_freq = {t: len(docs) for t, docs in term_docs.items()}

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self._docs:
            return []

        query_tokens = [t for t in jieba.cut(query) if t.strip()]
        scores = [0.0] * self._doc_count

        for token in set(query_tokens):
            if token not in self._term_doc_freq:
                continue
            df = self._term_doc_freq[token]
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)

            for doc_idx in range(self._doc_count):
                tf = self._doc_terms[doc_idx].get(token, 0)
                if tf == 0:
                    continue
                dl = sum(self._doc_terms[doc_idx].values())
                score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl))
                scores[doc_idx] += score

        scored = [(scores[i], i) for i in range(self._doc_count)]
        scored.sort(key=lambda x: -x[0])

        results = []
        for score, idx in scored[:top_k]:
            if score > 0:
                doc = dict(self._docs[idx])
                doc["score"] = round(score, 4)
                doc["method"] = "bm25"
                results.append(doc)

        return results
