# input:  text string, model name (from config)
# output: dense vector (list of float, 384-dim)
# pos:    摄入管道 + 召回层 → 文本向量化，供 Milvus 索引与检索

from typing import List, Optional

from src.core.config import settings


class Embedder:
    _model = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model

    def _load_model(self):
        if Embedder._model is None:
            from sentence_transformers import SentenceTransformer
            Embedder._model = SentenceTransformer(self.model_name)
        return Embedder._model

    @property
    def dim(self) -> int:
        return self._load_model().get_embedding_dimension()

    def encode(self, text: str) -> List[float]:
        model = self._load_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        model = self._load_model()
        vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return [v.tolist() for v in vecs]
