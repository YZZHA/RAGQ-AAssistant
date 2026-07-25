# retrieval/ — 多路召回层
BM25 稀疏检索 + Embedding 稠密检索 + 结果融合。
依赖: core/config, memory/long_term

## 文件清单
| 文件 | 地位 | 功能 |
|------|------|------|
| bm25.py | 核心 | BM25 稀疏向量检索（Milvus 内置） |
| embedding.py | 核心 | 稠密向量相似度检索 |
| fusion.py | 核心 | RRF / 加权结果融合 |
| rewrite.py | 核心 | Query Rewrite（LLM + 规则） |
