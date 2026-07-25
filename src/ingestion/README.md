# ingestion/ — 数据摄入管道
文档加载→切分→嵌入→写入 Milvus 的全链路编排。
依赖: core/config, retrieval/bm25（备选）

## 文件清单
| 文件 | 地位 | 功能 |
|------|------|------|
| loader.py | 核心 | 多格式文档解析（PDF/Word/MD/HTML） |
| chunker.py | 核心 | 递归字符/语义切分策略 |
| embedder.py | 核心 | sentence-transformers 嵌入封装 |
| indexer.py | 核心 | 写入 Milvus Collection |
| pipeline.py | 核心 | 摄入流水线编排入口 |
