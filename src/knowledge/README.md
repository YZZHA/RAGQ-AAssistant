# knowledge/ — 知识生产模块
缺口检测 → 草稿生成 → LLM 校验 → 自动入库，支撑知识库持续生长。
依赖: core/config, ingestion, retrieval

## 文件清单
| 文件 | 地位 | 功能 |
|------|------|------|
| gap_log.py | 核心 | 缺口日志 JSONL 读写 |
| gap_detector.py | 核心 | 检索分数阈值 + 用户反馈双触发 |
| draft.py | 核心 | LLM 将缺口 Q&A 标准化 |
| validator.py | 核心 | LLM 自检，防幻觉、防冲突 |
| auto_ingest.py | 核心 | chunk → embed → 重建 BM25 + FAISS |
| llm_chunker.py | 核心 | LLM 语义切分 + MD5 缓存 + 降级到正则 |
