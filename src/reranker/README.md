# reranker/ — 精排模块
Cross-Encoder 对候选 chunk 做 token 级相关性重排序。
依赖: core/config

## 文件清单
| 文件 | 地位 | 功能 |
|------|------|------|
| cross_encoder.py | 核心 | BGE-Reranker 封装 + Dynamic Top-K |
