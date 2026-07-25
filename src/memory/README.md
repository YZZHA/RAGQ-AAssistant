# memory/ — 记忆管理层
短时记忆（Redis 会话）+ 长时记忆（Milvus 检索）。
依赖: core/config

## 文件清单
| 文件 | 地位 | 功能 |
|------|------|------|
| short_term.py | 核心 | Redis 会话历史读写 |
| long_term.py | 核心 | Milvus 知识库检索 |
