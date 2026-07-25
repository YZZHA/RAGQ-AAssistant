# tenant/ — 多租户隔离层
API 入口解析租户 → 检索时附加 tenant_id 过滤。
依赖: core/config

## 文件清单
| 文件 | 地位 | 功能 |
|------|------|------|
| resolver.py | 核心 | 租户识别（API Key / Header） |
| filter.py | 核心 | Milvus metadata 过滤表达式构建 |
