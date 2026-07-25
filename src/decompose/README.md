# decompose/ — 查询分解模块
检测复杂/多实体问题 → LLM 拆分为独立子查询。
依赖: core/config, retrieval/rewrite

## 文件清单
| 文件 | 地位 | 功能 |
|------|------|------|
| rules.py | 核心 | 关键词+实体计数规则检测 |
| llm_splitter.py | 核心 | LLM 驱动子查询拆分 |
