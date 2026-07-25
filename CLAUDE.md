# FDI 部门智能 RAG 问答助手

## 1. 项目概述

| 项 | 说明 |
|----|------|
| **项目名称** | FDI 部门智能 RAG 问答助手 |
| **项目目标** | 构建面向FDI部门的垂直领域智能问答Agent，解决产品相关问题解答效率低下的痛点，实现从数据清洗到精准问答的全链路闭环 |
| **目标用户** | FDI部门内部员工（产品、运营、销售、技术支持等） |
| **技术栈** | Python 3.11+, LangChain, Milvus, OpenAI / Qwen, Redis, Elasticsearch |

---

## 2. 技术栈明细

| 层级 | 技术选型 | 用途 |
|------|----------|------|
| 应用框架 | FastAPI + Uvicorn | API 服务 |
| AI 框架 | LangChain + LangGraph | 编排Agent、Prompt管理 |
| 向量数据库 | Milvus 2.4+ | 稠密向量 + 稀疏向量混合检索 |
| 全文检索引擎 | Elasticsearch（可选） | BM25 精确匹配（Milvus 2.4+ 已内置 Sparse BM25，ES 作为备选方案） |
| 缓存与内存 | Redis | 会话短时记忆 + 语义缓存 |
| 嵌入模型 | sentence-transformers / text2vec-large-chinese | 文档与查询向量化 |
| LLM | OpenAI（GPT-4o） / Qwen-Max（阿里百炼） | 生成回答 |
| 精排 | Cross-Encoder (BGE-Reranker) | 候选chunk重排序 |
| 可观测性 | LangFuse | 调用链追踪、Token消耗监控 |
| 报文格式 | Protobuf | 内部服务间通信 |

---

## 3. 系统架构

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Rewrite 模块                            │
│         模糊问题标准化 + Query Decomposition 查询分解            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      多路召回引擎                                │
│   ┌──────────────┐              ┌──────────────┐                │
│   │ BM25 稀疏召回 │  ────────▶  │   结果融合    │                │
│   │ (关键词精确)  │              │  (合并去重)   │                │
│   └──────────────┘              └──────────────┘                │
│   ┌──────────────┐                   │                          │
│   │Embedding 稠密 │  ───────────────▶                            │
│   │  (语义召回)   │                                             │
│   └──────────────┘                                             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│              Cross-Encoder Rerank 精排模块                      │
│          对候选chunk进行token级相关性重新排序                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│              Dynamic Top-K 调度模块                              │
│       根据分数分布动态决定送入LLM的片段数量，降低Token消耗        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  上下文组装 + LLM生成                            │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │  短时记忆   │  │  长时记忆   │  │  召回文档   │            │
│   │  (Redis)    │  │  (Milvus)   │  │ (精排结果)  │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
│                         │                                       │
│                         ▼                                       │
│                ┌─────────────────┐                              │
│                │  LLM 生成回答   │                              │
│                │  (OpenAI/Qwen)  │                              │
│                └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
  最终答案输出
```

---

## 4. 目录结构

```
rag-qa-assistant/
├── CLAUDE.md                  # 本文件
├── pyproject.toml             # 项目配置 + 依赖
├── docker-compose.yml         # 基础设施服务编排
├── src/
│   ├── __init__.py
│   ├── api/                   # FastAPI 接口层
│   │   ├── __init__.py
│   │   ├── routes.py          # API 路由定义
│   │   ├── schemas.py         # Pydantic 请求/响应模型
│   │   └── dependencies.py    # 依赖注入（获取组件实例）
│   ├── core/                  # 全局配置
│   │   ├── __init__.py
│   │   ├── config.py          # 环境变量读取、配置类
│   │   └── logging.py         # 日志配置
│   ├── ingestion/             # 数据摄入管道
│   │   ├── __init__.py
│   │   ├── loader.py          # 文档加载（PDF/Word/Markdown/HTML）
│   │   ├── chunker.py         # 文本切分策略
│   │   ├── embedder.py        # 嵌入模型封装
│   │   ├── indexer.py         # 写入 Milvus / ES
│   │   └── pipeline.py        # 摄入流水线编排
│   ├── retrieval/             # 多路召回
│   │   ├── __init__.py
│   │   ├── bm25.py            # BM25 稀疏检索
│   │   ├── embedding.py       # 稠密向量检索
│   │   ├── fusion.py          # 结果融合（RRF / 加权融合）
│   │   └── rewrite.py         # Query Rewrite
│   ├── reranker/              # 精排模块
│   │   ├── __init__.py
│   │   └── cross_encoder.py   # Cross-Encoder Reranker
│   ├── generator/             # LLM 生成
│   │   ├── __init__.py
│   │   ├── llm.py             # LLM 客户端封装（OpenAI/Qwen）
│   │   ├── prompts.py         # Prompt 模板
│   │   └── guard.py           # 输出校验（后续可选）
│   ├── memory/                # 记忆管理
│   │   ├── __init__.py
│   │   ├── short_term.py      # Redis 会话记忆
│   │   └── long_term.py       # Milvus 长期记忆查询
│   ├── decompose/             # Query Decomposition 查询分解
│   │   ├── __init__.py
│   │   ├── rules.py           # 规则检测（关键词、实体计数）
│   │   └── llm_splitter.py    # LLM 驱动子查询拆分
│   ├── evaluate/              # 检索评估体系
│   │   ├── __init__.py
│   │   ├── metrics.py         # recall@k, MRR, NDCG
│   │   ├── dataset.py         # 标注问答数据集加载
│   │   └── runner.py          # 评估运行器
│   └── tenant/                # 多租户隔离
│       ├── __init__.py
│       ├── resolver.py        # 租户识别（API Key / Header）
│       └── filter.py          # Milvus metadata 过滤
├── tests/
│   ├── conftest.py            # Pytest fixtures
│   ├── test_retrieval/        # 召回模块测试
│   ├── test_reranker/         # 精排模块测试
│   ├── test_ingestion/        # 摄入管道测试
│   ├── test_decompose/        # 查询分解测试
│   └── test_evaluate/         # 评估框架测试
├── data/
│   ├── raw/                   # 原始文档
│   ├── processed/             # 处理后数据
│   └── qa_dataset/            # 标注问答数据集（评估用）
├── scripts/
│   ├── init_milvus.py         # Milvus Collection 初始化
│   ├── batch_ingest.py        # 批量文档摄入
│   ├── run_eval.py            # 运行评估
│   └── embedding_cache.py     # 嵌入缓存预热
└── config/
    ├── prompts/               # Prompt 模板 YAML
    ├── chunking.yaml          # 切分策略配置
    └── tenant_mapping.yaml    # 租户-知识库映射
```

---

## 5. 核心模块详解

### 5.1 Query Rewrite（查询重写）

- **职责**：将模糊、口语化或指代不明的问题标准化为精确检索查询
- **实现**：LLM + Prompt模板 + 规则辅助（实体识别补充）
- **输入**：用户原始问题 + 会话历史指针
- **输出**：经重写的标准查询

### 5.2 Query Decomposition（查询分解）

- **触发条件**：关键词检测（"对比"、"区别"、"列出"、"分别"）+ 实体数量 ≥2 + LLM意图判断
- **实现流程**：
  1. 规则辅助检测（关键词匹配 + 实体计数）
  2. LLM 驱动拆分（Prompt：判断是否需要拆分 → 输出子查询列表）
  3. 每个子查询独立走完整 RAG 流程
  4. 子结果聚合（表格对比 / 列表汇总）
- **容灾**：LLM不可用时，降级为纯规则拆分

### 5.3 多路召回（Multi-Path Retrieval）

- **BM25（稀疏向量）**：精确匹配专有名词、产品型号、缩写词
- **Embedding（稠密向量）**：捕捉语义相似性，覆盖同义表述
- **融合策略**：RRF (Reciprocal Rank Fusion) 或线性加权融合
- **注意**：Milvus 2.4+ 内置 Sparse BM25 能力，无需外部依赖 Elasticsearch。ES 作为备选方案保留。

### 5.4 精排 Rerank

- **模型**：BGE-Reranker-v2 / bge-reranker-large
- **实现**：Cross-Encoder Transformer，对 Top-N 候选 chunk 做 token 级相关性评分
- **目标**：将最相关的 3-5 个 chunk 排在队首

### 5.5 Dynamic Top-K

- **逻辑**：分析精排后的相关性分数分布
- **规则**：
  - 头部 chunk 分数显著高于后续 → 只取 2-3 个
  - 分数均匀分布 → 适当增加数量
  - 分数整体偏低 → 可能未命中，触发降级策略
- **价值**：保证召回率的同时降低 Token 消耗，减少上下文噪音

### 5.6 父子索引（Parent-Child Indexing）

- **子 Chunk**：小粒度（200-400 tokens），用于确保召回精度
- **父 Paragraph**：完整段落（800-1500 tokens），回溯保证上下文完整
- **流程**：子chunk 命中 → 检索对应父段落 → 送入 LLM

### 5.7 多租户隔离

- **实现**：Milvus Collection Metadata 字段 `tenant_id`
- **流程**：API 入口解析租户 → 所有检索查询附加 `filter: tenant_id == "xxx"`
- **优势**：零架构成本，天然与 Milvus 索引兼容

---

## 6. 数据摄入管道（Ingestion Pipeline）

### 6.1 整体流程

```
原始文档 (PDF/Word/Markdown/TXT/HTML)
    │
    ▼
┌─────────────────┐
│  Document Loader │  解析文件，提取原始文本 + 元数据
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Chunker         │  按段落/标题/语义边界切分
│  策略: 递归字符 │  → 父段落 (800-1500 tokens)
│  + 语义分块     │  → 子 chunk (200-400 tokens)
│                  │  父子关联ID维护
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Metadata Extr   │  提取来源、标题、租户、时间、版本
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Embedder        │  Dense Vector: text2vec-large-chinese
│                  │  Sparse Vector: BGE-M3 / Milvus BM25
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Indexer         │  写入 Milvus Collection
│  (增量/全量)    │  全量: drop & recreate
│                  │  增量: upsert by doc_id
└─────────────────┘
```

### 6.2 Chunk 策略

| 策略 | 适用场景 | 参数 |
|------|----------|------|
| 递归字符分割 | 通用文档 | chunk_size=400, overlap=50 |
| 语义分块 | 技术文档、FAQ | 按标题层级切分 |
| 固定长度 | 纯文本 | chunk_size=300, overlap=30 |

### 6.3 Metadata 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| tenant_id | VARCHAR | 租户标识 |
| doc_id | VARCHAR | 文档唯一ID |
| doc_title | VARCHAR | 文档标题/文件名 |
| chunk_id | VARCHAR | 子chunk唯一ID |
| parent_chunk_id | VARCHAR | 父段落ID |
| source_type | VARCHAR | PDF/Word/MD/TXT |
| updated_at | INT64 | 更新时间戳 |
| version | VARCHAR | 文档版本号 |

---

## 7. 检索评估体系

### 7.1 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **Recall@K** | 前K个结果中包含正确文档的比例 | `|relevant ∩ retrieved_K| / |relevant|` |
| **MRR** (Mean Reciprocal Rank) | 第一个正确结果的排名倒数均值 | `1/N * Σ(1/rank_i)` |
| **NDCG@K** | 归一化折损累计增益 | 考虑排序位置权重 |
| **Answer Accuracy** | LLM 生成的答案是否准确 | 人工标注 / LLM-as-Judge |
| **Latency P50/P95** | 端到端延迟分布 | 毫秒 |

### 7.2 评估数据集

- **格式**：`{ "question": "...", "relevant_docs": ["doc_id_1", ...], "expected_answer": "..." }`
- **规模**：初期 ≥100 条标注样本
- **来源**：历史问答记录抽取 + 人工标注

### 7.3 评估流程

```
标注数据集
    │
    ▼
┌──────────────────┐
│ 1. 检索评估       │  对每个问题运行检索，计算 Recall@K / MRR / NDCG
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 2. 端到端评估     │  完整RAG流程，LLM-as-Judge 或人工评分
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 3. 输出报告       │  JSON + Markdown 报告，追踪指标趋势
└──────────────────┘
```

---

## 8. 可观测性（LangFuse）

### 8.1 追踪维度

| 追踪项 | 内容 |
|--------|------|
| 请求元数据 | tenant_id, session_id, timestamp |
| Query Rewrite | 原始query → 重写query，耗时 |
| 召回阶段 | BM25 候选数 / Embedding 候选数 / 融合后数量 |
| 精排 | Top-K 输入/输出，每chunk得分 |
| LLM 生成 | Prompt tokens / Completion tokens / 总耗时 |
| 错误 | 异常类型、堆栈、频率 |

### 8.2 接入方式

```python
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    secret_key=settings.LANGFUSE_SECRET_KEY,
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    host=settings.LANGFUSE_HOST,
)
```

---

## 9. 开发环境搭建

### 9.1 前置条件

- Python 3.11+
- Docker & Docker Compose
- Git

### 9.2 一键启动基础设施

```bash
docker compose up -d
```

### 9.3 安装依赖

```bash
pip install -e ".[dev]"
```

### 9.4 环境变量

复制 `.env.example` 为 `.env`，填入：

| 变量 | 说明 | 示例 |
|------|------|------|
| MILVUS_HOST | Milvus 地址 | localhost |
| MILVUS_PORT | Milvus 端口 | 19530 |
| REDIS_URL | Redis 连接 | redis://localhost:6379 |
| OPENAI_API_KEY | OpenAI 密钥 | sk-xxx |
| QWEN_API_KEY | Qwen 密钥 | sk-xxx |
| LANGFUSE_SECRET_KEY | LangFuse 密钥 | sk-lf-xxx |
| EMBEDDING_MODEL | 嵌入模型名 | text2vec-large-chinese |

### 9.5 初始化

```bash
# 创建 Milvus Collection
python scripts/init_milvus.py

# 批量摄入文档
python scripts/batch_ingest.py --source data/raw/

# 运行评估
python scripts/run_eval.py
```

### 9.6 启动 API 服务

```bash
uvicorn src.api.routes:app --reload --port 8000
```

---

## 10. 编码规范

- **Python 版本**：3.11+，使用 Type Hints
- **格式化**：ruff format（替代 black）
- **Lint**：ruff check
- **类型检查**：mypy --strict
- **测试框架**：pytest + pytest-asyncio
- **文档字符串**：Google Style Docstring
- **配置管理**：pydantic-settings，所有配置从环境变量读取
- **模块依赖原则**：`ingestion` → `retrieval` → `reranker` → `generator` 单向依赖，不可反向引用
- **错误处理**：自定义异常类，全局异常中间件统一捕获
- **异步优先**：所有 I/O 操作使用 async/await（Milvus SDK 同步限制的除外）

---

## 11. 文本分词

采用jieba分词库进行分词，作用如下：
- BM25 候选词分词
- 问题内实体提取
- Query rewrite使用词语拼写纠错

---

## 12. 后续可选优化

### 12.1 结果缓存层

高维语义缓存：对历史问答结果做 Embedding，新问题与缓存问题相似度 ≥ 阈值时直接返回缓存答案。

- **利**：高频问题秒级返回，显著降低 LLM 调用成本
- **弊**：需要设计缓存失效策略（文档更新触发刷新），边缘情况多

### 12.2 Guard 安全兜底

输出校验与拦截：规则 + LLM 双重检测，防止幻觉、越权、不当回答。

- **利**：企业级安全合规
- **弊**：额外延迟（+200-500ms），误拦截需要调优

---

## 13. 用户交互

### 13.1 交互方式

采用 Web 聊天界面（方案 A），技术选型：

| 层级 | 技术 | 说明 |
|------|------|------|
| 后端框架 | FastAPI | 提供 API 接口 |
| 模板引擎 | Jinja2 | 服务端渲染 HTML |
| 前端样式 | Bootstrap 5 + 自定义 CSS | 响应式布局，支持移动端 |
| 实时通信 | Server-Sent Events (SSE) | 流式输出 LLM 回答 |

### 13.2 页面设计

| 页面 | 路由 | 说明 |
|------|------|------|
| 聊天页面 | `/` 或 `/chat` | 对话主界面，消息列表 + 输入框 |
| 会话列表 | `/sessions` | 历史会话管理，新建/切换/删除 |
| 文档管理 | `/documents` | 知识库文档查看（可选） |

### 13.3 交互流程

1. 用户打开页面 → 创建/选择会话
2. 在输入框输入问题 → 点击发送
3. 系统显示"正在思考..." → SSE 流式返回回答片段
4. 回答完成 → 显示参考文档来源
5. 支持追问（承载在同一会话上下文中）

---

## 14. API 设计

### 14.1 接口列表

| 接口 | 方法 | 说明 | 实现状态 |
|------|------|------|----------|
| `/api/chat` | POST | 发送问题，返回答案 | ✅ 本次实现 |
| `/api/sessions` | GET | 获取会话列表 | 📋 待实现 |
| `/api/sessions/{id}` | GET | 获取会话历史 | 📋 待实现 |
| `/api/sessions` | POST | 创建新会话 | 📋 待实现 |
| `/api/sessions/{id}` | DELETE | 删除会话 | 📋 待实现 |
| `/api/documents` | GET | 文档列表 | 📋 待实现 |
| `/api/documents/upload` | POST | 上传文档 | 📋 待实现 |
| `/api/evaluate/run` | POST | 运行离线评估 | 📋 待实现 |

### 14.2 核心接口：POST /api/chat

**请求体**

```json
{
  "session_id": "sess_abc123",
  "question": "A产品和B产品的区别？",
  "tenant_id": "fdi_dept"
}
```

**响应体（SSE 流式）**

```
event: token
data: {"token": "A产品"}

event: token
data: {"token": "主要用于"}

event: done
data: {"sources": [{"doc_id": "doc_001", "title": "产品A说明"}], "tokens_used": 342}
```

**错误响应**

```json
{
  "error": {
    "code": "RETRIEVAL_EMPTY",
    "message": "未找到相关文档",
    "fallback": "llm_direct"
  }
}
```

### 14.3 错误码定义

| 错误码 | 说明 | HTTP 状态码 |
|--------|------|-------------|
| `RETRIEVAL_EMPTY` | 召回结果为空 | 200（降级） |
| `LLM_UNAVAILABLE` | LLM 服务不可用 | 503 |
| `INPUT_INVALID` | 输入违规 | 400 |
| `RATE_LIMITED` | 请求频率过高 | 429 |
| `TENANT_MISMATCH` | 租户信息异常 | 403 |

---

## 15. 边缘场景处理

### 15.1 召回为空（RETRIEVAL_EMPTY）

**触发条件**：多路召回 + 精排后无任何 chunk 的相关性分数超过阈值。

**处理策略**：
1. 记录当前 query 到"未命中日志"（供后续分析）
2. 降级为 LLM 直接回答，并在回答末尾追加提示：
   > "以上回答基于通用知识，未在部门知识库中找到精确匹配内容"
3. 触发主动学习流程（可选）：通知知识库管理员补充文档

### 15.2 LLM 不可用（LLM_UNAVAILABLE）

**触发条件**：
- API 返回 429/5xx 错误
- API 超时（超过 30 秒无响应）
- API Key 无效或额度用尽

**处理策略**：
1. 检查结果缓存是否有 hit
   - 命中 → 直接返回缓存答案
   - 未命中 → 进入步骤 2
2. 返回"服务暂不可用"提示，附带预计恢复时间（如有）
3. 记录故障时间、上下文到日志

### 15.3 输入违规（INPUT_INVALID）

**触发条件**：
- 输入包含 SQL 注入特征（SELECT、DROP、DELETE 等）
- 输入包含 XSS 特征（`<script>`、`onerror` 等）
- 输入长度超过上限（1000 字）
- 输入包含敏感词（需接入敏感词库）

**处理策略**：
1. 输入校验层拦截，不进入后续流程
2. 返回 400 错误码 + 具体违规说明
3. 不记录问题内容到会话历史（防泄漏）

### 15.4 生产环境必做（本 demo 暂不实现）

| 场景 | 说明 |
|------|------|
| 向量库故障 | Milvus 不可用时 Redis + 缓存兜底 |
| 跨租户数据泄漏 | 请求级 tenant_id 校验 + 审计日志 |
| 会话超时 | Redis TTL 到期自动清理 |
| 并发限流 | 基于 IP/租户的令牌桶限流 |
| 敏感数据脱敏 | 回答中手机号、身份证号自动掩码 |

---

## 16. 技术决策

### 16.1 向量数据库：Milvus vs FAISS

| 对比维度 | Milvus 2.4+ | FAISS |
|----------|-------------|-------|
| 部署方式 | 独立服务（Docker） | 嵌入式库 |
| 多租户过滤 | ✅ 原生 Metadata 过滤 | ❌ 需自建 |
| 混合检索 | ✅ 稠密+稀疏向量混合 | ❌ 仅稠密 |
| 分布式 | ✅ 支持 | ❌ 单机 |
| 运维成本 | 中等（需 etcd + minio） | 低 |

**结论**：选 Milvus。demo 虽可用 FAISS 更轻量，但后续扩展时 Milvus 不必重写。

### 16.2 稀疏检索：Milvus BM25 vs Elasticsearch

| 对比维度 | Milvus Sparse BM25 | Elasticsearch |
|----------|-------------------|---------------|
| 运维成本 | 零额外服务 | 需额外部署 ES |
| 与稠密向量融合 | 同一查询、同一引擎 | 需 RRF 融合 |
| 中文分词 | 需外部 jieba | 内置 IK 分词 |
| 查全率 | 基础 BM25 | 更成熟的全文检索 |

**结论**：demo 阶段选 Milvus 内置 BM25，省一个 ES 依赖。生产环境若 BM25 精度不足再引入 ES。

### 16.3 AI 框架：LangChain vs 直接调用 LLM

| 对比维度 | LangChain | 直接调用 |
|----------|-----------|----------|
| 开发效率 | 高（Prompt 模板、Chain、Callback 开箱即用） | 低（需自建） |
| 可观测性 | 原生 LangSmith/LangFuse 集成 | 需手动埋点 |
| 抽象开销 | 较多抽象层，调试困难 | 完全可控 |
| 灵活度 | 受框架约束 | 完全灵活 |

**结论**：选 LangChain。demo 阶段开发效率更重要，抽象开销通过 LangFuse 追踪缓解。

### 16.4 精排模型：BGE-Reranker vs Cohere Rerank

| 对比维度 | BGE-Reranker | Cohere Rerank |
|----------|-------------|---------------|
| 部署方式 | 本地运行 | API 调用 |
| 费用 | 免费 | 按量付费 |
| 精度 | ~90% | ~92% |
| 中文支持 | 良好 | 良好 |
| 延迟 | 本地 50-100ms | 网络 200-500ms |

**结论**：选 BGE-Reranker。demo 项目首选免费本地方案，精度差距可接受。

### 16.5 嵌入模型：text2vec-large-chinese vs bge-m3

| 对比维度 | text2vec-large-chinese | bge-m3 |
|----------|------------------------|--------|
| 维度 | 768 | 1024 |
| 中文语义 | 特化中文，效果更优 | 多语言通用 |
| 混合向量 | 仅稠密 | 支持稠密+稀疏 |
| 社区热度 | 较低 | 较高 |

**结论**：选 text2vec-large-chinese。demo 阶段专精中文更重要，bge-m3 的混合向量能力可通过 Milvus 内置 BM25 替代。

### 16.6 LLM：OpenAI GPT-4o-mini vs Qwen-Max

| 对比维度 | GPT-4o-mini | Qwen-Max |
|----------|-------------|----------|
| 中文质量 | 优秀 | 优秀 |
| 价格 | $0.15/1M input tokens | ¥0.02/1K tokens |
| 延迟 | 中等（海外节点） | 低（国内节点） |
| 合规性 | 需国际网络 | 国内合规 |
| API 兼容 | OpenAI 格式 | OpenAI 兼容 |

**结论**：优先选 Qwen-Max。国内合规、低延迟、无需梯子，demo 项目以可用为首要目标。

---

## 17. 分形文档结构

### 17.1 三级联动规则

本项目采用"根 → 文件夹 → 文件"三级联动文档体系，任何功能、架构、写法的变更，工作结束后必须沿此链路同步更新：

```
CLAUDE.md                              ← 主文档（根）
  └── src/{模块}/README.md             ← 模块文档（文件夹）
        └── src/{模块}/*.py 头注释     ← 文件自述（文件）
```

### 17.2 各层级职责

| 层级 | 位置 | 内容 | 更新触发条件 |
|------|------|------|-------------|
| 根 | `CLAUDE.md` | 全局架构、技术栈、数据流、编码规范、交互方式、API、边缘场景、技术决策 | 任何影响全局的功能/架构/写法变更 |
| 文件夹 | `src/{模块}/README.md` | 三行模块定位说明 + 文件清单（文件名、地位、功能） | 该模块内文件增删改 |
| 文件 | `.py` 文件前三行注释 | `input:` / `output:` / `pos:` 各一行 | 文件内容更新 |

### 17.3 变更同步流程

```
1. 修改了某个 .py 文件
   ↓
2. 更新该文件开头三行的 input / output / pos 注释
   ↓
3. 更新所在文件夹的 README.md（文件增删改都要改）
   ↓
4. 如果影响跨模块依赖 或 全局架构，更新 CLAUDE.md
```

### 17.4 文件头注释格式

每个 `.py` 文件开头必须包含三行注释：

```python
# input:  依赖的外部输入（模块、服务、数据源）
# output: 对外提供的产出（数据、接口、文件）
# pos:    在系统局部架构中的定位
```

示例：

```python
# input:  raw text chunks, Milvus client, embedder
# output: Milvus collection (inserted/updated)
# pos:    摄入管道 → 索引写入模块
```

注意：一旦文件被更新，必须同步更新这三行注释以及所属文件夹的 README.md。

---

## Appendix A: 词汇表

| 术语 | 说明 |
|------|------|
| **RAG** | Retrieval-Augmented Generation，检索增强生成 |
| **短时记忆** | Redis 中保存的当前会话上下文，用户关闭会话后清除 |
| **长时记忆** | Milvus 中持久化的知识库文档向量索引，随文档更新刷新 |
| **结果缓存** | 问答对缓存，相同/相似问题直接返回历史答案，省计算 |
| **Embedding / 稠密向量** | 文本的语义向量表示，768+维浮点数 |
| **BM25 / 稀疏向量** | 基于词频的倒排索引检索，精准匹配关键词 |
| **RRF** | Reciprocal Rank Fusion，多路结果的无监督融合算法 |
| **Cross-Encoder** | 同时对 query 和 passage 建模的 Transformer，精排用 |
| **Top-K** | 检索返回的候选结果数量 |
| **Chunk** | 文档切分后的文本片段 |
| **父子索引** | 小chunk召回 + 回溯父段落补充上下文 |
