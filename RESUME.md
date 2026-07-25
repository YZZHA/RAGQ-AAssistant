# FDI 部门智能 RAG 问答助手 — 简历项目说明

> **本文件是项目的"活简历"**，每完成一个功能模块后，思考是否值得写入此文档。
> 判断标准：是否体现了架构能力、技术深度、工程思维。
> 如果可以写，追加到对应章节下方。

---

## 一句话定位

**RAG 智能问答系统** — 基于 LangChain + FAISS + Qwen 的垂直领域检索增强生成系统，支持多路召回、Cross-Encoder 精排、SSE 流式输出、自动化评估。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 应用框架 | FastAPI + Uvicorn |
| AI 框架 | LangChain |
| 向量检索 | FAISS (IndexFlatIP) |
| 稀疏检索 | BM25 (jieba 分词) |
| 精排 | Cross-Encoder (MiniLM) |
| LLM | Qwen-Max (阿里百炼) |
| 评估 | DeepEval (4 维 RAG 指标) |
| 缓存/记忆 | Redis / fakeredis |
| 嵌入模型 | paraphrase-multilingual-MiniLM-L12-v2 (384dim) |

---

## 核心架构与亮点

### 全链路 RAG 流程

```
文档摄入 → BM25 + FAISS 双路召回 → RRF 融合
       → Cross-Encoder 精排 → Dynamic Top-K
       → Prompt 组装 → LLM 生成 (SSE 流式)
```

### 自动化知识生产 (Phase 1)

```
用户提问 → RAG 回答 → 缺口检测（分数 < 0.3 或 用户点击"未解决"）
→ 记录缺口日志 (JSONL) → 后续可自动生成 Q&A 草稿 → 校验 → 入库
```

- **分数阈值**：精排后最高分 < 0.3 自动触发缺口记录
- **用户反馈**：Web UI 上提供"未解决"按钮，用户手动标记
- **日志持久化**：`data/knowledge_gaps/gaps.jsonl`，记录问题、回答、触发方式、时间戳
- **异步执行**：缺口检测在后台线程执行，不阻塞用户响应
- **草稿生成**：LLM (Qwen-Max) 将用户提问 + 回答标准化为 Q&A JSON
- **LLM 自检**：Qwen 做质量审核（幻觉检测、一致性检查），通过率约 90%
- **自动入库**：校验通过后 → chunk → embed → 重建 BM25 + FAISS 索引，后续同类问题直接命中知识库
- **缺口管理页面**：`/gaps` 展示缺口列表（触发方式、分数、状态），支持一键清空
- **Chunk 优化**：三级标题树（H1/H2/H3）记录 heading_chain → 过短合并（<150 char）→ 过长拆分（按段落/句号/逗号）→ overlap 50 chars → LLM 语义切分（MD5 缓存，首次缓存后续零调 LLM）

### 关键设计决策

| 决策 | 选型 | 理由 |
|------|------|------|
| 检索策略 | BM25 + Embedding 双路 + RRF | 同时保证专有名词精确匹配和语义覆盖 |
| 精排 | Cross-Encoder (MiniLM) | 比 Bi-Encoder 精度更高，demo 规模可接受延迟 |
| 向量库 | FAISS 本地内存索引 | 规避 Milvus Lite Windows 兼容问题，零运维成本 |
| LLM | Qwen-Max | 国内合规、低延迟、无需 VPN |
| 评估 | DeepEval (4 指标) | 覆盖回答相关性、事实一致性、排序质量、召回完整度 |

---

## 量化成果

### DeepEval 评估结果 (10 条标注数据)

| 指标 | 得分率 | 含义 |
|------|--------|------|
| AnswerRelevancy | 9/10 ✅ | 回答与问题相关 |
| Faithfulness | 10/10 ✅ | 回答基于检索结果，无幻觉 |
| ContextualPrecision | 8/10 ✅ | 相关 chunk 排位靠前 |
| ContextualRecall | 8/10 ✅ | 检索覆盖了关键信息 |
| **综合通过率** | **80%** | BM25 + FAISS 双路召回 |

### 系统性能

| 指标 | 数据 |
|------|------|
| 知识库文档 | 12 篇 FDI 领域文档 |
| 索引 chunk 数 | 73 个 |
| 评估数据集 | 30 条标注问答对 |
| 首次启动到可用 | ~10s (含模型加载 + 双索引构建) |

---

## 简历描述

### 中文 (~300 字)

```
FDI 部门智能 RAG 问答助手
- 构建面向外商直接投资领域的垂直 RAG 系统，实现从文档摄入到精准问答的全链路闭环。
  采用 BM25 + FAISS 双路召回 + RRF 融合，结合 Cross-Encoder 精排 + Dynamic Top-K 动态截断，提升检索精度同时降低 Token 消耗。
- 实现三级标题树（H1/H2/H3）结构化切分 + 过短合并/过长拆分/overlap 策略，并引入 LLM 语义切分（MD5 缓存，首次切分后零额外调用），提升 Recalling 效果。
- 实现 Query Rewrite + Query Decomposition，处理模糊/多实体复杂问题；支持多轮对话和 SSE 流式输出。
- 搭建自动化知识生产系统：缺口检测（分数阈值 + 用户反馈）→ LLM 草稿生成 → LLM 自检验证 → 自动入库索引重建，形成持续生长的知识闭环。
- 集成 DeepEval 评估框架，4 维 RAG 指标量化评估，综合通过率 80%。
- 技术栈：Python, LangChain, FastAPI, FAISS, Qwen-Max, Redis, DeepEval
```

### English (~250 words)

```
FDI Domain Intelligent RAG Q&A Assistant
- Built a domain-specific RAG system for foreign direct investment, covering the full pipeline from document ingestion to precise Q&A.
  Implemented BM25 + FAISS dual-path retrieval with RRF fusion, Cross-Encoder re-ranking and Dynamic Top-K for accuracy and token efficiency.
- Developed a three-level heading tree (H1/H2/H3) chunk strategy with short-merge/long-split/overlap, plus LLM-powered semantic chunking with MD5 cache (zero extra LLM calls after initial run).
- Implemented Query Rewrite + Query Decomposition for ambiguous and multi-entity questions, with multi-turn conversation and SSE streaming.
- Built an automated knowledge production system: gap detection (score threshold + user feedback) → LLM draft generation → LLM self-validation → auto-ingest with index rebuild, forming a self-growing knowledge loop.
- Integrated DeepEval for automated evaluation (AnswerRelevancy / Faithfulness / ContextualPrecision / ContextualRecall), achieving 80% overall pass rate.
- Tech Stack: Python, LangChain, FastAPI, FAISS, Qwen-Max, Redis, DeepEval
```

---

## 面试可能的追问与回答

| 面试问题 | 回答要点 |
|----------|----------|
| 为什么用 BM25 + Embedding 双路？ | BM25 精确匹配专有名词（产品型号、政策名），Embedding 捕获语义相似性（"咋注册"→"登记流程"），RRF 融合无需调参 |
| Cross-Encoder 和 Bi-Encoder 区别？ | Bi-Encoder 提前向量化存库（FAISS），速度快；Cross-Encoder 实时 query-passage 联合编码，精度高但慢。项目里两者配合：FAISS 初筛 → Cross-Encoder 精排 |
| 评估指标为什么选这 4 个？ | AnswerRelevancy 测回答质量，Faithfulness 防幻觉，ContextualPrecision 测排序好不好，ContextualRecall 测有没有漏召回。四者覆盖输入→检索→排序→生成全链路 |
| 2 个失败的用例是什么原因？ | chunk 粒度问题——正确答案在文档片段中被截断，Dynamic Top-K 只取 2-3 个 chunk 时被边缘化。增大 min_k 或优化 chunk 策略可改善 |
| 为什么没有用 Milvus？ | Windows 下 Milvus Lite 有文件锁兼容问题。替代方案 FAISS 在 demo 规模下性能完全够用，且零运维成本。生产环境可平滑切回 Milvus |

---

## 更新记录

| 日期 | 变更 | 版本 |
|------|------|------|
| 2026-07-18 | 初始版本，完成全链路 RAG + DeepEval 评估 + Web UI | v1.0 |
| 2026-07-18 | 自动化知识生产 Phase 1：缺口检测 + 用户反馈 + 日志持久化 | v1.1 |
| 2026-07-18 | 自动化知识生产 Phase 2：LLM 草稿生成 + 自检验证 + FAISS 增量索引 | v1.2 |
| 2026-07-18 | 自动化知识生产 Phase 3：缺口管理页面 + API + 全链路闭环 | v1.3 |
| 2026-07-18 | Chunk 优化：H1/H2/H3 标题树 + heading_chain + 过短合并 + 过长拆分 + overlap + LLM 语义切分 + MD5 缓存 | v1.4 |

> **每次完成一个新功能后，问自己：**
> 1. 这体现了什么技术深度？（架构/选型/优化）
> 2. 有量化数据支撑吗？（指标/性能/对比）
> 3. 面试官会感兴趣吗？（工业实践/踩坑经验/决策依据）
>
> 如果至少满足两项，写入此文档。
