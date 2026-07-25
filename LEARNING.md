# 自顶向下学习路径

> 沿着"入口 → 流程 → 模块 → 细节"逐层深入，每层都能回答"这一层我理解了什么"。

---

## 第一层：全局理解（读 2 个文件）

| 文件 | 读什么 | 读完能回答 |
|------|--------|-----------|
| `CLAUDE.md` | 第 3 章 架构图、第 5 章 核心模块、第 6 章 摄入管道 | 整个系统分几步？每步谁负责？ |
| `RESUME.md` | 一句话定位 + 全链路 RAG 流程 | 我给面试官怎么说？ |

**验证**：能画出来完整的 RAG 流程图（用户提问 → ... → 回答）。

---

## 第二层：请求跟踪（读 5 个文件）

从用户输入到返回的全路径：

```
用户请求 → ① src/api/routes.py
  → ② src/retrieval/ (rewrite → BM25/FAISS → fusion)
  → ③ src/reranker/cross_encoder.py
  → ④ src/generator/ (prompts → llm)
  → 返回 SSE
```

| 步骤 | 文件 | 关键函数 |
|------|------|----------|
| ① 入口 | `src/api/routes.py` | `chat()` → `_chat_stream()` |
| ② 召回 | `src/retrieval/bm25.py` `embedding.py` `fusion.py` `rewrite.py` | `rewrite()` → `.search()` → `rrf_fusion()` |
| ③ 精排 | `src/reranker/cross_encoder.py` | `rerank()` → `dynamic_top_k()` |
| ④ 生成 | `src/generator/llm.py` `prompts.py` | `build_rag_messages()` → `.generate_stream()` |

**方法**：在 `_chat_stream` 函数里逐行打断点，跑一个请求，观察数据在每个环节的输入/输出。

---

## 第三层：数据流动（3 条线）

| 数据线 | 从哪里来到哪里去 | 关键文件夹 |
|--------|------------------|-----------|
| **文档→索引** | `data/raw/*.md` → loader → chunker → embedder → indexer | `src/ingestion/` |
| **问题→检索** | question → rewrite → decompose → BM25/FAISS → fusion | `src/retrieval/` |
| **缺口→知识** | gap_detector → draft → validator → auto_ingest | `src/knowledge/` |

**方法**：打开 `data/qa_dataset/eval.jsonl` 选一条问题，手动跟踪它经过的每个函数。

---

## 第四层：读测试用例

测试文件比文档更精确，直接展示每个函数的输入/输出期望。

| 测试文件 | 读完能回答 |
|----------|-----------|
| `tests/test_retrieval/test_bm25.py` | BM25 怎么搜索的？分数怎么算？ |
| `tests/test_ingestion/test_chunker.py` | 一个文档切出来几个 chunk？heading_chain 怎么生成的？ |
| `tests/test_evaluate/test_rag_eval.py` | 评估系统怎么打分？哪些指标？ |
| `tests/test_knowledge/test_gap_detector.py` | 什么情况算"知识缺口"？ |

**方法**：`python -m pytest` 跑一个测试文件，读断言，理解每个函数的"承诺"。

---

## 第五层：关键模块逐文件拆解

按执行顺序串读每个模块的核心文件：

```
入口 → src/api/routes.py (_chat_stream)
  ├─ rewrite    → src/retrieval/rewrite.py
  ├─ decompose  → src/decompose/rules.py + llm_splitter.py
  ├─ bm25       → src/retrieval/bm25.py (BM25Index.search)
  ├─ embedding  → src/retrieval/embedding.py + local_store.py (FAISS)
  ├─ fusion     → src/retrieval/fusion.py (rrf_fusion)
  ├─ reranker   → src/reranker/cross_encoder.py (rerank + dynamic_top_k)
  ├─ prompts    → src/generator/prompts.py (build_rag_messages)
  └─ llm        → src/generator/llm.py (generate_stream)
```

---

## 第六层：动手调试（推荐）

```bash
python -m uvicorn src.api.routes:app --reload --port 8000
```

在 VSCode 调试器中打开 `src/api/routes.py`，在 `_chat_stream` 函数（第 112 行左右）打上断点，发送一个问题，逐行步进。

---

## 自测清单

能回答这些问题，说明已经吃透了：

- [ ] 用户问"A产品和B产品的区别？"经过 decompose 会变成几个子查询？怎么合并？
- [ ] FAISS 的向量什么时候生成的？新增文档后如何更新索引？
- [ ] 缺口检测的触发条件有几种？用户点击"未解决"后发生了什么？
- [ ] RRF 融合解决了什么问题？为什么不用简单的拼接？
- [ ] Dynamic Top-K 怎么决定取几个 chunk？
- [ ] LLM 语义切分什么时候触发？如果 LLM 调用失败会怎样？
- [ ] 评估系统的 4 个指标分别测什么？为什么选这 4 个？

---

## 速查：每个模块定位

| 模块 | 一句话定位 |
|------|-----------|
| `src/api/` | HTTP 入口 + SSE 流式返回 |
| `src/core/` | 全局配置（环境变量、日志） |
| `src/ingestion/` | 文档 → chunk → 向量 → 写入索引 |
| `src/memory/` | 短时记忆（Redis 会话）+ 长时记忆（向量检索） |
| `src/retrieval/` | 多路召回（BM25 + FAISS）+ Query Rewrite |
| `src/reranker/` | Cross-Encoder 重排序 + 动态截断 |
| `src/decompose/` | 复杂问题拆分子查询 |
| `src/generator/` | LLM 调用 + Prompt 组装 |
| `src/knowledge/` | 自动知识生产（缺口→草稿→校验→入库） |
| `src/evaluate/` | DeepEval 评估（4 维 RAG 指标） |
| `src/tenant/` | 多租户隔离 |
