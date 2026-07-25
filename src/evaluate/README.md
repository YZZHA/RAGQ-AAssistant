# evaluate/ — 检索评估体系
离线评估：加载标注数据集 → 计算指标 → 输出报告。
依赖: core/config, retrieval, reranker, generator

## 文件清单
| 文件 | 地位 | 功能 |
|------|------|------|
| metrics.py | 核心 | Recall@K / MRR / NDCG 计算 |
| dataset.py | 核心 | 标注数据集加载与校验 |
| runner.py | 核心 | 评估运行器编排 |
| judge_model.py | 核心 | DeepEval 自定义 Judge LLM（Qwen） |

## DeepEval 集成

使用 `deepeval` 框架进行端到端 RAG 评估，支持 4 个指标：

| 指标 | 阈值 | 评估内容 |
|------|------|----------|
| AnswerRelevancy | 0.5 | 回答是否匹配问题 |
| Faithfulness | 0.5 | 回答是否基于检索结果 |
| ContextualPrecision | 0.3 | 相关 chunk 排名质量 |
| ContextualRecall | 0.3 | 检索结果覆盖度 |

运行方式：
```bash
# 完整评估（含 pytest + deepeval 报告）
deepeval test run tests/test_evaluate/test_rag_eval.py

# 或直接 pytest
python -m pytest tests/test_evaluate/test_rag_eval.py -v
```
