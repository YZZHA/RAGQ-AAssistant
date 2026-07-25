# generator/ — LLM 生成层
Prompt 组装 + LLM 调用 + 输出校验。
依赖: core/config, memory/short_term

## 文件清单
| 文件 | 地位 | 功能 |
|------|------|------|
| llm.py | 核心 | OpenAI/Qwen 客户端封装 |
| prompts.py | 核心 | Prompt 模板管理 |
| guard.py | 后续可选 | 输出安全校验 |
