# input:  raw user query, session history (optional)
# output: rewritten standard query
# pos:    召回层前置 → 模糊/口语化问题标准化，提高检索命中率

import json
import re
from typing import List, Dict, Optional

from src.core.config import settings
from src.core.logging import logger


def rewrite(
    raw_query: str,
    history: Optional[List[Dict]] = None,
    model: str = "qwen-max",
) -> str:
    history_context = ""
    if history:
        recent = history[-4:]
        turns = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            turns.append(f"{role}: {msg['content']}")
        history_context = "\n".join(turns)

    prompt = _build_prompt(raw_query, history_context)

    try:
        rewritten = _call_llm(prompt, model)
        cleaned = rewritten.strip().strip('"').strip("'")
        if cleaned:
            logger.info("Query rewrite: '%s' → '%s'", raw_query, cleaned)
            return cleaned
    except Exception as e:
        logger.warning("LLM rewrite 失败，使用原 query: %s", e)

    return _rule_fallback(raw_query)


def _build_prompt(query: str, history_context: str) -> str:
    parts = ["你是一个FDI业务问答助手的查询改写器。请将用户的问题改写为更清晰、更适合检索的标准问句。"]

    if history_context:
        parts.append(f"\n对话历史:\n{history_context}")

    parts.append(f"""
改写规则:
1. 补充指代不明的内容（如"它"→具体产品名，"那个"→具体系统名）
2. 将口语化表达转为书面检索语
3. 保持原意的同时增加关键词
4. 直接输出改写后的问句，不要解释

用户问题: {query}

改写结果:""")

    return "\n".join(parts)


def _call_llm(prompt: str, model: str) -> str:
    from openai import OpenAI

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = settings.qwen_api_key

    if not api_key:
        raise ValueError("QWEN_API_KEY 未设置")

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=256,
    )
    return resp.choices[0].message.content or ""


def _rule_fallback(query: str) -> str:
    query = query.strip()
    query = re.sub(r"这个|那个|它(们)?|该|此", "", query)
    query = re.sub(r"我想问一下|请问|帮我|能不能|怎么|如何", "", query)
    query = query.strip()
    return query if query else "未识别的问题"
