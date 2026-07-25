# input:  user query, rules detection result
# output: list of sub-queries (or original query if not decomposed)
# pos:    查询分解核心 → LLM 驱动子查询拆分，聚合结果

import json
import re
from typing import List, Dict, Optional

from src.core.config import settings
from src.core.logging import logger
from src.decompose.rules import detect


DECOMPOSE_PROMPT = """你是一个FDI业务问答助手的查询分解器。
如果用户问题涉及多个实体或需要对比/列举，请将其拆分为独立的子查询。
每个子查询应该是一个可以单独检索的完整问题。

规则:
1. 如果问题只涉及一个实体或不需要拆分，输出 [{"query": "原问题"}]
2. 如果涉及多个实体，每个实体拆分为独立的子查询
3. 每个子查询必须完整（含产品名、系统名等），不要用"它"代替
4. 输出严格的 JSON 数组格式

示例:
用户：A产品和B产品的区别？
输出：[{"query": "A产品有哪些功能特点？"}, {"query": "B产品有哪些功能特点？"}]

用户：怎么办理外商投资登记？
输出：[{"query": "怎么办理外商投资登记？"}]

用户问题：{query}
输出："""


def decompose(query: str, model: str = "qwen-max") -> List[str]:
    rules_result = detect(query)

    if not rules_result["should_decompose"]:
        return [query]

    if not settings.qwen_api_key:
        logger.info("LLM 不可用，使用规则兜底拆分")
        return _rule_split(query, rules_result)

    try:
        return _llm_split(query, model)
    except Exception as e:
        logger.warning("LLM 分解失败 (%s)，使用规则兜底", e)
        return _rule_split(query, rules_result)


def _llm_split(query: str, model: str) -> List[str]:
    from openai import OpenAI

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(api_key=settings.qwen_api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个严格输出 JSON 的查询分解器。"},
            {"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    text = resp.choices[0].message.content or ""

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        items = json.loads(text)
        if isinstance(items, list):
            return [item["query"] for item in items if "query" in item]
    except (json.JSONDecodeError, KeyError):
        pass

    return [query]


def _rule_split(query: str, rules_result: Dict) -> List[str]:
    entities = rules_result.get("entities", [])
    if len(entities) < 2:
        return [query]

    sub_queries = []
    for entity in entities:
        sub = query
        for other in entities:
            if other != entity:
                sub = sub.replace(other, f"【{other}】")
        sub = re.sub(r"对比|区别|分别|比较|列出|各个|不同|差异", "", sub).strip()
        sub = sub.replace("【", "").replace("】", "和")
        sub = re.sub(r"和$", "", sub).strip()
        sub_queries.append(f"{entity}的功能特点是什么？")

    return sub_queries if sub_queries else [query]
