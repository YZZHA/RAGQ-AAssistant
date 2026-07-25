# input:  raw user query
# output: boolean (should decompose?) + keyword/entity detection result
# pos:    查询分解前置 → 规则快速过滤，降低 LLM 调用频率

import re
from typing import List, Dict

SPLIT_KEYWORDS = ["对比", "区别", "分别", "比较", "列出", "各个", "不同", "差异"]
ENTITY_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]{2,}")


def detect(query: str) -> Dict:
    result = {
        "should_decompose": False,
        "reason": "",
        "keywords": [],
        "entities": [],
        "entity_count": 0,
    }

    matched_keywords = [kw for kw in SPLIT_KEYWORDS if kw in query]
    if matched_keywords:
        result["keywords"] = matched_keywords
        result["reason"] = f"检测到拆分关键词: {', '.join(matched_keywords)}"

    entities = list(set(ENTITY_PATTERN.findall(query)))
    common_words = {"的", "是", "了", "在", "有", "和", "或", "与", "吗", "呢", "吧",
                    "什么", "怎么", "如何", "为什么", "哪个", "哪些", "谁"}
    entities = [e for e in entities if e not in common_words and len(e) >= 2]
    result["entities"] = entities
    result["entity_count"] = len(entities)

    if result["entity_count"] >= 2:
        result["should_decompose"] = True
        result["reason"] = f"检测到 {result['entity_count']} 个实体，可能为对比/列举类问题"

    if result["keywords"] and not result["should_decompose"]:
        result["should_decompose"] = True

    return result
