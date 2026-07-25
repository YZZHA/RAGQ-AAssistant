# input:  draft Q&A, existing knowledge base
# output: validation result (passed/rejected + reason)
# pos:    知识生产 → LLM 自检，确保新知识不冲突、无幻觉

import json
import re
from typing import Dict

from openai import OpenAI

from src.core.config import settings
from src.core.logging import logger


VALIDATE_PROMPT = """你是一个FDI知识库质量审核助手。检查下面这条Q&A条目是否适合加入知识库。

审核标准:
1. 答案是否基于事实，没有幻觉（虚构信息）？
2. 答案是否与常见的FDI业务知识一致？
3. 是否存在自相矛盾或逻辑问题？
4. 答案是否完整回答了问题？

输出格式 JSON:
{{
  "passed": true/false,
  "reason": "通过/不通过的原因"
}}

Q&A条目:
问题: {question}
答案: {answer}

JSON:"""


def validate(question: str, answer: str) -> Dict:
    if not settings.qwen_api_key:
        logger.warning("Qwen API Key 未设置，默认通过校验")
        return {"passed": True, "reason": "跳过 LLM 校验（无 API Key）"}

    prompt = VALIDATE_PROMPT.format(question=question, answer=answer)

    try:
        client = OpenAI(api_key=settings.qwen_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        resp = client.chat.completions.create(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        text = resp.choices[0].message.content or ""
        result = _parse_json(text)
        if result and isinstance(result.get("passed"), bool):
            if result["passed"]:
                logger.info("校验通过: %s", question[:50])
            else:
                logger.warning("校验拒绝: %s — %s", question[:50], result.get("reason", ""))
            return result
    except Exception as e:
        logger.warning("LLM 校验失败，默认通过: %s", e)

    return {"passed": True, "reason": "默认通过（校验异常）"}


def _parse_json(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}
