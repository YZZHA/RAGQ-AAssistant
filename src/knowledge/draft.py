# input:  gap record (question + original LLM answer)
# output: structured Q&A JSON ({question, answer})
# pos:    知识生产 → LLM 将缺口转化为标准 Q&A 草稿

import json
import re
from typing import Dict, Optional

from openai import OpenAI

from src.core.config import settings
from src.core.logging import logger


DRAFT_PROMPT = """你是一个FDI业务知识库编辑助手。请基于用户的提问和你已给出的回答，生成一条标准化的Q&A知识条目。

要求:
1. 问题: 改写为通用、清晰的标准问句
2. 答案: 整理为完整、准确的回答，包含关键细节
3. 输出格式: 严格的 JSON，不要包含其他文字
4. 如果原始回答不完整，可以参考通用知识补充，但不要编造

输出格式:
{{
  "question": "标准问题",
  "answer": "标准答案"
}}

用户问题: {user_question}
原始回答: {original_answer}

JSON:"""


def generate_draft(user_question: str, original_answer: str) -> Optional[Dict]:
    if not settings.qwen_api_key:
        logger.warning("Qwen API Key 未设置，无法生成草稿")
        return None

    prompt = DRAFT_PROMPT.format(user_question=user_question, original_answer=original_answer)

    try:
        client = OpenAI(api_key=settings.qwen_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        resp = client.chat.completions.create(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content or ""
        return _parse_json(text)
    except Exception as e:
        logger.warning("草稿生成失败: %s", e)
        return None


def _parse_json(text: str) -> Optional[Dict]:
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
    return None
