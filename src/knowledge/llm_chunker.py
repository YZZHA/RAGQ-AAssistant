# input:  document text + doc_id, Qwen API
# output: chunk list from LLM with heading_chain, or None (fallback to regex)
# pos:    知识生产 → LLM 语义切分，结果缓存到文件，避免重复调用

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI

from src.core.config import settings
from src.core.logging import logger


CACHE_PATH = Path("config/chunk_cache.json")

LLM_CHUNK_PROMPT = """你是一个文档切分助手。请将以下文档按语义边界拆分为独立的段落。

要求:
1. 每个段落是一个完整、独立的信息单元
2. 在自然语义边界处切分（主题切换、新概念引入等）
3. 保留原标题层级信息（# 标题, ## 子标题, ### 子子标题）
4. 每个段落 200-800 字，太长的段落需要拆分，太短的段落合并到相邻段落
5. 仅输出 JSON 数组，不要包含其他文字

输出格式:
{heading_chain_example}

文档：
{document_text}

JSON："""

LLM_CHUNK_PROMPT_EXAMPLE = """[
  {"heading_chain": ["标题1"], "content": "段落内容"},
  {"heading_chain": ["标题1", "子标题"], "content": "段落内容"}
]"""


def llm_chunk_document(text: str, doc_id: str, force_refresh: bool = False) -> Optional[List[Dict]]:
    text_hash = _md5(text)

    if not force_refresh:
        cached = _read_cache(doc_id, text_hash)
        if cached is not None:
            logger.info("Chunk 缓存命中: %s", doc_id)
            return cached

    if not settings.qwen_api_key:
        logger.warning("Qwen API Key 未设置，使用正则切分: %s", doc_id)
        return None

    prompt = LLM_CHUNK_PROMPT.format(document_text=text[:6000], heading_chain_example=LLM_CHUNK_PROMPT_EXAMPLE)

    try:
        client = OpenAI(api_key=settings.qwen_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        resp = client.chat.completions.create(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
        )
        raw = resp.choices[0].message.content or ""
        chunks = _parse_json_to_chunks(raw, doc_id)
        if chunks is None:
            logger.warning("LLM 返回格式异常，降级到正则: %s", doc_id)
            return None

        for i, c in enumerate(chunks):
            c["chunk_id"] = f"{doc_id}_llm_{i + 1:03d}"
            c["doc_id"] = doc_id
            c["parent_chunk_id"] = ""
            c["char_count"] = len(c["content"])
            c["is_parent"] = True

        _write_cache(doc_id, text_hash, chunks)
        logger.info("LLM chunk 完成: %s → %d chunks", doc_id, len(chunks))
        return chunks

    except Exception as e:
        logger.warning("LLM 切分失败，降级到正则: %s — %s", doc_id, e)
        return None


def _parse_json_to_chunks(raw: str, doc_id: str) -> Optional[List[Dict]]:
    raw = raw.strip()
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return None
        for item in items:
            if "content" not in item:
                return None
            if "heading_chain" not in item:
                item["heading_chain"] = []
        return items
    except json.JSONDecodeError:
        return None


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _read_cache(doc_id: str, text_hash: str) -> Optional[List[Dict]]:
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text("utf-8"))
        doc_hashes = data.get("doc_hashes", {})
        if doc_hashes.get(doc_id) != text_hash:
            return None
        chunks = data.get("chunks", {}).get(doc_id)
        return chunks if chunks else None
    except (json.JSONDecodeError, KeyError):
        return None


def _write_cache(doc_id: str, text_hash: str, chunks: List[Dict]):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        data = json.loads(CACHE_PATH.read_text("utf-8"))
    else:
        data = {"chunk_cache_version": 1, "doc_hashes": {}, "chunks": {}}
    data["doc_hashes"][doc_id] = text_hash
    data["chunks"][doc_id] = chunks
    CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def rebuild_all_cache():
    from src.ingestion.loader import load_directory
    from src.ingestion.chunker import chunk_document as regex_chunk

    docs = load_directory("data/raw")
    for text, meta in docs:
        doc_id = meta["doc_id"]
        result = llm_chunk_document(text, doc_id)
        if result is None:
            logger.info("缓存正则结果（LLM 不可用）: %s", doc_id)
            chunks = regex_chunk(text, doc_id)
            _write_cache(doc_id, _md5(text), chunks)
    logger.info("Chunk 缓存重建完成")
