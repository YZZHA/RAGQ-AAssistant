# input:  raw text, doc_id, chunking config
# output: list of chunk dicts with heading tree + parent-child linkage
# pos:    摄入管道 → 文本切分器，支持 H1/H2/H3 标题树 + 合并/拆分 + LLM 语义切分降级

import re
from typing import List, Dict

HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def chunk_document(
    text: str,
    doc_id: str,
    parent_max: int = 1200,
    parent_min: int = 600,
    child_max: int = 400,
    child_min: int = 150,
    overlap: int = 50,
    use_llm: bool = True,
) -> List[Dict]:
    text = text.strip()
    if not text:
        return []

    if use_llm and not doc_id.startswith("auto_"):
        from src.knowledge.llm_chunker import llm_chunk_document
        result = llm_chunk_document(text, doc_id)
        if result is not None:
            return result

    return _regex_chunk(text, doc_id, parent_max, child_min, child_max, overlap)


def _regex_chunk(text, doc_id, parent_max=1200, child_min=150, child_max=400, overlap=50):
    raw_sections = _split_by_all_headings(text)
    chunks = _sections_to_chunks(raw_sections, doc_id, parent_max, child_max)
    chunks = _merge_short_chunks(chunks, child_min)
    chunks = _split_long_chunks(chunks, parent_max)
    chunks = _apply_overlap(chunks, overlap)
    return chunks


def _split_by_all_headings(text: str) -> List[Dict]:
    """按 # / ## / ### 切分，追踪标题层级"""
    lines = text.split("\n")
    sections = []
    current_heading = []
    current_lines = []

    for line in lines:
        m = HEADING_PATTERN.match(line)
        if m:
            if current_lines:
                sections.append({
                    "heading_chain": list(current_heading),
                    "content": "\n".join(current_lines).strip(),
                })
                current_lines = []

            level = len(m.group(1))
            title = m.group(2).strip()
            current_heading = current_heading[:level - 1] + [title]
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "heading_chain": list(current_heading),
            "content": "\n".join(current_lines).strip(),
        })

    return [s for s in sections if s["content"]]


def _sections_to_chunks(
    sections: List[Dict], doc_id: str, parent_max: int, child_max: int
) -> List[Dict]:
    chunks = []
    for i, sec in enumerate(sections):
        content = sec["content"]
        char_count = len(content)
        chunk_id = f"{doc_id}_p_{i + 1:03d}"
        heading_chain = sec["heading_chain"]

        if char_count <= parent_max:
            chunks.append(_make_chunk(chunk_id, doc_id, "", content, char_count, True, heading_chain))
        else:
            sub_parts = _split_by_subheadings(content)
            buffer = []
            buf_len = 0
            sub_seq = 0

            for part in sub_parts:
                pl = len(part)
                if buf_len + pl > child_max and buffer:
                    sub_seq += 1
                    cid = f"{doc_id}_c_{i + 1:03d}_{sub_seq:03d}"
                    chunks.append(_make_chunk(cid, doc_id, chunk_id, "\n".join(buffer).strip(), buf_len, False, heading_chain))
                    buffer = [part]
                    buf_len = pl
                else:
                    buffer.append(part)
                    buf_len += pl

            if buffer:
                sub_seq += 1
                cid = f"{doc_id}_c_{i + 1:03d}_{sub_seq:03d}"
                chunks.append(_make_chunk(cid, doc_id, chunk_id, "\n".join(buffer).strip(), buf_len, False, heading_chain))

    return chunks


def _merge_short_chunks(chunks: List[Dict], min_chars: int) -> List[Dict]:
    if not chunks:
        return []
    result = [chunks[0]]
    for c in chunks[1:]:
        if not c["is_parent"] and c["char_count"] < min_chars and result:
            prev = result[-1]
            prev["content"] += "\n" + c["content"]
            prev["char_count"] += c["char_count"]
        else:
            result.append(c)
    return result


def _split_long_chunks(chunks: List[Dict], max_chars: int) -> List[Dict]:
    result = []
    for c in chunks:
        if c["char_count"] > max_chars:
            parts = _split_text(c["content"], max_chars)
            for part in parts:
                new_c = dict(c)
                new_c["content"] = part
                new_c["char_count"] = len(part)
                result.append(new_c)
        else:
            result.append(c)
    return result


def _split_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    for sep in ["\n\n", "。", "，"]:
        if sep in text:
            parts = []
            for p in text.split(sep):
                p = p.strip()
                if p:
                    parts.append(p + sep if sep != "\n\n" else p)
            merged = []
            buf = ""
            for p in parts:
                if len(buf) + len(p) > max_chars and buf:
                    merged.append(buf.strip())
                    buf = p
                else:
                    buf += p
            if buf:
                merged.append(buf.strip())
            if len(merged) > 1:
                return merged

    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _apply_overlap(chunks: List[Dict], overlap_chars: int) -> List[Dict]:
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        if prev["char_count"] > overlap_chars:
            tail = prev["content"][-overlap_chars:]
            chunks[i]["content"] = tail + "\n" + chunks[i]["content"]
            chunks[i]["char_count"] = len(chunks[i]["content"])
    return chunks


def _split_by_subheadings(text: str) -> List[str]:
    parts = re.split(r"\n(?=### )", text)
    result = []
    for part in parts:
        trimmed = part.strip()
        if not trimmed:
            continue
        if len(trimmed) > 400:
            paragraphs = [p.strip() for p in trimmed.split("\n\n") if p.strip()]
            result.extend(paragraphs)
        else:
            result.append(trimmed)
    return result


def _make_chunk(
    chunk_id: str, doc_id: str, parent_id: str,
    content: str, char_count: int, is_parent: bool,
    heading_chain: List[str],
) -> Dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "parent_chunk_id": parent_id if not is_parent else "",
        "content": content,
        "char_count": char_count,
        "is_parent": is_parent,
        "heading_chain": heading_chain,
    }
