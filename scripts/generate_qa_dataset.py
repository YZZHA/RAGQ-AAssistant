"""
FDI RAG 合成问答数据集生成工具

基于文档目录，调用 LLM 自动生成 Q&A 评估数据集。
支持 OpenAI / Qwen / Ollama 三种后端（兼容 OpenAI API 格式）。

用法:
  python scripts/generate_qa_dataset.py --source-dir data/raw --output data/qa_dataset/eval.jsonl
  python scripts/generate_qa_dataset.py --source-dir data/raw --backend qwen --model qwen-max
  python scripts/generate_qa_dataset.py --source-dir data/raw --backend ollama --model qwen2.5:7b --base-url http://localhost:11434/v1
  python scripts/generate_qa_dataset.py --source-dir data/raw --questions-per-chunk 5 --max-chunks 50
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional


def chunk_markdown(text: str, doc_id: str, min_chars: int = 200, max_chars: int = 800) -> List[Dict]:
    """将 Markdown 文档按标题层级切分为独立 chunk。"""
    lines = text.split("\n")
    chunks = []
    current_sections = []  # 当前标题层级栈
    current_content = []
    in_front_matter = True

    def flush():
        if not current_content:
            return
        content = "\n".join(current_content).strip()
        if len(content) < min_chars:
            return
        heading = " > ".join(current_sections) if current_sections else doc_id
        chunks.append({
            "doc_id": doc_id,
            "heading": heading,
            "content": content,
            "char_count": len(content),
        })

    for line in lines:
        stripped = line.strip()
        if in_front_matter:
            if stripped.startswith("---"):
                in_front_matter = False
            continue

        if stripped.startswith("## "):
            flush()
            current_content = [line]
            current_sections = [stripped.lstrip("#").strip()]
        elif stripped.startswith("### "):
            flush()
            current_content = [line]
            if current_sections:
                current_sections = current_sections[:1] + [stripped.lstrip("#").strip()]
            else:
                current_sections = [stripped.lstrip("#").strip()]
        else:
            current_content.append(line)

    flush()

    result = []
    for chunk in chunks:
        if chunk["char_count"] > max_chars:
            sub_chunks = _split_long_chunk(chunk, max_chars)
            result.extend(sub_chunks)
        else:
            result.append(chunk)

    return result


def _split_long_chunk(chunk: Dict, max_chars: int) -> List[Dict]:
    """将超长 chunk 按空行进一步拆分。"""
    paragraphs = re.split(r"\n\n+", chunk["content"])
    sub_chunks = []
    buffer = []
    buffer_len = 0

    for para in paragraphs:
        para_len = len(para)
        if not para.strip():
            continue
        if buffer_len + para_len > max_chars and buffer:
            sub_chunks.append({
                "doc_id": chunk["doc_id"],
                "heading": chunk["heading"],
                "content": "\n\n".join(buffer).strip(),
                "char_count": buffer_len,
            })
            buffer = [para]
            buffer_len = para_len
        else:
            buffer.append(para)
            buffer_len += para_len

    if buffer:
        sub_chunks.append({
            "doc_id": chunk["doc_id"],
            "heading": chunk["heading"],
            "content": "\n\n".join(buffer).strip(),
            "char_count": buffer_len,
        })

    return sub_chunks


def load_documents(source_dir: str) -> List[Dict]:
    """加载目录下的所有 .md/.txt 文件。"""
    docs = []
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"[错误] 目录不存在: {source_dir}")
        sys.exit(1)

    for file_path in sorted(source_path.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        doc_id = file_path.stem
        docs.append({"doc_id": doc_id, "filename": file_path.name, "text": text})
        print(f"  ✓ {file_path.name} ({len(text)} chars)")

    for file_path in sorted(source_path.glob("*.txt")):
        if file_path.suffix == ".txt":
            text = file_path.read_text(encoding="utf-8")
            doc_id = file_path.stem
            docs.append({"doc_id": doc_id, "filename": file_path.name, "text": text})
            print(f"  ✓ {file_path.name} ({len(text)} chars)")

    return docs


def build_prompt(chunk_content: str, questions_per_chunk: int) -> str:
    """构建生成 Q&A 的 Prompt。"""
    return f"""你是一个FDI（外商直接投资）部门的业务专家。以下是产品文档中的一段内容。

请基于这段文档内容，生成 {questions_per_chunk} 个部门员工可能会问的问题和对应的标准答案。

要求：
1. 问题必须仅基于文档内容回答，不要引入外部知识
2. 问题尽量覆盖不同类型：事实查询、操作步骤、规则询问等
3. 答案要准确完整，包含关键细节（数字、条件、例外情况等）
4. 严格使用 JSON 格式输出，不要包含其他文字

文档内容：
{chunk_content}

输出格式（严格 JSON 数组，不要包含其他文字）：
[
  {{"question": "问题内容", "expected_answer": "标准答案"}},
  {{"question": "问题内容", "expected_answer": "标准答案"}}
]
"""


def call_llm(prompt: str, backend: str, model: str, api_key: str, base_url: str, max_retries: int = 3) -> Optional[str]:
    """调用 LLM 生成问答对。"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个FDI业务专家助手，严格按照指令输出JSON格式数据。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [重试 {attempt + 1}/{max_retries}] {e}，{wait}秒后重试...")
            time.sleep(wait)

    return None


def parse_qa_response(response_text: str) -> List[Dict]:
    """从 LLM 响应中解析 Q&A 列表。"""
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        items = json.loads(text)
        if isinstance(items, list):
            return items
        return []
    except json.JSONDecodeError:
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []


def main():
    parser = argparse.ArgumentParser(description="FDI RAG 合成问答数据集生成工具")
    parser.add_argument("--source-dir", default="data/raw", help="文档目录路径（默认: data/raw）")
    parser.add_argument("--output", default="data/qa_dataset/eval.jsonl", help="输出文件路径（默认: data/qa_dataset/eval.jsonl）")
    parser.add_argument("--backend", default="openai", choices=["openai", "qwen", "ollama"], help="LLM 后端（默认: openai）")
    parser.add_argument("--model", default=None, help="模型名称（默认: openai=gpt-4o-mini, qwen=qwen-max, ollama=qwen2.5:7b）")
    parser.add_argument("--base-url", default=None, help="API 地址（默认: openai=https://api.openai.com/v1, qwen=https://dashscope.aliyuncs.com/compatible-mode/v1, ollama=http://localhost:11434/v1）")
    parser.add_argument("--questions-per-chunk", type=int, default=3, help="每个 chunk 生成的问题数（默认: 3）")
    parser.add_argument("--max-chunks", type=int, default=0, help="最多处理的 chunk 数（0=不限，默认: 0）")
    parser.add_argument("--api-key", default=None, help="API Key（默认从环境变量读取）")
    args = parser.parse_args()

    backend_configs = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "env_key": "OPENAI_API_KEY",
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-max",
            "env_key": "QWEN_API_KEY",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5:7b",
            "env_key": "",
        },
    }

    config = backend_configs[args.backend]
    base_url = args.base_url or config["base_url"]
    model = args.model or config["model"]
    api_key = args.api_key or os.environ.get(config["env_key"], "")

    if args.backend != "ollama" and not api_key:
        print(f"[错误] 请设置 {config['env_key']} 环境变量或使用 --api-key 参数")
        print(f"  提示: 也可以使用 --backend ollama 调用本地模型")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  FDI RAG 合成数据集生成工具")
    print(f"{'='*60}")
    print(f"  文档目录: {args.source_dir}")
    print(f"  输出文件: {args.output}")
    print(f"  LLM 后端: {args.backend}")
    print(f"  模型名称: {model}")
    print(f"  API 地址: {base_url}")
    print(f"  每chunk问题数: {args.questions_per_chunk}")
    print(f"{'='*60}\n")

    print("[1/4] 加载文档...")
    docs = load_documents(args.source_dir)
    if not docs:
        print("[错误] 未找到任何文档，请检查目录路径")
        sys.exit(1)
    print(f"  共加载 {len(docs)} 篇文档\n")

    print("[2/4] 文档分块...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_markdown(doc["text"], doc["doc_id"])
        all_chunks.extend(chunks)
        print(f"  {doc['filename']}: {len(chunks)} 个 chunk")

    if args.max_chunks > 0:
        all_chunks = all_chunks[:args.max_chunks]

    print(f"  总计 {len(all_chunks)} 个 chunk\n")

    print("[3/4] LLM 生成 Q&A 对...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_questions = 0
    failed_chunks = 0

    for i, chunk in enumerate(all_chunks):
        doc_id = chunk["doc_id"]
        heading = chunk["heading"]
        content = chunk["content"]
        char_count = chunk["char_count"]

        print(f"  [{i+1}/{len(all_chunks)}] {doc_id} > {heading} ({char_count} chars)")

        prompt = build_prompt(content, args.questions_per_chunk)
        response = call_llm(prompt, args.backend, model, api_key, base_url)

        if response is None:
            print(f"    ✗ 生成失败（已达最大重试次数）")
            failed_chunks += 1
            continue

        qa_pairs = parse_qa_response(response)

        if not qa_pairs:
            print(f"    ✗ 解析响应失败")
            print(f"    原始响应: {response[:200]}...")
            failed_chunks += 1
            continue

        with open(output_path, "a", encoding="utf-8") as f:
            for qa in qa_pairs:
                question = qa.get("question", "").strip()
                expected_answer = qa.get("expected_answer", "").strip()
                if question and expected_answer:
                    record = {
                        "question": question,
                        "relevant_docs": [doc_id],
                        "expected_answer": expected_answer,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_questions += 1

        print(f"    ✓ 生成 {len(qa_pairs)} 个问题")
        time.sleep(0.5)

    print(f"\n[4/4] 完成")
    print(f"  成功生成: {total_questions} 个 Q&A 对")
    print(f"  失败 chunk: {failed_chunks} 个")
    print(f"  输出文件: {output_path.resolve()}")
    print(f"  查看内容: type {output_path}")
    print()

    if total_questions == 0:
        print("[警告] 没有生成任何 Q&A 对，请检查 LLM 配置是否正确。")
        print("  常见问题:")
        print("  - API Key 未设置或无效")
        print("  - 模型名称不正确")
        print("  - 网络连接问题")
        print()


if __name__ == "__main__":
    main()
