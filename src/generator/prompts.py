# input:  prompt template name + variables (context, question, history)
# output: compiled prompt string
# pos:    生成层 → Prompt 模板管理与组装

from typing import List, Dict, Optional


RAG_SYSTEM_PROMPT = """你是一个FDI（外商直接投资）部门的专业业务助手。
请基于提供的知识文档内容回答用户问题。

回答规则:
1. 仅基于提供的文档内容回答，不要编造信息
2. 如果文档内容不足以回答问题，请明确说明"文档中未找到相关信息"
3. 引用相关文档时，在答案末尾标注来源文档名称
4. 使用清晰的中文，分段组织答案
5. 涉及数字、金额、日期等信息时，确保与文档完全一致"""


def build_rag_messages(
    question: str,
    context_chunks: List[Dict],
    history: Optional[List[Dict]] = None,
) -> List[Dict]:
    context_parts = []
    for i, c in enumerate(context_chunks, 1):
        title = c.get("doc_title", c.get("doc_id", "未知"))
        context_parts.append(f"[{i}] 来源: {title}\n{c.get('content', '')}")

    context_text = "\n\n".join(context_parts)

    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

    if history:
        for msg in history[-6:]:
            messages.append(msg)

    user_content = f"参考文档:\n{context_text}\n\n问题: {question}"
    messages.append({"role": "user", "content": user_content})

    return messages
