# input:  (none — package marker)
# output: generator module init
# pos:    生成层 → LLM调用包导入

from src.generator.llm import LLM
from src.generator.prompts import build_rag_messages

__all__ = ["LLM", "build_rag_messages"]
