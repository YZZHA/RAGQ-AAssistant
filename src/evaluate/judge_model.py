# input:  DeepEval prompts, Qwen API
# output: LLM-as-Judge responses for evaluation
# pos:    评估层 → DeepEval 自定义 Judge 模型，替代默认 OpenAI

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI

from src.core.config import settings


class QwenJudgeModel(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def get_model_name(self) -> str:
        return "Qwen-Max (Judge)"

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
