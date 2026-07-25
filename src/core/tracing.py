# input:  trace events (rewrite/retrieval/rerank/generate), LangFuse config
# output: trace data sent to LangFuse (or no-op if disabled)
# pos:    核心层 → 可观测性埋点，默认关闭，开启后不阻塞主流程

from typing import Optional

from src.core.config import settings
from src.core.logging import logger


_tracer: Optional["Langfuse"] = None


def get_tracer():
    global _tracer
    if not settings.langfuse_enabled:
        return None
    if _tracer is None and settings.langfuse_secret_key:
        try:
            from langfuse import Langfuse
            _tracer = Langfuse(
                secret_key=settings.langfuse_secret_key,
                public_key=settings.langfuse_public_key,
                host=settings.langfuse_host,
            )
            logger.info("LangFuse 追踪已启用")
        except Exception as e:
            logger.warning("LangFuse 初始化失败: %s", e)
    return _tracer


def trace_event(name: str, **kwargs):
    tracer = get_tracer()
    if tracer is None:
        return None
    try:
        return tracer.trace(name=name, **kwargs)
    except Exception as e:
        logger.debug("LangFuse 追踪异常: %s", e)
        return None


def trace_generation(input: str, output: str, model: str, **kwargs):
    tracer = get_tracer()
    if tracer is None:
        return
    try:
        generation = tracer.generation(
            name="llm-call",
            model=model,
            input=input,
            output=output,
            **kwargs,
        )
        return generation
    except Exception:
        pass
