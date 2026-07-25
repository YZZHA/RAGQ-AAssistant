# input:  FastAPI Depends, core/config, retrieval, generator, memory
# output: 路由依赖注入函数
# pos:    API 层 → 组件实例化与注入，解耦路由和实现

import uuid

from src.core.logging import logger
from src.memory.short_term import create_session, get_history, add_message


def generate_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


def get_or_create_session(session_id: str, tenant_id: str = "default") -> str:
    if not session_id:
        session_id = generate_session_id()
        create_session(session_id, tenant_id)
        logger.info("新会话创建: %s", session_id)
    else:
        history = get_history(session_id)
        if not history:
            create_session(session_id, tenant_id)
    return session_id
