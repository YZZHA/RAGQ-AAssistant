# input:  Redis client, session_id, message history
# output: session history stored/retrieved (list of messages)
# pos:    记忆层 → 短时记忆，保存当前会话上下文

import json
import hashlib
from typing import List, Dict, Optional

from src.core.redis_helper import get_redis_client


SESSION_TTL = 3600  # 1 hour
MAX_HISTORY = 50  # max messages per session


def _key(session_id: str) -> str:
    return f"session:{session_id}"


def create_session(session_id: str, tenant_id: str = "default") -> bool:
    r = get_redis_client()
    key = _key(session_id)
    result = r.set(key, json.dumps([]), ex=SESSION_TTL)
    r.hset(f"{key}:meta", mapping={"tenant_id": tenant_id, "created_at": str(__import__("time").time())})
    return True


def get_history(session_id: str) -> List[Dict]:
    r = get_redis_client()
    raw = r.get(_key(session_id))
    if raw is None:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def add_message(session_id: str, role: str, content: str) -> List[Dict]:
    r = get_redis_client()
    key = _key(session_id)
    history = get_history(session_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    r.set(key, json.dumps(history), ex=SESSION_TTL)
    return history


def delete_session(session_id: str) -> bool:
    r = get_redis_client()
    key = _key(session_id)
    r.delete(key)
    r.delete(f"{key}:meta")
    return True


def get_tenant_id(session_id: str) -> Optional[str]:
    r = get_redis_client()
    meta_key = f"{_key(session_id)}:meta"
    tenant_id = r.hget(meta_key, "tenant_id")
    return tenant_id if tenant_id else None
