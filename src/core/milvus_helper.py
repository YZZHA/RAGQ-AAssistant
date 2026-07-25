# input:  Settings.milvus_host / milvus_lite_path
# output: Milvus connection URI + auto-start embedded server
# pos:    核心层 → Milvus 连接管理，开发环境用嵌入式，生产环境用远程

import os
import atexit
import tempfile
import uuid
from pathlib import Path

from src.core.config import settings
from src.core.logging import logger


_milvus_uri: str | None = None
_milvus_data_dir: str | None = None


def get_milvus_uri() -> str:
    global _milvus_uri, _milvus_data_dir
    if _milvus_uri is not None:
        return _milvus_uri

    remote_uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
    try:
        from pymilvus import connections
        connections.connect(alias="_probe", host=settings.milvus_host, port=settings.milvus_port, timeout=2)
        connections.disconnect("_probe")
        _milvus_uri = remote_uri
        logger.info("Milvus 远程连接成功: %s", remote_uri)
        return _milvus_uri
    except Exception:
        logger.info("远程 Milvus 不可用，启动嵌入式 Milvus Lite")

    try:
        from milvus_lite import server_manager_instance
        _milvus_data_dir = tempfile.mkdtemp(prefix="milvus_data_")
        _milvus_uri = server_manager_instance.start_and_get_uri(_milvus_data_dir)

        def _cleanup():
            try:
                server_manager_instance.release_server(_milvus_data_dir)
                import shutil
                shutil.rmtree(_milvus_data_dir, ignore_errors=True)
            except Exception:
                pass

        atexit.register(_cleanup)
        logger.info("Milvus Lite 已启动: %s", _milvus_uri)
    except Exception as e:
        logger.error("Milvus Lite 启动失败: %s", e)
        raise

    return _milvus_uri
