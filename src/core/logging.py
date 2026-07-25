# input:  Settings.log_level
# output: 全局 logger 实例
# pos:    核心层 → 日志基础设施，被所有模块引用

import logging
import sys

from src.core.config import settings


def setup_logger(name: str = "rag_qa") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


logger = setup_logger()
