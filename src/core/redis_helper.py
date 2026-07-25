# input:  Settings.redis_url / redis_use_fake
# output: Redis client (real or fake)
# pos:    核心层 → Redis 连接管理，开发环境用 fakeredis，生产环境用真实 Redis

from src.core.config import settings
from src.core.logging import logger


_redis_client = None


def get_redis_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    if settings.redis_use_fake:
        try:
            import fakeredis
            _redis_client = fakeredis.FakeRedis(decode_responses=True)
            logger.info("Redis: 使用 fakeredis（内存模拟）")
            return _redis_client
        except ImportError:
            logger.warning("fakeredis 未安装，尝试连接真实 Redis")

    try:
        import redis as r
        _redis_client = r.from_url(settings.redis_url, decode_responses=True)
        _redis_client.ping()
        logger.info("Redis: 真实连接成功 %s", settings.redis_url)
    except Exception as e:
        logger.warning("Redis 连接失败 (%s)，降级为 fakeredis", e)
        import fakeredis
        _redis_client = fakeredis.FakeRedis(decode_responses=True)

    return _redis_client
