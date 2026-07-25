# input:  HTTP request (client IP)
# output: 429 or pass-through (FastAPI dependency)
# pos:    核心层 → 基于 IP 的令牌桶限流，防接口滥用

import time
from collections import defaultdict

from fastapi import HTTPException, Request


class TokenBucket:
    def __init__(self, rate: float = 10, capacity: int = 20):
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, dict] = defaultdict(lambda: {"tokens": capacity, "last": time.time()})

    def consume(self, key: str) -> bool:
        now = time.time()
        bucket = self._buckets[key]
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
        bucket["last"] = now
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False


rate_limiter = TokenBucket(rate=10, capacity=20)


async def rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.consume(client_ip):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")
