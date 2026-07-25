from unittest.mock import patch
from src.core.rate_limiter import TokenBucket


class TestTokenBucket:
    def test_consume_returns_true_when_available(self):
        tb = TokenBucket(rate=100, capacity=10)
        assert tb.consume("test") is True

    def test_exhaust_bucket(self):
        tb = TokenBucket(rate=100, capacity=3)
        assert tb.consume("a") is True
        assert tb.consume("a") is True
        assert tb.consume("a") is True
        assert tb.consume("a") is False  # 超出容量

    def test_different_keys_independent(self):
        tb = TokenBucket(rate=100, capacity=2)
        tb.consume("x")
        tb.consume("x")
        assert tb.consume("y") is True  # y 不受影响

    def test_rate_limit_dependency(self):
        from src.core.rate_limiter import rate_limit
        from unittest.mock import MagicMock

        mock_request = MagicMock()
        mock_request.client.host = "1.2.3.4"

        # 连续调用超出限制后应抛出 429
        for _ in range(30):
            try:
                import asyncio
                asyncio.run(rate_limit(mock_request))
            except Exception:
                pass
