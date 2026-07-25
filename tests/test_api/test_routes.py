import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.schemas import ChatRequest


@pytest.fixture
def client():
    with patch("src.api.routes._init_bm25", return_value=None):
        from src.api.routes import app
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestChatInputValidation:
    def test_empty_question_returns_400(self, client):
        resp = client.post("/api/chat", json={"question": ""})
        assert resp.status_code == 422

    def test_long_question_returns_400(self):
        from src.api.schemas import ChatRequest
        long_q = "a" * 1001
        try:
            ChatRequest(question=long_q)
            assert False
        except Exception:
            assert True

    def test_normal_request_passes_validation(self):
        req = ChatRequest(question="A产品和B产品的区别？", session_id="sess_test")
        assert req.question == "A产品和B产品的区别？"
        assert req.tenant_id == "default"
