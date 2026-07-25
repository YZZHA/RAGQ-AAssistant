import pytest
from src.memory.short_term import (
    create_session,
    get_history,
    add_message,
    delete_session,
    get_tenant_id,
)


SESSION_ID = "test_sess_001"


class TestCreateSession:
    def test_create_returns_true(self):
        result = create_session(SESSION_ID, "fdi_dept")
        assert result is True
        delete_session(SESSION_ID)

    def test_create_empty_history(self):
        create_session(SESSION_ID)
        history = get_history(SESSION_ID)
        assert history == []
        delete_session(SESSION_ID)


class TestAddMessage:
    def test_add_user_message(self):
        create_session(SESSION_ID)
        history = add_message(SESSION_ID, "user", "A产品和B产品的区别？")
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "A产品和B产品的区别？"
        delete_session(SESSION_ID)

    def test_add_assistant_message(self):
        create_session(SESSION_ID)
        add_message(SESSION_ID, "user", "你好")
        history = add_message(SESSION_ID, "assistant", "你好，有什么可以帮助？")
        assert len(history) == 2
        assert history[1]["role"] == "assistant"
        delete_session(SESSION_ID)


class TestGetHistory:
    def test_get_history_multiple_messages(self):
        create_session(SESSION_ID)
        add_message(SESSION_ID, "user", "Q1")
        add_message(SESSION_ID, "assistant", "A1")
        add_message(SESSION_ID, "user", "Q2")
        history = get_history(SESSION_ID)
        assert len(history) == 3
        delete_session(SESSION_ID)

    def test_get_non_existent_session_returns_empty(self):
        history = get_history("non_existent_session")
        assert history == []


class TestDeleteSession:
    def test_delete_clears_history(self):
        create_session(SESSION_ID)
        add_message(SESSION_ID, "user", "test")
        delete_session(SESSION_ID)
        history = get_history(SESSION_ID)
        assert history == []


class TestTenant:
    def test_get_tenant_id(self):
        create_session(SESSION_ID, "fdi_dept")
        tid = get_tenant_id(SESSION_ID)
        assert tid == "fdi_dept"
        delete_session(SESSION_ID)

    def test_get_tenant_id_non_existent(self):
        tid = get_tenant_id("non_existent")
        assert tid is None
