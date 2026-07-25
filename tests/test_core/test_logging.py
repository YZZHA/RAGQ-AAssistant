import logging

from src.core.logging import setup_logger


class TestLoggingBasic:
    def test_setup_logger_returns_logger(self):
        logger = setup_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_setup_logger_name(self):
        logger = setup_logger("test_name")
        assert logger.name == "test_name"

    def test_setup_logger_level(self):
        logger = setup_logger("test_level")
        assert logger.level == logging.DEBUG

    def test_setup_logger_handlers(self):
        logger = setup_logger("test_handlers")
        assert len(logger.handlers) >= 1


class TestLoggingSingleton:
    def test_same_logger_reuses_handlers(self):
        logger1 = setup_logger("singleton_test")
        handler_count = len(logger1.handlers)
        logger2 = setup_logger("singleton_test")
        assert len(logger2.handlers) == handler_count


class TestLoggingOutput:
    def test_logger_does_not_crash(self, caplog):
        logger = setup_logger("crash_test")
        logger.info("this is a test message")
        assert "this is a test message" in caplog.text
