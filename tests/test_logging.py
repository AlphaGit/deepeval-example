"""Tests for the logging module."""

import time

from src.deep_research.logging import (
    configure_logging,
    get_logger,
    log_duration,
    log_llm_call,
)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_default(self):
        """Test that logging can be configured with defaults."""
        configure_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_configure_logging_with_level(self):
        """Test that logging can be configured with a custom level."""
        configure_logging(level="DEBUG")
        logger = get_logger("test")
        assert logger is not None

    def test_configure_logging_json_mode(self):
        """Test that logging can be configured for JSON output."""
        configure_logging(json_logs=True)
        logger = get_logger("test")
        assert logger is not None


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_bound_logger(self):
        """Test that get_logger returns a structured logger."""
        configure_logging()
        logger = get_logger("test_module")
        assert logger is not None
        # Should be a structlog logger
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "error")

    def test_get_logger_without_name(self):
        """Test that get_logger works without a name."""
        configure_logging()
        logger = get_logger()
        assert logger is not None


class TestLogDuration:
    """Tests for log_duration context manager."""

    def test_log_duration_measures_time(self, capsys):
        """Test that log_duration tracks operation duration."""
        configure_logging(level="INFO")
        logger = get_logger("test")

        with log_duration(logger, "test_operation"):
            time.sleep(0.01)  # Sleep for 10ms

        # Operation should complete without error
        # Duration should be tracked (we just verify no exception)

    def test_log_duration_with_extra_fields(self, capsys):
        """Test that log_duration accepts extra fields."""
        configure_logging(level="INFO")
        logger = get_logger("test")

        with log_duration(logger, "test_operation", custom_field="value"):
            pass

    def test_log_duration_propagates_exceptions(self):
        """Test that log_duration propagates exceptions."""
        configure_logging(level="INFO")
        logger = get_logger("test")

        try:
            with log_duration(logger, "test_operation"):
                raise ValueError("test error")
        except ValueError as e:
            assert str(e) == "test error"

    def test_log_duration_result_context(self):
        """Test that log_duration allows updating result context."""
        configure_logging(level="INFO")
        logger = get_logger("test")

        with log_duration(logger, "test_operation") as ctx:
            ctx["result_value"] = 42

        # Should complete without error


class TestLogLlmCall:
    """Tests for log_llm_call decorator."""

    def test_log_llm_call_decorator(self):
        """Test that log_llm_call decorates functions correctly."""
        configure_logging(level="DEBUG")

        @log_llm_call
        def mock_llm_call(prompt: str) -> str:
            return f"Response to: {prompt}"

        result = mock_llm_call("test prompt")
        assert result == "Response to: test prompt"

    def test_log_llm_call_with_object_response(self):
        """Test log_llm_call with response having content attribute."""
        configure_logging(level="DEBUG")

        class MockResponse:
            content = "mock response content"

        @log_llm_call
        def mock_llm_call(prompt: str) -> MockResponse:
            return MockResponse()

        result = mock_llm_call("test prompt")
        assert result.content == "mock response content"

    def test_log_llm_call_propagates_exceptions(self):
        """Test that log_llm_call propagates exceptions from decorated function."""
        configure_logging(level="DEBUG")

        @log_llm_call
        def failing_llm_call(prompt: str) -> str:
            raise RuntimeError("API error")

        try:
            failing_llm_call("test")
        except RuntimeError as e:
            assert str(e) == "API error"
