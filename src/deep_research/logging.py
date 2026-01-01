"""Structured logging configuration for the deep research agent."""

import logging
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any

import structlog


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_logs: If True, output JSON logs; otherwise, use colored console output.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors for all output types
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        # JSON output for production/parsing
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Pretty console output for development
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(
                    colors=True,
                    exception_formatter=structlog.dev.plain_traceback,
                ),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Optional logger name (typically __name__).

    Returns:
        A configured structured logger.
    """
    return structlog.get_logger(name)


@contextmanager
def log_duration(logger: Any, event: str, **extra_fields):
    """Context manager to log the duration of an operation.

    Args:
        logger: The logger instance to use.
        event: The event name to log.
        **extra_fields: Additional fields to include in the log.

    Yields:
        A dict that can be updated with additional fields before the end log.
    """
    start_time = time.perf_counter()
    context = {"status": "started", **extra_fields}
    logger.info(event, **context)

    result_fields: dict = {}
    try:
        yield result_fields
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            event,
            status="completed",
            duration_ms=round(duration_ms, 2),
            **extra_fields,
            **result_fields,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            event,
            status="failed",
            duration_ms=round(duration_ms, 2),
            error=str(e),
            error_type=type(e).__name__,
            **extra_fields,
        )
        raise


def log_llm_call(func):
    """Decorator to log LLM API calls with timing.

    Logs the start, completion, and any errors during LLM invocations.
    """
    logger = get_logger("llm")

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to extract meaningful context from args
        call_context = {}
        if args:
            first_arg = args[0]
            if isinstance(first_arg, str):
                # Truncate long prompts for logging
                call_context["prompt_preview"] = (
                    first_arg[:100] + "..." if len(first_arg) > 100 else first_arg
                )
                call_context["prompt_length"] = len(first_arg)

        start_time = time.perf_counter()
        logger.debug(
            "llm_call", status="started", function=func.__name__, **call_context
        )

        try:
            result = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log response details
            response_context = {"duration_ms": round(duration_ms, 2)}
            if hasattr(result, "content"):
                response_context["response_length"] = len(result.content)

            logger.debug(
                "llm_call",
                status="completed",
                function=func.__name__,
                **response_context,
            )
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "llm_call",
                status="failed",
                function=func.__name__,
                duration_ms=round(duration_ms, 2),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    return wrapper
