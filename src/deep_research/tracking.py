"""MLflow tracking integration for the deep research agent."""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlflow
import mlflow.langchain

from .logging import get_logger

logger = get_logger("tracking")

# Track whether tracing has been enabled
_tracing_enabled = False


@dataclass
class EvaluationResult:
    """Result from running an evaluator."""

    name: str
    score: float
    passed: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    """Container for a research run result with evaluation data."""

    question: str
    report: str
    iteration_count: int
    section_count: int
    report_length: int
    web_search_count: int = 0
    llm_search_count: int = 0
    total_sources: int = 0
    evaluations: list[EvaluationResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def configure_tracking(
    experiment_name: str = "deep-research",
    tracking_uri: str | None = None,
    enable_tracing: bool = True,
) -> None:
    """Configure MLflow tracking and tracing.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: MLflow tracking server URI. Defaults to MLFLOW_TRACKING_URI
                     env var or SQLite database.
        enable_tracing: Whether to enable LangChain tracing for LLM calls.
    """
    global _tracing_enabled

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    # Enable LangChain autologging for tracing
    if enable_tracing and not _tracing_enabled:
        mlflow.langchain.autolog(
            log_traces=True,  # Enable trace logging
            silent=True,  # Suppress MLflow warnings during autologging
        )
        _tracing_enabled = True
        logger.info("tracing_enabled", backend="langchain")

    logger.info(
        "tracking_configured",
        experiment=experiment_name,
        tracking_uri=uri,
        tracing_enabled=enable_tracing,
    )


@contextmanager
def track_research_run(
    question: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
):
    """Context manager for tracking a research run in MLflow.

    Args:
        question: The research question being investigated.
        run_name: Optional name for the MLflow run.
        tags: Optional tags to add to the run.

    Yields:
        ResearchResult object to populate with results and evaluations.
    """
    result = ResearchResult(
        question=question,
        report="",
        iteration_count=0,
        section_count=0,
        report_length=0,
    )

    with mlflow.start_run(run_name=run_name) as run:
        logger.info("tracking_run_started", run_id=run.info.run_id)

        # Log input parameters
        mlflow.log_param("question", question[:250])  # MLflow param limit
        mlflow.log_param("question_length", len(question))

        if tags:
            mlflow.set_tags(tags)

        try:
            yield result

            # Log research metrics
            mlflow.log_metric("iteration_count", result.iteration_count)
            mlflow.log_metric("section_count", result.section_count)
            mlflow.log_metric("report_length", result.report_length)
            mlflow.log_metric("web_search_count", result.web_search_count)
            mlflow.log_metric("llm_search_count", result.llm_search_count)
            mlflow.log_metric("total_sources", result.total_sources)

            # Calculate web search usage ratio
            if result.section_count > 0:
                web_search_ratio = result.web_search_count / result.section_count
                mlflow.log_metric("web_search_ratio", web_search_ratio)

            # Log evaluation results
            for eval_result in result.evaluations:
                mlflow.log_metric(f"eval_{eval_result.name}_score", eval_result.score)
                mlflow.log_metric(
                    f"eval_{eval_result.name}_passed",
                    1.0 if eval_result.passed else 0.0,
                )

                # Log evaluation metadata
                for key, value in eval_result.metadata.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"eval_{eval_result.name}_{key}", value)
                    elif isinstance(value, str):
                        # Log string metadata as params (truncate to MLflow limit)
                        mlflow.log_param(f"eval_{eval_result.name}_{key}", value[:250])
                    elif isinstance(value, (list, dict)):
                        # Log complex types as JSON params
                        import json

                        mlflow.log_param(
                            f"eval_{eval_result.name}_{key}",
                            json.dumps(value)[:250],
                        )

            # Log artifacts
            if result.report:
                mlflow.log_text(result.report, "report.md")
                mlflow.log_text(result.question, "question.txt")

            # Log any additional metadata
            for key, value in result.metadata.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                elif isinstance(value, str):
                    mlflow.log_param(key, value[:250])

            # Calculate overall pass rate
            if result.evaluations:
                pass_rate = sum(1 for e in result.evaluations if e.passed) / len(
                    result.evaluations
                )
                mlflow.log_metric("eval_pass_rate", pass_rate)

            logger.info(
                "tracking_run_completed",
                run_id=run.info.run_id,
                evaluation_count=len(result.evaluations),
            )

        except Exception as e:
            mlflow.log_param("error", str(e)[:250])
            mlflow.set_tag("status", "failed")
            logger.error(
                "tracking_run_failed",
                run_id=run.info.run_id,
                error=str(e),
            )
            raise


def log_evaluation(
    result: ResearchResult,
    name: str,
    score: float,
    passed: bool,
    **metadata: Any,
) -> EvaluationResult:
    """Log an evaluation result to a research result.

    Args:
        result: The ResearchResult to add the evaluation to.
        name: Name of the evaluator.
        score: Numeric score from the evaluator (0.0 to 1.0 recommended).
        passed: Whether the evaluation passed.
        **metadata: Additional metadata to log.

    Returns:
        The created EvaluationResult.
    """
    eval_result = EvaluationResult(
        name=name,
        score=score,
        passed=passed,
        metadata=metadata,
    )
    result.evaluations.append(eval_result)

    logger.info(
        "evaluation_logged",
        evaluator=name,
        score=score,
        passed=passed,
    )

    return eval_result


def disable_tracing() -> None:
    """Disable MLflow LangChain tracing.

    Call this if you want to disable automatic tracing of LLM calls.
    """
    global _tracing_enabled

    if _tracing_enabled:
        mlflow.langchain.autolog(disable=True)
        _tracing_enabled = False
        logger.info("tracing_disabled")


def is_tracing_enabled() -> bool:
    """Check if MLflow tracing is currently enabled.

    Returns:
        True if tracing is enabled, False otherwise.
    """
    return _tracing_enabled
