"""Integrated research evaluation with MLflow tracking."""

from .agent import run_research
from .evaluators import Evaluator, get_default_evaluators, run_evaluators
from .logging import get_logger
from .tracking import (
    ResearchResult,
    configure_tracking,
    log_evaluation,
    track_research_run,
)

logger = get_logger("evaluate")


def run_research_with_evaluation(
    question: str,
    max_iterations: int = 2,
    evaluators: list[Evaluator] | None = None,
    include_llm_evaluators: bool = False,
    experiment_name: str = "deep-research",
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    enable_tracing: bool = True,
) -> ResearchResult:
    """Run research with automatic evaluation and MLflow tracking.

    This is the main entry point for running research with full evaluation
    and tracking capabilities. LLM traces are automatically captured and
    stored in MLflow.

    Args:
        question: The research question to investigate.
        max_iterations: Maximum research iterations.
        evaluators: Custom evaluators to use. If None, uses default evaluators.
        include_llm_evaluators: Whether to include LLM-based evaluators in defaults.
        experiment_name: MLflow experiment name.
        run_name: Optional name for the MLflow run.
        tags: Optional tags to add to the run.
        enable_tracing: Whether to enable LLM tracing (default True).

    Returns:
        ResearchResult with report and all evaluation results.

    Example:
        >>> from src.deep_research import run_research_with_evaluation
        >>> result = run_research_with_evaluation(
        ...     "What are the benefits of renewable energy?",
        ...     include_llm_evaluators=True,
        ... )
        >>> print(f"Report length: {result.report_length}")
        >>> for eval in result.evaluations:
        ...     print(f"{eval.name}: {eval.score:.2f} ({'PASS' if eval.passed else 'FAIL'})")

    To view traces, run: mlflow ui
    """
    # Configure tracking with tracing enabled
    configure_tracking(experiment_name=experiment_name, enable_tracing=enable_tracing)

    # Get evaluators
    if evaluators is None:
        evaluators = get_default_evaluators(include_llm=include_llm_evaluators)

    logger.info(
        "research_with_evaluation_start",
        question_length=len(question),
        evaluator_count=len(evaluators),
    )

    with track_research_run(question, run_name=run_name, tags=tags) as result:
        # Run the research
        report = run_research(question, max_iterations=max_iterations)

        # Populate result
        result.report = report
        result.report_length = len(report)
        result.metadata["max_iterations"] = max_iterations

        # Run evaluators
        eval_results = run_evaluators(question, report, evaluators)

        # Log each evaluation
        for evaluator, eval_result in zip(evaluators, eval_results, strict=True):
            log_evaluation(
                result,
                name=evaluator.name,
                score=eval_result.score,
                passed=eval_result.passed,
                reason=eval_result.reason,
                **eval_result.metadata,
            )

        logger.info(
            "research_with_evaluation_complete",
            report_length=result.report_length,
            evaluations_passed=sum(1 for e in result.evaluations if e.passed),
            evaluations_total=len(result.evaluations),
        )

    return result


def evaluate_report(
    question: str,
    report: str,
    evaluators: list[Evaluator] | None = None,
    include_llm_evaluators: bool = False,
    track: bool = True,
    experiment_name: str = "deep-research-evaluation",
    enable_tracing: bool = True,
) -> ResearchResult:
    """Evaluate an existing research report.

    Use this function when you already have a report and want to evaluate it
    without running the research agent again.

    Args:
        question: The original research question.
        report: The research report to evaluate.
        evaluators: Custom evaluators to use. If None, uses default evaluators.
        include_llm_evaluators: Whether to include LLM-based evaluators in defaults.
        track: Whether to track the evaluation in MLflow.
        experiment_name: MLflow experiment name.
        enable_tracing: Whether to enable LLM tracing (default True).

    Returns:
        ResearchResult with evaluation results.
    """
    if evaluators is None:
        evaluators = get_default_evaluators(include_llm=include_llm_evaluators)

    result = ResearchResult(
        question=question,
        report=report,
        iteration_count=0,
        section_count=0,
        report_length=len(report),
    )

    if track:
        configure_tracking(
            experiment_name=experiment_name, enable_tracing=enable_tracing
        )

        with track_research_run(question, run_name="evaluation-only") as tracked_result:
            tracked_result.report = report
            tracked_result.report_length = len(report)

            eval_results = run_evaluators(question, report, evaluators)

            for evaluator, eval_result in zip(evaluators, eval_results, strict=True):
                log_evaluation(
                    tracked_result,
                    name=evaluator.name,
                    score=eval_result.score,
                    passed=eval_result.passed,
                    reason=eval_result.reason,
                    **eval_result.metadata,
                )

            result = tracked_result
    else:
        eval_results = run_evaluators(question, report, evaluators)

        for evaluator, eval_result in zip(evaluators, eval_results, strict=True):
            from .tracking import EvaluationResult

            result.evaluations.append(
                EvaluationResult(
                    name=evaluator.name,
                    score=eval_result.score,
                    passed=eval_result.passed,
                    metadata={
                        "reason": eval_result.reason,
                        **eval_result.metadata,
                    },
                )
            )

    return result
