"""Tests for the tracking module."""

import shutil
import tempfile

import pytest

from src.deep_research.tracking import (
    EvaluationResult,
    ResearchResult,
    configure_tracking,
    disable_tracing,
    is_tracing_enabled,
    log_evaluation,
    track_research_run,
)


@pytest.fixture
def temp_mlflow_dir():
    """Create a temporary SQLite database for MLflow tracking."""
    temp_dir = tempfile.mkdtemp()
    db_path = f"{temp_dir}/mlflow.db"
    yield f"sqlite:///{db_path}"
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestResearchResult:
    """Tests for ResearchResult dataclass."""

    def test_research_result_creation(self):
        """Test creating a ResearchResult."""
        result = ResearchResult(
            question="What is AI?",
            report="AI is artificial intelligence.",
            iteration_count=2,
            section_count=3,
            report_length=30,
        )

        assert result.question == "What is AI?"
        assert result.report == "AI is artificial intelligence."
        assert result.iteration_count == 2
        assert result.section_count == 3
        assert result.report_length == 30
        assert result.evaluations == []
        assert result.metadata == {}

    def test_research_result_with_evaluations(self):
        """Test ResearchResult with evaluations."""
        result = ResearchResult(
            question="test",
            report="test report",
            iteration_count=1,
            section_count=1,
            report_length=11,
        )

        eval_result = EvaluationResult(
            name="length",
            score=0.8,
            passed=True,
            metadata={"word_count": 100},
        )
        result.evaluations.append(eval_result)

        assert len(result.evaluations) == 1
        assert result.evaluations[0].name == "length"


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_evaluation_result_creation(self):
        """Test creating an EvaluationResult."""
        result = EvaluationResult(
            name="test_evaluator",
            score=0.75,
            passed=True,
            metadata={"key": "value"},
        )

        assert result.name == "test_evaluator"
        assert result.score == 0.75
        assert result.passed is True
        assert result.metadata == {"key": "value"}

    def test_evaluation_result_defaults(self):
        """Test EvaluationResult with default metadata."""
        result = EvaluationResult(
            name="test",
            score=0.5,
            passed=False,
        )

        assert result.metadata == {}


class TestConfigureTracking:
    """Tests for configure_tracking function."""

    def test_configure_tracking_default(self, temp_mlflow_dir):
        """Test configuring tracking with defaults."""
        configure_tracking(
            experiment_name="test-experiment",
            tracking_uri=temp_mlflow_dir,
        )
        # Should not raise any exceptions

    def test_configure_tracking_custom_experiment(self, temp_mlflow_dir):
        """Test configuring tracking with custom experiment name."""
        configure_tracking(
            experiment_name="custom-experiment",
            tracking_uri=temp_mlflow_dir,
        )
        # Should not raise any exceptions


class TestTrackResearchRun:
    """Tests for track_research_run context manager."""

    def test_track_research_run_basic(self, temp_mlflow_dir):
        """Test basic research run tracking."""
        configure_tracking(
            experiment_name="test-tracking",
            tracking_uri=temp_mlflow_dir,
        )

        with track_research_run("What is Python?") as result:
            result.report = "Python is a programming language."
            result.report_length = len(result.report)
            result.iteration_count = 1
            result.section_count = 1

        assert result.question == "What is Python?"
        assert result.report == "Python is a programming language."

    def test_track_research_run_with_evaluations(self, temp_mlflow_dir):
        """Test tracking with evaluations."""
        configure_tracking(
            experiment_name="test-tracking-eval",
            tracking_uri=temp_mlflow_dir,
        )

        with track_research_run("Test question") as result:
            result.report = "Test report content."
            result.report_length = 20
            result.iteration_count = 1
            result.section_count = 1

            log_evaluation(result, "length", 0.9, True, word_count=100)
            log_evaluation(result, "structure", 0.8, True, heading_count=3)

        assert len(result.evaluations) == 2
        assert result.evaluations[0].name == "length"
        assert result.evaluations[1].name == "structure"

    def test_track_research_run_with_tags(self, temp_mlflow_dir):
        """Test tracking with custom tags."""
        configure_tracking(
            experiment_name="test-tracking-tags",
            tracking_uri=temp_mlflow_dir,
        )

        tags = {"model": "gpt-4", "version": "1.0"}

        with track_research_run("Test question", tags=tags) as result:
            result.report = "Report."
            result.report_length = 7

        # Should complete without errors

    def test_track_research_run_with_metadata(self, temp_mlflow_dir):
        """Test tracking with additional metadata."""
        configure_tracking(
            experiment_name="test-tracking-metadata",
            tracking_uri=temp_mlflow_dir,
        )

        with track_research_run("Test question") as result:
            result.report = "Report content."
            result.report_length = 15
            result.metadata["custom_metric"] = 42
            result.metadata["custom_param"] = "value"

        assert result.metadata["custom_metric"] == 42


class TestLogEvaluation:
    """Tests for log_evaluation function."""

    def test_log_evaluation_basic(self):
        """Test basic evaluation logging."""
        result = ResearchResult(
            question="test",
            report="test",
            iteration_count=0,
            section_count=0,
            report_length=4,
        )

        eval_result = log_evaluation(result, "test_eval", 0.85, True)

        assert eval_result.name == "test_eval"
        assert eval_result.score == 0.85
        assert eval_result.passed is True
        assert len(result.evaluations) == 1

    def test_log_evaluation_with_metadata(self):
        """Test evaluation logging with metadata."""
        result = ResearchResult(
            question="test",
            report="test",
            iteration_count=0,
            section_count=0,
            report_length=4,
        )

        eval_result = log_evaluation(
            result,
            "complex_eval",
            0.7,
            True,
            detail1="value1",
            detail2=123,
        )

        assert eval_result.metadata["detail1"] == "value1"
        assert eval_result.metadata["detail2"] == 123

    def test_log_multiple_evaluations(self):
        """Test logging multiple evaluations."""
        result = ResearchResult(
            question="test",
            report="test",
            iteration_count=0,
            section_count=0,
            report_length=4,
        )

        log_evaluation(result, "eval1", 0.9, True)
        log_evaluation(result, "eval2", 0.6, False)
        log_evaluation(result, "eval3", 0.75, True)

        assert len(result.evaluations) == 3
        assert [e.name for e in result.evaluations] == ["eval1", "eval2", "eval3"]


class TestTracing:
    """Tests for tracing functions."""

    def test_configure_tracking_enables_tracing(self, temp_mlflow_dir):
        """Test that configure_tracking enables tracing by default."""
        # Reset tracing state
        disable_tracing()
        assert is_tracing_enabled() is False

        configure_tracking(
            experiment_name="test-tracing",
            tracking_uri=temp_mlflow_dir,
            enable_tracing=True,
        )

        assert is_tracing_enabled() is True

    def test_configure_tracking_can_disable_tracing(self, temp_mlflow_dir):
        """Test that tracing can be disabled."""
        # First enable tracing
        configure_tracking(
            experiment_name="test-tracing-disabled",
            tracking_uri=temp_mlflow_dir,
            enable_tracing=True,
        )

        # Now disable it
        disable_tracing()
        assert is_tracing_enabled() is False

    def test_is_tracing_enabled_returns_bool(self, temp_mlflow_dir):
        """Test that is_tracing_enabled returns a boolean."""
        result = is_tracing_enabled()
        assert isinstance(result, bool)
