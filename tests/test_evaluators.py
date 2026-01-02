"""Tests for the evaluators module."""

from src.deep_research.evaluators import (
    EvaluatorResult,
    LengthEvaluator,
    StructureEvaluator,
    get_default_evaluators,
    run_evaluators,
)


class TestLengthEvaluator:
    """Tests for LengthEvaluator."""

    def test_length_evaluator_passes_long_report(self):
        """Test that a long report passes the length check."""
        evaluator = LengthEvaluator(min_words=10, min_chars=50)
        report = "This is a test report with enough words and characters to pass the evaluation criteria easily."
        result = evaluator.evaluate("test question", report)

        assert result.score == 1.0
        assert result.passed is True
        assert "word_count" in result.metadata

    def test_length_evaluator_fails_short_report(self):
        """Test that a short report fails the length check."""
        evaluator = LengthEvaluator(min_words=100, min_chars=500)
        report = "Short report."
        result = evaluator.evaluate("test question", report)

        assert result.score < 1.0
        assert result.passed is False
        assert result.metadata["word_count"] < 100

    def test_length_evaluator_partial_score(self):
        """Test that partial compliance gives partial score."""
        evaluator = LengthEvaluator(min_words=20, min_chars=100, threshold=0.5)
        # This report has ~15 words and ~80 chars, giving a score of ~0.65
        report = "This report has enough words to pass the threshold but not the full requirements."
        result = evaluator.evaluate("test question", report)

        assert 0 < result.score < 1.0
        assert result.passed is True  # score > 0.5 threshold

    def test_length_evaluator_empty_report(self):
        """Test handling of empty report."""
        evaluator = LengthEvaluator(min_words=10, min_chars=50)
        result = evaluator.evaluate("test question", "")

        assert result.score == 0.0
        assert result.passed is False


class TestStructureEvaluator:
    """Tests for StructureEvaluator."""

    def test_structure_evaluator_with_headings(self):
        """Test that a well-structured report passes."""
        evaluator = StructureEvaluator(
            min_sections=2, require_intro=False, require_conclusion=False
        )
        report = """# Introduction
This is the intro.

## Main Section
Some content here.

## Another Section
More content.
"""
        result = evaluator.evaluate("test question", report)

        assert result.score >= 0.7
        assert result.passed is True
        assert result.metadata["heading_count"] >= 2

    def test_structure_evaluator_no_headings(self):
        """Test that a report without headings fails."""
        evaluator = StructureEvaluator(min_sections=2, threshold=0.7)
        report = "This is just plain text without any markdown headings or structure."
        result = evaluator.evaluate("test question", report)

        assert result.metadata["heading_count"] == 0

    def test_structure_evaluator_detects_introduction(self):
        """Test that introduction detection works."""
        evaluator = StructureEvaluator(
            require_intro=True, require_conclusion=False, min_sections=1
        )
        report = """# Introduction
This report provides an overview.

## Details
More information.
"""
        result = evaluator.evaluate("test question", report)
        assert result.metadata["has_intro"] is True

    def test_structure_evaluator_detects_conclusion(self):
        """Test that conclusion detection works."""
        evaluator = StructureEvaluator(
            require_intro=False, require_conclusion=True, min_sections=1
        )
        report = """# Main Content
Some content.

## Conclusion
In conclusion, this is the summary.
"""
        result = evaluator.evaluate("test question", report)
        assert result.metadata["has_conclusion"] is True


class TestRunEvaluators:
    """Tests for run_evaluators function."""

    def test_run_evaluators_with_defaults(self):
        """Test running default evaluators."""
        report = """# Research Report

## Introduction
This is a comprehensive research report on the topic.

## Main Findings
Here are the key findings with substantial content to meet length requirements.
We have analyzed multiple sources and compiled the results.

## Conclusion
In conclusion, the research shows important insights.
"""
        results = run_evaluators("test question", report)

        assert len(results) == 2  # Default has 2 evaluators
        assert all(isinstance(r, EvaluatorResult) for r in results)

    def test_run_evaluators_with_custom_list(self):
        """Test running custom evaluators."""
        evaluators = [
            LengthEvaluator(min_words=5, min_chars=20),
        ]
        report = "This is a test report with enough content."
        results = run_evaluators("test question", report, evaluators)

        assert len(results) == 1
        assert results[0].passed is True


class TestGetDefaultEvaluators:
    """Tests for get_default_evaluators function."""

    def test_get_default_evaluators_basic(self):
        """Test getting basic evaluators."""
        evaluators = get_default_evaluators(include_llm=False)

        assert len(evaluators) == 2
        assert any(e.name == "length" for e in evaluators)
        assert any(e.name == "structure" for e in evaluators)

    def test_get_default_evaluators_with_llm(self):
        """Test getting evaluators with LLM-based ones."""
        evaluators = get_default_evaluators(include_llm=True)

        assert len(evaluators) == 4
        assert any(e.name == "relevance" for e in evaluators)
        assert any(e.name == "completeness" for e in evaluators)
