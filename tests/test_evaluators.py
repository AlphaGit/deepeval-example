"""Tests for the evaluators module."""

from unittest.mock import MagicMock, patch

from src.deep_research.evaluators import (
    EvaluatorResult,
    LengthEvaluator,
    PlanAdherenceEvaluator,
    PlanQualityEvaluator,
    ReasoningEvaluator,
    StructureEvaluator,
    get_default_evaluators,
    run_evaluators,
    run_evaluators_with_reasoning,
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

    def test_get_default_evaluators_with_reasoning(self):
        """Test getting evaluators with reasoning ones."""
        evaluators = get_default_evaluators(include_reasoning=True)

        assert (
            len(evaluators) == 5
        )  # 2 base + 3 reasoning (PlanQuality, PlanAdherence, ReasoningTrace)
        assert any(e.name == "plan_quality" for e in evaluators)
        assert any(e.name == "plan_adherence" for e in evaluators)
        assert any(e.name == "reasoning_trace" for e in evaluators)

    def test_get_default_evaluators_with_all(self):
        """Test getting all evaluators."""
        evaluators = get_default_evaluators(include_llm=True, include_reasoning=True)

        assert len(evaluators) == 7  # 2 base + 2 llm + 3 reasoning
        assert any(e.name == "length" for e in evaluators)
        assert any(e.name == "structure" for e in evaluators)
        assert any(e.name == "relevance" for e in evaluators)
        assert any(e.name == "completeness" for e in evaluators)
        assert any(e.name == "plan_quality" for e in evaluators)
        assert any(e.name == "plan_adherence" for e in evaluators)


class TestPlanQualityEvaluator:
    """Tests for PlanQualityEvaluator."""

    def test_plan_quality_evaluator_without_plan(self):
        """Test that evaluator fails gracefully without a plan."""
        evaluator = PlanQualityEvaluator()
        result = evaluator.evaluate("test question", "test report", plan=None)

        assert result.score == 0.0
        assert result.passed is False
        assert "No plan provided" in result.reason
        assert result.metadata.get("error") == "missing_plan"

    def test_plan_quality_evaluator_is_reasoning_evaluator(self):
        """Test that PlanQualityEvaluator is a ReasoningEvaluator."""
        evaluator = PlanQualityEvaluator()
        assert isinstance(evaluator, ReasoningEvaluator)
        assert evaluator.name == "plan_quality"

    @patch("src.deep_research.evaluators.ChatOpenAI")
    def test_plan_quality_evaluator_with_plan(self, mock_chat_openai):
        """Test evaluator with a valid plan."""
        # Setup mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"score": 0.85, "reason": "Good plan", "strengths": ["comprehensive"], "weaknesses": []}'
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model

        evaluator = PlanQualityEvaluator(threshold=0.7)
        plan = [
            "What are the key benefits of renewable energy?",
            "What are the environmental impacts?",
            "What are the economic considerations?",
        ]

        result = evaluator.evaluate(
            "Benefits of renewable energy", "test report", plan=plan
        )

        assert result.score == 0.85
        assert result.passed is True
        assert result.reason == "Good plan"
        assert result.metadata["plan_size"] == 3
        assert "comprehensive" in result.metadata["strengths"]

    @patch("src.deep_research.evaluators.ChatOpenAI")
    def test_plan_quality_evaluator_handles_json_in_code_block(self, mock_chat_openai):
        """Test evaluator handles JSON wrapped in code blocks."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '```json\n{"score": 0.75, "reason": "Okay plan", "strengths": [], "weaknesses": ["missing depth"]}\n```'
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model

        evaluator = PlanQualityEvaluator()
        plan = ["Query 1", "Query 2"]

        result = evaluator.evaluate("question", "report", plan=plan)

        assert result.score == 0.75
        assert "missing depth" in result.metadata["weaknesses"]

    @patch("src.deep_research.evaluators.ChatOpenAI")
    def test_plan_quality_evaluator_handles_errors(self, mock_chat_openai):
        """Test evaluator handles LLM errors gracefully."""
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("API Error")
        mock_chat_openai.return_value = mock_model

        evaluator = PlanQualityEvaluator()
        plan = ["Query 1"]

        result = evaluator.evaluate("question", "report", plan=plan)

        assert result.score == 0.5
        assert result.passed is False
        assert "Evaluation failed" in result.reason


class TestPlanAdherenceEvaluator:
    """Tests for PlanAdherenceEvaluator."""

    def test_plan_adherence_evaluator_without_plan(self):
        """Test that evaluator fails gracefully without a plan."""
        evaluator = PlanAdherenceEvaluator()
        execution = [{"topic": "Topic 1", "content": "Content"}]
        result = evaluator.evaluate(
            "test question", "test report", plan=None, execution=execution
        )

        assert result.score == 0.0
        assert result.passed is False
        assert "No plan provided" in result.reason
        assert result.metadata.get("error") == "missing_plan"

    def test_plan_adherence_evaluator_without_execution(self):
        """Test that evaluator fails gracefully without execution data."""
        evaluator = PlanAdherenceEvaluator()
        plan = ["Query 1", "Query 2"]
        result = evaluator.evaluate(
            "test question", "test report", plan=plan, execution=None
        )

        assert result.score == 0.0
        assert result.passed is False
        assert "No execution data provided" in result.reason
        assert result.metadata.get("error") == "missing_execution"

    def test_plan_adherence_evaluator_is_reasoning_evaluator(self):
        """Test that PlanAdherenceEvaluator is a ReasoningEvaluator."""
        evaluator = PlanAdherenceEvaluator()
        assert isinstance(evaluator, ReasoningEvaluator)
        assert evaluator.name == "plan_adherence"

    @patch("src.deep_research.evaluators.ChatOpenAI")
    def test_plan_adherence_evaluator_with_data(self, mock_chat_openai):
        """Test evaluator with valid plan and execution data."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"score": 0.9, "reason": "Good adherence", "covered_queries": 3, "total_queries": 3, "deviations": []}'
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model

        evaluator = PlanAdherenceEvaluator(threshold=0.7)
        plan = ["Query 1", "Query 2", "Query 3"]
        execution = [
            {"topic": "Query 1", "content": "Research on query 1..."},
            {"topic": "Query 2", "content": "Research on query 2..."},
            {"topic": "Query 3", "content": "Research on query 3..."},
        ]

        result = evaluator.evaluate(
            "question", "report", plan=plan, execution=execution
        )

        assert result.score == 0.9
        assert result.passed is True
        assert result.metadata["plan_size"] == 3
        assert result.metadata["execution_size"] == 3
        assert result.metadata["covered_queries"] == 3

    @patch("src.deep_research.evaluators.ChatOpenAI")
    def test_plan_adherence_evaluator_with_deviations(self, mock_chat_openai):
        """Test evaluator detects deviations from plan."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"score": 0.6, "reason": "Some deviations", "covered_queries": 2, "total_queries": 3, "deviations": ["Off-topic section added"]}'
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model

        evaluator = PlanAdherenceEvaluator(threshold=0.7)
        plan = ["Query 1", "Query 2", "Query 3"]
        execution = [
            {"topic": "Query 1", "content": "Content 1"},
            {"topic": "Off-topic", "content": "Off-topic content"},
        ]

        result = evaluator.evaluate(
            "question", "report", plan=plan, execution=execution
        )

        assert result.score == 0.6
        assert result.passed is False  # Below threshold
        assert "Off-topic section added" in result.metadata["deviations"]

    @patch("src.deep_research.evaluators.ChatOpenAI")
    def test_plan_adherence_evaluator_handles_errors(self, mock_chat_openai):
        """Test evaluator handles LLM errors gracefully."""
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("API Error")
        mock_chat_openai.return_value = mock_model

        evaluator = PlanAdherenceEvaluator()
        plan = ["Query 1"]
        execution = [{"topic": "Query 1", "content": "Content"}]

        result = evaluator.evaluate(
            "question", "report", plan=plan, execution=execution
        )

        assert result.score == 0.5
        assert result.passed is False
        assert "Evaluation failed" in result.reason


class TestRunEvaluatorsWithReasoning:
    """Tests for run_evaluators_with_reasoning function."""

    def test_run_evaluators_with_reasoning_regular_evaluators(self):
        """Test running regular evaluators through the reasoning function."""
        evaluators = [
            LengthEvaluator(min_words=5, min_chars=20),
            StructureEvaluator(
                min_sections=1, require_intro=False, require_conclusion=False
            ),
        ]
        report = "# Test\nThis is a test report with enough content."
        results = run_evaluators_with_reasoning(
            "test question", report, evaluators, plan=None, execution=None
        )

        assert len(results) == 2
        assert all(isinstance(r, EvaluatorResult) for r in results)

    @patch("src.deep_research.evaluators.ChatOpenAI")
    def test_run_evaluators_with_reasoning_mixed_evaluators(self, mock_chat_openai):
        """Test running a mix of regular and reasoning evaluators."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"score": 0.8, "reason": "Good", "strengths": [], "weaknesses": []}'
        )
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model

        evaluators = [
            LengthEvaluator(min_words=5, min_chars=20),
            PlanQualityEvaluator(),
        ]
        report = "This is a test report with enough content."
        plan = ["Query 1", "Query 2"]

        results = run_evaluators_with_reasoning(
            "test question", report, evaluators, plan=plan, execution=None
        )

        assert len(results) == 2
        # First evaluator is LengthEvaluator
        assert results[0].passed is True
        # Second evaluator is PlanQualityEvaluator
        assert results[1].score == 0.8

    def test_run_evaluators_with_reasoning_defaults(self):
        """Test run_evaluators_with_reasoning with default evaluators."""
        report = """# Research Report

## Introduction
This is a comprehensive research report on the topic.

## Main Findings
Here are the key findings with substantial content to meet length requirements.
We have analyzed multiple sources and compiled the results.

## Conclusion
In conclusion, the research shows important insights.
"""
        results = run_evaluators_with_reasoning("test question", report)

        assert len(results) == 2  # Default has 2 evaluators
        assert all(isinstance(r, EvaluatorResult) for r in results)


class TestReasoningTraceEvaluator:
    """Tests for ReasoningTraceEvaluator."""

    def test_reasoning_trace_evaluator_without_plan(self):
        """Test that evaluator fails gracefully without a plan."""
        from src.deep_research.evaluators import ReasoningTraceEvaluator

        evaluator = ReasoningTraceEvaluator()
        execution = [{"topic": "Topic 1", "content": "Content", "tool_used": "llm"}]
        result = evaluator.evaluate(
            "test question", "test report", plan=None, execution=execution
        )

        assert result.score == 0.0
        assert result.passed is False
        assert "Missing plan or execution data" in result.reason
        assert result.metadata.get("error") == "missing_context"

    def test_reasoning_trace_evaluator_without_execution(self):
        """Test that evaluator fails gracefully without execution data."""
        from src.deep_research.evaluators import ReasoningTraceEvaluator

        evaluator = ReasoningTraceEvaluator()
        plan = ["Query 1", "Query 2"]
        result = evaluator.evaluate(
            "test question", "test report", plan=plan, execution=None
        )

        assert result.score == 0.0
        assert result.passed is False
        assert "Missing plan or execution data" in result.reason
        assert result.metadata.get("error") == "missing_context"

    def test_reasoning_trace_evaluator_is_reasoning_evaluator(self):
        """Test that ReasoningTraceEvaluator is a ReasoningEvaluator."""
        from src.deep_research.evaluators import (
            ReasoningEvaluator,
            ReasoningTraceEvaluator,
        )

        evaluator = ReasoningTraceEvaluator()
        assert isinstance(evaluator, ReasoningEvaluator)
        assert evaluator.name == "reasoning_trace"

    @patch("src.deep_research.evaluators.ChatOpenAI")
    def test_reasoning_trace_evaluator_with_data(self, mock_chat_openai):
        """Test evaluator with valid plan and execution data."""
        from src.deep_research.evaluators import ReasoningTraceEvaluator

        mock_model = MagicMock()
        mock_response = MagicMock()
        # Mock a comprehensive evaluation response
        mock_response.content = """{
  "logical_coherence": {"score": 0.9, "explanation": "Queries follow logical progression"},
  "goal_alignment": {"score": 0.85, "explanation": "All queries align with the research goal"},
  "efficiency": {"score": 0.8, "explanation": "Minimal redundancy in queries"},
  "reasoning_faithfulness": {"score": 0.9, "explanation": "Execution aligned with plan"},
  "intermediate_state_accuracy": {"score": 0.85, "explanation": "Dependencies handled well"},
  "hypothesis_generation": {"score": 0.8, "explanation": "Multiple perspectives explored"},
  "overall_assessment": "Strong reasoning process with good coherence and goal alignment",
  "strengths": ["Logical query progression", "Comprehensive coverage"],
  "weaknesses": ["Could explore more alternatives"],
  "recommendations": ["Consider broader perspectives"]
}"""
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model

        evaluator = ReasoningTraceEvaluator(threshold=0.7)
        plan = ["What is X?", "How does X work?", "What are benefits of X?"]
        execution = [
            {
                "topic": "What is X?",
                "content": "X is a technology...",
                "tool_used": "web_search",
                "source_count": 3,
            },
            {
                "topic": "How does X work?",
                "content": "X works by...",
                "tool_used": "llm",
                "source_count": 0,
            },
            {
                "topic": "What are benefits of X?",
                "content": "Benefits include...",
                "tool_used": "web_search",
                "source_count": 2,
            },
        ]

        result = evaluator.evaluate(
            "Tell me about X",
            "A comprehensive report on X...",
            plan=plan,
            execution=execution,
        )

        # Check overall score (average of dimensions)
        expected_score = (0.9 + 0.85 + 0.8 + 0.9 + 0.85 + 0.8) / 6
        assert abs(result.score - expected_score) < 0.01
        assert result.passed is True

        # Check metadata contains all dimension scores
        assert result.metadata["logical_coherence_score"] == 0.9
        assert result.metadata["goal_alignment_score"] == 0.85
        assert result.metadata["efficiency_score"] == 0.8
        assert result.metadata["reasoning_faithfulness_score"] == 0.9
        assert result.metadata["intermediate_state_accuracy_score"] == 0.85
        assert result.metadata["hypothesis_generation_score"] == 0.8

        # Check explanations are present
        assert "logical progression" in result.metadata["logical_coherence_explanation"]
        assert (
            "align with the research goal"
            in result.metadata["goal_alignment_explanation"]
        )

        # Check overall assessment and findings
        assert "Strong reasoning process" in result.metadata["overall_assessment"]
        assert result.metadata["strengths"] == [
            "Logical query progression",
            "Comprehensive coverage",
        ]
        assert result.metadata["weaknesses"] == ["Could explore more alternatives"]
        assert result.metadata["recommendations"] == ["Consider broader perspectives"]

        # Check context metrics
        assert result.metadata["plan_size"] == 3
        assert result.metadata["execution_steps"] == 3

    @patch("src.deep_research.evaluators.ChatOpenAI")
    def test_reasoning_trace_evaluator_handles_errors(self, mock_chat_openai):
        """Test evaluator handles LLM errors gracefully."""
        from src.deep_research.evaluators import ReasoningTraceEvaluator

        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("API Error")
        mock_chat_openai.return_value = mock_model

        evaluator = ReasoningTraceEvaluator()
        plan = ["Query 1"]
        execution = [{"topic": "Query 1", "content": "Content", "tool_used": "llm"}]

        result = evaluator.evaluate(
            "question", "report", plan=plan, execution=execution
        )

        assert result.score == 0.5
        assert result.passed is False
        assert "Evaluation failed" in result.reason
        assert "error" in result.metadata
