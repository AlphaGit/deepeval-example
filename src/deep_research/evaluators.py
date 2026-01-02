"""Evaluators for assessing research report quality."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from .logging import get_logger

logger = get_logger("evaluators")


@dataclass
class EvaluatorResult:
    """Result from an evaluator."""

    score: float  # 0.0 to 1.0
    passed: bool
    reason: str
    metadata: dict[str, Any]


class Evaluator(ABC):
    """Base class for evaluators."""

    name: str = "base"
    threshold: float = 0.5

    @abstractmethod
    def evaluate(self, question: str, report: str) -> EvaluatorResult:
        """Evaluate a research report.

        Args:
            question: The original research question.
            report: The generated research report.

        Returns:
            EvaluatorResult with score, pass/fail, and reasoning.
        """
        pass


class LengthEvaluator(Evaluator):
    """Evaluates if the report meets minimum length requirements."""

    name = "length"

    def __init__(
        self,
        min_words: int = 100,
        min_chars: int = 500,
        threshold: float = 1.0,
    ):
        """Initialize the length evaluator.

        Args:
            min_words: Minimum number of words required.
            min_chars: Minimum number of characters required.
            threshold: Score threshold to pass (default 1.0 = must meet both).
        """
        self.min_words = min_words
        self.min_chars = min_chars
        self.threshold = threshold

    def evaluate(self, question: str, report: str) -> EvaluatorResult:
        word_count = len(report.split())
        char_count = len(report)

        word_score = (
            min(1.0, word_count / self.min_words) if self.min_words > 0 else 1.0
        )
        char_score = (
            min(1.0, char_count / self.min_chars) if self.min_chars > 0 else 1.0
        )

        score = (word_score + char_score) / 2
        passed = score >= self.threshold

        logger.debug(
            "length_evaluation",
            word_count=word_count,
            char_count=char_count,
            score=score,
            passed=passed,
        )

        return EvaluatorResult(
            score=score,
            passed=passed,
            reason=f"Word count: {word_count}/{self.min_words}, Char count: {char_count}/{self.min_chars}",
            metadata={
                "word_count": word_count,
                "char_count": char_count,
                "min_words": self.min_words,
                "min_chars": self.min_chars,
            },
        )


class StructureEvaluator(Evaluator):
    """Evaluates if the report has proper structure (headings, sections)."""

    name = "structure"

    def __init__(
        self,
        min_sections: int = 2,
        require_intro: bool = True,
        require_conclusion: bool = True,
        threshold: float = 0.7,
    ):
        """Initialize the structure evaluator.

        Args:
            min_sections: Minimum number of markdown headings required.
            require_intro: Whether an introduction section is required.
            require_conclusion: Whether a conclusion section is required.
            threshold: Score threshold to pass.
        """
        self.min_sections = min_sections
        self.require_intro = require_intro
        self.require_conclusion = require_conclusion
        self.threshold = threshold

    def evaluate(self, question: str, report: str) -> EvaluatorResult:
        # Count markdown headings (# ## ### etc.)
        heading_pattern = r"^#{1,6}\s+.+$"
        headings = re.findall(heading_pattern, report, re.MULTILINE)
        heading_count = len(headings)

        # Check for introduction
        intro_patterns = [
            r"(?i)^#{1,6}\s*(introduction|overview|summary|background)",
            r"(?i)^(this report|this research|in this)",
        ]
        has_intro = any(re.search(p, report, re.MULTILINE) for p in intro_patterns)

        # Check for conclusion
        conclusion_patterns = [
            r"(?i)^#{1,6}\s*(conclusion|summary|final\s*thoughts|key\s*takeaways)",
            r"(?i)(in\s*conclusion|to\s*summarize|in\s*summary)",
        ]
        has_conclusion = any(
            re.search(p, report, re.MULTILINE) for p in conclusion_patterns
        )

        # Calculate scores
        section_score = (
            min(1.0, heading_count / self.min_sections)
            if self.min_sections > 0
            else 1.0
        )
        intro_score = 1.0 if (not self.require_intro or has_intro) else 0.0
        conclusion_score = (
            1.0 if (not self.require_conclusion or has_conclusion) else 0.0
        )

        score = (section_score + intro_score + conclusion_score) / 3
        passed = score >= self.threshold

        logger.debug(
            "structure_evaluation",
            heading_count=heading_count,
            has_intro=has_intro,
            has_conclusion=has_conclusion,
            score=score,
            passed=passed,
        )

        return EvaluatorResult(
            score=score,
            passed=passed,
            reason=f"Headings: {heading_count}/{self.min_sections}, Intro: {has_intro}, Conclusion: {has_conclusion}",
            metadata={
                "heading_count": heading_count,
                "has_intro": has_intro,
                "has_conclusion": has_conclusion,
                "headings": headings[:10],  # First 10 headings
            },
        )


class RelevanceEvaluator(Evaluator):
    """Evaluates if the report is relevant to the question using an LLM."""

    name = "relevance"

    def __init__(
        self,
        model: ChatOpenAI | None = None,
        threshold: float = 0.7,
    ):
        """Initialize the relevance evaluator.

        Args:
            model: ChatOpenAI model to use for evaluation.
            threshold: Score threshold to pass.
        """
        self.model = model
        self.threshold = threshold

    def evaluate(self, question: str, report: str) -> EvaluatorResult:
        if self.model is None:
            import os

            from langchain_openai import ChatOpenAI

            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.model = ChatOpenAI(model=model_name, temperature=0)

        prompt = f"""You are an expert evaluator assessing if a research report adequately addresses a given question.

Question: {question}

Report:
{report[:4000]}  # Truncate to avoid token limits

Evaluate the relevance of the report to the question on a scale of 0.0 to 1.0:
- 1.0: Perfectly addresses all aspects of the question
- 0.8: Addresses most aspects with good depth
- 0.6: Addresses the main topic but missing some aspects
- 0.4: Partially relevant but significant gaps
- 0.2: Tangentially related
- 0.0: Completely irrelevant

Respond with ONLY a JSON object in this exact format:
{{"score": <float>, "reason": "<brief explanation>"}}"""

        try:
            response = self.model.invoke(prompt)
            import json

            # Parse JSON from response
            content = response.content.strip()
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = re.sub(r"```(?:json)?\n?", "", content)
                content = content.strip()

            result = json.loads(content)
            score = float(result.get("score", 0.5))
            reason = result.get("reason", "No reason provided")

            passed = score >= self.threshold

            logger.debug(
                "relevance_evaluation",
                score=score,
                passed=passed,
                reason=reason,
            )

            return EvaluatorResult(
                score=score,
                passed=passed,
                reason=reason,
                metadata={"model_response": content[:500]},
            )

        except Exception as e:
            logger.error("relevance_evaluation_error", error=str(e))
            return EvaluatorResult(
                score=0.5,
                passed=False,
                reason=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)},
            )


class CompletenessEvaluator(Evaluator):
    """Evaluates if the report covers key aspects of the topic using an LLM."""

    name = "completeness"

    def __init__(
        self,
        model: ChatOpenAI | None = None,
        threshold: float = 0.6,
    ):
        """Initialize the completeness evaluator.

        Args:
            model: ChatOpenAI model to use for evaluation.
            threshold: Score threshold to pass.
        """
        self.model = model
        self.threshold = threshold

    def evaluate(self, question: str, report: str) -> EvaluatorResult:
        if self.model is None:
            import os

            from langchain_openai import ChatOpenAI

            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.model = ChatOpenAI(model=model_name, temperature=0)

        prompt = f"""You are an expert evaluator assessing the completeness of a research report.

Question: {question}

Report:
{report[:4000]}

Evaluate the completeness of the report on a scale of 0.0 to 1.0:
- 1.0: Comprehensive coverage with depth on all key aspects
- 0.8: Good coverage with minor gaps
- 0.6: Covers main points but lacks depth in some areas
- 0.4: Covers basics but missing important aspects
- 0.2: Superficial coverage
- 0.0: Incomplete or empty

Consider:
1. Does it cover multiple perspectives/aspects?
2. Does it provide supporting evidence or examples?
3. Does it address potential counterarguments or limitations?
4. Is there sufficient depth for each point made?

Respond with ONLY a JSON object in this exact format:
{{"score": <float>, "reason": "<brief explanation>", "missing_aspects": ["aspect1", "aspect2"]}}"""

        try:
            response = self.model.invoke(prompt)
            import json

            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"```(?:json)?\n?", "", content)
                content = content.strip()

            result = json.loads(content)
            score = float(result.get("score", 0.5))
            reason = result.get("reason", "No reason provided")
            missing = result.get("missing_aspects", [])

            passed = score >= self.threshold

            logger.debug(
                "completeness_evaluation",
                score=score,
                passed=passed,
                missing_aspects=missing,
            )

            return EvaluatorResult(
                score=score,
                passed=passed,
                reason=reason,
                metadata={
                    "missing_aspects": missing,
                    "model_response": content[:500],
                },
            )

        except Exception as e:
            logger.error("completeness_evaluation_error", error=str(e))
            return EvaluatorResult(
                score=0.5,
                passed=False,
                reason=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)},
            )


def run_evaluators(
    question: str,
    report: str,
    evaluators: list[Evaluator] | None = None,
) -> list[EvaluatorResult]:
    """Run multiple evaluators on a research report.

    Args:
        question: The original research question.
        report: The generated research report.
        evaluators: List of evaluators to run. Defaults to basic evaluators.

    Returns:
        List of EvaluatorResult objects.
    """
    if evaluators is None:
        evaluators = [
            LengthEvaluator(),
            StructureEvaluator(),
        ]

    results = []
    for evaluator in evaluators:
        logger.info("running_evaluator", evaluator=evaluator.name)
        result = evaluator.evaluate(question, report)
        results.append(result)

        logger.info(
            "evaluator_complete",
            evaluator=evaluator.name,
            score=result.score,
            passed=result.passed,
        )

    return results


# Convenience function to get default evaluators
def get_default_evaluators(include_llm: bool = False) -> list[Evaluator]:
    """Get a list of default evaluators.

    Args:
        include_llm: Whether to include LLM-based evaluators.

    Returns:
        List of Evaluator instances.
    """
    evaluators: list[Evaluator] = [
        LengthEvaluator(min_words=100, min_chars=500),
        StructureEvaluator(min_sections=2),
    ]

    if include_llm:
        evaluators.extend(
            [
                RelevanceEvaluator(),
                CompletenessEvaluator(),
            ]
        )

    return evaluators
