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


class ReasoningEvaluator(Evaluator):
    """Base class for reasoning evaluators that assess agent planning and execution.

    These evaluators require additional context beyond just the question and report:
    - plan: The agent's planned actions (e.g., search queries)
    - execution: The actual actions taken (e.g., research sections)
    """

    def evaluate(
        self,
        question: str,
        report: str,
        plan: list[str] | None = None,
        execution: list[dict[str, Any]] | None = None,
    ) -> EvaluatorResult:
        """Evaluate with reasoning context.

        Args:
            question: The original research question.
            report: The generated research report.
            plan: The agent's plan (e.g., search queries).
            execution: The execution results (e.g., research sections).

        Returns:
            EvaluatorResult with score, pass/fail, and reasoning.
        """
        pass


class PlanQualityEvaluator(ReasoningEvaluator):
    """Evaluates if the agent's plan is logical, complete, and efficient.

    Assesses whether the research queries/plan adequately cover the research
    question with appropriate scope and granularity.
    """

    name = "plan_quality"

    def __init__(
        self,
        model: ChatOpenAI | None = None,
        threshold: float = 0.7,
    ):
        """Initialize the plan quality evaluator.

        Args:
            model: ChatOpenAI model to use for evaluation.
            threshold: Score threshold to pass.
        """
        self.model = model
        self.threshold = threshold

    def evaluate(
        self,
        question: str,
        report: str,
        plan: list[str] | None = None,
        execution: list[dict[str, Any]] | None = None,
    ) -> EvaluatorResult:
        if plan is None:
            return EvaluatorResult(
                score=0.0,
                passed=False,
                reason="No plan provided for evaluation",
                metadata={"error": "missing_plan"},
            )

        if self.model is None:
            import os

            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.model = ChatOpenAI(model=model_name, temperature=0)

        plan_str = "\n".join(f"- {query}" for query in plan)

        prompt = f"""You are an expert evaluator assessing the quality of a research plan.

Original Research Question: {question}

Research Plan (Search Queries):
{plan_str}

Evaluate the quality of this research plan on a scale of 0.0 to 1.0:
- 1.0: Plan is comprehensive, logical, and covers all key aspects
- 0.8: Plan is well-structured with good coverage, minor gaps
- 0.6: Plan covers main points but missing some perspectives
- 0.4: Plan is incomplete or has logical issues
- 0.2: Plan is poorly structured or overly narrow
- 0.0: Plan is irrelevant or completely inadequate

Consider these criteria:
1. Logical coherence: Do the queries build a complete picture?
2. Coverage: Are all key aspects of the question addressed?
3. Efficiency: Are queries focused without unnecessary overlap?
4. Granularity: Is the level of detail appropriate?
5. Dependencies: Are there clear research priorities?

Respond with ONLY a JSON object in this exact format:
{{"score": <float>, "reason": "<brief explanation>", "strengths": ["strength1", "strength2"], "weaknesses": ["weakness1", "weakness2"]}}"""

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
            strengths = result.get("strengths", [])
            weaknesses = result.get("weaknesses", [])

            passed = score >= self.threshold

            logger.debug(
                "plan_quality_evaluation",
                score=score,
                passed=passed,
                plan_size=len(plan),
            )

            return EvaluatorResult(
                score=score,
                passed=passed,
                reason=reason,
                metadata={
                    "plan_size": len(plan),
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "model_response": content[:500],
                },
            )

        except Exception as e:
            logger.error("plan_quality_evaluation_error", error=str(e))
            return EvaluatorResult(
                score=0.5,
                passed=False,
                reason=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)},
            )


class PlanAdherenceEvaluator(ReasoningEvaluator):
    """Evaluates if the agent follows its plan during execution.

    Assesses whether the research execution (sections) aligns with
    the planned queries and doesn't deviate from the intended strategy.
    """

    name = "plan_adherence"

    def __init__(
        self,
        model: ChatOpenAI | None = None,
        threshold: float = 0.7,
    ):
        """Initialize the plan adherence evaluator.

        Args:
            model: ChatOpenAI model to use for evaluation.
            threshold: Score threshold to pass.
        """
        self.model = model
        self.threshold = threshold

    def evaluate(
        self,
        question: str,
        report: str,
        plan: list[str] | None = None,
        execution: list[dict[str, Any]] | None = None,
    ) -> EvaluatorResult:
        if plan is None:
            return EvaluatorResult(
                score=0.0,
                passed=False,
                reason="No plan provided for evaluation",
                metadata={"error": "missing_plan"},
            )

        if execution is None:
            return EvaluatorResult(
                score=0.0,
                passed=False,
                reason="No execution data provided for evaluation",
                metadata={"error": "missing_execution"},
            )

        if self.model is None:
            import os

            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.model = ChatOpenAI(model=model_name, temperature=0)

        plan_str = "\n".join(f"- {query}" for query in plan)

        # Format execution data (research sections)
        execution_str = ""
        for i, section in enumerate(execution, 1):
            topic = section.get("topic", "Unknown topic")
            content_preview = section.get("content", "")[:200]
            execution_str += (
                f"\n{i}. Topic: {topic}\n   Content preview: {content_preview}...\n"
            )

        prompt = f"""You are an expert evaluator assessing if an AI agent followed its research plan.

Original Research Question: {question}

Research Plan (Planned Queries):
{plan_str}

Execution Results (Research Sections):
{execution_str}

Evaluate how well the execution adheres to the plan on a scale of 0.0 to 1.0:
- 1.0: Perfect adherence - all planned topics researched, no deviations
- 0.8: Strong adherence - most topics covered, minor reasonable additions
- 0.6: Moderate adherence - some topics missed or significant additions
- 0.4: Weak adherence - notable deviations from plan
- 0.2: Poor adherence - execution barely follows plan
- 0.0: No adherence - completely different from plan

Consider:
1. Coverage: Were all planned queries researched?
2. Alignment: Do the section topics match the planned queries?
3. Deviations: Are there unexplained off-topic sections?
4. Completeness: Did execution fulfill the plan's intent?

Respond with ONLY a JSON object in this exact format:
{{"score": <float>, "reason": "<brief explanation>", "covered_queries": <int>, "total_queries": <int>, "deviations": ["deviation1"]}}"""

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
            covered = result.get("covered_queries", len(execution))
            total = result.get("total_queries", len(plan))
            deviations = result.get("deviations", [])

            passed = score >= self.threshold

            logger.debug(
                "plan_adherence_evaluation",
                score=score,
                passed=passed,
                covered_queries=covered,
                total_queries=total,
            )

            return EvaluatorResult(
                score=score,
                passed=passed,
                reason=reason,
                metadata={
                    "plan_size": len(plan),
                    "execution_size": len(execution),
                    "covered_queries": covered,
                    "total_queries": total,
                    "deviations": deviations,
                    "model_response": content[:500],
                },
            )

        except Exception as e:
            logger.error("plan_adherence_evaluation_error", error=str(e))
            return EvaluatorResult(
                score=0.5,
                passed=False,
                reason=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)},
            )


class SourceQualityReasoningEvaluator(ReasoningEvaluator):
    """Evaluates source quality with access to research sections."""

    name = "source_quality"

    def __init__(
        self,
        min_sources: int = 3,
        min_unique_domains: int = 2,
        threshold: float = 0.6,
    ):
        """Initialize the source quality evaluator.

        Args:
            min_sources: Minimum number of sources expected.
            min_unique_domains: Minimum number of unique domains.
            threshold: Score threshold to pass.
        """
        self.min_sources = min_sources
        self.min_unique_domains = min_unique_domains
        self.threshold = threshold

    def evaluate(
        self,
        question: str,
        report: str,
        plan: list[str] | None = None,
        execution: list[dict[str, Any]] | None = None,
    ) -> EvaluatorResult:
        """Evaluate source quality and diversity.

        Args:
            question: The original research question.
            report: The generated research report.
            plan: The agent's plan (not used).
            execution: The execution results (research sections).

        Returns:
            EvaluatorResult with score, pass/fail, and source metrics.
        """
        if execution is None:
            return EvaluatorResult(
                score=0.0,
                passed=False,
                reason="No execution data provided",
                metadata={"error": "missing_execution"},
            )

        # Extract all sources from execution
        all_sources = []
        web_search_count = 0
        llm_count = 0

        for section in execution:
            sources = section.get("sources", [])
            all_sources.extend(sources)

            tool_used = section.get("tool_used", "llm")
            if tool_used == "web_search":
                web_search_count += 1
            else:
                llm_count += 1

        # Remove "LLM Knowledge Base" pseudo-source
        real_sources = [s for s in all_sources if s != "LLM Knowledge Base"]
        unique_sources = list(set(real_sources))

        # Extract unique domains
        from urllib.parse import urlparse

        unique_domains = set()
        for source in real_sources:
            try:
                domain = urlparse(source).netloc
                if domain:
                    unique_domains.add(domain)
            except Exception:
                pass

        # Calculate scores
        source_count_score = (
            min(1.0, len(unique_sources) / self.min_sources)
            if self.min_sources > 0
            else 1.0
        )
        domain_diversity_score = (
            min(1.0, len(unique_domains) / self.min_unique_domains)
            if self.min_unique_domains > 0
            else 1.0
        )

        # Bonus for using web search
        web_search_bonus = 0.1 if web_search_count > 0 else 0.0

        score = (source_count_score + domain_diversity_score) / 2 + web_search_bonus
        score = min(1.0, score)  # Cap at 1.0

        passed = score >= self.threshold

        logger.debug(
            "source_quality_evaluation",
            unique_sources=len(unique_sources),
            unique_domains=len(unique_domains),
            web_search_count=web_search_count,
            llm_count=llm_count,
            score=score,
            passed=passed,
        )

        return EvaluatorResult(
            score=score,
            passed=passed,
            reason=f"Sources: {len(unique_sources)}/{self.min_sources}, Domains: {len(unique_domains)}/{self.min_unique_domains}, Web searches: {web_search_count}",
            metadata={
                "unique_sources": len(unique_sources),
                "unique_domains": len(unique_domains),
                "total_sources": len(real_sources),
                "web_search_count": web_search_count,
                "llm_count": llm_count,
                "source_list": unique_sources[:10],  # First 10 sources
                "domain_list": list(unique_domains),
            },
        )


class ToolSelectionEvaluator(ReasoningEvaluator):
    """Evaluates whether the agent selected appropriate tools for each query."""

    name = "tool_selection"

    def __init__(
        self,
        model: ChatOpenAI | None = None,
        threshold: float = 0.7,
    ):
        """Initialize the tool selection evaluator.

        Args:
            model: ChatOpenAI model to use for evaluation.
            threshold: Score threshold to pass.
        """
        self.model = model
        self.threshold = threshold

    def evaluate(
        self,
        question: str,
        report: str,
        plan: list[str] | None = None,
        execution: list[dict[str, Any]] | None = None,
    ) -> EvaluatorResult:
        """Evaluate the appropriateness of tool selections.

        Args:
            question: The original research question.
            report: The generated research report (not used).
            plan: The agent's plan (search queries).
            execution: The execution results (research sections with tool info).

        Returns:
            EvaluatorResult with score, pass/fail, and tool selection analysis.
        """
        if execution is None or plan is None:
            return EvaluatorResult(
                score=0.0,
                passed=False,
                reason="No execution or plan data provided",
                metadata={"error": "missing_context"},
            )

        if self.model is None:
            import os

            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.model = ChatOpenAI(model=model_name, temperature=0)

        # Build summary of tool choices
        tool_choices = []
        for section in execution:
            query = section.get("topic", "Unknown")
            tool = section.get("tool_used", "llm")
            source_count = section.get("source_count", 0)
            tool_choices.append(
                f"- Query: '{query}' → Tool: {tool} (sources: {source_count})"
            )

        choices_str = "\n".join(tool_choices)

        prompt = f"""You are an expert evaluator assessing whether an AI research agent made good decisions about which tools to use.

Original Research Question: {question}

Tool Choices Made:
{choices_str}

Evaluate the appropriateness of these tool choices on a scale of 0.0 to 1.0:
- 1.0: Excellent choices - used web search for factual/current queries, LLM for conceptual
- 0.8: Good choices - mostly appropriate with minor room for improvement
- 0.6: Acceptable - some suboptimal choices but generally reasonable
- 0.4: Poor - several queries used wrong tool
- 0.2: Very poor - most queries used inappropriate tools
- 0.0: Completely inappropriate tool selection

Consider:
1. Were factual, current, or data-driven queries researched via web search?
2. Were conceptual or explanatory queries handled by the LLM?
3. Is there good balance between web and LLM research?
4. Did web searches produce actual sources?

Respond with ONLY a JSON object in this exact format:
{{"score": <float>, "reason": "<brief explanation>", "good_choices": ["query1"], "poor_choices": ["query2"]}}"""

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
            good_choices = result.get("good_choices", [])
            poor_choices = result.get("poor_choices", [])

            passed = score >= self.threshold

            logger.debug(
                "tool_selection_evaluation",
                score=score,
                passed=passed,
                good_choices_count=len(good_choices),
                poor_choices_count=len(poor_choices),
            )

            return EvaluatorResult(
                score=score,
                passed=passed,
                reason=reason,
                metadata={
                    "good_choices": good_choices,
                    "poor_choices": poor_choices,
                    "web_search_count": sum(
                        1 for s in execution if s.get("tool_used") == "web_search"
                    ),
                    "llm_count": sum(
                        1 for s in execution if s.get("tool_used") == "llm"
                    ),
                    "model_response": content[:500],
                },
            )

        except Exception as e:
            logger.error("tool_selection_evaluation_error", error=str(e))
            return EvaluatorResult(
                score=0.5,
                passed=False,
                reason=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)},
            )


class ReasoningTraceEvaluator(ReasoningEvaluator):
    """Comprehensive evaluator for agent reasoning and planning quality.

    Based on principles from "Evaluating Reasoning and Planning in Agentic LLM Systems":
    https://apxml.com/courses/agentic-llm-memory-architectures/chapter-6-evaluation-optimization-agentic-systems/evaluating-reasoning-planning

    This evaluator assesses the internal reasoning process rather than just outputs.
    As the article emphasizes: "A sound but slightly suboptimal answer demonstrates more
    value than a correct answer derived through flawed reasoning."

    Evaluates:
    - Logical Coherence: Do successive thoughts follow rationally?
    - Goal Alignment: Is reasoning directed toward the task objective?
    - Efficiency: Does reasoning avoid unnecessary loops or redundancy?
    - Reasoning Faithfulness: Do execution results align with the plan?
    - Intermediate State Accuracy: Are necessary facts identified before synthesis?
    - Hypothesis Generation: Are plausible alternatives explored systematically?

    Provides detailed explanations suitable for MLflow reporting with:
    - Individual dimension scores (0.0-1.0) with explanations
    - Overall assessment summary
    - Identified strengths and weaknesses
    - Actionable recommendations for improvement
    """

    name = "reasoning_trace"

    def __init__(
        self,
        model: ChatOpenAI | None = None,
        threshold: float = 0.7,
    ):
        """Initialize the reasoning trace evaluator.

        Args:
            model: ChatOpenAI model to use for evaluation.
            threshold: Score threshold to pass.
        """
        self.model = model
        self.threshold = threshold

    def evaluate(
        self,
        question: str,
        report: str,
        plan: list[str] | None = None,
        execution: list[dict[str, Any]] | None = None,
    ) -> EvaluatorResult:
        """Evaluate the agent's reasoning trace comprehensively.

        Args:
            question: The original research question.
            report: The generated research report.
            plan: The agent's planned search queries.
            execution: The execution results (research sections).

        Returns:
            EvaluatorResult with detailed reasoning analysis and MLflow-ready metadata.
        """
        if plan is None or execution is None:
            return EvaluatorResult(
                score=0.0,
                passed=False,
                reason="Missing plan or execution data for reasoning evaluation",
                metadata={"error": "missing_context"},
            )

        if self.model is None:
            import os

            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.model = ChatOpenAI(model=model_name, temperature=0)

        # Build comprehensive trace context
        plan_str = "\n".join(f"{i + 1}. {query}" for i, query in enumerate(plan))

        execution_str = []
        for i, section in enumerate(execution, 1):
            topic = section.get("topic", "Unknown")
            tool = section.get("tool_used", "unknown")
            source_count = section.get("source_count", 0)
            content_preview = section.get("content", "")[:200] + "..."
            execution_str.append(
                f"{i}. Query: {topic}\n"
                f"   Tool: {tool}\n"
                f"   Sources: {source_count}\n"
                f"   Result Preview: {content_preview}"
            )

        execution_text = "\n\n".join(execution_str)

        prompt = f"""You are an expert evaluator assessing the quality of an AI agent's reasoning and planning process.

Based on principles from "Evaluating Reasoning and Planning in Agentic LLM Systems", evaluate the internal reasoning process, not just the final output. A sound but slightly suboptimal answer demonstrates more value than a correct answer derived through flawed reasoning.

## Research Question
{question}

## Agent's Plan (Search Queries)
{plan_str}

## Execution Trace (Actions Taken)
{execution_text}

## Final Report Quality
Length: {len(report)} characters
Preview: {report[:300]}...

---

Evaluate this reasoning trace across these dimensions (0.0-1.0 scale for each):

1. **Logical Coherence** (0.0-1.0)
   - Do successive queries follow rationally from the research goal?
   - Is there a clear thought progression?
   - Are there contradictions or logical gaps?

2. **Goal Alignment** (0.0-1.0)
   - Is each query consistently directed toward answering the question?
   - Do queries explore relevant aspects or drift off-topic?
   - Is the scope appropriate?

3. **Efficiency** (0.0-1.0)
   - Does the plan avoid unnecessary redundancy?
   - Are there wasteful loops or duplicate efforts?
   - Is the number of queries appropriate ({len(plan)} queries)?

4. **Reasoning Faithfulness** (0.0-1.0)
   - Do execution results align with planned intent?
   - Did the agent adapt appropriately when results differed from expectations?
   - Was tool selection (web vs LLM) appropriate for each query?

5. **Intermediate State Accuracy** (0.0-1.0)
   - Did the agent correctly identify necessary facts before synthesis?
   - Were dependencies between queries handled properly?
   - Did later queries build on earlier findings?

6. **Hypothesis Generation** (0.0-1.0)
   - Were plausible alternatives explored where appropriate?
   - Is there evidence of consideration of multiple perspectives?
   - Did the agent decompose complex questions effectively?

Provide your evaluation in this exact JSON format:
{{
  "logical_coherence": {{"score": <0.0-1.0>, "explanation": "<detailed reasoning>"}},
  "goal_alignment": {{"score": <0.0-1.0>, "explanation": "<detailed reasoning>"}},
  "efficiency": {{"score": <0.0-1.0>, "explanation": "<detailed reasoning>"}},
  "reasoning_faithfulness": {{"score": <0.0-1.0>, "explanation": "<detailed reasoning>"}},
  "intermediate_state_accuracy": {{"score": <0.0-1.0>, "explanation": "<detailed reasoning>"}},
  "hypothesis_generation": {{"score": <0.0-1.0>, "explanation": "<detailed reasoning>"}},
  "overall_assessment": "<comprehensive summary of reasoning quality>",
  "strengths": ["<strength 1>", "<strength 2>", ...],
  "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
  "recommendations": ["<improvement 1>", "<improvement 2>", ...]
}}"""

        try:
            response = self.model.invoke(prompt)
            import json

            content = response.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r"```(?:json)?\n?", "", content)
                content = content.strip()

            result = json.loads(content)

            # Extract dimension scores
            dimensions = {
                "logical_coherence": result.get("logical_coherence", {}),
                "goal_alignment": result.get("goal_alignment", {}),
                "efficiency": result.get("efficiency", {}),
                "reasoning_faithfulness": result.get("reasoning_faithfulness", {}),
                "intermediate_state_accuracy": result.get(
                    "intermediate_state_accuracy", {}
                ),
                "hypothesis_generation": result.get("hypothesis_generation", {}),
            }

            # Calculate overall score as weighted average
            dimension_scores = [
                dimensions["logical_coherence"].get("score", 0.5),
                dimensions["goal_alignment"].get("score", 0.5),
                dimensions["efficiency"].get("score", 0.5),
                dimensions["reasoning_faithfulness"].get("score", 0.5),
                dimensions["intermediate_state_accuracy"].get("score", 0.5),
                dimensions["hypothesis_generation"].get("score", 0.5),
            ]
            overall_score = sum(dimension_scores) / len(dimension_scores)

            # Extract explanations for detailed reporting
            explanations = {
                dim: data.get("explanation", "No explanation provided")
                for dim, data in dimensions.items()
            }

            passed = overall_score >= self.threshold

            # Build detailed reason text
            reason_parts = [result.get("overall_assessment", "No assessment provided")]
            if result.get("strengths"):
                reason_parts.append(
                    "Strengths: " + "; ".join(result.get("strengths", []))
                )
            if result.get("weaknesses"):
                reason_parts.append(
                    "Weaknesses: " + "; ".join(result.get("weaknesses", []))
                )

            reason = " | ".join(reason_parts)

            logger.debug(
                "reasoning_trace_evaluation",
                overall_score=overall_score,
                passed=passed,
                dimension_count=len(dimensions),
            )

            # Prepare MLflow-ready metadata
            metadata = {
                # Individual dimension scores
                "logical_coherence_score": dimensions["logical_coherence"].get(
                    "score", 0.0
                ),
                "goal_alignment_score": dimensions["goal_alignment"].get("score", 0.0),
                "efficiency_score": dimensions["efficiency"].get("score", 0.0),
                "reasoning_faithfulness_score": dimensions[
                    "reasoning_faithfulness"
                ].get("score", 0.0),
                "intermediate_state_accuracy_score": dimensions[
                    "intermediate_state_accuracy"
                ].get("score", 0.0),
                "hypothesis_generation_score": dimensions["hypothesis_generation"].get(
                    "score", 0.0
                ),
                # Explanations for each dimension
                "logical_coherence_explanation": explanations["logical_coherence"],
                "goal_alignment_explanation": explanations["goal_alignment"],
                "efficiency_explanation": explanations["efficiency"],
                "reasoning_faithfulness_explanation": explanations[
                    "reasoning_faithfulness"
                ],
                "intermediate_state_accuracy_explanation": explanations[
                    "intermediate_state_accuracy"
                ],
                "hypothesis_generation_explanation": explanations[
                    "hypothesis_generation"
                ],
                # Overall assessment
                "overall_assessment": result.get("overall_assessment", "No assessment"),
                "strengths": result.get("strengths", []),
                "weaknesses": result.get("weaknesses", []),
                "recommendations": result.get("recommendations", []),
                # Context metrics
                "plan_size": len(plan),
                "execution_steps": len(execution),
                "full_evaluation_response": content[:1000],  # Truncate for storage
            }

            return EvaluatorResult(
                score=overall_score,
                passed=passed,
                reason=reason,
                metadata=metadata,
            )

        except Exception as e:
            logger.error("reasoning_trace_evaluation_error", error=str(e))
            return EvaluatorResult(
                score=0.5,
                passed=False,
                reason=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e), "error_type": type(e).__name__},
            )


# Convenience function to get default evaluators
def get_default_evaluators(
    include_llm: bool = False,
    include_reasoning: bool = False,
    include_web_search: bool = False,
) -> list[Evaluator]:
    """Get a list of default evaluators.

    Args:
        include_llm: Whether to include LLM-based evaluators.
        include_reasoning: Whether to include reasoning evaluators (includes
            PlanQuality, PlanAdherence, and ReasoningTrace evaluators).
        include_web_search: Whether to include web search evaluators.

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

    if include_reasoning:
        evaluators.extend(
            [
                PlanQualityEvaluator(),
                PlanAdherenceEvaluator(),
                ReasoningTraceEvaluator(),
            ]
        )

    if include_web_search:
        evaluators.extend(
            [
                SourceQualityReasoningEvaluator(),
                ToolSelectionEvaluator(),
            ]
        )

    return evaluators


def run_evaluators_with_reasoning(
    question: str,
    report: str,
    evaluators: list[Evaluator] | None = None,
    plan: list[str] | None = None,
    execution: list[dict[str, Any]] | None = None,
) -> list[EvaluatorResult]:
    """Run multiple evaluators on a research report, including reasoning evaluators.

    Args:
        question: The original research question.
        report: The generated research report.
        evaluators: List of evaluators to run. Defaults to basic evaluators.
        plan: The agent's plan (search queries) for reasoning evaluators.
        execution: The execution results (research sections) for reasoning evaluators.

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

        # Check if this is a reasoning evaluator that needs extra context
        if isinstance(evaluator, ReasoningEvaluator):
            result = evaluator.evaluate(
                question, report, plan=plan, execution=execution
            )
        else:
            result = evaluator.evaluate(question, report)

        results.append(result)

        logger.info(
            "evaluator_complete",
            evaluator=evaluator.name,
            score=result.score,
            passed=result.passed,
        )

    return results
