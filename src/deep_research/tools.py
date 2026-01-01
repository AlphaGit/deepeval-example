"""Tools for the deep research agent."""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .logging import get_logger, log_duration
from .prompts import RESEARCH_PROMPT

logger = get_logger("tools")


def create_deep_research_tool(model: ChatOpenAI):
    """Create a deep research tool that uses an LLM for research.

    Args:
        model: The ChatOpenAI model to use for research.

    Returns:
        A tool function that performs deep research on a query.
    """

    @tool
    def deep_research(query: str) -> str:
        """Perform deep research on a topic using an LLM.

        This tool takes a research query and uses an LLM to provide
        comprehensive research findings based on its knowledge.

        Args:
            query: The research query to investigate.

        Returns:
            A detailed research summary for the given query.
        """
        logger.debug(
            "deep_research_tool",
            status="started",
            query_preview=query[:80] + "..." if len(query) > 80 else query,
        )

        prompt = RESEARCH_PROMPT.format(query=query)

        with log_duration(logger, "llm_call", tool="deep_research") as result_ctx:
            response = model.invoke(prompt)
            result_ctx["response_length"] = len(response.content)

        logger.debug(
            "deep_research_tool",
            status="completed",
            response_length=len(response.content),
        )

        return response.content

    return deep_research


def create_query_generator(model: ChatOpenAI):
    """Create a function that generates search queries from a question.

    Args:
        model: The ChatOpenAI model to use for query generation.

    Returns:
        A function that generates search queries.
    """
    import json

    from .prompts import QUERY_GENERATION_PROMPT

    def generate_queries(question: str, num_queries: int = 3) -> list[str]:
        """Generate search queries for a research question.

        Args:
            question: The research question.
            num_queries: Number of queries to generate.

        Returns:
            A list of search query strings.
        """
        logger.debug(
            "query_generator",
            status="started",
            num_queries=num_queries,
        )

        prompt = QUERY_GENERATION_PROMPT.format(
            question=question, num_queries=num_queries
        )

        with log_duration(logger, "llm_call", tool="query_generator") as result_ctx:
            response = model.invoke(prompt)
            result_ctx["response_length"] = len(response.content)

        try:
            queries = json.loads(response.content)
            if isinstance(queries, list):
                logger.debug(
                    "query_generator",
                    status="completed",
                    query_count=len(queries),
                    queries=queries,
                )
                return queries
        except json.JSONDecodeError as e:
            logger.warning(
                "query_generator_json_error",
                error=str(e),
                raw_response=response.content[:200],
                fallback="using original question",
            )

        return [question]

    return generate_queries


def create_synthesizer(model: ChatOpenAI):
    """Create a function that synthesizes research findings.

    Args:
        model: The ChatOpenAI model to use for synthesis.

    Returns:
        A function that synthesizes research sections into a report.
    """
    from .prompts import SYNTHESIS_PROMPT

    def synthesize(question: str, research_sections: list[dict]) -> str:
        """Synthesize research sections into a final report.

        Args:
            question: The original research question.
            research_sections: List of research section dictionaries.

        Returns:
            A synthesized research report.
        """
        logger.debug(
            "synthesizer",
            status="started",
            section_count=len(research_sections),
        )

        sections_text = "\n\n".join(
            f"## {section['topic']}\n{section['content']}"
            for section in research_sections
        )
        prompt = SYNTHESIS_PROMPT.format(
            question=question, research_sections=sections_text
        )

        with log_duration(
            logger, "llm_call", tool="synthesizer", section_count=len(research_sections)
        ) as result_ctx:
            response = model.invoke(prompt)
            result_ctx["response_length"] = len(response.content)

        logger.debug(
            "synthesizer",
            status="completed",
            report_length=len(response.content),
        )

        return response.content

    return synthesize
