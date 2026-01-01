"""Tools for the deep research agent."""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .prompts import RESEARCH_PROMPT


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
        prompt = RESEARCH_PROMPT.format(query=query)
        response = model.invoke(prompt)
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
        prompt = QUERY_GENERATION_PROMPT.format(
            question=question, num_queries=num_queries
        )
        response = model.invoke(prompt)
        try:
            queries = json.loads(response.content)
            if isinstance(queries, list):
                return queries
        except json.JSONDecodeError:
            pass
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
        sections_text = "\n\n".join(
            f"## {section['topic']}\n{section['content']}"
            for section in research_sections
        )
        prompt = SYNTHESIS_PROMPT.format(
            question=question, research_sections=sections_text
        )
        response = model.invoke(prompt)
        return response.content

    return synthesize
