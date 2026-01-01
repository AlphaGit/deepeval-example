"""State definitions for the deep research agent."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class ResearchSection(TypedDict):
    """A section of research findings."""

    topic: str
    content: str
    sources: list[str]


class ResearchState(TypedDict):
    """State for the deep research agent.

    Attributes:
        messages: The conversation history with the user.
        question: The original research question.
        search_queries: Generated search queries for research.
        research_sections: Collected research findings by section.
        final_report: The synthesized final research report.
        iteration: Current research iteration count.
        max_iterations: Maximum allowed research iterations.
    """

    messages: Annotated[list, add_messages]
    question: str
    search_queries: list[str]
    research_sections: list[ResearchSection]
    final_report: str
    iteration: int
    max_iterations: int
