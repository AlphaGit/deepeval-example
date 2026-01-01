"""Deep research agent implementation using LangGraph."""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .state import ResearchSection, ResearchState
from .tools import create_deep_research_tool, create_query_generator, create_synthesizer

load_dotenv()


def create_research_agent(
    model_name: str | None = None,
    max_iterations: int = 2,
):
    """Create a deep research agent using LangGraph.

    The agent follows a deep research pattern:
    1. Generate search queries from the research question
    2. Execute deep research for each query in parallel
    3. Synthesize findings into a comprehensive report
    4. Optionally iterate for more depth

    Args:
        model_name: The OpenAI model to use (defaults to OPENAI_MODEL env var or gpt-4o-mini).
        max_iterations: Maximum research iterations allowed.

    Returns:
        A compiled LangGraph StateGraph for research.
    """
    model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    model = ChatOpenAI(model=model_name, temperature=0.7)

    deep_research = create_deep_research_tool(model)
    generate_queries = create_query_generator(model)
    synthesize = create_synthesizer(model)

    def initialize(state: ResearchState) -> ResearchState:
        """Initialize the research state from the user's question."""
        messages = state.get("messages", [])
        question = state.get("question", "")

        if not question and messages:
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                question = last_message.content
            elif hasattr(last_message, "content"):
                question = last_message.content

        return {
            "question": question,
            "search_queries": [],
            "research_sections": [],
            "final_report": "",
            "iteration": 0,
            "max_iterations": state.get("max_iterations", max_iterations),
        }

    def generate_queries_node(state: ResearchState) -> ResearchState:
        """Generate search queries for the research question."""
        question = state["question"]
        queries = generate_queries(question, num_queries=3)
        return {"search_queries": queries, "iteration": state["iteration"] + 1}

    def execute_research(state: ResearchState) -> ResearchState:
        """Execute deep research for each query."""
        queries = state["search_queries"]
        sections: list[ResearchSection] = []

        for query in queries:
            result = deep_research.invoke(query)
            section: ResearchSection = {
                "topic": query,
                "content": result,
                "sources": ["LLM Knowledge Base"],
            }
            sections.append(section)

        existing_sections = state.get("research_sections", [])
        return {"research_sections": existing_sections + sections}

    def synthesize_report(state: ResearchState) -> ResearchState:
        """Synthesize research sections into a final report."""
        question = state["question"]
        sections = state["research_sections"]
        report = synthesize(question, sections)
        return {
            "final_report": report,
            "messages": [AIMessage(content=report)],
        }

    def should_continue(state: ResearchState) -> Literal["continue", "synthesize"]:
        """Determine if more research iterations are needed."""
        iteration = state["iteration"]
        max_iter = state["max_iterations"]

        if iteration >= max_iter:
            return "synthesize"

        sections = state.get("research_sections", [])
        if len(sections) >= 6:
            return "synthesize"

        return "continue"

    graph = StateGraph(ResearchState)

    graph.add_node("initialize", initialize)
    graph.add_node("generate_queries", generate_queries_node)
    graph.add_node("execute_research", execute_research)
    graph.add_node("synthesize", synthesize_report)

    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "generate_queries")
    graph.add_edge("generate_queries", "execute_research")
    graph.add_conditional_edges(
        "execute_research",
        should_continue,
        {"continue": "generate_queries", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_research(question: str, max_iterations: int = 2) -> str:
    """Run a research query and return the final report.

    Args:
        question: The research question to investigate.
        max_iterations: Maximum research iterations.

    Returns:
        The final research report as a string.
    """
    agent = create_research_agent(max_iterations=max_iterations)
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "search_queries": [],
            "research_sections": [],
            "final_report": "",
            "iteration": 0,
            "max_iterations": max_iterations,
        }
    )
    return result["final_report"]
