"""Deep research agent implementation using LangGraph."""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .logging import get_logger, log_duration
from .prompts import TOOL_SELECTION_PROMPT
from .state import ResearchSection, ResearchState
from .tools import (
    create_deep_research_tool,
    create_query_generator,
    create_synthesizer,
    create_web_search_tool,
)

load_dotenv()

logger = get_logger("agent")


def select_research_tool(query: str, model: ChatOpenAI) -> str:
    """Determine which research tool to use for a query.

    Uses an LLM to decide whether to use web search or LLM-based research
    based on the characteristics of the query.

    Args:
        query: The research query.
        model: The LLM to use for decision making.

    Returns:
        Either "web_search" or "llm".
    """
    prompt = TOOL_SELECTION_PROMPT.format(query=query)

    try:
        response = model.invoke(prompt)
        decision = response.content.strip().lower()

        if "web_search" in decision:
            return "web_search"
        else:
            return "llm"
    except Exception as e:
        logger.warning(
            "tool_selection_failed",
            error=str(e),
            fallback="llm",
        )
        return "llm"  # Safe fallback


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

    # Create tools
    deep_research = create_deep_research_tool(model)
    web_search = create_web_search_tool(max_results=5)
    generate_queries = create_query_generator(model)
    synthesize = create_synthesizer(model)

    logger.info(
        "agent_created",
        model=model_name,
        max_iterations=max_iterations,
        web_search_enabled=web_search is not None,
    )

    def initialize(state: ResearchState) -> ResearchState:
        """Initialize the research state from the user's question."""
        logger.info("node_enter", node="initialize")

        messages = state.get("messages", [])
        question = state.get("question", "")

        if not question and messages:
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                question = last_message.content
            elif hasattr(last_message, "content"):
                question = last_message.content

        logger.info(
            "node_exit",
            node="initialize",
            question_length=len(question),
            question_preview=question[:80] + "..." if len(question) > 80 else question,
        )

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
        iteration = state["iteration"] + 1
        logger.info(
            "node_enter",
            node="generate_queries",
            iteration=iteration,
        )

        question = state["question"]

        with log_duration(logger, "query_generation", iteration=iteration):
            queries = generate_queries(question, num_queries=3)

        logger.info(
            "node_exit",
            node="generate_queries",
            iteration=iteration,
            query_count=len(queries),
            queries=queries,
        )

        return {"search_queries": queries, "iteration": iteration}

    def execute_research(state: ResearchState) -> ResearchState:
        """Execute deep research for each query."""
        queries = state["search_queries"]
        iteration = state["iteration"]

        logger.info(
            "node_enter",
            node="execute_research",
            iteration=iteration,
            query_count=len(queries),
            web_search_available=web_search is not None,
        )

        sections: list[ResearchSection] = []

        for i, query in enumerate(queries, 1):
            logger.info(
                "research_query_start",
                iteration=iteration,
                query_index=i,
                query_total=len(queries),
                query=query,
            )

            # Decide which tool to use
            if web_search is not None:
                # Force web search for the first query in the first iteration
                # This ensures research always starts with current, real-world information
                if iteration == 1 and i == 1:
                    tool_choice = "web_search"
                    logger.info(
                        "tool_selected",
                        query=query,
                        tool="web_search",
                        reason="forced_first_query",
                    )
                else:
                    tool_choice = select_research_tool(query, model)
                    logger.info("tool_selected", query=query, tool=tool_choice)
            else:
                tool_choice = "llm"
                logger.info(
                    "tool_selected",
                    query=query,
                    tool="llm",
                    reason="web_search_unavailable",
                )

            # Execute research with selected tool
            with log_duration(
                logger,
                "research_query",
                iteration=iteration,
                query_index=i,
                tool=tool_choice,
            ) as result_ctx:
                if tool_choice == "web_search":
                    content, sources = web_search.invoke(query)
                    result_ctx["content_length"] = len(content)
                    result_ctx["source_count"] = len(sources)
                else:
                    content = deep_research.invoke(query)
                    sources = ["LLM Knowledge Base"]
                    result_ctx["content_length"] = len(content)
                    result_ctx["source_count"] = 0

            # Calculate real source count (exclude "LLM Knowledge Base")
            real_source_count = len([s for s in sources if s != "LLM Knowledge Base"])

            section: ResearchSection = {
                "topic": query,
                "content": content,
                "sources": sources,
                "tool_used": tool_choice,
                "source_count": real_source_count,
            }
            sections.append(section)

        existing_sections = state.get("research_sections", [])
        total_sections = len(existing_sections) + len(sections)

        # Count tool usage
        web_search_count = sum(1 for s in sections if s["tool_used"] == "web_search")
        llm_count = sum(1 for s in sections if s["tool_used"] == "llm")

        logger.info(
            "node_exit",
            node="execute_research",
            iteration=iteration,
            new_sections=len(sections),
            total_sections=total_sections,
            web_searches=web_search_count,
            llm_searches=llm_count,
        )

        return {"research_sections": existing_sections + sections}

    def synthesize_report(state: ResearchState) -> ResearchState:
        """Synthesize research sections into a final report."""
        logger.info(
            "node_enter",
            node="synthesize",
            total_sections=len(state.get("research_sections", [])),
        )

        question = state["question"]
        sections = state["research_sections"]

        with log_duration(
            logger, "synthesis", section_count=len(sections)
        ) as result_ctx:
            report = synthesize(question, sections)
            result_ctx["report_length"] = len(report)

        logger.info(
            "node_exit",
            node="synthesize",
            report_length=len(report),
        )

        return {
            "final_report": report,
            "messages": [AIMessage(content=report)],
        }

    def should_continue(state: ResearchState) -> Literal["continue", "synthesize"]:
        """Determine if more research iterations are needed."""
        iteration = state["iteration"]
        max_iter = state["max_iterations"]
        sections = state.get("research_sections", [])

        if iteration >= max_iter:
            decision = "synthesize"
            reason = f"max iterations reached ({iteration}/{max_iter})"
        elif len(sections) >= 6:
            decision = "synthesize"
            reason = f"sufficient sections collected ({len(sections)})"
        else:
            decision = "continue"
            reason = f"continuing research (iteration {iteration}/{max_iter}, sections {len(sections)})"

        logger.info(
            "routing_decision",
            decision=decision,
            reason=reason,
            iteration=iteration,
            max_iterations=max_iter,
            section_count=len(sections),
        )

        return decision

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
    result = run_research_with_state(question, max_iterations=max_iterations)
    return result["final_report"]


def run_research_with_state(question: str, max_iterations: int = 2) -> ResearchState:
    """Run a research query and return the full state including plan and execution.

    This function returns the complete agent state, which is useful for
    reasoning evaluations that need access to:
    - search_queries: The agent's plan (what it intended to research)
    - research_sections: The execution results (what was actually researched)

    Args:
        question: The research question to investigate.
        max_iterations: Maximum research iterations.

    Returns:
        The full ResearchState dictionary containing all agent state.
    """
    logger.info(
        "research_start",
        question_length=len(question),
        max_iterations=max_iterations,
    )

    agent = create_research_agent(max_iterations=max_iterations)

    with log_duration(
        logger, "research_complete", max_iterations=max_iterations
    ) as result_ctx:
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
        result_ctx["report_length"] = len(result["final_report"])
        result_ctx["total_sections"] = len(result.get("research_sections", []))

    return result
