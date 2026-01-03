"""Tools for the deep research agent."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import trafilatura
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .logging import get_logger, log_duration
from .prompts import RESEARCH_PROMPT

logger = get_logger("tools")

# Web search configuration
WEB_SEARCH_TIMEOUT = 10  # seconds per URL fetch
WEB_SEARCH_MAX_CONTENT_LENGTH = 8000  # characters per article
WEB_SEARCH_FETCH_TOP_N = 3  # fetch full content for top N results
WEB_SEARCH_PARALLEL_FETCHES = True  # fetch URLs in parallel
WEB_SEARCH_MAX_RETRIES = 3  # max retry attempts for DuckDuckGo
WEB_SEARCH_RETRY_DELAY = 2  # seconds between retries


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


def fetch_url_content(url: str, timeout: int = WEB_SEARCH_TIMEOUT) -> tuple[str, bool]:
    """Fetch and extract main content from a URL.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (content_text, success_bool).
    """
    try:
        logger.debug("fetch_url_start", url=url)

        # Fetch HTML
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
            allow_redirects=True,
        )
        response.raise_for_status()

        # Extract main content using trafilatura
        content = trafilatura.extract(
            response.content,
            include_comments=False,
            include_tables=True,
            no_fallback=False,  # Try harder to extract content
        )

        if content:
            # Truncate if too long
            if len(content) > WEB_SEARCH_MAX_CONTENT_LENGTH:
                content = content[:WEB_SEARCH_MAX_CONTENT_LENGTH] + "...[truncated]"

            logger.debug(
                "fetch_url_success",
                url=url,
                content_length=len(content),
            )
            return content, True
        else:
            logger.warning("fetch_url_no_content", url=url)
            return "", False

    except requests.Timeout:
        logger.warning("fetch_url_timeout", url=url, timeout=timeout)
        return "", False
    except requests.RequestException as e:
        logger.warning(
            "fetch_url_error",
            url=url,
            error=str(e),
            error_type=type(e).__name__,
        )
        return "", False
    except Exception as e:
        logger.error(
            "fetch_url_unexpected_error",
            url=url,
            error=str(e),
            error_type=type(e).__name__,
        )
        return "", False


def create_web_search_tool(max_results: int = 5):
    """Create a web search tool using DuckDuckGo.

    Uses DuckDuckGo for open-source web search. No API key required.

    Args:
        max_results: Maximum number of search results to return (default 5).

    Returns:
        A tool function that performs web searches, or None if unavailable.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning(
            "web_search_unavailable",
            reason="ddgs not installed",
            fallback="web search disabled",
        )
        return None

    @tool
    def web_search(query: str) -> tuple[str, list[str]]:
        """Search the web for current information on a topic.

        Use this tool when you need:
        - Current events or recent information
        - Factual data, statistics, or numbers
        - Information about specific companies, products, or people
        - Recent developments in fast-moving fields

        Args:
            query: The search query.

        Returns:
            A tuple of (formatted_content, list_of_source_urls).
        """
        logger.debug(
            "web_search_tool",
            status="started",
            query_preview=query[:80] + "..." if len(query) > 80 else query,
        )

        try:
            # Retry logic for DuckDuckGo rate limiting
            results = []
            for attempt in range(WEB_SEARCH_MAX_RETRIES):
                with log_duration(
                    logger, "web_search_call", tool="web_search"
                ) as result_ctx:
                    # Use DDGS for web search
                    results = list(DDGS().text(query, max_results=max_results))
                    result_ctx["result_count"] = len(results)
                    result_ctx["attempt"] = attempt + 1

                if results:
                    # Success - got results
                    break
                elif attempt < WEB_SEARCH_MAX_RETRIES - 1:
                    # Rate limited - retry after delay
                    logger.warning(
                        "web_search_rate_limited",
                        query=query,
                        attempt=attempt + 1,
                        retry_after=WEB_SEARCH_RETRY_DELAY,
                    )
                    time.sleep(WEB_SEARCH_RETRY_DELAY)

            if not results:
                logger.warning(
                    "web_search_no_results",
                    query=query,
                    attempts=WEB_SEARCH_MAX_RETRIES,
                )
                return "No search results found.", []

            # Fetch full content for top N results
            fetch_count = min(WEB_SEARCH_FETCH_TOP_N, len(results))
            content_map = {}  # url -> content

            if WEB_SEARCH_PARALLEL_FETCHES:
                # Parallel fetching
                with ThreadPoolExecutor(max_workers=fetch_count) as executor:
                    future_to_url = {
                        executor.submit(
                            fetch_url_content, results[i].get("href", "")
                        ): i
                        for i in range(fetch_count)
                        if results[i].get("href")
                    }

                    for future in as_completed(future_to_url):
                        idx = future_to_url[future]
                        url = results[idx].get("href", "")
                        content, success = future.result()
                        if success and content:
                            content_map[url] = content
            else:
                # Sequential fetching
                for i in range(fetch_count):
                    url = results[i].get("href", "")
                    if url:
                        content, success = fetch_url_content(url)
                        if success and content:
                            content_map[url] = content

            # Format results
            sources = []
            content_parts = []
            fetch_success_count = len(content_map)

            for i, result in enumerate(results, 1):
                title = result.get("title", "Unknown")
                url = result.get("href", "")
                snippet = result.get("body", "")

                if url:
                    sources.append(url)

                # Use full content if available, else fallback to snippet
                if url in content_map:
                    main_content = content_map[url]
                    content_parts.append(
                        f"[{i}] {title}\n{main_content}\nSource: {url}\n"
                    )
                else:
                    content_parts.append(f"[{i}] {title}\n{snippet}\nSource: {url}\n")

            formatted_content = "\n".join(content_parts)

            logger.debug(
                "web_search_tool",
                status="completed",
                result_count=len(sources),
                content_length=len(formatted_content),
                fetch_attempted=fetch_count,
                fetch_success=fetch_success_count,
                fetch_rate=f"{fetch_success_count}/{fetch_count}",
            )

            return formatted_content, sources

        except Exception as e:
            logger.error(
                "web_search_error",
                error=str(e),
                error_type=type(e).__name__,
                query=query,
            )
            # Return empty results on error, don't fail the entire research
            return f"Web search failed: {str(e)}", []

    return web_search
