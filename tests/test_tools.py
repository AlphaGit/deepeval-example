"""Tests for the tools module."""

from unittest.mock import MagicMock

import requests


class TestCreateDeepResearchTool:
    """Tests for the deep research tool factory."""

    def test_deep_research_tool_creation(self):
        """Test that deep research tool can be created."""
        from src.deep_research.tools import create_deep_research_tool

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="Research results")

        tool = create_deep_research_tool(mock_model)
        assert tool is not None
        assert callable(tool.invoke)

    def test_deep_research_tool_invocation(self):
        """Test that deep research tool calls the model correctly."""
        from src.deep_research.tools import create_deep_research_tool

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content="Detailed research on Python"
        )

        tool = create_deep_research_tool(mock_model)
        result = tool.invoke("Python programming")

        assert result == "Detailed research on Python"
        mock_model.invoke.assert_called_once()
        call_args = mock_model.invoke.call_args[0][0]
        assert "Python programming" in call_args


class TestCreateQueryGenerator:
    """Tests for the query generator factory."""

    def test_query_generator_creation(self):
        """Test that query generator can be created."""
        from src.deep_research.tools import create_query_generator

        mock_model = MagicMock()
        generator = create_query_generator(mock_model)
        assert callable(generator)

    def test_query_generator_returns_list(self):
        """Test that query generator returns a list of queries."""
        from src.deep_research.tools import create_query_generator

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content='["query 1", "query 2", "query 3"]'
        )

        generator = create_query_generator(mock_model)
        queries = generator("What is machine learning?", num_queries=3)

        assert isinstance(queries, list)
        assert len(queries) == 3
        assert queries == ["query 1", "query 2", "query 3"]

    def test_query_generator_handles_invalid_json(self):
        """Test that query generator handles invalid JSON gracefully."""
        from src.deep_research.tools import create_query_generator

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="not valid json")

        generator = create_query_generator(mock_model)
        queries = generator("What is AI?")

        assert queries == ["What is AI?"]


class TestCreateSynthesizer:
    """Tests for the synthesizer factory."""

    def test_synthesizer_creation(self):
        """Test that synthesizer can be created."""
        from src.deep_research.tools import create_synthesizer

        mock_model = MagicMock()
        synthesizer = create_synthesizer(mock_model)
        assert callable(synthesizer)

    def test_synthesizer_combines_sections(self):
        """Test that synthesizer properly combines research sections."""
        from src.deep_research.tools import create_synthesizer

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(
            content="Synthesized report about the topic"
        )

        synthesizer = create_synthesizer(mock_model)
        sections = [
            {"topic": "Topic 1", "content": "Content 1"},
            {"topic": "Topic 2", "content": "Content 2"},
        ]
        result = synthesizer("Research question", sections)

        assert result == "Synthesized report about the topic"
        mock_model.invoke.assert_called_once()
        call_args = mock_model.invoke.call_args[0][0]
        assert "Research question" in call_args
        assert "Topic 1" in call_args
        assert "Topic 2" in call_args


class TestFetchUrlContent:
    """Tests for URL content fetching."""

    def test_fetch_url_content_success(self, monkeypatch):
        """Test successful content extraction."""
        from src.deep_research.tools import fetch_url_content

        mock_html = """
        <html>
        <body>
            <article>
                <h1>Main Article</h1>
                <p>This is the main content.</p>
            </article>
        </body>
        </html>
        """

        mock_response = MagicMock()
        mock_response.content = mock_html.encode("utf-8")
        mock_response.raise_for_status = MagicMock()

        monkeypatch.setattr("requests.get", lambda *a, **k: mock_response)

        content, success = fetch_url_content("https://example.com")

        assert success is True
        assert "main content" in content.lower()

    def test_fetch_url_content_timeout(self, monkeypatch):
        """Test timeout handling."""
        from src.deep_research.tools import fetch_url_content

        def mock_get(*args, **kwargs):
            raise requests.Timeout("Connection timeout")

        monkeypatch.setattr("requests.get", mock_get)

        content, success = fetch_url_content("https://example.com")

        assert success is False
        assert content == ""

    def test_fetch_url_content_truncation(self, monkeypatch):
        """Test content truncation for large pages."""
        from src.deep_research.tools import (
            WEB_SEARCH_MAX_CONTENT_LENGTH,
            fetch_url_content,
        )

        long_content = "x" * (WEB_SEARCH_MAX_CONTENT_LENGTH + 1000)
        mock_html = f"<html><body><p>{long_content}</p></body></html>"

        mock_response = MagicMock()
        mock_response.content = mock_html.encode("utf-8")
        mock_response.raise_for_status = MagicMock()

        monkeypatch.setattr("requests.get", lambda *a, **k: mock_response)

        content, success = fetch_url_content("https://example.com")

        assert success is True
        assert "[truncated]" in content


class TestEnhancedWebSearch:
    """Tests for the enhanced web search with content fetching."""

    def test_web_search_with_content_fetching(self, monkeypatch):
        """Test web search fetches and includes full content."""
        from src.deep_research.tools import create_web_search_tool

        mock_results = [
            {
                "title": "Article 1",
                "href": "https://example.com/article1",
                "body": "Short snippet 1",
            }
        ]

        # Mock DuckDuckGo
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = mock_results

        monkeypatch.setattr("ddgs.DDGS", lambda: mock_ddgs)

        # Mock fetch_url_content
        def mock_fetch(url, timeout=10):
            return "Full content of article 1 with much more detail.", True

        monkeypatch.setattr(
            "src.deep_research.tools.fetch_url_content",
            mock_fetch,
        )

        tool = create_web_search_tool(max_results=2)
        content, sources = tool.invoke("test query")

        assert "Full content of article 1" in content
        assert "much more detail" in content
