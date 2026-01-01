"""Tests for the tools module."""

from unittest.mock import MagicMock


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
