"""Tests for the prompts module."""

from src.deep_research.prompts import (
    FOLLOW_UP_PROMPT,
    QUERY_GENERATION_PROMPT,
    RESEARCH_PROMPT,
    SYNTHESIS_PROMPT,
)


class TestPrompts:
    """Tests for prompt templates."""

    def test_query_generation_prompt_has_placeholders(self):
        """Test that query generation prompt has required placeholders."""
        assert "{num_queries}" in QUERY_GENERATION_PROMPT
        assert "{question}" in QUERY_GENERATION_PROMPT

    def test_query_generation_prompt_formatting(self):
        """Test that query generation prompt can be formatted."""
        formatted = QUERY_GENERATION_PROMPT.format(
            num_queries=3, question="What is Python?"
        )
        assert "3" in formatted
        assert "What is Python?" in formatted

    def test_research_prompt_has_placeholders(self):
        """Test that research prompt has required placeholders."""
        assert "{query}" in RESEARCH_PROMPT

    def test_research_prompt_formatting(self):
        """Test that research prompt can be formatted."""
        formatted = RESEARCH_PROMPT.format(query="Python programming")
        assert "Python programming" in formatted

    def test_synthesis_prompt_has_placeholders(self):
        """Test that synthesis prompt has required placeholders."""
        assert "{question}" in SYNTHESIS_PROMPT
        assert "{research_sections}" in SYNTHESIS_PROMPT

    def test_synthesis_prompt_formatting(self):
        """Test that synthesis prompt can be formatted."""
        formatted = SYNTHESIS_PROMPT.format(
            question="What is AI?",
            research_sections="## Section 1\nContent here",
        )
        assert "What is AI?" in formatted
        assert "## Section 1" in formatted

    def test_follow_up_prompt_has_placeholders(self):
        """Test that follow-up prompt has required placeholders."""
        assert "{question}" in FOLLOW_UP_PROMPT
        assert "{current_summary}" in FOLLOW_UP_PROMPT

    def test_follow_up_prompt_formatting(self):
        """Test that follow-up prompt can be formatted."""
        formatted = FOLLOW_UP_PROMPT.format(
            question="What is cloud computing?",
            current_summary="Cloud computing is...",
        )
        assert "What is cloud computing?" in formatted
        assert "Cloud computing is..." in formatted
