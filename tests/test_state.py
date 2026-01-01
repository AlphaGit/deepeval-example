"""Tests for the research state module."""

from src.deep_research.state import ResearchSection, ResearchState


class TestResearchSection:
    """Tests for ResearchSection TypedDict."""

    def test_research_section_structure(self):
        """Test that ResearchSection has expected keys."""
        section: ResearchSection = {
            "topic": "Test Topic",
            "content": "Test content about the topic.",
            "sources": ["Source 1", "Source 2"],
        }
        assert section["topic"] == "Test Topic"
        assert section["content"] == "Test content about the topic."
        assert section["sources"] == ["Source 1", "Source 2"]

    def test_research_section_empty_sources(self):
        """Test ResearchSection with empty sources list."""
        section: ResearchSection = {
            "topic": "Topic",
            "content": "Content",
            "sources": [],
        }
        assert section["sources"] == []


class TestResearchState:
    """Tests for ResearchState TypedDict."""

    def test_research_state_structure(self):
        """Test that ResearchState has expected structure."""
        state: ResearchState = {
            "messages": [],
            "question": "What is machine learning?",
            "search_queries": ["ML definition", "ML applications"],
            "research_sections": [],
            "final_report": "",
            "iteration": 0,
            "max_iterations": 2,
        }
        assert state["question"] == "What is machine learning?"
        assert state["iteration"] == 0
        assert state["max_iterations"] == 2

    def test_research_state_with_sections(self):
        """Test ResearchState with populated research sections."""
        section: ResearchSection = {
            "topic": "ML Basics",
            "content": "Machine learning is...",
            "sources": ["Wikipedia"],
        }
        state: ResearchState = {
            "messages": [],
            "question": "What is ML?",
            "search_queries": [],
            "research_sections": [section],
            "final_report": "Final report content",
            "iteration": 1,
            "max_iterations": 2,
        }
        assert len(state["research_sections"]) == 1
        assert state["research_sections"][0]["topic"] == "ML Basics"
        assert state["final_report"] == "Final report content"
