"""Tests for the deep research agent."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage


class TestCreateResearchAgent:
    """Tests for the research agent factory."""

    @patch("src.deep_research.agent.ChatOpenAI")
    def test_agent_creation(self, mock_openai):
        """Test that research agent can be created."""
        from src.deep_research.agent import create_research_agent

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content='["query1"]')
        mock_openai.return_value = mock_model

        agent = create_research_agent()
        assert agent is not None

    @patch("src.deep_research.agent.ChatOpenAI")
    def test_agent_creation_with_custom_model(self, mock_openai):
        """Test agent creation with custom model name."""
        from src.deep_research.agent import create_research_agent

        mock_model = MagicMock()
        mock_openai.return_value = mock_model

        create_research_agent(model_name="gpt-4")
        mock_openai.assert_called_with(model="gpt-4", temperature=0.7)

    @patch("src.deep_research.agent.ChatOpenAI")
    def test_agent_creation_with_max_iterations(self, mock_openai):
        """Test agent creation with custom max iterations."""
        from src.deep_research.agent import create_research_agent

        mock_model = MagicMock()
        mock_openai.return_value = mock_model

        agent = create_research_agent(max_iterations=5)
        assert agent is not None


class TestRunResearch:
    """Tests for the run_research function."""

    @patch("src.deep_research.agent.create_research_agent")
    def test_run_research_returns_report(self, mock_create_agent):
        """Test that run_research returns a final report."""
        from src.deep_research.agent import run_research

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"final_report": "This is the research report"}
        mock_create_agent.return_value = mock_agent

        result = run_research("What is Python?")

        assert result == "This is the research report"
        mock_agent.invoke.assert_called_once()

    @patch("src.deep_research.agent.create_research_agent")
    def test_run_research_passes_question(self, mock_create_agent):
        """Test that run_research passes the question correctly."""
        from src.deep_research.agent import run_research

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"final_report": "Report"}
        mock_create_agent.return_value = mock_agent

        run_research("How does AI work?")

        call_args = mock_agent.invoke.call_args[0][0]
        assert call_args["question"] == "How does AI work?"

    @patch("src.deep_research.agent.create_research_agent")
    def test_run_research_with_max_iterations(self, mock_create_agent):
        """Test that run_research respects max_iterations."""
        from src.deep_research.agent import run_research

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"final_report": "Report"}
        mock_create_agent.return_value = mock_agent

        run_research("Question", max_iterations=5)

        mock_create_agent.assert_called_with(max_iterations=5)
        call_args = mock_agent.invoke.call_args[0][0]
        assert call_args["max_iterations"] == 5


class TestAgentNodes:
    """Tests for individual agent node functions."""

    @patch("src.deep_research.agent.ChatOpenAI")
    def test_initialize_extracts_question_from_messages(self, mock_openai):
        """Test that initialize node extracts question from messages."""
        from src.deep_research.agent import create_research_agent

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content='["q1", "q2", "q3"]')
        mock_openai.return_value = mock_model

        agent = create_research_agent(max_iterations=1)

        mock_model.invoke.return_value = MagicMock(content="Research content")

        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Test question")],
                "question": "",
                "search_queries": [],
                "research_sections": [],
                "final_report": "",
                "iteration": 0,
                "max_iterations": 1,
            }
        )

        assert "final_report" in result
