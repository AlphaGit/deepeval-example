# Deep Research Agent

A LangGraph-based deep research agent that uses OpenAI LLMs to perform comprehensive research on any topic.

## Features

- **Deep Research Pattern**: Multi-iteration research workflow with query generation, parallel research, and synthesis
- **LangGraph Integration**: Built on LangGraph's StateGraph for structured agent workflows
- **OpenAI LLMs**: Uses OpenAI models for research and synthesis
- **Configurable**: Customize model, iterations, and research depth

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd deepeval-example

# Install dependencies with uv
uv sync

# For development (includes test dependencies)
uv sync --extra dev
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   OPENAI_MODEL=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
   ```

## Usage

### Command Line

```bash
# Run with a question as argument
uv run python main.py "What are the key principles of machine learning?"

# Or run interactively
uv run python main.py
```

### Programmatic Usage

```python
from src.deep_research import run_research, create_research_agent

# Simple usage
report = run_research("What is quantum computing?", max_iterations=2)
print(report)

# Advanced usage with custom agent
agent = create_research_agent(model_name="gpt-4o", max_iterations=3)
result = agent.invoke({
    "messages": [],
    "question": "Explain neural networks",
    "search_queries": [],
    "research_sections": [],
    "final_report": "",
    "iteration": 0,
    "max_iterations": 3,
})
print(result["final_report"])
```

## Architecture

The agent follows the deep research pattern:

1. **Initialize**: Extract the research question from user input
2. **Generate Queries**: Create multiple search queries to explore different aspects
3. **Execute Research**: Use LLM to perform deep research on each query
4. **Synthesize**: Combine findings into a comprehensive report
5. **Iterate**: Optionally repeat for more depth

```
+------------------+
|   Initialize     |
+--------+---------+
         |
         v
+------------------+
| Generate Queries |<---------+
+--------+---------+          |
         |                    |
         v                    |
+------------------+          |
| Execute Research |          |
+--------+---------+          |
         |                    |
         v                    |
    +--------+                |
    | More?  |--Yes-----------+
    +---+----+
        |
        No
        |
        v
+------------------+
|   Synthesize     |
+--------+---------+
         |
         v
      [END]
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_agent.py -v
```

## Linting

```bash
# Run ruff linter
uv run ruff check .

# Run ruff formatter
uv run ruff format .

# Auto-fix linting issues
uv run ruff check --fix .
```

## Pre-commit Hooks

Pre-commit hooks are configured to run tests and linting on each commit.

```bash
# Install pre-commit hooks (first time setup)
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files
```

## Project Structure

```
deepeval-example/
├── main.py                 # CLI entry point
├── src/
│   └── deep_research/
│       ├── __init__.py     # Package exports
│       ├── agent.py        # LangGraph agent implementation
│       ├── prompts.py      # LLM prompt templates
│       ├── state.py        # State type definitions
│       └── tools.py        # Research tools
├── tests/
│   ├── conftest.py         # Test configuration
│   ├── test_agent.py       # Agent tests
│   ├── test_prompts.py     # Prompt tests
│   ├── test_state.py       # State tests
│   └── test_tools.py       # Tools tests
├── .env.example            # Environment template
├── .pre-commit-config.yaml # Pre-commit hooks config
├── .gitignore
├── pyproject.toml
└── README.md
```

## License

MIT
