# Deep Research Agent

A LangGraph-based deep research agent that uses OpenAI LLMs to perform comprehensive research on any topic, with integrated evaluation, MLflow tracking, and structured logging for full observability.

## Features

- **Deep Research Pattern**: Multi-iteration research workflow with query generation, parallel research, and synthesis
- **LangGraph Integration**: Built on LangGraph's StateGraph for structured agent workflows
- **OpenAI LLMs**: Uses OpenAI models for research and synthesis
- **Automated Evaluation**: Multiple evaluators to assess report quality:
  - Length and structure validation
  - LLM-based relevance and completeness checking
  - Reasoning evaluators for plan quality and adherence
- **MLflow Tracking**: Automatic experiment tracking with LLM tracing, metrics, and artifact storage
- **Structured Logging**: JSON and colored console logging with structlog for observability
- **Configurable**: Customize model, iterations, research depth, and evaluation criteria

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

2. Edit `.env` and configure the following:
   ```bash
   # Required: OpenAI API key
   OPENAI_API_KEY=your-api-key-here

   # Optional: OpenAI model (defaults to gpt-4o-mini)
   OPENAI_MODEL=gpt-4o-mini

   # Optional: Logging configuration
   LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
   LOG_FORMAT=console  # "json" for JSON output, anything else for colored console

   # Optional: MLflow tracking (defaults to sqlite:///mlflow.db)
   MLFLOW_TRACKING_URI=sqlite:///mlflow.db
   ```

## Usage

### Command Line

The CLI automatically runs research with evaluation and MLflow tracking:

```bash
# Run with a question as argument
uv run python main.py "What are the key principles of machine learning?"

# Or run interactively
uv run python main.py
```

After running, view detailed results including LLM traces in the MLflow UI:

```bash
mlflow ui
# Then open http://localhost:5000
```

### Programmatic Usage

```python
from src.deep_research import (
    run_research_with_evaluation,
    run_research,
    create_research_agent,
)

# Full evaluation with MLflow tracking (recommended)
result = run_research_with_evaluation(
    "What is quantum computing?",
    max_iterations=2,
    include_llm_evaluators=True,
    include_reasoning_evaluators=True,
)
print(result.report)
for eval in result.evaluations:
    print(f"{eval.name}: {eval.score:.2f} ({'PASS' if eval.passed else 'FAIL'})")

# Simple usage without evaluation
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

The agent follows the deep research pattern with integrated evaluation and tracking:

### Core Research Flow

1. **Initialize**: Extract the research question from user input
2. **Generate Queries**: Create multiple search queries to explore different aspects
3. **Execute Research**: Use LLM to perform deep research on each query
4. **Synthesize**: Combine findings into a comprehensive report
5. **Iterate**: Optionally repeat for more depth

### Evaluation System

The system includes multiple evaluator types:

- **Basic Evaluators**: Length and structure validation
- **LLM-based Evaluators**: Relevance and completeness assessment using GPT models
- **Reasoning Evaluators**: Plan quality and adherence tracking for agent transparency

### Observability

- **Structured Logging**: All operations logged with structlog for debugging and monitoring
- **MLflow Tracking**: Automatic experiment tracking with:
  - LLM traces for every API call
  - Metrics for iteration count, section count, and evaluation scores
  - Artifacts including the final report and original question
  - Evaluation metadata and pass/fail rates

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
├── main.py                 # CLI entry point with evaluation
├── src/
│   └── deep_research/
│       ├── __init__.py     # Package exports
│       ├── agent.py        # LangGraph agent implementation
│       ├── prompts.py      # LLM prompt templates
│       ├── state.py        # State type definitions
│       ├── tools.py        # Research tools
│       ├── evaluators.py   # Report evaluation system
│       ├── evaluate.py     # Integrated evaluation workflow
│       ├── logging.py      # Structured logging configuration
│       └── tracking.py     # MLflow tracking integration
├── tests/
│   ├── conftest.py         # Test configuration
│   ├── test_agent.py       # Agent tests
│   ├── test_prompts.py     # Prompt tests
│   ├── test_state.py       # State tests
│   ├── test_tools.py       # Tools tests
│   ├── test_evaluators.py # Evaluator tests
│   ├── test_logging.py     # Logging tests
│   └── test_tracking.py    # Tracking tests
├── mlflow.db               # SQLite database for MLflow (created on first run)
├── mlruns/                 # MLflow experiment data and traces
├── .env.example            # Environment template
├── .pre-commit-config.yaml # Pre-commit hooks config
├── .gitignore
├── pyproject.toml
├── CLAUDE.MD               # Project context and learnings
└── README.md
```

## License

MIT
