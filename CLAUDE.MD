# Claude Development Guide

This file contains instructions and context for AI agents developing this project.

## Development Workflow

The established workflow for this project is:
1. User requests a change
2. Perform the change and test it (testing is mandatory)
3. Ask user to review the changes
4. **Always ask for permission before committing** - never commit without explicit user approval
5. Once user approves without feedback, commit the changes to the repository

## Project Overview

This is a LangGraph-based deep research agent with integrated evaluation, MLflow tracking, and structured logging.

### Tech Stack
- Python 3.12 (managed with pyenv)
- Package manager: uv
- Agent framework: LangGraph
- LLM provider: OpenAI
- Tracking: MLflow (SQLite backend)
- Logging: structlog

### Core Architecture Patterns

**State Management**: All agent state is defined in `state.py` using TypedDict for type safety.

**Evaluation System**: Three-tier evaluator architecture:
- Basic evaluators (synchronous, rule-based): Length, Structure
- LLM evaluators (async, model-based): Relevance, Completeness
- Reasoning evaluators (require plan/execution context): PlanQuality, PlanAdherence

**Observability**: Every operation should be observable:
- Use structured logging via `get_logger(__name__)` from `logging.py`
- MLflow automatically tracks experiments via `track_research_run` context manager
- LLM traces are captured automatically when tracing is enabled

## Development Guidelines

### Adding New Features

**New Evaluators**:
- Inherit from `Evaluator` or `ReasoningEvaluator` in `evaluators.py`
- Follow existing evaluator patterns for structure
- Add tests to `tests/test_evaluators.py`
- Export from `__init__.py`

**New Agent Nodes**:
- Add to `agent.py` following LangGraph patterns
- Update state in `state.py` if new fields are needed
- Add prompts to `prompts.py` for consistency
- Use structured logging throughout

**MLflow Integration**:
- Use `track_research_run()` context manager for any research operations
- MLflow uses SQLite backend (mlflow.db) not filesystem
- All LLM calls are automatically traced when enabled

### Code Quality

**Testing**: All tests must pass before committing
```bash
uv run pytest                    # Run all tests
uv run pytest -v                 # Verbose output
uv run pytest tests/test_*.py    # Specific test file
```

**Linting**: Use Ruff for formatting and linting
```bash
uv run ruff check .              # Check for issues
uv run ruff check --fix .        # Auto-fix issues
uv run ruff format .             # Format code
```

**Pre-commit Hooks**: Automatically run on commit
- Ruff linter with auto-fix
- Ruff formatter
- Pytest (all tests must pass)
- Install once with: `uv run pre-commit install`

### Important Conventions

- Use structured logging: `logger = get_logger(__name__)`
- Type hints required for all function signatures
- Docstrings for all public functions and classes
- Tests are mandatory for all new features
- Follow existing patterns in the codebase
- Export new public APIs from `__init__.py`
