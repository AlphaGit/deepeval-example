# Claude Development Guide

## Workflow (CRITICAL)
1. Make change + test (mandatory: `uv run pytest`)
2. Ask user to review
3. **NEVER commit without explicit user approval**
4. Commit only after user approves

## Tech Stack
Python 3.12 + uv | LangGraph agent | OpenAI | MLflow (SQLite) | structlog

## Architecture
- **State**: `state.py` (TypedDict)
- **Evaluators**: 3 tiers - Basic (sync), LLM (async), Reasoning (requires plan context)
- **Logging**: `get_logger(__name__)` from `logging.py`
- **Tracking**: Wrap operations in `track_research_run()` context manager
- **MLflow**: Uses SQLite backend (mlflow.db), not filesystem

## Commands
```bash
uv run pytest              # Test (required before commit)
uv run ruff check --fix .  # Lint + auto-fix
uv run ruff format .       # Format
```

## Conventions
- Type hints + docstrings for all public APIs
- Export new APIs from `__init__.py`
- Follow existing patterns (check similar files first)
- Structured logging required
- Tests mandatory for new features

## File Locations
- Evaluators: `evaluators.py` → tests in `tests/test_evaluators.py`
- Agent nodes: `agent.py` (update `state.py` if needed)
- Prompts: `prompts.py`
