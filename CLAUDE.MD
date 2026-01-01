# Claude Learning Context

This file tracks important learnings and context provided by the user throughout the project.

## Instructions
- Every new learning provided by the user should be included in this file unless explicitly told to avoid it
- This helps maintain context and consistency across sessions
- Organize learnings by topic or chronologically as appropriate

## Learnings

### Project Setup (2026-01-01)
- Project initialized with Python 3.12 using pyenv
- Using uv for package management
- Git repository initialized

### Development Workflow (2026-01-01)
The established workflow for this project is:
1. User requests a change
2. Claude performs the change and tests it (testing is mandatory)
3. Claude asks user to review the changes
4. Once user approves without feedback, Claude commits the changes to the repository

## Application Usage

### Setup
```bash
# Install dependencies
uv sync --extra dev

# Copy environment template and add your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=your-key-here
```

### Running the Application
```bash
# Run with a question as argument
uv run python main.py "What are the key principles of machine learning?"

# Run interactively (will prompt for question)
uv run python main.py
```

### Programmatic Usage
```python
from src.deep_research import run_research

report = run_research("Your research question", max_iterations=2)
print(report)
```

## Testing

### Run All Tests
```bash
uv run pytest
```

### Run Tests with Verbose Output
```bash
uv run pytest -v
```

### Run Specific Test File
```bash
uv run pytest tests/test_agent.py -v
```

### Run Tests with Coverage (if coverage is installed)
```bash
uv run pytest --cov=src/deep_research
```

## Linting

### Run Ruff Linter
```bash
uv run ruff check .
```

### Run Ruff Formatter
```bash
uv run ruff format .
```

### Auto-fix Linting Issues
```bash
uv run ruff check --fix .
```

## Pre-commit Hooks

Pre-commit hooks are configured to run automatically on `git commit`. They run:
1. Ruff linter (with auto-fix)
2. Ruff formatter
3. Pytest (all tests must pass)

### Manual Pre-commit Run
```bash
uv run pre-commit run --all-files
```

### Install Pre-commit Hooks (first time setup)
```bash
uv run pre-commit install
```
