"""Main entry point for the deep research agent."""

import os
import sys

from src.deep_research import run_research
from src.deep_research.logging import configure_logging


def main():
    """Run the deep research agent with a question from command line or interactively."""
    # Configure logging from environment or default to INFO
    log_level = os.getenv("LOG_LEVEL", "INFO")
    json_logs = os.getenv("LOG_FORMAT", "").lower() == "json"
    configure_logging(level=log_level, json_logs=json_logs)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Deep Research Agent")
        print("=" * 40)
        question = input("Enter your research question: ").strip()

    if not question:
        print("No question provided. Exiting.")
        return

    print(f"\nResearching: {question}")
    print("-" * 40)

    try:
        report = run_research(question, max_iterations=2)
        print("\n" + "=" * 40)
        print("RESEARCH REPORT")
        print("=" * 40)
        print(report)
    except Exception as e:
        print(f"Error during research: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
