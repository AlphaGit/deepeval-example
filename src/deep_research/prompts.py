"""Prompts for the deep research agent."""

QUERY_GENERATION_PROMPT = """You are a research assistant that generates search queries.

Given a research question, generate {num_queries} diverse search queries that would help
gather comprehensive information to answer the question.

The queries should:
1. Cover different aspects of the topic
2. Use varied phrasing to capture different perspectives
3. Include both broad and specific queries

Research Question: {question}

Respond with a JSON array of search query strings, nothing else.
Example: ["query 1", "query 2", "query 3"]
"""

RESEARCH_PROMPT = """You are a deep research assistant. Your task is to research a topic
and provide comprehensive, well-structured findings.

Research Query: {query}

Based on your knowledge, provide a detailed research summary covering:
1. Key facts and information
2. Different perspectives or viewpoints
3. Recent developments (if applicable)
4. Important considerations

Provide your research findings in a clear, organized format.
"""

SYNTHESIS_PROMPT = """You are a research synthesis expert. Your task is to synthesize
multiple research findings into a comprehensive, coherent report.

Original Question: {question}

Research Findings:
{research_sections}

Create a final research report that:
1. Directly answers the original question
2. Integrates insights from all research sections
3. Highlights key findings and conclusions
4. Notes any conflicting information or uncertainties
5. Provides actionable insights where applicable

Write a well-structured report with clear sections and conclusions.
"""

FOLLOW_UP_PROMPT = """Based on the current research findings, determine if more research
is needed to fully answer the question.

Original Question: {question}

Current Research Summary:
{current_summary}

Should more research be conducted? Consider:
1. Are there gaps in the current findings?
2. Are there aspects of the question not yet addressed?
3. Would additional research significantly improve the answer?

Respond with JSON: {{"needs_more_research": true/false, "reason": "explanation", "suggested_queries": ["query1", "query2"] if needed}}
"""
