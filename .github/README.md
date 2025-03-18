# GitHub Actions Workflows

This directory contains GitHub Actions workflows that automatically run tests for the osmosis-wrap package.

## Workflows

### test.yml

This workflow runs on:
- Every push to the main branch
- Every pull request to the main branch

It performs the following steps:
1. Sets up Python environments (3.8, 3.9, 3.10)
2. Installs the required dependencies
3. Runs the pytest tests in test.py

The workflow validates that the osmosis-wrap functionality works correctly with various LLM API clients including:
- Anthropic
- OpenAI
- LangChain

## Test Skip Behavior

Some tests are conditionally skipped if the corresponding packages are not installed. This allows the test suite to run even if optional dependencies are not present. 