# GitHub Actions Workflows

This directory contains GitHub Actions workflows that automatically run tests for the osmosis-ai package.

## Workflows

### test.yml

This workflow runs on:
- Every push to the main branch
- Every pull request to the main branch

It performs the following steps:
1. Sets up Python environments (3.8, 3.9, 3.10)
2. Caches dependencies to speed up subsequent runs
3. Installs the required dependencies
4. Runs the pytest tests in test.py

The workflow validates that the osmosis-ai functionality works correctly with various LLM API clients including:
- Anthropic
- OpenAI
- LangChain

### Caching

The workflow implements caching to speed up CI/CD:
- Pip package cache - Reduces time spent downloading packages
- Pytest cache - Preserves test session information between runs

Caching is implemented based on:
- Python version
- Operating system
- Dependencies in setup.py

## Test Skip Behavior

Some tests are conditionally skipped if the corresponding packages are not installed. This allows the test suite to run even if optional dependencies are not present. 