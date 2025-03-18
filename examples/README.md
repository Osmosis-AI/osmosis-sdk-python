# Osmosis Wrap Examples

This directory contains example code demonstrating how to use osmosis-wrap with various LLM libraries.

## Prerequisites

Before running these examples, make sure you have:

1. Installed osmosis-wrap and its dependencies
2. Set up environment variables for your API keys

```bash
# Copy the sample .env file (from the project root)
cp ../.env.sample .env

# Edit the .env file with your API keys
```

Edit the `.env` file to add your API keys:

```
# Required for logging
HOOVER_API_KEY=your_hoover_api_key_here

# Optional: Only needed if you're using these services
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Examples

### Comprehensive Example

`combined_example.py` provides a full demonstration of osmosis-wrap with multiple LLM providers:

```bash
python examples/combined_example.py
```

This example:
- Initializes osmosis-wrap
- Makes API calls to Anthropic Claude (if available)
- Makes API calls to OpenAI GPT (if available)
- Demonstrates LangChain integration (if available)
- Shows how to toggle Hoover logging on and off

### Individual Provider Examples

Each example demonstrates osmosis-wrap with a specific provider:

- **Anthropic**: `anthropic_example.py` - Shows how to use osmosis-wrap with the Anthropic Claude API
- **OpenAI**: `openai_example.py` - Shows how to use osmosis-wrap with the OpenAI API
- **LangChain**: `langchain_example.py` - Shows how to use osmosis-wrap with LangChain

Run any example with:

```bash
python examples/[example_file].py
```

### Tests

`test_examples.py` contains unit tests for osmosis-wrap functionality. Run with:

```bash
pytest examples/test_examples.py
```

## Troubleshooting

If you encounter errors:

1. **API Key Issues**: Make sure you've set up all required API keys in the `.env` file
2. **Missing Dependencies**: Install dependencies for the LLM provider you want to use:
   ```bash
   # For Anthropic
   pip install anthropic
   
   # For OpenAI
   pip install openai
   
   # For LangChain
   pip install langchain-core
   ```
3. **Hoover Connectivity**: Ensure your Hoover API key is valid and the service is accessible 