[![Run Tests](https://github.com/Gulp-AI/osmosis-wrap/actions/workflows/test.yml/badge.svg)](https://github.com/Gulp-AI/osmosis-wrap/actions/workflows/test.yml)

# Osmosis Wrap

A Python library that monkey patches LLM client libraries to send all prompts and responses to the Hoover API for logging and monitoring.

## Supported Libraries

- **Anthropic**: Logs all Claude API requests and responses (both sync and async clients)
- **OpenAI**: Logs all OpenAI API requests and responses (supports v1 and v2 API versions, both sync and async clients)
- **LangChain**: Currently supports prompt template logging (LLM and ChatModel support varies by LangChain version)

## Installation

[pypi](https://pypi.org/project/osmosis-wrap/)

```bash
# Basic installation with minimal dependencies
pip install osmosis-wrap

# Install with specific provider support
pip install "osmosis-wrap[openai]"     # Only OpenAI support
pip install "osmosis-wrap[anthropic]"  # Only Anthropic support

# Install with LangChain support
pip install "osmosis-wrap[langchain]"         # Base LangChain support
pip install "osmosis-wrap[langchain-openai]"  # LangChain + OpenAI support
pip install "osmosis-wrap[langchain-anthropic]" # LangChain + Anthropic support

# Install with all dependencies
pip install "osmosis-wrap[all]"
```

Or install from source:

```bash
git clone https://github.com/your-username/osmosis-wrap.git
cd osmosis-wrap
pip install -e .
```

For development, you can install all dependencies using:

```bash
pip install -r requirements.txt
```

## Environment Setup

Osmosis Wrap requires a Hoover API key to log LLM usage. Create a `.env` file in your project directory:

```bash
# Copy the sample .env file
cp .env.sample .env

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

## Usage

First, import and initialize Osmosis Wrap with your Hoover API key:

```python
import os
import osmosis_wrap

# Initialize with your Hoover API key
osmosis_wrap.init("your-hoover-api-key")

# Or load from environment variable
hoover_api_key = os.environ.get("HOOVER_API_KEY")
osmosis_wrap.init(hoover_api_key)
```

Once you import `osmosis_wrap` and initialize it, the library automatically patches the supported LLM clients. You can then use your LLM clients normally, and all API calls will be logged to Hoover:

### Anthropic Example

```python
# Import osmosis_wrap first and initialize it
import osmosis_wrap
osmosis_wrap.init(os.environ.get("HOOVER_API_KEY"))

# Then import and use Anthropic as normal
from anthropic import Anthropic

# Create and use the Anthropic client as usual - it's already patched
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# All API calls will now be logged to Hoover automatically
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

# Async client is also supported and automatically patched
from anthropic import AsyncAnthropic
import asyncio

async def call_claude_async():
    async_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = await async_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": "Hello, async Claude!"}
        ]
    )
    return response

# All async API calls will be logged to Hoover as well
asyncio.run(call_claude_async())
```

### OpenAI Example

```python
# Import osmosis_wrap first and initialize it
import osmosis_wrap
osmosis_wrap.init(os.environ.get("HOOVER_API_KEY"))

# Then import and use OpenAI as normal
from openai import OpenAI

# Create and use the OpenAI client as usual - it's already patched
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# All API calls will now be logged to Hoover automatically
response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=150,
    messages=[
        {"role": "user", "content": "Hello, GPT!"}
    ]
)

# Async client is also supported and automatically patched
from openai import AsyncOpenAI
import asyncio

async def call_openai_async():
    async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=150,
        messages=[
            {"role": "user", "content": "Hello, async GPT!"}
        ]
    )
    return response

# All async API calls will be logged to Hoover as well
asyncio.run(call_openai_async())
```

### LangChain Example

```python
# Import osmosis_wrap first and initialize it
import osmosis_wrap
osmosis_wrap.init(os.environ.get("HOOVER_API_KEY"))

# Then use LangChain as normal
from langchain_core.prompts import PromptTemplate

# Use LangChain prompt templates as usual
template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short paragraph about {topic}."
)

# Formatting the prompt will be logged to Hoover automatically
formatted_prompt = template.format(topic="artificial intelligence")
print(f"Formatted prompt: {formatted_prompt}")

# Multiple prompt templates are also captured
template2 = PromptTemplate(
    input_variables=["name", "profession"],
    template="My name is {name} and I work as a {profession}."
)
formatted_prompt2 = template2.format(name="Alice", profession="data scientist")
print(f"Formatted prompt 2: {formatted_prompt2}")
```

## Configuration

You can configure the behavior of the library by modifying the following variables:

```python
import osmosis_wrap

# Disable logging to Hoover (default: True)
osmosis_wrap.enabled = False
```

## How it Works

This library uses monkey patching to override the LLM clients' methods that make API calls. When you import the `osmosis_wrap` module, it automatically patches the supported LLM client libraries. When methods are called on these clients, the library intercepts the calls and sends the request parameters and response data to the Hoover API for logging and monitoring.

The data sent to Hoover includes:
- Timestamp (UTC)
- Request parameters
- Response data
- HTTP status code

## License

MIT 
