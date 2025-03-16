# Osmosis Wrap

A Python library that monkey patches LLM client libraries to print all prompts and responses.

## Supported Libraries

- **Anthropic**: Prints all Claude API requests and responses
- **OpenAI**: Prints all OpenAI API requests and responses (supports both v1 and legacy clients)

## Installation

```bash
pip install osmosis-wrap
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

Osmosis Wrap looks for API keys in environment variables. Create a `.env` file in your project directory:

```bash
# Copy the sample .env file
cp .env.sample .env

# Edit the .env file with your API keys
```

Edit the `.env` file to add your API keys:

```
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Usage

Simply import `osmosis_wrap` before importing any LLM client libraries:

```python
import osmosis_wrap
```

### Anthropic Example

```python
import osmosis_wrap
from anthropic import Anthropic

# Create and use the Anthropic client as usual
client = Anthropic(api_key="your-api-key")  # Or use environment variable

# All API calls will now print prompts and responses
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
```

### OpenAI Example

```python
import osmosis_wrap
from openai import OpenAI

# Create and use the OpenAI client as usual
client = OpenAI(api_key="your-api-key")  # Or use environment variable

# All API calls will now print prompts and responses
response = client.chat.completions.create(
    model="gpt-4",
    max_tokens=150,
    messages=[
        {"role": "user", "content": "Hello, GPT!"}
    ]
)
```

### Using Environment Variables

The library can automatically use API keys from environment variables:

```python
import osmosis_wrap
from osmosis_wrap.utils import get_api_key
from anthropic import Anthropic

# Get API key from environment variables (ANTHROPIC_API_KEY)
api_key = get_api_key("anthropic")

# Create client with the environment API key
client = Anthropic(api_key=api_key)
```

## Configuration

You can configure the behavior of the library by modifying the following variables:

```python
import osmosis_wrap

# Disable printing (default: True)
osmosis_wrap.enabled = False

# Print to stderr instead of stdout (default: False)
osmosis_wrap.use_stderr = True

# Disable pretty printing of JSON (default: True)
osmosis_wrap.pretty_print = False

# Change indent for pretty printing (default: 2)
osmosis_wrap.indent = 4

# Disable printing of responses, only print requests (default: True)
osmosis_wrap.print_messages = False
```

### Configuration via Environment Variables

You can also configure the library using environment variables in your `.env` file:

```
# Enable logging to stderr instead of stdout
OSMOSIS_USE_STDERR=true

# Disable pretty printing of JSON
OSMOSIS_PRETTY_PRINT=false

# Change JSON indentation level
OSMOSIS_INDENT=4

# Disable response printing (requests will still be printed)
OSMOSIS_PRINT_RESPONSES=false
```

## How it Works

This library uses monkey patching to override the LLM clients' methods that make API calls. When these methods are called, the library prints the request parameters and response data before returning the response to the caller.

## License

MIT 