# Osmosis Wrap

A Python library that monkey patches LLM client libraries to send all prompts and responses to the Hoover API for logging and monitoring.

## Supported Libraries

- **Anthropic**: Logs all Claude API requests and responses
- **OpenAI**: Logs all OpenAI API requests and responses (supports both v1 and legacy clients)

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

First, initialize Osmosis Wrap with your Hoover API key:

```python
import osmosis_wrap

# Initialize with your Hoover API key
osmosis_wrap.init("your-hoover-api-key")

# Or load from environment variable
from generic_util import get_api_key

hoover_api_key = get_api_key("hoover")
osmosis_wrap.init(hoover_api_key)
```

Then use your LLM clients as usual:

### Anthropic Example

```python
from anthropic import Anthropic

# Create and use the Anthropic client as usual
client = Anthropic(api_key="your-api-key")  # Or use environment variable

# All API calls will now be logged to Hoover
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
```

### OpenAI Example

```python
from openai import OpenAI

# Create and use the OpenAI client as usual
client = OpenAI(api_key="your-api-key")  # Or use environment variable

# All API calls will now be logged to Hoover
response = client.chat.completions.create(
    model="gpt-4o-mini",
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

# Initialize with Hoover API key from environment
hoover_api_key = get_api_key("hoover")
osmosis_wrap.init(hoover_api_key)

# Get LLM API key from environment
api_key = get_api_key("anthropic")
client = Anthropic(api_key=api_key)
```

## Configuration

You can configure the behavior of the library by modifying the following variables:

```python
import osmosis_wrap

# Disable logging to Hoover (default: True)
osmosis_wrap.enabled = False
```

## How it Works

This library uses monkey patching to override the LLM clients' methods that make API calls. When these methods are called, the library sends the request parameters and response data to the Hoover API for logging and monitoring.

The data sent to Hoover includes:
- Timestamp (UTC)
- Request parameters
- Response data
- HTTP status code

## License

MIT 