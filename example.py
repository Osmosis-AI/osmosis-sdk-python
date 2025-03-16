"""
Example usage of osmosis-wrap

This example demonstrates how to use the osmosis-wrap library to print
prompts and responses when using various LLM APIs.
"""

# Import osmosis_wrap before importing any LLM libraries
import osmosis_wrap
from osmosis_wrap.utils import get_api_key

# Optional: Configure osmosis_wrap behavior
osmosis_wrap.pretty_print = True
osmosis_wrap.indent = 2

# Example 1: Using Anthropic
print("\n=== Anthropic Example ===")
try:
    from anthropic import Anthropic
    
    # Get API key from environment variables
    api_key = get_api_key("anthropic")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not found in environment variables.")
        api_key = "YOUR_ANTHROPIC_API_KEY"  # Replace with your actual API key if not using environment variables
    
    # Create Anthropic client as usual
    client = Anthropic(api_key=api_key)
    
    # Make an API call - the prompt and response will be printed automatically
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": "Explain what monkey patching is in Python in one sentence."}
        ]
    )
    
    # Use the response as usual
    print("\nActual response content:")
    print(response.content[0].text)
except ImportError:
    print("Anthropic library not installed. Run 'pip install anthropic' to use this example.")

# Example 2: Using OpenAI
print("\n=== OpenAI Example ===")
try:
    from openai import OpenAI
    
    # Get API key from environment variables
    api_key = get_api_key("openai")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual API key if not using environment variables
    
    # Create OpenAI client as usual
    client = OpenAI(api_key=api_key)
    
    # Make an API call - the prompt and response will be printed automatically
    response = client.chat.completions.create(
        model="gpt-4",
        max_tokens=150,
        messages=[
            {"role": "user", "content": "Explain what monkey patching is in Python in one sentence."}
        ]
    )
    
    # Use the response as usual
    print("\nActual response content:")
    print(response.choices[0].message.content)
except ImportError:
    print("OpenAI library not installed. Run 'pip install openai' to use this example.")

# Example of disabling printing for specific calls
print("\nDisabling printing for the next API call:")
osmosis_wrap.enabled = False

# Try with any available client
try:
    from anthropic import Anthropic
    api_key = get_api_key("anthropic") or "YOUR_ANTHROPIC_API_KEY"
    client = Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        messages=[{"role": "user", "content": "This request won't be printed."}]
    )
except ImportError:
    try:
        from openai import OpenAI
        api_key = get_api_key("openai") or "YOUR_OPENAI_API_KEY"
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=150,
            messages=[{"role": "user", "content": "This request won't be printed."}]
        )
    except ImportError:
        print("Neither OpenAI nor Anthropic libraries are installed.")

# Re-enable printing
osmosis_wrap.enabled = True

print("\nPrinting is now re-enabled for future API calls.") 