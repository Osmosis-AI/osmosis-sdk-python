"""
Example demonstrating how to use osmosis-wrap with OpenAI
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import and initialize osmosis_wrap
import osmosis_wrap

# Initialize with Hoover API key
hoover_api_key = os.environ.get("HOOVER_API_KEY")
osmosis_wrap.init(hoover_api_key)

# Print messages to console for demonstration
osmosis_wrap.print_messages = True

print("OpenAI Integration Example\n")

try:
    # Import OpenAI after osmosis_wrap is initialized
    from openai import OpenAI
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Create the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Example with streaming
    print("\nStreaming Example:")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=150,
        messages=[
            {"role": "user", "content": "Write a short poem about artificial intelligence."}
        ],
        stream=True
    )
    
    print("\nStreaming response from GPT:")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
    print("\n")
    
    print("\nAll interactions above have been logged via osmosis_wrap!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc() 