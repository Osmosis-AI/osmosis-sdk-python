"""
Example demonstrating how to use osmosis-wrap with OpenAI
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import and initialize osmosisai
import osmosisai

# Initialize with OSMOSIS API key
osmosis_api_key = os.environ.get("OSMOSIS_API_KEY")
osmosisai.init(osmosis_api_key)

# Print messages to console for demonstration
osmosisai.print_messages = True

print("OpenAI Integration Example\n")

try:
    # Import OpenAI after osmosisai is initialized
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
    
    print("\nAll interactions above have been logged via osmosisai!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc() 