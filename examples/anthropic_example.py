"""
Example demonstrating how to use osmosis-ai with Anthropic
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import and initialize osmosis_ai
import osmosis_ai

# Initialize with OSMOSIS API key
osmosis_api_key = os.environ.get("OSMOSIS_API_KEY")
osmosis_ai.init(osmosis_api_key)

# Print messages to console for demonstration
osmosis_ai.set_log_destination(osmosis_ai.LogDestination.STDOUT)

print("Anthropic Integration Example\n")

try:
    # Import Anthropic after osmosis_ai is initialized
    from anthropic import Anthropic
    
    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Create the Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Make a request to Claude
    print("Making request to Claude...")
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=150,
        messages=[
            {"role": "user", "content": "Hello, Claude! What are three interesting facts about neural networks?"}
        ]
    )
    
    # Print the response
    print("\nResponse from Claude:")
    print(response.content[0].text)
    
    print("\nAll interactions above have been logged via osmosis_ai!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc() 