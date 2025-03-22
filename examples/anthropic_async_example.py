"""
Example demonstrating how to use osmosis-ai with Anthropic
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import and initialize osmosis_ai first
import osmosis_ai

# Initialize with OSMOSIS API key
osmosis_api_key = os.environ.get("OSMOSIS_API_KEY")
osmosis_ai.init(osmosis_api_key)

# Print messages to console for demonstration
osmosis_ai.set_log_destination(osmosis_ai.LogDestination.STDOUT)

print("Anthropic Integration Example\n")

# Import Anthropic AFTER osmosis_ai initialization
from anthropic import AsyncAnthropic

async def call_claude_async():
    try:
        # Get API key from environment
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        print("\nAsync Client Example")
        async_client = AsyncAnthropic(api_key=api_key)
        
        response = await async_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            messages=[
                {"role": "user", "content": "Hello, async Claude! What are the key differences between CNNs and RNNs?"}
            ]
        )
        print("\nAsync Response from Claude:")
        print(response.content[0].text)
        return response
    except Exception as e:
        print(f"Error in async call: {str(e)}")
        import traceback
        traceback.print_exc()

# Main function to run everything
def main():
    try:
        # Run the async example
        asyncio.run(call_claude_async())
        print("\nAll interactions above have been logged via osmosis_ai!")
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the program
if __name__ == "__main__":
    main() 