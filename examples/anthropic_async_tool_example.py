"""
Example demonstrating how to use osmosis-ai with Anthropic tool calls using async API
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

print("Anthropic Async Tool Use Integration Example\n")

# Import Anthropic AFTER osmosis_ai initialization
from anthropic import AsyncAnthropic

# Define a tool for weather information
weather_tool = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g., San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature to use",
            },
        },
        "required": ["location"],
    },
}


# Define the function to be called when the tool is used
async def handle_tool_use(tool_name, tool_input):
    if tool_name == "get_current_weather":
        location = tool_input.get("location")
        unit = tool_input.get("unit", "fahrenheit")
        # In a real application, you would call a weather API here
        print(f"Tool called: Getting weather for {location} in {unit}")
        return {
            "temperature": 72 if unit == "fahrenheit" else 22,
            "condition": "sunny",
            "humidity": "45%",
            "location": location,
        }
    return {"error": "Unknown tool"}


async def call_claude_with_tools():
    try:
        # Get API key from environment
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        # Create the async Anthropic client
        async_client = AsyncAnthropic(api_key=api_key)

        # Make a request to Claude with tool use
        print("Making async request to Claude with tool use...")
        response = await async_client.messages.create(
            model="claude-3-opus-20240229",  # Using a model that supports tools
            max_tokens=130,
            tools=[weather_tool],
            messages=[
                {
                    "role": "user",
                    "content": "What's the current weather in San Francisco?",
                }
            ],
        )

        # Process the response and handle any tool calls
        final_response = response
        if hasattr(response, "content") and len(response.content) > 0:
            for content in response.content:
                if content.type == "tool_use":
                    # Access properties directly on the content object
                    print(f"\nClaude requested to use tool: {content.name}")
                    print(f"Tool input: {content.input}")

                    # Handle the tool use
                    tool_result = await handle_tool_use(content.name, content.input)

                    # Continue the conversation with the tool result
                    print(f"Tool result: {tool_result}")

                    # Send tool result back to Claude
                    final_response = await async_client.messages.create(
                        model="claude-3-opus-20240229",
                        max_tokens=130,
                        tools=[weather_tool],
                        messages=[
                            {
                                "role": "user",
                                "content": "What's the current weather in San Francisco?",
                            },
                            {"role": "assistant", "content": [content]},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": content.id,
                                        "result": tool_result,
                                    }
                                ],
                            },
                        ],
                    )

        # Print the final response
        print("\nFinal response from Claude:")
        for content in final_response.content:
            if content.type == "text":
                print(content.text)

        return final_response
    except Exception as e:
        print(f"Error in async tool call: {str(e)}")
        import traceback

        traceback.print_exc()


# Main function to run everything
def main():
    try:
        # Run the async example
        asyncio.run(call_claude_with_tools())
        print("\nAll interactions above have been logged via osmosis_ai!")
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback

        traceback.print_exc()


# Run the program
if __name__ == "__main__":
    main()
