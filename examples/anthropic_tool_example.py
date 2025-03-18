"""
Example demonstrating how to use osmosis-wrap with Anthropic tool calls
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

print("Anthropic Tool Use Integration Example\n")

try:
    # Import Anthropic after osmosis_wrap is initialized
    from anthropic import Anthropic
    
    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Create the Anthropic client
    client = Anthropic(api_key=api_key)
    
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
                    "description": "The unit of temperature to use"
                },
            },
            "required": ["location"],
        },
    }
    
    # Define the function to be called when the tool is used
    def handle_tool_use(tool_name, tool_input):
        if tool_name == "get_current_weather":
            location = tool_input.get("location")
            unit = tool_input.get("unit", "fahrenheit")
            # In a real application, you would call a weather API here
            print(f"Tool called: Getting weather for {location} in {unit}")
            return {
                "temperature": 72 if unit == "fahrenheit" else 22,
                "condition": "sunny",
                "humidity": "45%",
                "location": location
            }
        return {"error": "Unknown tool"}
    
    # Make a request to Claude with tool use
    print("Making request to Claude with tool use...")
    response = client.messages.create(
        model="claude-3-opus-20240229",  # Using a model that supports tools
        max_tokens=130,
        tools=[weather_tool],
        messages=[
            {"role": "user", "content": "What's the current weather in San Francisco?"}
        ]
    )
    
    # Process the response and handle any tool calls
    final_response = response
    if hasattr(response, 'content') and len(response.content) > 0:
        for content in response.content:
            if content.type == "tool_use":
                # Access properties directly on the content object
                print(f"\nClaude requested to use tool: {content.name}")
                print(f"Tool input: {content.input}")
                
                # Handle the tool use
                tool_result = handle_tool_use(content.name, content.input)
                
                # Continue the conversation with the tool result
                print(f"Tool result: {tool_result}")
                
                # Send tool result back to Claude
                final_response = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=130,
                    tools=[weather_tool],
                    messages=[
                        {"role": "user", "content": "What's the current weather in San Francisco?"},
                        {"role": "assistant", "content": [content]},
                        {"role": "user", "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "result": tool_result
                            }
                        ]}
                    ]
                )
    
    # Print the final response
    print("\nFinal response from Claude:")
    for content in final_response.content:
        if content.type == "text":
            print(content.text)
    
    print("\nAll interactions above have been logged via osmosis_wrap!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc() 