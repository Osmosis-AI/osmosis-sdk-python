"""
Example demonstrating how to use osmosis-wrap with LangChain-Anthropic and LangChain-OpenAI
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import and initialize osmosis_wrap
import osmosis_wrap

# Initialize with OSMOSIS API key
osmosis_api_key = os.getenv("OSMOSIS_API_KEY")
osmosis_wrap.init(osmosis_api_key)

# Set to True to print messages to console
osmosis_wrap.print_messages = True

# Import the specific langchain adapters
from osmosis_wrap.adapters.langchain_anthropic import wrap_langchain_anthropic
from osmosis_wrap.adapters.langchain_openai import wrap_langchain_openai

# Initialize the adapters - do this before importing any langchain models
wrap_langchain_anthropic()
wrap_langchain_openai()

print("LangChain-Anthropic and LangChain-OpenAI Integration Example\n")

try:
    # Import langchain components after osmosis_wrap adapters are initialized
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.language_models.llms import LLM
        print("Using langchain_core for imports")
    except ImportError:
        try:
            from langchain.prompts import PromptTemplate
            from langchain.llms.base import BaseLLM as LLM
            print("Using legacy langchain for imports")
        except ImportError:
            print("Error: langchain or langchain_core is not installed.")
            print("Install with: pip install langchain-core")
            sys.exit(1)

    # Try to import OpenAI-specific models
    try:
        from langchain_openai import OpenAI, ChatOpenAI
        print("Successfully imported from langchain_openai")
    except ImportError:
        try:
            from langchain.llms.openai import OpenAI
            from langchain.chat_models.openai import ChatOpenAI
            print("Using legacy langchain for OpenAI models")
        except ImportError:
            print("Warning: langchain-openai is not installed.")
            print("Install with: pip install langchain-openai")
            OpenAI = None
            ChatOpenAI = None

    # Try to import Anthropic-specific models
    try:
        from langchain_anthropic import ChatAnthropic
        print("Successfully imported from langchain_anthropic")
    except ImportError:
        try:
            from langchain.chat_models.anthropic import ChatAnthropic
            print("Using legacy langchain for Anthropic models")
        except ImportError:
            print("Warning: langchain-anthropic is not installed.")
            print("Install with: pip install langchain-anthropic")
            ChatAnthropic = None
    
    # Use a prompt template
    print("\n--- Prompt Template Usage ---")
    template = PromptTemplate(
        input_variables=["topic"],
        template="Write a short paragraph about {topic}."
    )
    
    # Format the prompt
    prompt = template.format(topic="artificial intelligence")
    print(f"Formatted prompt: {prompt}")
    
    # Test another prompt template
    template2 = PromptTemplate(
        input_variables=["name", "profession"],
        template="My name is {name} and I work as a {profession}."
    )
    
    prompt2 = template2.format(name="Alice", profession="data scientist")
    print(f"Formatted prompt 2: {prompt2}")
    
    # Create a simple mock LLM to demonstrate wrapping
    class MockLLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "mock_llm"
        
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            print(f"Mock LLM received prompt: {prompt}")
            return f"Mock response to: {prompt}"
    
    # Create and use a LLM
    print("\n--- LLM Usage ---")
    llm = MockLLM()
    
    # Use the invoke method (modern LangChain)
    result = llm.invoke("What is the capital of France?")
    print(f"LLM response: {result}")
    
    # Use the LLM with a formatted prompt template
    formatted_prompt = template.format(topic="machine learning")
    result = llm.invoke(formatted_prompt)
    print(f"LLM response with template: {result}")
    
    # Example with OpenAI - will be logged through the langchain-openai adapter
    if OpenAI:
        print("\n--- OpenAI LLM Example ---")
        # Replace with your OpenAI API key or set it in your .env file
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            openai_llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key)
            try:
                print("Sending a test prompt to OpenAI...")
                openai_response = openai_llm.invoke("Tell me a short joke")
                print(f"OpenAI response: {openai_response}")
            except Exception as e:
                print(f"Error with OpenAI: {str(e)}")
        else:
            print("OpenAI API key not found. Skipping OpenAI example.")
    
    # Example with ChatOpenAI - will be logged through the langchain-openai adapter
    if ChatOpenAI:
        print("\n--- ChatOpenAI Example ---")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            chat_openai = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
            try:
                print("Sending a test message to ChatOpenAI...")
                from langchain_core.messages import HumanMessage
                chat_response = chat_openai.invoke([
                    HumanMessage(content="What's the best way to learn programming?")
                ])
                print(f"ChatOpenAI response: {chat_response}")
            except Exception as e:
                print(f"Error with ChatOpenAI: {str(e)}")
        else:
            print("OpenAI API key not found. Skipping ChatOpenAI example.")
    
    # Example with ChatAnthropic - will be logged through the langchain-anthropic adapter
    if ChatAnthropic:
        print("\n--- ChatAnthropic Example ---")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            chat_anthropic = ChatAnthropic(model="claude-2", anthropic_api_key=anthropic_api_key)
            try:
                print("Sending a test message to ChatAnthropic...")
                from langchain_core.messages import HumanMessage
                anthropic_response = chat_anthropic.invoke([
                    HumanMessage(content="Explain the concept of deep learning in simple terms.")
                ])
                print(f"ChatAnthropic response: {anthropic_response}")
            except Exception as e:
                print(f"Error with ChatAnthropic: {str(e)}")
        else:
            print("Anthropic API key not found. Skipping ChatAnthropic example.")
    
    print("\nAll interactions above should have been logged via osmosis_wrap!")
    print("The langchain-openai and langchain-anthropic adapters have captured model-specific details.")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 