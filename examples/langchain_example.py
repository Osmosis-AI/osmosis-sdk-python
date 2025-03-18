"""
Example demonstrating how to use osmosis-wrap with LangChain
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import and initialize osmosis_wrap
import osmosis_wrap

# Initialize with Hoover API key (or a placeholder if not available)
hoover_api_key = os.getenv("HOOVER_API_KEY", "test-hoover-key")
osmosis_wrap.init(hoover_api_key)

# Set to True to print messages to console
osmosis_wrap.print_messages = True

print("LangChain Integration Example\n")

try:
    # Import langchain components after osmosis_wrap is initialized
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
    
    print("\nAll interactions above should have been logged via osmosis_wrap!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 