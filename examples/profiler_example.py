"""
Profiler example that measures performance differences between integrations 
with and without osmosis-wrap.

This script measures and compares execution times for:
- OpenAI
- Anthropic
- LangChain-OpenAI
- LangChain-Anthropic

Each integration is tested both with and without osmosis-wrap enabled.
"""

import os
import time
import statistics
from dotenv import load_dotenv
import importlib
import sys

# Load environment variables from .env file
load_dotenv()

# Color formatting for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{BLUE}{BOLD}{'='*80}{RESET}")
    print(f"{BLUE}{BOLD} {text} {RESET}")
    print(f"{BLUE}{BOLD}{'='*80}{RESET}")

def print_section(text):
    """Print a formatted section header."""
    print(f"\n{YELLOW}{BOLD} {text} {RESET}")
    print(f"{YELLOW}{'-'*80}{RESET}")

def time_execution(func, num_runs=3):
    """Time the execution of a function multiple times and return statistics."""
    times = []
    results = []
    
    for i in range(num_runs):
        start_time = time.time()
        result = func()
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        results.append(result)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "avg": avg_time,
        "median": median_time,
        "min": min_time,
        "max": max_time,
        "all": times,
        "results": results
    }

def compare_times(with_osmosis, without_osmosis):
    """Compare times with and without osmosis-wrap."""
    diff = without_osmosis["avg"] - with_osmosis["avg"]
    percent = (diff / without_osmosis["avg"]) * 100
    
    if diff > 0:
        print(f"  {GREEN}Osmosis-wrap is faster by {abs(diff):.4f}s ({abs(percent):.2f}%){RESET}")
    else:
        print(f"  {RED}Osmosis-wrap is slower by {abs(diff):.4f}s ({abs(percent):.2f}%){RESET}")
    
    return diff, percent

def reset_modules():
    """Reset modules to ensure clean testing environment."""
    modules_to_remove = [
        'osmosisai', 
        'openai', 
        'anthropic',
        'langchain_openai',
        'langchain_anthropic',
        'langchain_core'
    ]
    
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]

# Sample prompt for testing
PROMPT = "Explain the concept of quantum computing in two sentences."
NUM_RUNS = 3

# Test timings for OpenAI integration
def test_openai():
    print_section("Testing OpenAI Integration")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Skipping test.")
        return
    
    # Test without osmosis-wrap
    reset_modules()
    print("Without osmosis-wrap:")
    
    def run_openai_without():
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=50,
            messages=[{"role": "user", "content": PROMPT}]
        )
        return response.choices[0].message.content
    
    without_times = time_execution(run_openai_without, NUM_RUNS)
    
    # Test with osmosis-wrap
    reset_modules()
    print("With osmosis-wrap:")
    
    def run_openai_with():
        import osmosisai
        osmosisai.init(os.environ.get("OSMOSIS_API_KEY"))
        osmosisai.log_destination = "none"
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=50,
            messages=[{"role": "user", "content": PROMPT}]
        )
        return response.choices[0].message.content
    
    with_times = time_execution(run_openai_with, NUM_RUNS)
    
    # Compare results
    compare_times(with_times, without_times)
    return with_times, without_times

# Test timings for Anthropic integration
def test_anthropic():
    print_section("Testing Anthropic Integration")
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Anthropic API key not found. Skipping test.")
        return
    
    # Test without osmosis-wrap
    reset_modules()
    print("Without osmosis-wrap:")
    
    def run_anthropic_without():
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[{"role": "user", "content": PROMPT}]
        )
        return response.content[0].text
    
    without_times = time_execution(run_anthropic_without, NUM_RUNS)
    
    # Test with osmosis-wrap
    reset_modules()
    print("With osmosis-wrap:")
    
    def run_anthropic_with():
        import osmosisai
        osmosisai.init(os.environ.get("OSMOSIS_API_KEY"))
        osmosisai.log_destination = "none"
        
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[{"role": "user", "content": PROMPT}]
        )
        return response.content[0].text
    
    with_times = time_execution(run_anthropic_with, NUM_RUNS)
    
    # Compare results
    compare_times(with_times, without_times)
    return with_times, without_times

# Test timings for LangChain-OpenAI integration
def test_langchain_openai():
    print_section("Testing LangChain-OpenAI Integration")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Skipping test.")
        return
    
    # Test without osmosis-wrap
    reset_modules()
    print("Without osmosis-wrap:")
    
    def run_langchain_openai_without():
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            
            chat_openai = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
            response = chat_openai.invoke([HumanMessage(content=PROMPT)])
            return response.content
        except ImportError:
            print("LangChain-OpenAI not installed. Skipping test.")
            return "LangChain-OpenAI not installed"
    
    without_times = time_execution(run_langchain_openai_without, NUM_RUNS)
    
    # Test with osmosis-wrap
    reset_modules()
    print("With osmosis-wrap:")
    
    def run_langchain_openai_with():
        try:
            import osmosisai
            osmosisai.init(os.environ.get("OSMOSIS_API_KEY"))
            osmosisai.log_destination = "none"
            
            from osmosisai.adapters.langchain_openai import wrap_langchain_openai
            wrap_langchain_openai()
            
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            
            chat_openai = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
            response = chat_openai.invoke([HumanMessage(content=PROMPT)])
            return response.content
        except ImportError:
            print("LangChain-OpenAI not installed. Skipping test.")
            return "LangChain-OpenAI not installed"
    
    with_times = time_execution(run_langchain_openai_with, NUM_RUNS)
    
    # Compare results
    compare_times(with_times, without_times)
    return with_times, without_times

# Test timings for LangChain-Anthropic integration
def test_langchain_anthropic():
    print_section("Testing LangChain-Anthropic Integration")
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Anthropic API key not found. Skipping test.")
        return
    
    # Test without osmosis-wrap
    reset_modules()
    print("Without osmosis-wrap:")
    
    def run_langchain_anthropic_without():
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage
            
            chat_anthropic = ChatAnthropic(model="claude-3-haiku-20240307", anthropic_api_key=api_key)
            response = chat_anthropic.invoke([HumanMessage(content=PROMPT)])
            return response.content
        except ImportError:
            print("LangChain-Anthropic not installed. Skipping test.")
            return "LangChain-Anthropic not installed"
    
    without_times = time_execution(run_langchain_anthropic_without, NUM_RUNS)
    
    # Test with osmosis-wrap
    reset_modules()
    print("With osmosis-wrap:")
    
    def run_langchain_anthropic_with():
        try:
            import osmosisai
            osmosisai.init(os.environ.get("OSMOSIS_API_KEY"))
            osmosisai.log_destination = "none"
            
            from osmosisai.adapters.langchain_anthropic import wrap_langchain_anthropic
            wrap_langchain_anthropic()
            
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage
            
            chat_anthropic = ChatAnthropic(model="claude-3-haiku-20240307", anthropic_api_key=api_key)
            response = chat_anthropic.invoke([HumanMessage(content=PROMPT)])
            return response.content
        except ImportError:
            print("LangChain-Anthropic not installed. Skipping test.")
            return "LangChain-Anthropic not installed"
    
    with_times = time_execution(run_langchain_anthropic_with, NUM_RUNS)
    
    # Compare results
    compare_times(with_times, without_times)
    return with_times, without_times

# Run all tests
if __name__ == "__main__":
    print_header("OSMOSIS-WRAP PROFILER")
    print(f"This script compares execution times for different integrations with and without osmosis-wrap.")
    print(f"Each test will run {NUM_RUNS} times to get an average.")
    
    results = {}
    
    try:
        # Run all tests
        results["openai"] = test_openai()
        results["anthropic"] = test_anthropic()
        results["langchain_openai"] = test_langchain_openai()
        results["langchain_anthropic"] = test_langchain_anthropic()
        
        # Print summary
        print_header("SUMMARY")
        
        for integration, times in results.items():
            if times:
                with_times, without_times = times
                diff, percent = compare_times(with_times, without_times)
                
                if diff > 0:
                    performance = f"{GREEN}FASTER by {abs(diff):.4f}s ({abs(percent):.2f}%){RESET}"
                else:
                    performance = f"{RED}SLOWER by {abs(diff):.4f}s ({abs(percent):.2f}%){RESET}"
                
                print(f"{BOLD}{integration.upper()}{RESET}: Osmosis-wrap is {performance}")
    
    except Exception as e:
        print(f"{RED}Error: {str(e)}{RESET}")
        import traceback
        traceback.print_exc() 