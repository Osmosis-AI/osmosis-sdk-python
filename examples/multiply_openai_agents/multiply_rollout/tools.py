from agents import function_tool


@function_tool
async def multiply_tool(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b
