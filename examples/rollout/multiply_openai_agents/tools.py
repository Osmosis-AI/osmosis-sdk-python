from agents import function_tool


@function_tool
def multiply_tool(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The product of the two numbers.
    """
    return a * b
