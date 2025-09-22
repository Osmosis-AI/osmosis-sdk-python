
def osmosis_reward(func: Callable) -> Callable:
    """
    Decorator for reward functions.
    
    Args:
        func: The reward function to be wrapped
        
    Returns:
        The wrapped function
        
    Example:
        @osmosis_reward
        def calculate_reward(state, action):
            return state.score + action.value
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper
