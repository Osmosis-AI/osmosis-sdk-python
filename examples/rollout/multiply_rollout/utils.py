import re


def extract_solution(solution_str):
    solution = re.search(r"####\s*([-+]?\d*\.?\d+)", solution_str)
    if not solution:
        return None
    return solution.group(1)
