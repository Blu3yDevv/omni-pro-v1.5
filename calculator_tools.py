# calculator_tools.py
from __future__ import annotations

import math
import re


_SAFE_EXPR = re.compile(r"^[0-9\+\-\*\/\(\)\.\s\^]+$")


def evaluate_expression(expr: str) -> str:
    """
    Very small, safe-ish arithmetic evaluator.

    Supports: +, -, *, /, parentheses, ^ for power.
    Returns a string with the result, or a short error message.

    NOTE:
    - This is used by agents._detect_simple_math(), which already filters
      the input heavily. In normal operation, users won't see the error
      messages; the router will simply fall back to the full pipeline.
    """
    expr = expr.strip()
    if not expr:
        return "No expression provided."

    if not _SAFE_EXPR.match(expr):
        return "Expression contains unsupported characters."

    # Convert ^ to ** for Python
    expr = expr.replace("^", "**")

    try:
        # Use Python's eval in a *very* restricted environment.
        result = eval(expr, {"__builtins__": {}}, {"math": math})
    except Exception as e:
        return f"Could not evaluate expression: {e}"

    return str(result)
