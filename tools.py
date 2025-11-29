# tools.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from calculator_tools import evaluate_expression
from time_tools import get_time_for_human_query

from search_client import deep_web_search as _deep_web_search_raw
from search_client import format_search_results_for_prompt


# ---------------------------------------------------------------------------
# Tool Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """
    Standardised result returned by all tools.

    - success: whether the tool ran successfully.
    - answer: human-readable answer (already formatted for the user).
    - tool_name: identifier string for logging / traces.
    """
    success: bool
    answer: str
    tool_name: str


# ---------------------------------------------------------------------------
# Individual tool wrappers
# ---------------------------------------------------------------------------

def _simple_calculator_tool(user_input: str) -> ToolResult:
    """
    Very small calculator wrapper.

    Accepts natural-ish queries like:
      - "what is 5 + 7"
      - "calculate 2^10"
      - "3 * (4 + 5)"

    It tries to extract an expression and passes it to evaluate_expression().
    """
    # Strip leading phrases
    text = user_input.strip().lower()

    # Remove common prefixes
    prefixes = [
        "what is", "what's", "whats",
        "calculate", "calc",
        "solve", "answer",
    ]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()
            break

    # Remove leading '=' if present
    if text.startswith("="):
        text = text[1:].strip()

    if not text:
        return ToolResult(
            success=False,
            answer="I couldn’t see a maths expression to calculate.",
            tool_name="calculator",
        )

    result = evaluate_expression(text)
    return ToolResult(
        success=True,
        answer=f"{text} = {result}",
        tool_name="calculator",
    )


def _time_tool(user_input: str) -> ToolResult:
    """
    Wrapper around get_time_for_human_query().

    Handles queries like:
      - "what time is it"
      - "what time is it in seychelles"
      - "time in Mahé"
    """
    answer = get_time_for_human_query(user_input)
    return ToolResult(
        success=True,
        answer=answer,
        tool_name="time",
    )


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

# Each tool receives the original user_input and returns ToolResult.
ToolFn = Callable[[str], ToolResult]

_TOOL_REGISTRY: Dict[str, ToolFn] = {
    "calculator": _simple_calculator_tool,
    "time": _time_tool,
    # Future tools can be registered here, e.g.:
    # "unit_converter": _unit_converter_tool,
    # "weather": _weather_tool,
}


def list_tools() -> Dict[str, str]:
    """
    Lightweight description of available tools.
    Can be used later if you add a meta "tools" command.
    """
    return {
        "calculator": "Evaluate basic arithmetic expressions (+, -, *, /, ^, parentheses).",
        "time": "Tell the current time for a few known locations (e.g. Seychelles) or UTC.",
    }


def deep_web_search(
    query: str,
    max_results: int | None = None,
    max_chars: int | None = None,
) -> str:
    """
    Text-oriented web search wrapper used by agents.

    Internally:
      - calls search_client.deep_web_search(...) to get raw results
      - formats them via format_search_results_for_prompt(...)
      - optionally truncates to max_chars.

    This restores the old 'deep_web_search' symbol that agents expect,
    but keeps the new 'tools.py' layout clean.
    """
    if not query:
        return "Top web results (none):\nNo query provided."

    if max_results is None or max_results <= 0:
        max_results = 8  # sensible default

    # Raw list[dict] from your existing search_client implementation
    raw_results = _deep_web_search_raw(query, max_results=max_results)

    # Turn into an LLM-friendly text block
    text = format_search_results_for_prompt(raw_results)

    # Enforce a character cap if requested (for token safety)
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]

    return text



# ---------------------------------------------------------------------------
# Top-level router for TRIVIAL toolable queries
# ---------------------------------------------------------------------------

_MATH_LIKE = re.compile(r"^[0-9\+\-\*\/\^\(\)\.\s]+$")


def fast_tool_router(user_input: str) -> Tuple[bool, str, str]:
    """
    Fast, *pre-planner* tool router.

    The goal is to catch the trivial stuff before we spin up the
    whole multi-agent stack.

    Returns:
      (used_tool: bool, answer: str, tool_name: str)

    If no suitable tool is found, returns (False, "", "").
    """
    text = (user_input or "").strip().lower()
    if not text:
        return False, "", ""

    # 1) Calculator: either explicit math expression, or text that looks like math.
    #    If this hits, we do NOT run the planner at all.
    # -----------------------------------------------------------------------
    if _MATH_LIKE.fullmatch(text.replace(" ", "")):
        tool = _TOOL_REGISTRY["calculator"]
        res = tool(user_input)
        return res.success, res.answer, res.tool_name

    # Common calculator prefixes
    if any(text.startswith(p) for p in ["what is", "what's", "whats", "calculate", "calc", "solve", "answer"]):
        tool = _TOOL_REGISTRY["calculator"]
        res = tool(user_input)
        return res.success, res.answer, res.tool_name

    # 2) Time questions – very cheap to answer locally.
    # -----------------------------------------------------------------------
    # Extremely rough heuristic: any mention of "time" plus "what" or "now".
    if "time" in text and ("what" in text or "now" in text or "current" in text):
        tool = _TOOL_REGISTRY["time"]
        res = tool(user_input)
        return res.success, res.answer, res.tool_name

    # 3) No tool matched → let the main pipeline handle it.
    # -----------------------------------------------------------------------
    return False, "", ""
