# orchestrator.py
from __future__ import annotations

from typing import Any, Dict, Tuple

from agents import run_multi_agent
from guardrails import (
    is_disallowed,
    safe_refusal_message,
    preprocess_user_input,
    postprocess_model_output,
)
from logging_utils import log_interaction
from session_memory import get_session_memory


def omni_pro_turn(raw_user_input: str) -> Tuple[str, Dict[str, Any]]:
    """
    High-level, single-turn orchestrator for Omni-Pro.

    Responsibilities:
    - Run guardrails (disallowed checks).
    - Update / read session memory.
    - Call multi-agent pipeline with appropriate chat_history.
    - Post-process output.
    - Log interaction.
    - Return (final_answer, agent_traces).
    """
    memory = get_session_memory()
    agent_traces: Dict[str, Any] = {}

    text = (raw_user_input or "").strip()
    if not text:
        return "I didn’t receive any content to respond to.", agent_traces

    # Store user turn in memory BEFORE processing so planner sees it.
    memory.add_turn("user", text)

    # Guardrails – global safety
    if is_disallowed(text):
        answer = safe_refusal_message()
        final_answer = postprocess_model_output(answer)
        memory.add_turn("assistant", final_answer)
        try:
            log_interaction(
                user_input=text,
                final_answer=final_answer,
                agent_traces=agent_traces,
            )
        except Exception:
            pass
        return final_answer, agent_traces

    # Preprocess input for the pipeline
    processed_input = preprocess_user_input(text)

    # Build chat_history from memory
    chat_history = memory.to_chat_history()

    # Multi-agent pipeline (with tools + judge inside)
    final_answer, agent_traces = run_multi_agent(
        processed_input,
        chat_history=chat_history,
    )

    # Postprocess output for the user
    final_answer = postprocess_model_output(final_answer)

    # Store assistant turn in memory
    memory.add_turn("assistant", final_answer)

    # Log interaction (non-fatal)
    try:
        log_interaction(
            user_input=text,
            final_answer=final_answer,
            agent_traces=agent_traces,
        )
    except Exception:
        pass

    return final_answer, agent_traces
