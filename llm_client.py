# llm_client.py
from __future__ import annotations

import os
from typing import Callable, Optional

from langchain_groq import ChatGroq

from config import config


def _build_groq_client(model_name: str, temperature: float, streaming: bool = False) -> ChatGroq:
    """
    Helper to build a configured ChatGroq client for a given model.

    The `streaming` flag enables token streaming where supported. For normal
    (non-streaming) calls we keep streaming=False.
    """
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_TOKEN")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY (or GROQ_API_TOKEN) environment variable is not set."
        )

    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_api_key=api_key,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# Model instances
# ---------------------------------------------------------------------------

_fast_llm = _build_groq_client(
    config.llm.fast_model,
    config.llm.temperature_fast,
)

_planner_llm = _build_groq_client(
    config.llm.planner_model,
    config.llm.temperature_planner,
)

_research_llm = _build_groq_client(
    config.llm.research_model,
    config.llm.temperature_research,
)

_evidence_llm = _build_groq_client(
    config.llm.evidence_model,
    config.llm.temperature_evidence,
)

_reviewer_llm = _build_groq_client(
    config.llm.reviewer_model,
    config.llm.temperature_reviewer,
)

_final_llm = _build_groq_client(
    config.llm.final_model,
    config.llm.temperature_final,
)

# Streaming variant of the final model, used only when explicitly requested.
_final_stream_llm = _build_groq_client(
    config.llm.final_model,
    config.llm.temperature_final,
    streaming=True,
)


# ---------------------------------------------------------------------------
# Core invocation helpers
# ---------------------------------------------------------------------------


def _invoke_llm(llm: ChatGroq, system_prompt: str, user_prompt: str) -> str:
    """
    Normal (non-streaming) invocation: returns the full answer as a string.
    """
    messages = [
        ("system", system_prompt),
        ("user", user_prompt),
    ]
    response = llm.invoke(messages)
    # ChatGroq returns an AIMessage-like object; the text is in `.content`.
    content = getattr(response, "content", None)
    if content is None:
        # Fallback: convert whole object to string.
        return str(response)
    return str(content)


def call_llm_final_stream(
    system_prompt: str,
    user_prompt: str,
    on_chunk: Optional[callable] = None,
) -> str:
    """
    Streaming wrapper for the final model.

    - Calls the Groq chat model in streaming mode.
    - For each chunk of content, calls on_chunk(text) if provided.
    - Returns the full concatenated string at the end.
    """
    messages = [
        ("system", system_prompt),
        ("user", user_prompt),
    ]
    full_text_parts: list[str] = []

    # Use the underlying LangChain ChatGroq streaming interface.
    for msg in _final_llm.stream(messages):
        chunk = getattr(msg, "content", "") or ""
        if not chunk:
            continue
        full_text_parts.append(chunk)
        if on_chunk is not None:
            on_chunk(chunk)

    return "".join(full_text_parts)



def _stream_llm(
    llm: ChatGroq,
    system_prompt: str,
    user_prompt: str,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Streaming invocation helper.

    - Iterates over `llm.stream(...)` to receive incremental chunks.
    - Optionally calls `on_token(chunk_text)` for each text fragment.
    - Returns the full concatenated content at the end.
    """
    messages = [
        ("system", system_prompt),
        ("user", user_prompt),
    ]

    full_text_parts: list[str] = []
    for chunk in llm.stream(messages):
        # Each `chunk` is an AIMessageChunk-like object with `.content`.
        piece = getattr(chunk, "content", "") or ""
        if not isinstance(piece, str):
            piece = str(piece)
        if piece:
            full_text_parts.append(piece)
            if on_token is not None:
                on_token(piece)

    return "".join(full_text_parts)


# ---------------------------------------------------------------------------
# Public wrapper functions used by the agents
# ---------------------------------------------------------------------------


def call_llm_fast(system_prompt: str, user_prompt: str) -> str:
    return _invoke_llm(_fast_llm, system_prompt, user_prompt)


def call_llm_planner(system_prompt: str, user_prompt: str) -> str:
    return _invoke_llm(_planner_llm, system_prompt, user_prompt)


def call_llm_research(system_prompt: str, user_prompt: str) -> str:
    return _invoke_llm(_research_llm, system_prompt, user_prompt)


def call_llm_evidence(system_prompt: str, user_prompt: str) -> str:
    return _invoke_llm(_evidence_llm, system_prompt, user_prompt)


def call_llm_reviewer(system_prompt: str, user_prompt: str) -> str:
    return _invoke_llm(_reviewer_llm, system_prompt, user_prompt)


def call_llm_final(system_prompt: str, user_prompt: str) -> str:
    return _invoke_llm(_final_llm, system_prompt, user_prompt)


def call_llm_final_stream(
    system_prompt: str,
    user_prompt: str,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Streaming variant of `call_llm_final`.

    - Uses a separate ChatGroq client with `streaming=True`.
    - Calls `on_token(text)` for each partial text chunk (if provided).
    - Returns the full concatenated answer string at the end.

    This is ideal for web / CLI UIs that want to show the Finalizer agent's
    answer as it is being generated, while still getting the full text back
    for logging, judging, and memory.
    """
    return _stream_llm(_final_stream_llm, system_prompt, user_prompt, on_token=on_token)
