# guardrails.py
"""
Guardrails and safety hooks for Omni-Pro.

Right now:
- is_disallowed: placeholder that always returns False (you can add rules).
- safe_refusal_message: generic refusal message.
- preprocess_user_input/postprocess_model_output: identity functions (hooks for later).
"""

from __future__ import annotations


def is_disallowed(text: str) -> bool:
    """
    Decide whether the user text is disallowed.

    For now, we don't block anything here; safety is handled by underlying models.
    You can add keyword-based rules here later if you want.
    """
    # Example skeleton:
    # lowered = text.lower()
    # if "how to make a bomb" in lowered:
    #     return True
    return False


def safe_refusal_message() -> str:
    """
    Generic refusal message used when is_disallowed() returns True.
    """
    return (
        "I’m not able to help with that request. "
        "If you have another topic or question, feel free to ask."
    )


def preprocess_user_input(text: str) -> str:
    """
    Hook to normalize or clean user input before routing.
    Currently a no-op.
    """
    return text


def postprocess_model_output(text: str) -> str:
    """
    Hook to clean or adapt model output before printing.
    Currently a no-op.
    """
    return text
