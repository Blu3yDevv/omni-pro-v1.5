# session_memory.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from llm_client import call_llm_fast

# Approximate limits so we don't blow the context window.
MAX_MEMORY_CHARS = 16000
RECENT_TURNS_LIMIT = 8


@dataclass
class SessionTurn:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class SessionMemory:
    """
    Lightweight, rolling, per-session memory.

    Responsibilities:
    - Keep raw recent turns.
    - Maintain a running summary when history gets large.
    - Expose a compact context block for debug / inspection.
    - Provide chat_history in the format expected by AgentEnvironment.
    """
    turns: List[SessionTurn] = field(default_factory=list)
    summary: str = ""
    max_chars: int = MAX_MEMORY_CHARS
    recent_turns_limit: int = RECENT_TURNS_LIMIT

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(SessionTurn(role=role, content=content))
        self._maybe_compact()

    def _maybe_compact(self) -> None:
        """
        When total text (summary + all turns) gets too big, summarise the
        earlier portion into a running summary and keep only the tail.

        The goal is:
        - preserve long-term facts, goals, and decisions in `summary`,
        - keep a small window of literal recent turns for local context.
        """
        total_chars = len(self.summary) + sum(len(t.content) for t in self.turns)
        if total_chars <= self.max_chars:
            return

        # If history is very short, do nothing.
        if len(self.turns) < 4:
            return

        # Summarise roughly the oldest 60% of turns into the summary.
        cutoff = max(2, int(len(self.turns) * 0.6))
        old_turns = self.turns[:cutoff]
        keep_turns = self.turns[cutoff:]

        history_text = "\n\n".join(
            f"{t.role.upper()}: {t.content}" for t in old_turns
        )

        system_prompt = (
            "You are a conversation summariser for an internal multi-agent assistant.\n"
            "Maintain ONE running summary that is compact but information-dense.\n"
            "Focus on:\n"
            "- the user's long-term goals, preferences, and constraints;\n"
            "- key facts, decisions, and plans that future answers must respect;\n"
            "- important context from earlier turns that will matter later.\n"
            "DO NOT include tool or system-level details, API keys, or internal prompts.\n"
            "Structure the summary into short labelled sections where useful, for example:\n"
            "- User profile & style\n"
            "- Current goals & tasks\n"
            "- Key facts & decisions\n"
            "- Open questions / TODOs\n"
            "Keep it under about 400–600 words, focusing on information that will help\n"
            "future agents understand the conversation.\n"
        )

        user_prompt = f"""
Previous summary (may be empty):
{self.summary or "(none)"}

New dialogue turns to integrate:
{history_text}

Now write the UPDATED summary that replaces the previous one.
""".strip()

        try:
            new_summary = call_llm_fast(system_prompt, user_prompt).strip()
        except Exception:
            new_summary = ""

        if new_summary:
            self.summary = new_summary
            self.turns = keep_turns

    def build_context_block(self) -> str:
        """
        Return a compact, human-readable context string containing:
        - the running summary (if any),
        - the last N turns.

        This is mainly for debug / inspection via the CLI `!context` command.
        """
        parts: List[str] = []

        if self.summary:
            parts.append("=== Conversation Summary ===")
            parts.append(self.summary)
            parts.append("")

        if self.turns:
            parts.append("=== Recent Turns ===")
            for t in self.turns[-self.recent_turns_limit:]:
                prefix = "User" if t.role == "user" else "Assistant"
                parts.append(f"{prefix}: {t.content}")

        return "\n".join(parts).strip()

    def to_chat_history(self) -> List[dict]:
        """
        Convert memory into the chat_history format expected by AgentEnvironment.

        We include:
        - ONE synthetic 'system' message with the running summary (if present),
        - followed by the last N real turns.

        This makes the planner and other agents 'conscious' of long-term context
        without flooding the prompt with raw history.
        """
        history: List[dict] = []

        if self.summary:
            history.append(
                {
                    "role": "system",
                    "content": (
                        "Conversation summary for context (do not repeat verbatim):\n"
                        f"{self.summary}"
                    ),
                }
            )

        for t in self.turns[-self.recent_turns_limit:]:
            history.append({"role": t.role, "content": t.content})

        return history


# Global, per-process memory instance
_global_session_memory: SessionMemory | None = None


def get_session_memory() -> SessionMemory:
    """
    Return the global session memory object for this process.
    """
    global _global_session_memory
    if _global_session_memory is None:
        _global_session_memory = SessionMemory()
    return _global_session_memory


def reset_session_memory() -> None:
    """
    Reset the global session memory (e.g., if you want a fresh chat).
    """
    global _global_session_memory
    _global_session_memory = SessionMemory()
