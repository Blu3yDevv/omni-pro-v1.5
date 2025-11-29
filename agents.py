# agents.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional

from llm_client import (
    call_llm_fast,
    call_llm_planner,
    call_llm_research,
    call_llm_evidence,
    call_llm_reviewer,
    call_llm_final,
    call_llm_final_stream,  # NEW: streaming finalizer
)

from tools import fast_tool_router
from tools import deep_web_search  # if this import exists in your version
  # canonical web search wrapper
from rag_db import retrieve_rag_context
from config import config
from terminal_ui import (
    print_box,
    print_status,
    FG_CYAN,
    FG_GREEN,
    FG_YELLOW,
    FG_MAGENTA,
)


# ---------------------------------------------------------------------------
# Constants for context limiting
# ---------------------------------------------------------------------------

MAX_RAG_CHARS = 1800
MAX_WEB_CHARS_RESEARCH = 2600
MAX_WEB_CHARS_EVIDENCE = 2600
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 1500


# ---------------------------------------------------------------------------
# Helper: chat history formatting
# ---------------------------------------------------------------------------

def _format_history_for_prompt(
    history: List[Dict[str, str]],
    max_turns: int = MAX_HISTORY_TURNS,
    max_chars: int = MAX_HISTORY_CHARS,
) -> str:
    if not history:
        return "None."
    tail = history[-max_turns:]
    lines: List[str] = []
    for msg in tail:
        role = msg.get("role", "user")
        prefix = "User" if role == "user" else "Assistant"
        content = msg.get("content", "").strip().replace("\n", " ")
        if not content:
            continue
        lines.append(f"{prefix}: {content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        # keep the most recent content
        text = text[-max_chars:]
    return text or "None."


# ---------------------------------------------------------------------------
# Helper: robust JSON extraction from noisy LLM output
# ---------------------------------------------------------------------------

def _extract_json_dict(raw: str) -> Dict[str, Any] | None:
    """
    Try to parse a dict from an LLM response that MAY contain extra prose.

    Strategy:
    1) First try json.loads(raw) directly.
    2) If that fails, find the first '{' and the last '}', slice, and try again.
    3) If that still fails, return None.
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    # Attempt 1: parse whole string
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Attempt 2: slice between first '{' and last '}'
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlannerDecision:
    """
    Routing + configuration decision made by the Planner.

    mode:
      - "short_answer"   → quick direct answer, minimal agents.
      - "full_answer"    → normal multi-agent answer (no web unless needed).
      - "math_or_time"   → use the math/time helper instead of heavy stack.
      - "research_heavy" → full research + evidence + review (+ optional judge).

    Flags:
      - use_researcher: whether to call the Researcher agent.
      - use_evidence:   whether to call the Evidence agent.
      - use_reviewer:   whether to call the Reviewer agent.
      - use_judge:      whether to call the Judge for a final audit / rewrite.

    Additional knobs (Phase 2):
      - context_level: "minimal" | "normal" | "full"
      - web_depth:     "none" | "light" | "deep" | "auto"
      - tools:         list of tool names to favour (e.g. ["calculator", "time"])
    """
    mode: str = "full_answer"
    use_researcher: bool = True
    use_evidence: bool = True
    use_reviewer: bool = True
    use_judge: bool = True

    context_level: str = "normal"   # "minimal" | "normal" | "full"
    web_depth: str = "auto"         # "none" | "light" | "deep" | "auto"
    tools: List[str] = field(default_factory=list)

    notes: str = ""


@dataclass
class ReviewDecision:
    """
    Reviewer routing decision for refinement.
    """
    needs_revision: bool = False
    target: str = "none"               # "none" | "researcher" | "evidence"
    message_for_target: str = ""
    review_notes: str = ""             # human-readable critique for logs


@dataclass
class JudgeVerdict:
    """
    Verdict from the Judge agent on the final answer.
    """
    accept: bool = True
    needs_rewrite: bool = False
    message_for_finalizer: str = ""
    critique_for_logs: str = ""


@dataclass
class AgentEnvironment:
    """
    Shared environment object passed between internal agents.

    It stores:
    - user_input: the current user question for this turn
    - chat_history: prior turns for context-aware reasoning
    - rag_context: internal knowledge base snippets
    - web_cache: cached web search results keyed by (label, query, max_results, max_chars)
    - agent_outputs: raw text outputs per agent name
    - activity_log: chronological log of what happened

    Phase 2 additions:
    - history_max_turns / history_max_chars: planner-controlled context window
    - web_depth: planner-controlled web search depth ("none"/"light"/"deep"/"auto")
    """
    user_input: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    rag_context: str = ""
    web_cache: Dict[str, str] = field(default_factory=dict)
    agent_outputs: Dict[str, str] = field(default_factory=dict)
    activity_log: List[str] = field(default_factory=list)

    # Planner-tuned context settings
    history_max_turns: int = MAX_HISTORY_TURNS
    history_max_chars: int = MAX_HISTORY_CHARS
    web_depth: str = "auto"

    # --- Logging helpers ---------------------------------------------------

    def log(self, message: str) -> None:
        self.activity_log.append(message)

    # alias for compatibility with older code
    def log_event(self, text: str) -> None:
        self.log(text)

    # --- Chat history helpers ----------------------------------------------

    def set_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> None:
        self.chat_history = chat_history or []

    def history_snippet(
        self,
        max_turns: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Return a formatted slice of recent chat history.

        If max_turns / max_chars are not passed, defaults to the planner-controlled
        history_max_turns / history_max_chars.
        """
        turns = max_turns if max_turns is not None else self.history_max_turns
        chars = max_chars if max_chars is not None else self.history_max_chars
        return _format_history_for_prompt(self.chat_history, turns, chars)

    # --- Knowledge helpers -------------------------------------------------

    def ensure_rag_loaded(self) -> str:
        """
        Load RAG context once per turn, including a slice of recent history
        to improve retrieval, and truncate to MAX_RAG_CHARS.

        Uses planner-controlled history_snippet() by default.
        """
        if not self.rag_context:
            query = self.user_input
            history = self.history_snippet()
            if history and history != "None.":
                query = f"{self.user_input}\n\nRecent context:\n{history}"
            self.rag_context = retrieve_rag_context(query) or ""
            if self.rag_context and len(self.rag_context) > MAX_RAG_CHARS:
                self.rag_context = self.rag_context[:MAX_RAG_CHARS]
            self.log(f"[env] RAG context loaded (len={len(self.rag_context)})")
        return self.rag_context

    def run_web_search(
        self,
        label: str,
        query: Optional[str] = None,
        max_results: int = 12,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Run a deep web search and cache by (label, query, max_results, max_chars).
        Returns the raw merged search context string.

        Respects planner-controlled web_depth:
          - "none": disables web search entirely.
          - "light": trims result count and snippet size.
          - "deep": keeps or slightly expands requested depth.
          - "auto": leaves parameters as requested.
        """
        if query is None:
            query = self.user_input

        depth = (self.web_depth or "auto").lower().strip()

        if depth == "none":
            msg = (
                "WEB SEARCH DISABLED BY PLANNER.\n"
                f"- label: {label}\n"
                f"- query: {query}"
            )
            self.log(f"[env] web search disabled for label={label}")
            return msg

        # Adjust for "light" searches (fewer results / smaller snippets)
        if depth == "light":
            max_results = min(max_results, 6)
            # If max_chars not provided, cap at ~half of our research window
            if max_chars is None:
                max_chars = MAX_WEB_CHARS_RESEARCH // 2
            else:
                max_chars = min(max_chars, MAX_WEB_CHARS_RESEARCH // 2)

        key = f"{label}::{query}::{max_results}::{max_chars}"
        if key in self.web_cache:
            self.log(f"[env] reused cached web search: {key}")
            return self.web_cache[key]

        self.log(f"[env] running web search: {key}")
        try:
            if max_chars is None:
                ctx = deep_web_search(query, max_results=max_results)
            else:
                ctx = deep_web_search(query, max_results=max_results, max_chars=max_chars)
        except Exception as e:
            ctx = f"[WEB SEARCH ERROR: {e}]"
        self.web_cache[key] = ctx
        return ctx

    # --- Agent output helpers ----------------------------------------------

    def set_agent_output(self, agent_name: str, text: str) -> None:
        self.agent_outputs[agent_name] = text
        self.log(f"[env] stored output for {agent_name} (len={len(text)})")

    # alias for compatibility with older code
    def record_agent_output(self, agent_name: str, content: str) -> None:
        self.set_agent_output(agent_name, content)

    def get_agent_output(self, agent_name: str) -> str:
        return self.agent_outputs.get(agent_name, "")

    def agent_outputs_snippet(self, max_chars: int = 1500) -> str:
        if not self.agent_outputs:
            return "None."
        pieces: List[str] = []
        for name, text in self.agent_outputs.items():
            clean = text.strip().replace("\n\n", "\n")
            pieces.append(f"[{name}]\n{clean}")
        joined = "\n\n".join(pieces)
        if len(joined) > max_chars:
            joined = joined[-max_chars:]
        return joined or "None."


    def all_agent_outputs(self) -> Dict[str, str]:
        """
        Return a shallow copy of all agent outputs.
        Useful when you want to return everything at once.
        """
        return dict(self.agent_outputs)



    # --- Snapshot for prompts / logging ------------------------------------

    def snapshot_for_prompt(self, for_agent: str) -> str:
        """
        Produce a textual snapshot of the environment.

        This is meant to be *read* by the language model as part of its context.
        We keep it information-dense but not insanely long:
        - full user question
        - recent chat history
        - short note about RAG length
        - list of agent outputs with truncated previews
        - web cache keys
        - tail of the activity log
        """
        lines: List[str] = []
        lines.append(f"ENVIRONMENT SNAPSHOT for {for_agent}")
        lines.append("")
        lines.append("[USER QUESTION]")
        lines.append(self.user_input.strip())
        lines.append("")
        lines.append("[RECENT CHAT HISTORY]")
        lines.append(self.history_snippet())
        lines.append("")

        # RAG summary (length only, actual content passed separately where needed)
        if self.rag_context:
            lines.append(f"[RAG STATUS] loaded (len={len(self.rag_context)})")
        else:
            lines.append("[RAG STATUS] not loaded yet")
        lines.append("")

        # Agent outputs preview
        if self.agent_outputs:
            lines.append("[AGENT OUTPUTS AVAILABLE]")
            for name, text in self.agent_outputs.items():
                preview = text.strip()
                if len(preview) > 600:
                    preview = preview[:600] + "\n...[preview truncated in snapshot]..."
                lines.append(f"--- {name} (len={len(text)}) ---")
                lines.append(preview)
                lines.append("")
        else:
            lines.append("[AGENT OUTPUTS AVAILABLE] none yet")
            lines.append("")

        # Web cache keys only (actual content passed separately where needed)
        if self.web_cache:
            lines.append("[WEB SEARCH CACHE KEYS]")
            for key in self.web_cache.keys():
                lines.append(f"- {key}")
            lines.append("")
        else:
            lines.append("[WEB SEARCH CACHE KEYS] none yet")
            lines.append("")

        # Activity log (tail)
        if self.activity_log:
            lines.append("[ACTIVITY LOG (last 15 entries)]")
            for entry in self.activity_log[-15:]:
                lines.append(f"- {entry}")
        else:
            lines.append("[ACTIVITY LOG] empty")

        return "\n".join(lines)

    def build_snapshot(self, agent_name: str, max_chars: int = 2800) -> str:
        """
        Build a JSON-style snapshot that agents can see as their 'environment'.
        This is optional and mainly for debugging / alternative prompts.
        """
        env = {
            "agent_name": agent_name,
            "user_input": self.user_input,
            "recent_chat_history": self.history_snippet(),
            "known_agent_outputs": self.agent_outputs,
            "recent_events": self.activity_log[-12:],
        }
        text = json.dumps(env, ensure_ascii=False, indent=2)
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    def to_log_string(self) -> str:
        return json.dumps(
            {
                "user_input": self.user_input,
                "chat_history_tail": self.history_snippet(),
                "agent_outputs": self.agent_outputs,
                "activity_log": self.activity_log,
            },
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------

def _planner_agent_env(env: AgentEnvironment) -> PlannerDecision:
    """
    Decide how heavy the pipeline should be and which agents to use.

    Behaviour:
    - Super fast path for trivial greetings.
    - Otherwise, call the LLM planner which sets mode, flags, context depth, web depth, etc.
    """
    env.log("[planner] starting")

    # ---- SUPER-FAST PATH FOR VERY TRIVIAL GREETINGS ONLY -------------------
    q_stripped = (env.user_input or "").strip().lower()
    trivial_greetings = {"hi", "hey", "hello", "yo", "sup", "hi!", "hey!", "hello!"}
    if q_stripped in trivial_greetings:
        decision = PlannerDecision(
            mode="short_answer",
            use_researcher=False,
            use_evidence=False,
            use_reviewer=False,
            use_judge=False,
            context_level="minimal",
            web_depth="none",
            tools=[],
            notes="Trivial greeting only. Respond briefly and invite user to ask something.",
        )
        env.set_agent_output(
            "planner",
            f"Mode: {decision.mode}\n"
            f"use_researcher={decision.use_researcher}, "
            f"use_evidence={decision.use_evidence}, "
            f"use_reviewer={decision.use_reviewer}, "
            f"use_judge={decision.use_judge}\n"
            f"context_level={decision.context_level}, "
            f"web_depth={decision.web_depth}, "
            f"tools={decision.tools}\n\n"
            f"Plan:\n{decision.notes}",
        )
        env.log("[planner] trivial greeting fast-path")
        env.log("[planner] finished")
        return decision

    # ---- NORMAL PLANNER USING LLM -----------------------------------------
    system_prompt = """
You are the Chief Planner and Router for an internal multi-stage assistant called Omni-Pro.

ROLE
- Read the user's message carefully (once, slowly).
- Consider the recent conversation history (do not treat the message as isolated).
- Decide HOW MUCH processing is needed, and HOW DEEP it should go.
- Decide which specialist stages to use: Researcher, Evidence Analyst, Reviewer, Judge.
- Decide context depth and web search depth.
- Produce a compact, information-dense internal plan in the "notes" field.

ABSOLUTE RESTRICTIONS
- You MUST NOT mention or rely on any "knowledge cutoff" date.
- You MUST NOT assume that something "does not exist" or "has not been released"
  just because you personally have not seen it or it sounds unfamiliar.
- Existence / release status must be determined later by the Researcher using web + RAG.

OUTPUT FORMAT
Return a SINGLE JSON object and NOTHING else, matching this schema exactly:

{
  "mode": "short_answer" | "full_answer" | "math_or_time" | "research_heavy",
  "use_researcher": true | false,
  "use_evidence": true | false,
  "use_reviewer": true | false,
  "use_judge": true | false,
  "context_level": "minimal" | "normal" | "full",
  "web_depth": "none" | "light" | "deep" | "auto",
  "tools": ["calculator" | "time" | "other-tool-names"],
  "notes": "..."
}

Do NOT:
- Wrap the JSON in code fences.
- Add comments, explanations, or extra keys.
- Use trailing commas.

ROUTING RULES (HIGH LEVEL)
- Goal: keep the pipeline LIGHT for normal chat, and HEAVY only when needed.

1) PURE GREETINGS / INTRODUCTION
   - Examples: "hi", "hello", "hey", "what is your name", "who are you".
   - mode: "short_answer"
   - use_researcher/use_evidence/use_reviewer/use_judge: false
   - context_level: "minimal"
   - web_depth: "none"

2) PURE MATH OR TIME (no external facts needed)
   - Examples: "77^3", "what is 5 + 5", "what time is it in Seychelles".
   - mode: "math_or_time"
   - all flags false
   - context_level: "minimal"
   - web_depth: "none"
   - tools: include "calculator" and/or "time" as appropriate.

3) GENERAL CHAT / SCHOOL EXPLANATIONS / PERSONAL ADVICE
   - Examples: "explain inheritance and variation for Year 9",
              "help me plan my study schedule",
              "how do I stay disciplined".
   - These usually do NOT need live web search.
   - mode: "short_answer" if the question is small and focused.
   - mode: "full_answer" if it has multiple parts or needs deeper reasoning.
   - use_researcher/use_evidence/use_reviewer/use_judge: false
   - context_level: "normal" (or "full" if prior turns matter).
   - web_depth: "none" (or "light" only if genuinely helpful).

4) REAL-WORLD FACTUAL QUERIES THAT BENEFIT FROM FRESH DATA
   - Examples: hardware specs and comparisons, GPU/CPU benchmarks,
               game payouts, release dates, "latest" information,
               finance/economy questions, laws, locations, recent events.
   - mode: "research_heavy"
   - use_researcher: true
   - use_evidence: true
   - use_reviewer: true
   - use_judge: true (by default)
   - context_level: "full" if prior turns matter, otherwise "normal".
   - web_depth: "deep" (or "light" if the question is simple).

UNCERTAINTY RULE
- If you are unsure whether heavy research is REQUIRED, prefer:
  - "short_answer" or "full_answer" WITHOUT research, and
  - "web_depth": "none" or "light",
  instead of defaulting to "research_heavy".

REQUIREMENTS FOR "notes"
"notes" must:
1) Restate the user question mentioning key entities.
2) List 3–8 internal steps (e.g. identify type of question, decide whether
   web is needed, decide how deep the context should be, structure final answer).
3) Capture any format/tone constraints from the user.

"notes" is internal only. Never write about JSON or routing there.
    """.strip()

    snapshot = env.snapshot_for_prompt("planner")

    user_prompt = f"""
[USER QUESTION]
{env.user_input}

[ENVIRONMENT SNAPSHOT]
{snapshot}

Return ONLY the JSON object described above.
    """.strip()

    raw = call_llm_planner(system_prompt, user_prompt)

    # Conservative default if parsing totally fails.
    decision = PlannerDecision(
        mode="research_heavy",
        use_researcher=True,
        use_evidence=True,
        use_reviewer=True,
        use_judge=True,
        context_level="normal",
        web_depth="deep",
        tools=[],
        notes=str(raw).strip(),
    )

    data = _extract_json_dict(raw)
    if data is not None:
        try:
            mode = str(data.get("mode", decision.mode)).strip() or decision.mode
            use_researcher = bool(data.get("use_researcher", decision.use_researcher))
            use_evidence = bool(data.get("use_evidence", decision.use_evidence))
            use_reviewer = bool(data.get("use_reviewer", decision.use_reviewer))
            use_judge = bool(data.get("use_judge", decision.use_judge))
            notes = str(data.get("notes", decision.notes)).strip() or decision.notes

            context_level = str(
                data.get("context_level", decision.context_level)
            ).strip().lower() or decision.context_level
            if context_level not in ("minimal", "normal", "full"):
                context_level = "normal"

            web_depth = str(
                data.get("web_depth", decision.web_depth)
            ).strip().lower() or decision.web_depth
            if web_depth not in ("none", "light", "deep", "auto"):
                web_depth = "auto"

            tools_list: List[str] = []
            raw_tools = data.get("tools", decision.tools)
            if isinstance(raw_tools, list):
                for t in raw_tools:
                    if isinstance(t, str):
                        t_clean = t.strip().lower()
                        if t_clean:
                            tools_list.append(t_clean)

            decision = PlannerDecision(
                mode=mode,
                use_researcher=use_researcher,
                use_evidence=use_evidence,
                use_reviewer=use_reviewer,
                use_judge=use_judge,
                context_level=context_level,
                web_depth=web_depth,
                tools=tools_list,
                notes=notes,
            )
        except Exception:
            env.log("[planner] parsed JSON but reconstruction failed, using fallback decision")
    else:
        env.log("[planner] JSON parse failed, using raw text in notes")

    # Hard invariants: math_or_time never uses heavy agents.
    if decision.mode == "math_or_time":
        decision.use_researcher = False
        decision.use_evidence = False
        decision.use_reviewer = False
        decision.use_judge = False
        decision.context_level = "minimal"
        decision.web_depth = "none"

    # If no researcher, auto-disable downstream heavy agents.
    if not decision.use_researcher:
        decision.use_evidence = False
        decision.use_reviewer = False
        decision.use_judge = False

    # Runtime overrides: balanced / turbo / deep
    runtime = getattr(config, "runtime", None)
    mode_setting = getattr(runtime, "mode", "balanced") if runtime is not None else "balanced"
    mode_setting = str(mode_setting).lower()

    if mode_setting == "turbo":
        if decision.mode not in ("research_heavy", "math_or_time"):
            if not decision.use_researcher:
                decision.use_evidence = False
                decision.use_reviewer = False
                decision.use_judge = False
                env.log("[planner] turbo mode: stripped evidence/reviewer/judge for non-research query")
    elif mode_setting == "deep":
        if decision.mode != "math_or_time":
            decision.use_researcher = True
            decision.use_evidence = True
            decision.use_reviewer = True
            decision.use_judge = True
            env.log("[planner] deep mode: forcing full stack for non-math query")

    if not decision.use_researcher:
        decision.use_evidence = False
        decision.use_reviewer = False
        decision.use_judge = False

    env.set_agent_output(
        "planner",
        (
            f"Mode: {decision.mode}\n"
            f"use_researcher={decision.use_researcher}, "
            f"use_evidence={decision.use_evidence}, "
            f"use_reviewer={decision.use_reviewer}, "
            f"use_judge={decision.use_judge}\n"
            f"context_level={decision.context_level}, "
            f"web_depth={decision.web_depth}, "
            f"tools={decision.tools}\n\n"
            f"Plan:\n{decision.notes}"
        ),
    )
    env.log("[planner] finished")
    return decision



def planner_agent(user_input: str) -> PlannerDecision:
    """
    Backwards-compatible wrapper that runs the planner in a fresh environment.

    NOTE: In the full pipeline you should use run_multi_agent(), which shares
    a single AgentEnvironment across all agents. This wrapper is only here in
    case other code imports planner_agent directly.
    """
    env = AgentEnvironment(user_input=user_input)
    return _planner_agent_env(env)


# ---------------------------------------------------------------------------
# Researcher Agent
# ---------------------------------------------------------------------------

def _researcher_agent_env(env: AgentEnvironment, plan: PlannerDecision) -> str:
    """
    Researcher:
    - Uses RAG + deep web search.
    - Produces a rich 'Research Note'.
    - Errs on the side of being long and exhaustive for non-trivial queries.

    Web depth is controlled by env.web_depth ("none"/"light"/"deep"/"auto").
    """
    env.log("[researcher] starting")

    rag_context = env.ensure_rag_loaded()

    # Primary web search with generous result count but limited chars
    web_context_main = env.run_web_search(
        "research_main",
        max_results=18,
        max_chars=MAX_WEB_CHARS_RESEARCH,
    )

    # Optional secondary search with a slightly modified query (more generic)
    web_context_alt = env.run_web_search(
        "research_alt",
        query=env.user_input + " details review",
        max_results=12,
        max_chars=MAX_WEB_CHARS_RESEARCH,
    )

    merged_web_context = (
        "[PRIMARY RESULTS]\n"
        f"{web_context_main}\n\n"
        "[SECONDARY RESULTS]\n"
        f"{web_context_alt}"
    )

    system_prompt = """
You are the Senior Researcher for Omni-Pro.

Your output is an INTERNAL RESEARCH REPORT. It is NOT shown directly to the user.

INPUTS YOU RECEIVE
- The user's question.
- The Planner's notes (internal strategy).
- INTERNAL KNOWLEDGE (RAG): trusted facts about Omni-Pro and other stored knowledge.
- WEB RESULTS: merged search snippets from multiple queries (may be disabled).
- An environment snapshot describing other agents' outputs and activity so far.

CRITICAL RESTRICTIONS
- You MUST actively use the WEB RESULTS whenever you make concrete external factual
  claims (specs, dates, prices, releases, payouts, etc.), IF they are available.
- You MUST NOT mention or rely on any "knowledge cutoff".
- You MUST NOT say that an entity "does not exist", "has not been released", or
  "is not real" purely because you did not see it in RAG or in your training.
- If web + RAG provide no strong evidence, say you could not find credible
  information, and treat any partial hints as speculative.

FACTUAL RIGOUR
- Prioritise concrete, checkable information:
  numbers, named entities, relationships, and clear context.
- For Omni-Pro itself, RAG is primary. For everything else, WEB RESULTS are primary.
- Prefer credible domains over random noise; ignore irrelevant hits.

STYLE AND TONE
- Do NOT address the user directly.
- Do NOT mention RAG, tools, prompts, "web search", or "APIs" explicitly.
- Be clear, neutral, and confident when evidence is strong.
- Be explicit about uncertainty or speculation when evidence is weak.

OUTPUT FORMAT (plain text, no JSON, no code fences):

Research Note:
  - 4–8 dense paragraphs that directly address the question content.
  - Explain key entities, relationships, and context.
  - For multi-entity questions, explicitly compare them.

Key Facts:
  - 8–20 bullet points of the most important checkable facts.

Uncertainties:
  - List anything genuinely unclear, speculative, or poorly supported.
  - If web results are disabled or do NOT clearly confirm existence/specs of an item,
    say so here instead of declaring it "not real".

Sources:
  - 6–15 lines of "domain or short title – URL" for the strongest sources.

LENGTH
- For non-trivial questions, err on the side of being detailed but focused on
  the actual question.
    """.strip()

    snapshot = env.snapshot_for_prompt("researcher")

    user_prompt = f"""
[USER QUESTION]
{env.user_input}

[PLANNER NOTES]
{plan.notes}

[ENVIRONMENT SNAPSHOT]
{snapshot}

[INTERNAL KNOWLEDGE (RAG)]
{rag_context}

[WEB RESULTS]
{merged_web_context}

Write the Research Note now.
    """.strip()

    research = call_llm_research(system_prompt, user_prompt).strip()
    env.set_agent_output("researcher", research)
    env.log("[researcher] finished")
    return research


def researcher_agent(plan: PlannerDecision, user_input: str) -> str:
    """
    Backwards-compatible wrapper that runs the researcher in a fresh environment.
    In the real pipeline, run_multi_agent() should be used instead.
    """
    env = AgentEnvironment(user_input=user_input)
    env.ensure_rag_loaded()
    return _researcher_agent_env(env, plan)


def _researcher_refine_agent(
    env: AgentEnvironment,
    plan: PlannerDecision,
    previous_summary: str,
    reviewer_message: str,
) -> str:
    """
    Refinement version of the Researcher.
    """
    env.log("[researcher_refine] starting")
    rag_context = env.ensure_rag_loaded()
    web_context_main = env.run_web_search(
        "research_refine_main",
        max_results=18,
        max_chars=MAX_WEB_CHARS_RESEARCH,
    )
    web_context_alt = env.run_web_search(
        "research_refine_alt",
        query=env.user_input + " updated info",
        max_results=12,
        max_chars=MAX_WEB_CHARS_RESEARCH,
    )
    merged_web_context = (
        "[PRIMARY RESULTS]\n"
        f"{web_context_main}\n\n"
        "[SECONDARY RESULTS]\n"
        f"{web_context_alt}"
    )

    system_prompt = """
You are the Senior Researcher for Omni-Pro, refining a previous internal Research Summary.

Your output is an INTERNAL RESEARCH REPORT. It is NOT shown directly to the user.

CRITICAL RESTRICTION: NO EXISTENCE CLAIMS FROM IGNORANCE
- You MUST NOT declare that something is "not real", "fake", "does not exist",
  or "has not been released" purely because evidence is weak or hard to find.
- Absence of evidence is NOT proof of non-existence.
- If you cannot confidently confirm that an entity exists, treat it as UNCERTAIN
  and document that clearly in the Uncertainties section instead of calling it fake.

INPUTS
- The original user question.
- The Planner's notes.
- Your previous Research Summary.
- Reviewer feedback.
- INTERNAL KNOWLEDGE (RAG).
- Fresh WEB RESULTS (may be disabled or light).
- An environment snapshot.

YOUR JOB
- Produce a clearly IMPROVED Research Summary that:
  * preserves correct and strong parts from the old one,
  * removes or fixes anything the reviewer flagged,
  * adds concrete facts, numbers, and better sources where needed,
  * improves clarity, comparisons, and structure,
  * handles unknown/uncertain entities honestly without guessing.

Do NOT:
- Address the user directly.
- Mention RAG, tools, prompts, or "web search" in the text.
- Talk about "the previous version" explicitly. Just output the improved version.

OUTPUT FORMAT (same as normal Researcher output):
Research Note:
Key Facts:
Uncertainties:
Sources:
    """.strip()

    snapshot = env.snapshot_for_prompt("researcher_refine")
    user_prompt = f"""
[USER QUESTION]
{env.user_input}

[PLANNER NOTES]
{plan.notes}

[PREVIOUS RESEARCH SUMMARY]
{previous_summary}

[REVIEWER FEEDBACK]
{reviewer_message}

[ENVIRONMENT SNAPSHOT]
{snapshot}

[INTERNAL KNOWLEDGE (RAG)]
{rag_context}

[WEB RESULTS]
{merged_web_context}

Write the refined Research Note now.
    """.strip()

    refined = call_llm_research(system_prompt, user_prompt).strip()
    env.set_agent_output("researcher_refine", refined)
    env.log("[researcher_refine] finished")
    return refined


# ---------------------------------------------------------------------------
# Evidence Agent
# ---------------------------------------------------------------------------

def _evidence_agent_env(env: AgentEnvironment, research_summary: str) -> str:
    """
    Evidence Analyst:
    - ALWAYS runs its own deep_web_search (unless disabled by planner).
    - Verifies main claims from the Research Summary.
    - Produces a dense Evidence Pack.
    """
    env.log("[evidence] starting")
    # Fresh web search focused on verification
    web_context_verify = env.run_web_search(
        "evidence_verify",
        max_results=18,
        max_chars=MAX_WEB_CHARS_EVIDENCE,
    )

    system_prompt = """
You are the Evidence Analyst for Omni-Pro.

Your output is an INTERNAL EVIDENCE PACK. It is NOT shown directly to the user.

CRITICAL RESTRICTION: NO EXISTENCE CLAIMS FROM IGNORANCE
- You MUST NOT declare that something is "not real", "fake", "does not exist",
  or "has not been released" purely because:
    * it is unfamiliar, or
    * evidence is sparse or hard to find.
- If you cannot find strong support that something exists, treat its status as
  UNCERTAIN and say that evidence is insufficient, NOT that it is fake or impossible.

YOU RECEIVE
- The user's question.
- A Research Summary from the Researcher.
- Fresh WEB RESULTS to verify and enrich that summary (may be disabled/light).
- An environment snapshot.

YOUR JOB
- Use the WEB RESULTS actively to:
  * Check important claims from the Research Summary.
  * Flag contradictions or weakly-supported statements.
  * Add missing but clearly relevant facts that have strong support.
- For each key claim, say whether it is supported, contradicted, or unclear.

Do NOT:
- Address the user directly.
- Mention RAG, tools, prompts, or "web search" explicitly.
- Invent URLs or sources that are not in the WEB RESULTS.

OUTPUT FORMAT (plain text):

Evidence Pack:
Key Claims:
  - Claim: <short statement>
    Status: supported | contradicted | unclear
    Evidence:
      - <domain> – <1–2 sentence snippet>
      - <domain> – <1–2 sentence snippet>

Uncertain / Conflicting Points:
  - Describe any real disagreements between sources.
  - Mention if some information is only speculative or poorly supported.
  - If the existence or release status of something is not firmly supported,
    explain that the evidence is inconclusive rather than calling it "not real".

Sources:
  - <domain or short title> – <URL>

STYLE AND DEPTH
- Focus on the 6–14 most important claims relative to the user question.
- Provide at least 1–2 evidence snippets per claim when possible.
- Prefer clear, concise evidence over long quotes.
- Drop low-quality or noisy sources when better ones exist.
    """.strip()

    snapshot = env.snapshot_for_prompt("evidence")
    user_prompt = f"""
[USER QUESTION]
{env.user_input}

[RESEARCH SUMMARY]
{research_summary}

[ENVIRONMENT SNAPSHOT]
{snapshot}

[WEB RESULTS FOR VERIFICATION]
{web_context_verify}

Write the Evidence Pack now.
    """.strip()

    evidence = call_llm_evidence(system_prompt, user_prompt).strip()
    env.set_agent_output("evidence", evidence)
    env.log("[evidence] finished")
    return evidence


def evidence_agent(user_input: str, research_summary: str) -> str:
    """
    Backwards-compatible wrapper that runs the evidence analyst in a fresh environment.
    """
    env = AgentEnvironment(user_input=user_input)
    return _evidence_agent_env(env, research_summary)


def _evidence_refine_agent(
    env: AgentEnvironment,
    research_summary: str,
    previous_evidence: str,
    reviewer_message: str,
) -> str:
    """
    Refinement version of Evidence Agent.
    """
    env.log("[evidence_refine] starting")
    web_context_verify = env.run_web_search(
        "evidence_refine_verify",
        max_results=18,
        max_chars=MAX_WEB_CHARS_EVIDENCE,
    )

    system_prompt = """
You are the Evidence Analyst for Omni-Pro, refining a previous INTERNAL Evidence Pack.

CRITICAL RESTRICTION: NO EXISTENCE CLAIMS FROM IGNORANCE
- Do NOT state that something is "not real", "fake", or "does not exist" simply
  because evidence is thin or hard to find.
- If evidence does not clearly confirm or deny an entity, treat it as UNCERTAIN
  and describe that lack of clarity in the Uncertain / Conflicting Points section.

YOU RECEIVE
- The user's question.
- The latest Research Summary.
- The previous Evidence Pack.
- Reviewer feedback.
- Fresh WEB RESULTS (may be disabled/light).
- An environment snapshot.

YOUR JOB
- Produce a stronger, cleaner Evidence Pack by:
  * fixing vague or unsupported claims,
  * removing noise and low-value details,
  * adding better or more precise sources where available,
  * tightening the mapping between claims and evidence,
  * handling uncertainty honestly instead of guessing.

Do NOT:
- Address the user directly.
- Mention RAG, tools, or "web search" explicitly.
- Talk about "the previous version" in the output. Just present the improved pack.

OUTPUT FORMAT (same as normal Evidence Pack):
Evidence Pack:
Key Claims:
  - Claim: ...
    Status: ...
    Evidence:
      - domain – snippet
Uncertain / Conflicting Points:
Sources:

Focus on clarity, trustworthiness, and direct relevance to the user question.
    """.strip()

    snapshot = env.snapshot_for_prompt("evidence_refine")
    user_prompt = f"""
[USER QUESTION]
{env.user_input}

[RESEARCH SUMMARY]
{research_summary}

[PREVIOUS EVIDENCE PACK]
{previous_evidence}

[REVIEWER FEEDBACK]
{reviewer_message}

[ENVIRONMENT SNAPSHOT]
{snapshot}

[WEB RESULTS FOR VERIFICATION]
{web_context_verify}

Write the refined Evidence Pack now.
    """.strip()

    refined = call_llm_evidence(system_prompt, user_prompt).strip()
    env.set_agent_output("evidence_refine", refined)
    env.log("[evidence_refine] finished")
    return refined


# ---------------------------------------------------------------------------
# Reviewer Agent
# ---------------------------------------------------------------------------

def _reviewer_decision_agent(env: AgentEnvironment, evidence_or_research: str) -> ReviewDecision:
    """
    Internal Reviewer:
    - Critiques content.
    - Decides if we should refine Researcher or Evidence once more.
    """
    env.log("[reviewer] starting")
    system_prompt = """
You are the Quality Control Reviewer for Omni-Pro.

Your output is INTERNAL and NOT shown to the user.

YOU RECEIVE
- The user's question.
- Either an Evidence Pack (preferred) or a Research Summary.
- An environment snapshot describing other agents' outputs and activity so far.

TASKS
1) Critically review the content for:
   - factual strength and support,
   - clarity and structure,
   - coverage of the user's question (including edge cases or constraints).
2) Decide whether we should trigger ONE more refinement round.
   You can send feedback either to:
   - the Researcher, or
   - the Evidence Analyst.

IMPORTANT
- Only request refinement if it will meaningfully improve accuracy, coverage,
  or clarity.
- Do NOT request refinement for tiny stylistic nitpicks.

OUTPUT FORMAT
Return a SINGLE JSON object and NOTHING else, matching this schema:

{
  "needs_revision": true | false,
  "target": "none" | "researcher" | "evidence",
  "message_for_target": "specific, concise instructions (<=120 words)",
  "review_notes": "human-readable critique for logs (<=200 words)"
}

CONSTRAINTS
- Do NOT wrap JSON in code fences.
- Do NOT add comments or extra keys.
- Do NOT use trailing commas.
- "target" must be one of: "none", "researcher", "evidence".

GUIDELINES
- If the content is basically accurate, well-supported, and answers the user
  clearly:
    needs_revision = false, target = "none".
- If facts are weak / missing / badly supported:
    needs_revision = true, target = "researcher".
- If the mapping between claims and sources is messy or unclear:
    needs_revision = true, target = "evidence".
    """.strip()

    snapshot = env.snapshot_for_prompt("reviewer")
    user_prompt = f"""
[USER QUESTION]
{env.user_input}

[CONTENT TO REVIEW]
{evidence_or_research}

[ENVIRONMENT SNAPSHOT]
{snapshot}

Return ONLY the JSON object.
    """.strip()

    raw = call_llm_reviewer(system_prompt, user_prompt).strip()

    # Fallback: treat raw text as review_notes only
    review = ReviewDecision(
        needs_revision=False,
        target="none",
        message_for_target="",
        review_notes=str(raw).strip(),
    )

    data = _extract_json_dict(raw)
    if data is not None:
        try:
            needs_revision = bool(data.get("needs_revision", review.needs_revision))
            target = str(data.get("target", review.target)).strip().lower() or review.target
            message_for_target = str(
                data.get("message_for_target", review.message_for_target)
            ).strip()
            review_notes = str(
                data.get("review_notes", review.review_notes)
            ).strip() or review.review_notes

            if target not in ("none", "researcher", "evidence"):
                target = "none"

            review = ReviewDecision(
                needs_revision=needs_revision,
                target=target,
                message_for_target=message_for_target,
                review_notes=review_notes,
            )
        except Exception:
            env.log("[reviewer] parsed JSON but reconstruction failed, keeping fallback")
    else:
        env.log("[reviewer] JSON parse failed, using raw text as review_notes")

    env.log(f"[reviewer] finished (needs_revision={review.needs_revision}, target={review.target})")
    return review


def reviewer_agent(evidence_pack: str, user_input: str) -> str:
    """
    Backwards-compatible wrapper returning just the review text (for logging).
    """
    env = AgentEnvironment(user_input=user_input)
    dec = _reviewer_decision_agent(env, evidence_pack)
    return dec.review_notes


# ---------------------------------------------------------------------------
# Judge Agent (post-finalizer gatekeeper)
# ---------------------------------------------------------------------------

def _judge_agent_env(
    env: AgentEnvironment,
    user_input: str,
    final_answer: str,
    research_summary: str,
    evidence_pack: str,
) -> JudgeVerdict:
    """
    Judge:
    - Evaluates the final answer for:
        * structural quality,
        * factual consistency with research/evidence,
        * citation use,
        * tone and over-assertion.
    - Can request ONE rewrite from the finalizer.
    """
    env.log("[judge] starting")

    system_prompt = """
You are the Judge for Omni-Pro's final answers.

Your job is to audit the FINAL ANSWER produced for the user.

YOU RECEIVE
- The user's question.
- The FINAL ANSWER that will be shown to the user.
- The internal Research Summary.
- The internal Evidence Pack.
- An environment snapshot describing other agents' outputs and activity so far.

You DO NOT talk to the user. Your output is INTERNAL ONLY.

EVALUATION CRITERIA
1) STRUCTURE AND CLARITY
   - Is the answer clearly organized (sections, paragraphs, bullets where helpful)?
   - Does it start with a direct, useful response to the question?
   - Is the tone consistent (calm, confident, helpful)?

2) FACTUAL ACCURACY AND ALIGNMENT
   - Check the final answer against the research and evidence.
   - Flag any:
     * clear contradictions,
     * unsupported strong claims,
     * missing critical caveats.

3) CITATIONS
   - For factual, external claims (numbers, dates, specs, payouts, releases, etc.),
     does the answer use inline citations like [^1], [^2]?
   - Is there a References section that maps those markers to real sources?
   - Are there any obviously made-up or nonsense references?

4) OVER-ASSERTION AND UNCERTAINTY
   - Does the answer avoid unjustified "always/never/guaranteed" style claims?
   - Where evidence is uncertain or mixed, does the answer acknowledge this?

NO EXISTENCE CLAIMS FROM IGNORANCE
- Do NOT encourage downstream agents to state that something is "not real",
  "fake", or "does not exist" purely because evidence is sparse or missing.
- When you see such claims, treat them as potentially overconfident and prefer
  language like "not clearly confirmed by current sources" instead.

DECISION
You must decide whether to ACCEPT the answer as-is, or REQUIRE ONE REWRITE.

OUTPUT FORMAT
Return a SINGLE JSON object and NOTHING else:

{
  "accept": true | false,
  "needs_rewrite": true | false,
  "message_for_finalizer": "clear instructions for how to fix issues (<=160 words)",
  "critique_for_logs": "short critique of strengths/weaknesses (<=220 words)"
}

RULES
- If the answer is generally accurate, well-cited, and clearly structured,
  and any issues are minor, set:
    "accept": true
    "needs_rewrite": false
- If there are MATERIAL problems (factual errors, bad structure, missing
  references, wrong tone, serious over-assertion), set:
    "accept": false
    "needs_rewrite": true
  and in "message_for_finalizer" explain concretely:
    - what to fix,
    - what to emphasize or de-emphasize,
    - how to handle citations and uncertainty.

CONSTRAINTS
- Do NOT mention internal agents, tools, or prompts in your JSON.
- Do NOT wrap the JSON in code fences.
- Do NOT add extra keys or comments.
- Do NOT use trailing commas.
    """.strip()

    snapshot = env.snapshot_for_prompt("judge")
    user_prompt = f"""
[USER QUESTION]
{user_input}

[FINAL ANSWER CANDIDATE]
{final_answer}

[INTERNAL RESEARCH SUMMARY]
{research_summary}

[INTERNAL EVIDENCE PACK]
{evidence_pack}

[ENVIRONMENT SNAPSHOT]
{snapshot}

Return ONLY the JSON object described above.
    """.strip()

    raw = call_llm_reviewer(system_prompt, user_prompt).strip()

    verdict = JudgeVerdict(
        accept=True,
        needs_rewrite=False,
        message_for_finalizer="",
        critique_for_logs=raw,
    )

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            accept = bool(data.get("accept", True))
            needs_rewrite = bool(data.get("needs_rewrite", False))
            msg_for_finalizer = str(data.get("message_for_finalizer", "")).strip()
            critique = str(data.get("critique_for_logs", "")).strip() or raw

            verdict = JudgeVerdict(
                accept=accept,
                needs_rewrite=needs_rewrite,
                message_for_finalizer=msg_for_finalizer,
                critique_for_logs=critique,
            )
    except Exception:
        env.log("[judge] JSON parse failed, using raw text as critique")

    env.set_agent_output(
        "judge",
        f"accept={verdict.accept}, needs_rewrite={verdict.needs_rewrite}\n\n{verdict.critique_for_logs}",
    )
    env.log(f"[judge] finished (accept={verdict.accept}, needs_rewrite={verdict.needs_rewrite})")
    return verdict


def judge_agent(
    user_input: str,
    final_answer: str,
    research_summary: str,
    evidence_pack: str,
) -> JudgeVerdict:
    """
    Backwards-compatible wrapper around the Judge agent.
    """
    env = AgentEnvironment(user_input=user_input)
    return _judge_agent_env(env, user_input, final_answer, research_summary, evidence_pack)


# ---------------------------------------------------------------------------
# Concluder Agent (Final Writer)
# ---------------------------------------------------------------------------

def _concluder_agent_env(
    env: AgentEnvironment,
    research_summary: str,
    evidence_pack: str,
    review_notes: str,
    judge_feedback: str = "",
    stream: bool = False,
    stream_callback: Optional[callable] = None,
) -> str:
    """
    Final user-facing agent.

    IMPORTANT:
    - Has access to ALL upstream information:
      research summary, evidence pack, review notes, RAG, and environment.
    - Must integrate key facts, not just give a vague summary.
    - Adds citations.
    - Strengthens cause–effect logic.
    - Maintains a consistent tone and avoids over-assertions.
    - May receive optional feedback from the Judge for a rewrite.
    - Must never mention agents, tools, internal prompts, or background routing.

    Streaming:
    - If stream=True and stream_callback is provided, tokens/chunks from the
      final LLM call are passed to stream_callback(text_chunk) as they arrive.
    - The full answer is still assembled and returned at the end.
    """
    env.log("[final] starting")
    rag_context = env.ensure_rag_loaded()

    system_prompt = """
You are Omni-Pro, the final user-facing assistant.

You receive INTERNAL MATERIAL:
- The user's original question.
- A Research Summary (rich explanation + key facts).
- An Evidence Pack (claims + sources).
- Internal Review / Judge notes.
- INTERNAL KNOWLEDGE (RAG).
- An environment snapshot describing internal processing so far.

The user NEVER sees the internal material. They only see YOUR final answer.

CORE OBJECTIVE
- Provide one high-quality, well-reasoned, well-cited answer that feels like it
  was written by a single expert assistant speaking directly to the user.
- By default, for any non-trivial question, be thorough and structured rather
  than minimal. A "premium" answer is clear, comprehensive, and organised.

INFORMATION USE
- Use high-value information from research, evidence, and RAG.
- Pull through key numbers, names, dates, and comparisons when they matter.
- Do not ignore useful relevant detail just to keep the answer short.

CITATIONS
- For external factual claims (numbers, dates, specs, payouts, releases, etc.),
  add inline markers [^1], [^2], etc.
- End with a "References" section mapping those markers to sources:
  [^1]: domain or short title – URL
- Reuse the same number for the same source if cited multiple times.
- Pure reasoning or RAG-only content does not require citations.

CAUSE–EFFECT LOGIC
- When you describe how or why something happens, make the causal links explicit.
- Avoid unexplained jumps from fact to conclusion.

CONFIDENCE AND UNCERTAINTY
- When evidence is strong, be direct and confident.
- When evidence is weak or speculative, say so clearly, but still give best-effort
  guidance.
- If web + RAG do NOT provide solid specs or existence for a product, say:
  you could not find credible, detailed information and avoid guessing.

BANNED META / SCRIPTED LANGUAGE
- Do NOT narrate your reasoning or planning steps. Do not say things like:
  "The user asked...", "Let me check...", "I will now...", "According to the research summary...",
  "According to the internal knowledge base...", etc.
- Do NOT mention RAG, research summary, evidence pack, documents, tools, or agents.
- Do NOT talk about "internal notes", "upstream analysis", or anything similar.
- Do NOT restate the full question before answering; just answer it.
- If you use phrases like "according to", they MUST refer only to external sources
  cited via [^n] in the References section (e.g. "According to [^1] ..."), NOT
  to internal docs or environment snapshots.

STRUCTURE AND LENGTH
- Greetings / identity only:
  - If the user only greets you or asks who/what you are (e.g. "hi", "hello",
    "what is your name", "who are you"):
      - Respond in 1–2 short sentences.
      - Be friendly and direct.
      - Do NOT describe your stack, architecture, or multi-stage system.
      - Do NOT mention years, versions, or implementation details unless explicitly asked.

- Explicitly brief requests:
  - If the user clearly asks for a brief answer (e.g. "short answer", "in one sentence",
    "TL;DR", "very short"), respect that and respond concisely while staying accurate.

- Default behaviour (everything else):
  - Start with a direct, high-level answer in 1–3 sentences so the user immediately
    sees the core result.
  - Then expand with a structured explanation. As a baseline:
      - Use clear Markdown headings (##, ###) for major sections when the topic is
        anything more than trivial.
      - Aim for multiple paragraphs that cover:
          * Key concepts and definitions.
          * Step-by-step reasoning or process (where helpful).
          * Relevant examples or scenarios.
          * Practical advice, trade-offs, or next steps for the user.
      - Use bullet points or numbered lists for steps, options, or pros/cons.
  - It is better to provide a full, well-organised mini-guide than a shallow paragraph,
    as long as the content stays relevant and non-repetitive.

ADDRESSING THE USER
- Talk directly to the user as "you".
- Use a calm, confident, natural tone (conversational but not chatty).
- Make it sound like a normal, human explanation, not like you’re reading from
  a script or spec sheet.
- Maintain Omni-Pro's premium, professional voice: refined, analytical, and thorough,
  not lazy or minimal.
    """.strip()
    snapshot = env.snapshot_for_prompt("final")

    user_prompt = f"""
[USER QUESTION]
{env.user_input}

[RESEARCH SUMMARY]
{research_summary}

[INTERNAL EVIDENCE PACK]
{evidence_pack}

[INTERNAL REVIEW / JUDGE NOTES]
{review_notes}

[INTERNAL JUDGE FEEDBACK (if any)]
{judge_feedback}

[INTERNAL KNOWLEDGE (RAG)]
{rag_context}

[ENVIRONMENT SNAPSHOT]
{snapshot}

Write the final answer for the user now.
    """.strip()

    # --- Normal vs streaming call -----------------------------------------
    if stream and stream_callback is not None:
        # Streaming version: callback gets chunks as they arrive
        final_answer = call_llm_final_stream(
            system_prompt,
            user_prompt,
            stream_callback,  # positional callback argument
        ).strip()
    else:
        # Non-streaming: single shot
        final_answer = call_llm_final(system_prompt, user_prompt).strip()

    env.set_agent_output("final", final_answer)
    env.log("[final] finished")
    return final_answer



def concluder_agent(
    user_input: str,
    research_summary: str,
    evidence_pack: str,
    review_notes: str,
) -> str:
    """
    Backwards-compatible wrapper that runs the final writer in a fresh environment.
    """
    env = AgentEnvironment(user_input=user_input)
    env.ensure_rag_loaded()
    return _concluder_agent_env(env, research_summary, evidence_pack, review_notes)


# ---------------------------------------------------------------------------
# Pipeline Orchestrator + Agent Status
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool-router shortcut (math/time etc.) before full agent stack
# ---------------------------------------------------------------------------

def try_tool_route(user_input: str) -> Tuple[bool, str, str]:
    """
    Lightweight shortcut before spinning up the full multi-agent pipeline.

    Delegates to fast_tool_router() from tools.py.

    Returns:
        used_tool: whether a tool produced a direct answer.
        answer:    the tool's final answer text (or "" if none).
        tool_name: short human-readable name of the tool used (for logging/UI).
    """
    try:
        result = fast_tool_router(user_input)
    except Exception:
        # If the tool router fails for any reason, fall back to normal agents.
        return False, "", ""

    # Allow a few possible shapes from fast_tool_router.
    if isinstance(result, tuple):
        if len(result) == 3:
            used, answer, name = result
        elif len(result) == 2:
            used, answer = result
            name = "tool"
        else:
            return False, "", ""
    else:
        # If router returns something unexpected, ignore and use full pipeline.
        return False, "", ""

    used = bool(used)
    answer = str(answer) if answer is not None else ""
    name = str(name) if name is not None else "tool"

    if not used or not answer.strip():
        return False, "", ""

    return True, answer, name


def run_multi_agent(
    user_input: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    stream_final: bool = False,
    stream_callback: Optional[callable] = None,
) -> Tuple[str, Dict[str, str]]:
    """
    Orchestrates:
      (tool shortcut) →
      Planner → (optional) Researcher → (optional) Evidence →
      Reviewer (with possible refinement) → Concluder → Judge (optional, non-stream).

    Streaming behaviour:
      - If stream_final=True and stream_callback is provided, ONLY the Final Writer
        streams tokens. No agent boxes or judge messages are printed after the
        streamed answer, so the answer stays as the last AI content on screen.
      - Tool fast paths are returned directly (no streaming from here).
    """
    agent_traces: Dict[str, str] = {}

    # 0) Tool shortcut (simple math / time / etc.)
    used_tool, tool_answer, tool_name = fast_tool_router(user_input)
    if used_tool:
        print_status(f"Tool route: {tool_name} used. Skipping agents.", color=FG_GREEN)

        # No streaming for tool route; just return the tool answer.
        if stream_final and stream_callback is not None:
            stream_callback(tool_answer)

        final_answer = tool_answer
        agent_traces["router"] = f"Tool route '{tool_name}' used; no agents invoked."
        agent_traces["final"] = final_answer
        return final_answer, agent_traces

    env = AgentEnvironment(user_input=user_input, chat_history=chat_history or [])

    # 1) Planner
    print_status("Planner Thinking...", color=FG_CYAN)
    t0 = time.time()
    decision = _planner_agent_env(env)
    planner_text = env.get_agent_output("planner")
    agent_traces["planner"] = planner_text
    if getattr(config.runtime, "debug", False):
        print_box("Planner Output", planner_text, color=FG_CYAN)
        print_box("Env Snapshot After Planner", env.snapshot_for_prompt("debug-planner"), color=FG_MAGENTA)
    print_status(f"Planner Finished. ({time.time() - t0:.2f}s)", color=FG_CYAN)

    # --- math_or_time fast path controlled by Planner ----------------------
    if decision.mode == "math_or_time":
        print_status("Planner selected math_or_time fast path. Using internal handler.", color=FG_GREEN)
        answer = _handle_math_or_time(env.user_input)
        env.set_agent_output("final", answer)

        if stream_final and stream_callback is not None:
            stream_callback(answer)

        agent_traces["final"] = answer
        agent_traces["mode"] = "math_or_time"
        return answer, agent_traces
    # -----------------------------------------------------------------------

    # 2) Optional Researcher
    research_summary = ""
    if decision.use_researcher:
        print_status("Researcher Searching Web & Internal DB...", color=FG_GREEN)
        t1 = time.time()
        research_summary = _researcher_agent_env(env, decision)
        agent_traces["researcher"] = research_summary
        if getattr(config.runtime, "debug", False):
            print_box("Researcher Output", research_summary, color=FG_GREEN)
            print_box("Env Snapshot After Researcher", env.snapshot_for_prompt("debug-researcher"), color=FG_MAGENTA)
        print_status(f"Researcher Summary Ready. ({time.time() - t1:.2f}s)", color=FG_GREEN)

    # 3) Optional Evidence
    evidence_pack = ""
    if decision.use_evidence and research_summary:
        print_status("Evidence Analyst Verifying Claims & Sources...", color=FG_MAGENTA)
        t2 = time.time()
        evidence_pack = _evidence_agent_env(env, research_summary)
        agent_traces["evidence"] = evidence_pack
        if getattr(config.runtime, "debug", False):
            print_box("Evidence Output", evidence_pack, color=FG_MAGENTA)
            print_box("Env Snapshot After Evidence", env.snapshot_for_prompt("debug-evidence"), color=FG_MAGENTA)
        print_status(f"Evidence Pack Ready. ({time.time() - t2:.2f}s)", color=FG_MAGENTA)

    # 4) Reviewer + optional refinement loop
    review_notes = ""
    max_review_cycles = 2
    cycle = 0

    if decision.use_reviewer:
        while True:
            cycle += 1
            content_to_review = evidence_pack or research_summary
            if not content_to_review:
                break

            print_status(f"Reviewer Checking Quality (cycle {cycle})...", color=FG_YELLOW)
            t3 = time.time()
            review_decision = _reviewer_decision_agent(env, content_to_review)
            review_notes = review_decision.review_notes
            agent_traces[f"reviewer_cycle_{cycle}"] = review_notes

            if getattr(config.runtime, "debug", False):
                label = "Reviewer Output" if cycle == 1 else f"Reviewer Output (cycle {cycle})"
                print_box(label, review_notes, color=FG_YELLOW)
                print_box(
                    f"Env Snapshot After Reviewer cycle {cycle}",
                    env.snapshot_for_prompt("debug-reviewer"),
                    color=FG_MAGENTA,
                )

            if (
                not review_decision.needs_revision
                or review_decision.target == "none"
                or cycle >= max_review_cycles
            ):
                print_status(
                    f"Reviewer Satisfied With Quality. ({time.time() - t3:.2f}s)",
                    color=FG_YELLOW,
                )
                break

            # If reviewer asks for more work:
            if review_decision.target == "researcher" and decision.use_researcher:
                print_status("Reviewer Requested Researcher Refinement...", color=FG_GREEN)
                refined_research = _researcher_refine_agent(
                    env=env,
                    plan=decision,
                    previous_summary=research_summary or content_to_review,
                    reviewer_message=review_decision.message_for_target,
                )
                research_summary = refined_research
                agent_traces[f"researcher_refined_{cycle}"] = refined_research
                if getattr(config.runtime, "debug", False):
                    print_box(
                        f"Researcher Refined (cycle {cycle})",
                        refined_research,
                        color=FG_GREEN,
                    )
                if decision.use_evidence:
                    print_status("Evidence Analyst Re-verifying After Research Refine...", color=FG_MAGENTA)
                    refined_evidence = _evidence_agent_env(env, research_summary)
                    evidence_pack = refined_evidence
                    agent_traces[f"evidence_after_research_refine_{cycle}"] = refined_evidence
                    if getattr(config.runtime, "debug", False):
                        print_box(
                            f"Evidence Output After Research Refine (cycle {cycle})",
                            refined_evidence,
                            color=FG_MAGENTA,
                        )

            elif review_decision.target == "evidence" and decision.use_evidence and research_summary:
                print_status("Reviewer Requested Evidence Refinement...", color=FG_MAGENTA)
                refined_evidence = _evidence_refine_agent(
                    env=env,
                    research_summary=research_summary,
                    previous_evidence=evidence_pack or content_to_review,
                    reviewer_message=review_decision.message_for_target,
                )
                evidence_pack = refined_evidence
                agent_traces[f"evidence_refined_{cycle}"] = refined_evidence
                if getattr(config.runtime, "debug", False):
                    print_box(
                        f"Evidence Refined (cycle {cycle})",
                        refined_evidence,
                        color=FG_MAGENTA,
                    )
            else:
                print_status("Reviewer Cannot Refine Further, Continuing...", color=FG_YELLOW)
                break

    # 5) Final answer – Final Writer (STREAMS if requested)
    print_status("Final Writer Composing Answer...", color=FG_CYAN)
    t4 = time.time()
    final_initial = _concluder_agent_env(
        env=env,
        research_summary=research_summary,
        evidence_pack=evidence_pack or research_summary,
        review_notes=review_notes,
        judge_feedback="",
        stream=stream_final,
        stream_callback=stream_callback,
    )
    agent_traces["final_initial"] = final_initial

    # IMPORTANT: do NOT print "Final Answer Draft Ready..." when streaming,
    # otherwise the answer is no longer the last thing on screen.
    if not stream_final:
        print_status(f"Final Answer Draft Ready. ({time.time() - t4:.2f}s)", color=FG_CYAN)

    final_answer = final_initial

    # 6) Judge: completely disabled in streaming mode to avoid a second answer
    runtime_cfg = getattr(config, "runtime", None)
    if runtime_cfg is not None:
        if hasattr(runtime_cfg, "enable_judge"):
            judge_allowed_by_runtime = bool(getattr(runtime_cfg, "enable_judge", True))
        else:
            judge_allowed_by_runtime = bool(getattr(runtime_cfg, "use_judge", True))
    else:
        judge_allowed_by_runtime = True

    if stream_final:
        # Keep the UI clean: no judge rewrite, no extra status lines.
        env.log("[judge] skipped (streaming mode: judge disabled to keep final answer last)")
        agent_traces["judge"] = "Judge skipped: streaming mode (no rewrite)."
    else:
        if decision.use_judge and judge_allowed_by_runtime and (research_summary or evidence_pack):
            print_status("Judge Auditing Final Answer...", color=FG_YELLOW)
            t5 = time.time()
            verdict = _judge_agent_env(
                env=env,
                user_input=user_input,
                final_answer=final_answer,
                research_summary=research_summary,
                evidence_pack=evidence_pack or research_summary,
            )
            agent_traces["judge"] = verdict.critique_for_logs

            if verdict.needs_rewrite and not verdict.accept:
                print_status("Judge Requested Rewrite From Final Writer...", color=FG_YELLOW)
                final_answer_rewritten = _concluder_agent_env(
                    env=env,
                    research_summary=research_summary,
                    evidence_pack=evidence_pack or research_summary,
                    review_notes=review_notes,
                    judge_feedback=verdict.message_for_finalizer,
                    stream=False,        # no streaming on second pass
                    stream_callback=None,
                )
                final_answer = final_answer_rewritten
                agent_traces["final_after_judge"] = final_answer_rewritten
                print_status(
                    f"Final Answer Rewritten After Judge Feedback. ({time.time() - t5:.2f}s)",
                    color=FG_YELLOW,
                )
            else:
                print_status(
                    f"Judge Completed. ({time.time() - t5:.2f}s)",
                    color=FG_YELLOW,
                )
        else:
            skip_reason = []
            if not decision.use_judge:
                skip_reason.append("planner disabled judge")
            if not judge_allowed_by_runtime:
                skip_reason.append("runtime disabled judge")
            if not (research_summary or evidence_pack):
                skip_reason.append("missing upstream research/evidence")
            reason_text = "; ".join(skip_reason) if skip_reason else "Judge skipped by planner/runtime."
            env.log(f"[judge] skipped ({reason_text})")
            agent_traces["judge"] = f"Judge skipped: {reason_text}"

    agent_traces["final"] = final_answer

    # Final environment log dump (debug only)
    if getattr(config.runtime, "debug", False):

        try:
            env_log = env.to_log_string()
            agent_traces["env"] = env_log
        except Exception:
            # Never let logging crash the run
            pass
    return final_answer, agent_traces




# ---------------------------------------------------------------------------
# Debug helper: pretty-print agent traces in terminal box UI
# ---------------------------------------------------------------------------

def pretty_print_agent_traces(agent_traces: Dict[str, str]) -> None:
    """
    Utility for your CLI: given the `agent_traces` dict returned by run_multi_agent,
    print each stage's output in a nice box using the terminal_ui helpers.

    NOTE:
    - We intentionally do NOT print the 'final' box because the final answer is
      already shown separately at the bottom of the terminal.
    """
    if not agent_traces:
        print_status("No agent traces available for this session.", color=FG_YELLOW)
        return

    # Order we care about most when debugging
    ordered_keys = [
        "planner",
        "researcher",
        "researcher_refined_1",
        "evidence",
        "evidence_refined_1",
        "reviewer_cycle_1",
        "reviewer_cycle_2",
        "final_initial",
        "judge",
        "final_after_judge",
        # deliberately omit "final" from ordered debug boxes
    ]

    used = set()

    def color_for(key: str) -> str:
        if key.startswith("planner"):
            return FG_CYAN
        if key.startswith("researcher"):
            return FG_GREEN
        if key.startswith("evidence"):
            return FG_MAGENTA
        if key.startswith("reviewer"):
            return FG_YELLOW
        if key.startswith("judge"):
            return FG_YELLOW
        if key.startswith("final"):
            return FG_CYAN
        return FG_MAGENTA

    # First, print in the preferred order (except 'final')
    for key in ordered_keys:
        if key not in agent_traces:
            continue
        if key == "final":
            continue  # never show FINAL box
        used.add(key)
        print_box(key.upper(), agent_traces[key], color=color_for(key))

    # Then, print anything else we logged (e.g. extra cycles, env logs),
    # still skipping 'final' so it never appears as a box.
    for key, value in agent_traces.items():
        if key in used:
            continue
        if key == "final":
            continue  # hide FINAL from debug view
        print_box(key.upper(), value, color=color_for(key))
