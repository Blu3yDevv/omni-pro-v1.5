from __future__ import annotations

print("1. Python started...")

import sys
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Force UTF-8 output to avoid weird console issues
sys.stdout.reconfigure(encoding="utf-8")

print("2. Importing System Libraries...")
load_dotenv()

print("3. Importing AI & Agent Logic...")
try:
    from agents import run_multi_agent, pretty_print_agent_traces
    from guardrails import (
        preprocess_user_input,
        postprocess_model_output,
        is_disallowed,
        safe_refusal_message,
    )
    from logging_utils import log_interaction
    from terminal_ui import (
        print_banner,
        print_box,
        FG_WHITE,
        FG_GREY,
        ITALIC,
        RESET,
    )
    from session_memory import get_session_memory

    print("4. Libraries Imported Successfully!")
except Exception as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)


def main() -> None:
    print("5. Entering Main Loop...")
    print("---------------------------------------")
    print("OMNI-PRO-V1")
    print("---------------------------------------")

    memory = get_session_memory()
    last_agent_traces: Dict[str, Any] = {}

    try:
        while True:
            print("\nType your request below (or 'exit' to quit).")
            print("Special commands: !context (show memory), !debug last (show last agent traces)")
            raw_input_text = input("User > ").strip()

            if not raw_input_text:
                # Ignore empty lines
                continue

            lowered = raw_input_text.lower()
            if lowered in {"exit", "quit", "q"}:
                print("Exiting...")
                break

            # Debug: show memory context
            if lowered in {"!context", "/context"}:
                ctx_block = memory.build_context_block()
                print_banner("SESSION CONTEXT", None, color=FG_WHITE)
                print_box("Memory", ctx_block or "(empty)", color=FG_WHITE)
                continue

            # Debug: show last agent traces in nice terminal UI
            if lowered in {"!debug last", "/debug"}:
                print_banner("AGENT TRACES (DEBUG)", None, color=FG_WHITE)
                if last_agent_traces:
                    pretty_print_agent_traces(last_agent_traces)
                else:
                    print_box("Agent Traces", "(no traces yet)", color=FG_WHITE)
                continue

            # Optional: add the user turn to memory BEFORE routing,
            # so agents see this in their session context.
            memory.add_turn("user", raw_input_text)

            # Global safety check: short-circuit disallowed content
            if is_disallowed(raw_input_text):
                answer = safe_refusal_message()
                agent_traces: Dict[str, Any] = {}
            else:
                user_input = preprocess_user_input(raw_input_text)

                # Build chat_history from session memory for context-aware reasoning
                chat_history = memory.to_chat_history()

                # --- Streaming setup for final answer ------------------------
                streamed_text_parts: list[str] = []

                in_think = False
                tag_open = "<think>"
                tag_close = "</think>"
                pending = ""

                def on_stream_chunk(chunk: str) -> None:
                    nonlocal in_think, pending

                    if not chunk:
                        return

                    # Accumulate raw for possible logging/debug
                    streamed_text_parts.append(chunk)

                    text = pending + chunk
                    pending = ""
                    i = 0
                    L = len(text)

                    while i < L:
                        ch = text[i]

                        # Possible tag start
                        if ch == "<":
                            remaining = text[i:]

                            # Full tags
                            if remaining.startswith(tag_open):
                                in_think = True
                                i += len(tag_open)
                                continue
                            if remaining.startswith(tag_close):
                                in_think = False
                                i += len(tag_close)
                                continue

                            # Partial tag at the end of the chunk?
                            if tag_open.startswith(remaining) or tag_close.startswith(remaining):
                                pending = remaining
                                break
                            # else: fall through and print "<" as a normal character

                        # Normal character, print with current style
                        if in_think:
                            sys.stdout.write(f"{FG_GREY}{ITALIC}{ch}{RESET}")
                        else:
                            sys.stdout.write(ch)

                        i += 1

                    sys.stdout.flush()

                # Show header BEFORE streaming starts
                print_banner("FINAL RESPONSE (Streaming)", None, color=FG_WHITE)
                print("Omni-Pro > ", end="", flush=True)

                # Multi-agent pipeline (with streaming final writer)
                answer, agent_traces = run_multi_agent(
                    user_input,
                    chat_history=chat_history,
                    stream_final=True,
                    stream_callback=on_stream_chunk,
                )

                # Ensure newline after streaming output
                print()

            # Postprocess output (guardrails)
            answer = postprocess_model_output(answer)

            # Store assistant turn in memory
            memory.add_turn("assistant", answer)

            # Cache traces for debug commands
            last_agent_traces = agent_traces

            # NOTE: we do NOT print the answer again here.
            # The only user-facing answer is the streamed text above.
            # Debug boxes are only shown via !debug last.

            # Log after each turn
            try:
                log_interaction(
                    user_input=raw_input_text,
                    final_answer=answer,
                    agent_traces=agent_traces,
                )
            except Exception:
                # Logging should never crash the main loop.
                pass

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
