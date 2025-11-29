# main_cli.py
from __future__ import annotations

import sys
from typing import Any, Dict

from dotenv import load_dotenv

from orchestrator import omni_pro_turn
from session_memory import get_session_memory
from terminal_ui import (
    print_banner,
    print_box,
    FG_WHITE,
)
from agents import pretty_print_agent_traces


def main() -> None:
    # Ensure UTF-8 console
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_dotenv()

    print("⚡ OMNI-PRO-V1: CUSTOM MULTI-AGENT SYSTEM ONLINE")
    print("---------------------------------------")

    memory = get_session_memory()
    last_agent_traces: Dict[str, Any] = {}

    try:
        while True:
            print("\nType your request below (or 'exit' to quit).")
            print("Special commands: !context (show memory), !debug last (show last agent traces)")
            raw_input_text = input("User > ").strip()

            if not raw_input_text:
                continue

            lowered = raw_input_text.lower()
            if lowered in {"exit", "quit", "q"}:
                print("Exiting...")
                break

            # Show current session context
            if lowered in {"!context", "/context"}:
                ctx_block = memory.build_context_block()
                print_banner("SESSION CONTEXT", None, color=FG_WHITE)
                print_box("Memory", ctx_block or "(empty)", color=FG_WHITE)
                continue

            # Show last agent traces
            if lowered in {"!debug last", "/debug"}:
                print_banner("LAST AGENT TRACES", None, color=FG_WHITE)
                if last_agent_traces:
                    pretty_print_agent_traces(last_agent_traces)
                else:
                    print_box("Agent Traces", "(no traces yet)", color=FG_WHITE)
                continue

            # Normal turn goes through the orchestrator
            final_answer, agent_traces = omni_pro_turn(raw_input_text)

            last_agent_traces = agent_traces

            # Final response box
            print_banner("FINAL RESPONSE", None, color=FG_WHITE)
            print_box("Omni-Pro", final_answer, color=FG_WHITE)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
