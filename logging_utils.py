# logging_utils.py
"""
Simple JSONL logging for Omni-Pro interactions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any


LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "omni_pro.log")


def _ensure_log_dir() -> None:
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def log_interaction(
    user_input: str,
    final_answer: str,
    agent_traces: Dict[str, Any],
) -> None:
    """
    Append one interaction to logs/omni_pro.log as JSONL.
    """
    try:
        _ensure_log_dir()
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_input": user_input,
            "final_answer": final_answer,
            "agent_traces": agent_traces,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging should never crash the main app; swallow errors.
        pass
