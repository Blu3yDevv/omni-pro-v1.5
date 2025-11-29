# time_tools.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict


# Minimal mapping for now. You can extend this if you want.
_LOCATION_OFFSETS: Dict[str, int] = {
    "seychelles": 4,
    "mahe": 4,
    "mahé": 4,
    "victoria": 4,
}


def get_time_for_human_query(query: str) -> str:
    """
    Very lightweight time helper.

    - Uses system UTC time.
    - Applies fixed offsets for a small set of known locations (Seychelles etc.).
    - Does NOT do DST or arbitrary timezone resolution.

    Called from agents._handle_math_or_time().
    """
    query_l = query.lower()

    now_utc = datetime.now(timezone.utc)

    for name, offset in _LOCATION_OFFSETS.items():
        if name in query_l:
            local = now_utc + timedelta(hours=offset)
            prefix = {
                "seychelles": "Seychelles",
                "mahe": "Mahé (Seychelles)",
                "mahé": "Mahé (Seychelles)",
                "victoria": "Victoria (Seychelles)",
            }.get(name, name.title())

            return (
                f"Current time in {prefix} (UTC+{offset:02d}:00) is "
                f"{local.strftime('%Y-%m-%d %H:%M:%S')}."
            )

    return (
        "I only have built-in support for a few locations (like Seychelles) right now. "
        f"Current UTC time is {now_utc.strftime('%Y-%m-%d %H:%M:%S')}."
    )
