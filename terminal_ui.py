# terminal_ui.py

from __future__ import annotations

import textwrap

# Basic ANSI colors
RESET = "\033[0m"
FG_WHITE = "\033[97m"
FG_CYAN = "\033[96m"
FG_GREEN = "\033[92m"
FG_YELLOW = "\033[93m"
FG_MAGENTA = "\033[95m"

# Extra styling for "thinking" sections
FG_GREY = "\033[90m"
ITALIC = "\033[3m"


def _colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _is_think_line(text: str) -> bool:
    """
    Detect lines that contain the internal thinking markers.
    """
    return "<think>" in text or "</think>" in text


def print_banner(title: str, subtitle: str | None = None, color: str = FG_WHITE, width: int = 118) -> None:
    """
    Big top-level banner, like your FINAL RESPONSE header.
    """
    if width < 20:
        width = 20

    top = "╔" + "═" * (width - 2) + "╗"
    bottom = "╚" + "═" * (width - 2) + "╝"

    print(_colorize(top, color))
    title_line = title.center(width - 2)
    print(_colorize(f"║{title_line}║", color))

    if subtitle:
        subtitle_line = subtitle.center(width - 2)
        print(_colorize(f"║{subtitle_line}║", color))

    print(_colorize(bottom, color))


def print_box(title: str, content: str, color: str = FG_WHITE, width: int = 118) -> None:
    """
    Content box with a title bar and wrapped text, like your Planner / Researcher outputs.

    Any text between <think> and </think> (including multi-line) is:
      - rendered in grey,
      - italic,
      - and the tags themselves are stripped.
    """
    if width < 30:
        width = 30

    # Frame lines
    top = "┌" + "─" * (width - 2) + "┐"
    title_line = f" {title} ".center(width - 2, " ")
    title_row = f"│{title_line}│"
    sep = "├" + "─" * (width - 2) + "┤"
    bottom = "└" + "─" * (width - 2) + "┘"

    print(_colorize(top, color))
    print(_colorize(title_row, color))
    print(_colorize(sep, color))

    # Build body_lines as a list of (text, is_think_section)
    if not content:
        body_lines: list[tuple[str, bool]] = [("", False)]
    else:
        body_lines: list[tuple[str, bool]] = []
        in_think = False

        for raw_line in content.splitlines():
            # Preserve blank lines, but keep current think state
            if not raw_line.strip():
                body_lines.append(("", in_think))
                continue

            line = raw_line

            # Detect opening/closing tags on this raw line
            opening = "<think>" in line
            closing = "</think>" in line

            # Strip tags from the text that will be displayed
            line = line.replace("<think>", "").replace("</think>", "")

            # This line should be styled as "think" if:
            # - we were already in a think block, or
            # - it opens a think block on this line.
            line_is_think = in_think or opening

            # Update global think state for subsequent lines
            if opening:
                in_think = True
            if closing:
                in_think = False

            # Wrap the stripped text
            wrapped = textwrap.wrap(
                line,
                width=width - 4,
                break_long_words=True,
                replace_whitespace=False,
            )

            if not wrapped:
                body_lines.append(("", line_is_think))
            else:
                for w in wrapped:
                    body_lines.append((w, line_is_think))

    # Print the body
    for text, is_think in body_lines:
        padded = text.ljust(width - 4)

        if is_think:
            # Grey + italic for inner text, keep borders in main color.
            left_border = _colorize("│ ", color)
            right_border = _colorize(" │", color)
            inner_text = f"{FG_GREY}{ITALIC}{padded}{RESET}"
            print(left_border + inner_text + right_border)
        else:
            print(_colorize(f"│ {padded} │", color))

    print(_colorize(bottom, color))


def print_status(message: str, color: str = FG_WHITE) -> None:
    """
    One-line agent status indicator, e.g.:
      [Planner] thinking...
      [Researcher] searching web & RAG...

    If the message itself contains <think> or </think>, it is also styled
    grey and italic (tags are removed in the displayed text).
    """
    if _is_think_line(message):
        clean = message.replace("<think>", "").replace("</think>", "")
        styled = f"{FG_GREY}{ITALIC}[{clean}]{RESET}"
        print(styled)
    else:
        print(_colorize(f"[{message}]", color))
