# search_client.py
from __future__ import annotations

import re
from typing import Dict, List, Optional

# Prefer new library name; fall back to old one if needed.
try:
    from ddgs import DDGS  # pip install ddgs
except ImportError:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS  # older package name
    except ImportError:
        DDGS = None  # Hard fallback – search will just return []


# Some obviously irrelevant / non-English-heavy domains we want to avoid
BLOCKED_DOMAINS = {
    "zhidao.baidu.com",
    "baidu.com",
    "zhihu.com",
}


def _looks_non_english(text: str) -> bool:
    """Skip results whose title/snippet are mostly CJK characters."""
    if not text:
        return False
    total = len(text)
    if total == 0:
        return False
    cjk = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff":
            cjk += 1
    return (cjk / total) > 0.30


def deep_web_search(query: str, max_results: int = 6) -> List[Dict[str, str]]:
    """
    Run an English-biased meta search using DDGS.

    Returns a list of dicts:
      {"title": ..., "url": ..., "snippet": ..., "domain": ...}
    """
    if DDGS is None:
        return []

    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            gen = ddgs.text(
                query,
                region="wt-wt",        # world-wide, neutral
                safesearch="moderate",
                timelimit="y",         # last year – decent default
                max_results=max_results,
            )
            for r in gen or []:
                url = (r.get("href") or r.get("url") or "").strip()
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()

                # Extract simple domain
                domain = ""
                m = re.search(r"https?://([^/]+)/?", url)
                if m:
                    domain = m.group(1).lower()

                # Filter obvious junk
                if any(b in domain for b in BLOCKED_DOMAINS):
                    continue
                snippet_source = (body or title)
                if _looks_non_english(snippet_source):
                    continue

                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": body,
                        "domain": domain,
                    }
                )
                if len(results) >= max_results:
                    break
    except Exception:
        return []

    return results


def format_search_results_for_prompt(results: List[Dict[str, str]]) -> str:
    """
    Turn raw search results into a compact, LLM-friendly text block.

    This is what gets fed into the Researcher / Evidence agents.
    """
    if not results:
        return (
            "Top web results (none):\n"
            "No high-quality web results were found for this query."
        )

    lines: List[str] = ["Top web results (most relevant first):"]
    for i, r in enumerate(results, 1):
        title = r.get("title") or "(no title)"
        domain = r.get("domain") or ""
        url = r.get("url") or ""
        snippet = (r.get("snippet") or "").replace("\n", " ").strip()
        if len(snippet) > 260:
            snippet = snippet[:257] + "..."

        lines.append(
            f"[{i}] {title} ({domain})\n"
            f"URL: {url}\n"
            f"Snippet: {snippet}"
        )

    return "\n\n".join(lines)
