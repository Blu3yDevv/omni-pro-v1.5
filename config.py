# config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """
    Settings for all LLM calls (Groq in your current setup).
    You can change model names here to use different open-source models on Groq.
    """
    # Model names – make sure these exist on Groq
    fast_model: str = "llama-3.1-8b-instant"

    # Phase 1: use a fast model for planner/reviewer;
    # keep strong models for research/evidence/final.
    planner_model: str = "llama-3.1-8b-instant"
    research_model: str = "qwen/qwen3-32b"
    evidence_model: str = "qwen/qwen3-32b"
    reviewer_model: str = "llama-3.1-8b-instant"
    final_model: str = "qwen/qwen3-32b"

    # Temperatures – you can tweak these later
    temperature_fast: float = 0.2
    temperature_planner: float = 0.1
    temperature_research: float = 0.1
    temperature_evidence: float = 0.1
    temperature_reviewer: float = 0.1
    temperature_final: float = 0.2

    # Optional: fallback API key config if env var is not set
    groq_api_key: str | None = None


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    """
    Runtime behaviour toggles for Omni-Pro.
    """
    mode: Literal["balanced", "turbo", "deep"] = "balanced"
    use_judge: bool = True
    debug: bool = True

# ---------------------------------------------------------------------------
# App-wide config container
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """
    Top-level config object.

    NOTE: We MUST use default_factory for nested dataclasses to avoid
    the 'mutable default' error from dataclasses.
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


# Single global config instance used everywhere
config = AppConfig()
