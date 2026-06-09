"""
piai/models_data.py — Model registry data, hand-seeded from models_data.json.
Phase 2: replace with scripts/generate_models.py that reads the JSON and writes this file.

DO NOT edit by hand — edit models_data.json and regenerate.
"""
from __future__ import annotations

from piai.api_registry import KnownApi

# Raw model definitions — loaded by models.py into Model dataclasses
MODELS_RAW: list[dict] = [
    # -----------------------------------------------------------------------
    # Anthropic
    # -----------------------------------------------------------------------
    {
        "id": "claude-opus-4-5",
        "name": "Claude Opus 4.5",
        "api": KnownApi.ANTHROPIC,
        "provider": "anthropic",
        "context_window": 200000,
        "max_tokens": 32000,
        "reasoning": True,
        "cost": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75},
        "thinking_level_map": {
            "minimal": {"budget_tokens": 1024},
            "low":     {"budget_tokens": 2048},
            "medium":  {"budget_tokens": 8192},
            "high":    {"budget_tokens": 16384},
            "xhigh":   {"budget_tokens": 16384},
        },
        "compat": {"batch_tool_results": True},
    },
    {
        "id": "claude-sonnet-4-5",
        "name": "Claude Sonnet 4.5",
        "api": KnownApi.ANTHROPIC,
        "provider": "anthropic",
        "context_window": 200000,
        "max_tokens": 8192,
        "reasoning": True,
        "cost": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
        "thinking_level_map": {
            "minimal": {"budget_tokens": 1024},
            "low":     {"budget_tokens": 2048},
            "medium":  {"budget_tokens": 8192},
            "high":    {"budget_tokens": 16384},
            "xhigh":   {"budget_tokens": 16384},
        },
        "compat": {"batch_tool_results": True},
    },
    {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "api": KnownApi.ANTHROPIC,
        "provider": "anthropic",
        "context_window": 200000,
        "max_tokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.8, "output": 4.0, "cache_read": 0.08, "cache_write": 1.0},
        "compat": {"batch_tool_results": True},
    },
    # -----------------------------------------------------------------------
    # OpenAI
    # -----------------------------------------------------------------------
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "api": KnownApi.OPENAI_COMPLETIONS,
        "provider": "openai",
        "context_window": 128000,
        "max_tokens": 16384,
        "reasoning": False,
        "cost": {"input": 2.5, "output": 10.0, "cache_read": 1.25},
    },
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "api": KnownApi.OPENAI_COMPLETIONS,
        "provider": "openai",
        "context_window": 128000,
        "max_tokens": 16384,
        "reasoning": False,
        "cost": {"input": 0.15, "output": 0.6, "cache_read": 0.075},
    },
    {
        "id": "o4-mini",
        "name": "OpenAI o4-mini",
        "api": KnownApi.OPENAI_COMPLETIONS,
        "provider": "openai",
        "context_window": 200000,
        "max_tokens": 100000,
        "reasoning": True,
        "thinking_level_map": {
            "minimal": {"reasoning_effort": "low"},
            "low":     {"reasoning_effort": "low"},
            "medium":  {"reasoning_effort": "medium"},
            "high":    {"reasoning_effort": "high"},
            "xhigh":   {"reasoning_effort": "high"},
        },
    },
    {
        "id": "o3",
        "name": "OpenAI o3",
        "api": KnownApi.OPENAI_COMPLETIONS,
        "provider": "openai",
        "context_window": 200000,
        "max_tokens": 100000,
        "reasoning": True,
        "thinking_level_map": {
            "minimal": {"reasoning_effort": "low"},
            "low":     {"reasoning_effort": "low"},
            "medium":  {"reasoning_effort": "medium"},
            "high":    {"reasoning_effort": "high"},
            "xhigh":   {"reasoning_effort": "high"},
        },
    },
    # -----------------------------------------------------------------------
    # Google Vertex AI
    # -----------------------------------------------------------------------
    {
        "id": "gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "api": KnownApi.GOOGLE_VERTEX,
        "provider": "google-vertex",
        "context_window": 1048576,
        "max_tokens": 65536,
        "reasoning": True,
        "cost": {"input": 1.25, "output": 10.0},
        "thinking_level_map": {
            "minimal": {"thinking_budget": 1024},
            "low":     {"thinking_budget": 2048},
            "medium":  {"thinking_budget": 8192},
            "high":    {"thinking_budget": 16384},
            "xhigh":   {"thinking_budget": 32768},
        },
    },
    {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "api": KnownApi.GOOGLE_VERTEX,
        "provider": "google-vertex",
        "context_window": 1048576,
        "max_tokens": 65536,
        "reasoning": True,
        "cost": {"input": 0.15, "output": 0.6},
        "thinking_level_map": {
            "minimal": {"thinking_budget": 1024},
            "low":     {"thinking_budget": 2048},
            "medium":  {"thinking_budget": 8192},
            "high":    {"thinking_budget": 16384},
            "xhigh":   {"thinking_budget": 32768},
        },
    },
    {
        "id": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash",
        "api": KnownApi.GOOGLE_VERTEX,
        "provider": "google-vertex",
        "context_window": 1048576,
        "max_tokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.1, "output": 0.4},
    },
    # -----------------------------------------------------------------------
    # Faux (mock)
    # -----------------------------------------------------------------------
    {
        "id": "faux",
        "name": "Faux (Mock) Provider",
        "api": KnownApi.FAUX,
        "provider": "faux",
        "context_window": 100000,
    },
]
