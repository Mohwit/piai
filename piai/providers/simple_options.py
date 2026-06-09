"""
piai/providers/simple_options.py — build_base_options(): SimpleStreamOptions → StreamOptions.
Direct port of simple-options.ts from the original.
"""
from __future__ import annotations

from piai.types import (
    SimpleStreamOptions,
    StreamOptions,
    ThinkingOptions,
    ReasoningOptions,
)


def build_base_options(simple: SimpleStreamOptions) -> StreamOptions:
    """
    Convert SimpleStreamOptions to full StreamOptions.

    The unified 'reasoning' ThinkingLevel is mapped to both ThinkingOptions
    and ReasoningOptions so each provider can pick the one it understands:
      - Anthropic / Google use thinking (ThinkingOptions)
      - OpenAI o* models use reasoning (ReasoningOptions)
    """
    thinking = None
    reasoning = None

    if simple.reasoning and simple.reasoning != "off":
        thinking = ThinkingOptions(level=simple.reasoning)
        reasoning = ReasoningOptions(effort=simple.reasoning)

    return StreamOptions(
        temperature=simple.temperature,
        max_tokens=simple.max_tokens,
        thinking=thinking,
        reasoning=reasoning,
        tools=simple.tools,
        tool_choice=simple.tool_choice,
        cache=simple.cache,
        abort_signal=simple.abort_signal,
        api_key=simple.api_key,
    )
