"""
piai/utils/overflow.py — Context window overflow detection.

Three detection mechanisms (mirrors isContextOverflow from the original):
  1. Error message regex match against 20+ provider-specific patterns
  2. Silent overflow: usage.input_tokens >= context_window
  3. Length truncation: output_tokens == 0 with non-empty input
"""
from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from piai.types import TokenUsage

# Provider-specific overflow error message patterns
OVERFLOW_PATTERNS: list[re.Pattern] = [
    # Anthropic
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"exceeds.*context.*window", re.IGNORECASE),
    re.compile(r"context_length_exceeded", re.IGNORECASE),
    # OpenAI
    re.compile(r"maximum context length", re.IGNORECASE),
    re.compile(r"context_length_exceeded", re.IGNORECASE),
    re.compile(r"This model's maximum context length is", re.IGNORECASE),
    re.compile(r"reduce the length of your messages", re.IGNORECASE),
    # Google / Vertex
    re.compile(r"Request payload size exceeds the limit", re.IGNORECASE),
    re.compile(r"exceeds the maximum allowed", re.IGNORECASE),
    re.compile(r"prompt_token_count.*exceeds", re.IGNORECASE),
    # Mistral
    re.compile(r"Tokens limit exceeded", re.IGNORECASE),
    re.compile(r"prompt_tokens.*exceeds.*max", re.IGNORECASE),
    # Bedrock
    re.compile(r"Input is too long", re.IGNORECASE),
    re.compile(r"ValidationException.*too many tokens", re.IGNORECASE),
    # Groq
    re.compile(r"Please reduce the length", re.IGNORECASE),
    # Generic
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"token limit", re.IGNORECASE),
    re.compile(r"context overflow", re.IGNORECASE),
    re.compile(r"input too long", re.IGNORECASE),
    re.compile(r"context window full", re.IGNORECASE),
]


def is_context_overflow(
    error: Optional[Exception] = None,
    usage: Optional["TokenUsage"] = None,
    context_window: Optional[int] = None,
) -> bool:
    """
    Detect context length exceeded via three mechanisms:

    1. Error message regex match (provider-specific error strings)
    2. Silent overflow: usage.input_tokens >= context_window (no error thrown)
    3. Length truncation: output_tokens == 0 with non-empty input
    """
    # Mechanism 1: error message pattern matching
    if error is not None:
        msg = str(error)
        for pattern in OVERFLOW_PATTERNS:
            if pattern.search(msg):
                return True

    if usage is None:
        return False

    # Mechanism 2: silent overflow — input exceeds context window
    if context_window is not None and usage.input_tokens >= context_window:
        return True

    # Mechanism 3: length truncation — server consumed all tokens, no output
    if usage.input_tokens > 0 and usage.output_tokens == 0:
        return True

    return False
