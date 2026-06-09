"""
piai/utils/hash_utils.py — short_hash + per-provider tool call ID normalization.

Per-provider normalization rules (applied in transform_messages.py):
  OpenAI:    max_len=64  (IDs can be 450+ chars)
  Anthropic: pass-through (max 64 chars native)
  Google:    generated, not normalized
  Mistral:   max_len=9
  Groq:      pass-through
"""
from __future__ import annotations

import hashlib


def short_hash(s: str, length: int = 64) -> str:
    """
    Deterministic hash of `s` truncated to `length` hex chars.
    Uses SHAKE-256 (variable-length output) so the full `length` is used,
    not a truncated SHA-256.

    Mirrors TypeScript shortHash() which uses MurmurHash3 → base-36.
    Python equivalent: SHAKE-256 hex, truncated to requested length.
    """
    digest = hashlib.shake_256(s.encode()).hexdigest(length // 2 + 1)
    return digest[:length]


def normalize_tool_call_id(tc_id: str, max_len: int) -> str:
    """
    Normalize a tool call ID to at most max_len characters.
    If already within limit, returns unchanged.
    If longer, hashes deterministically to exactly max_len chars.
    """
    if len(tc_id) <= max_len:
        return tc_id
    return short_hash(tc_id, max_len)


def normalize_openai_tool_call_id(tc_id: str) -> str:
    """OpenAI can return 450+ char IDs — normalize to 64."""
    return normalize_tool_call_id(tc_id, 64)


def normalize_mistral_tool_call_id(tc_id: str) -> str:
    """Mistral uses 9-char IDs — truncate incoming long IDs."""
    return tc_id[:9] if len(tc_id) > 9 else tc_id
