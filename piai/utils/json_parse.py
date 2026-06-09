"""
piai/utils/json_parse.py — Streaming and repair JSON parsing.
Mirrors parseStreamingJson + parseJsonWithRepair from the original.
"""
from __future__ import annotations

import json
import re


def parse_streaming_json(partial: str) -> dict:
    """
    Parse a partial (potentially incomplete) JSON string into a dict.
    Used for streaming tool arguments that arrive chunk by chunk.

    Falls back gracefully: tries the partial-json library first,
    then standard json, then returns an empty dict.
    """
    if not partial or not partial.strip():
        return {}

    try:
        import partial_json  # type: ignore[import]
        result = partial_json.loads(partial)
        if isinstance(result, dict):
            return result
        return {}
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: try to complete the JSON by closing unclosed brackets
    try:
        return json.loads(partial)
    except json.JSONDecodeError:
        pass

    # Last resort: close any unclosed structures and retry
    completed = _close_json(partial)
    try:
        result = json.loads(completed)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return {}


def parse_json_with_repair(text: str) -> dict:
    """
    Parse JSON that may contain control characters or other malformations.
    Mirrors parseJsonWithRepair from the original — handles escape sequences
    and unpaired surrogates from some providers.
    """
    if not text or not text.strip():
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix common issues: unescaped control characters
    repaired = _repair_json(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return {}


def _close_json(partial: str) -> str:
    """Attempt to close an incomplete JSON string by counting open brackets."""
    stack: list[str] = []
    in_string = False
    escape_next = False

    for char in partial:
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()

    # Close open string if needed
    if in_string:
        partial += '"'

    # Close open brackets in reverse order
    return partial + "".join(reversed(stack))


def _repair_json(text: str) -> str:
    """Escape unescaped control characters in JSON strings."""
    # Replace literal control characters with their escape sequences
    control_chars = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\b": "\\b",
        "\f": "\\f",
    }
    result = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == "\\" and in_string:
            result.append(char)
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        if in_string and char in control_chars:
            result.append(control_chars[char])
            continue
        result.append(char)

    return "".join(result)
