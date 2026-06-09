"""
piai/providers/transform_messages.py — Shared message normalization used by ALL providers.
Mirrors transform-messages.ts from @earendil-works/pi-ai.

Functions here are provider-agnostic and operate only on piai types.
Never put provider-specific logic here.
"""
from __future__ import annotations

import time
from typing import Any

from piai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
)
from piai.utils.hash_utils import normalize_openai_tool_call_id


# ---------------------------------------------------------------------------
# Anthropic: batch consecutive ToolResultMessages into one user message
# ---------------------------------------------------------------------------

def batch_tool_results(messages: list[Message]) -> list[Any]:
    """
    Anthropic requires that all tool results for a given assistant turn be
    sent as a single user message with multiple tool_result content blocks.

    This function converts the flat piai message list into the Anthropic
    wire format, batching consecutive ToolResultMessages.

    Returns a list of Anthropic-format message dicts (not piai types).
    """
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, ToolResultMessage):
            # Collect all consecutive tool result messages
            tool_blocks = []
            while i < len(messages) and isinstance(messages[i], ToolResultMessage):
                tr = messages[i]
                content_parts = []
                for part in tr.content:
                    if isinstance(part, TextContent):
                        content_parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.media_type or "image/jpeg",
                                "data": part.image if isinstance(part.image, str)
                                        else part.image.decode(),
                            },
                        })
                tool_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tr.tool_use_id,
                    "content": content_parts,
                })
                i += 1
            result.append({"role": "user", "content": tool_blocks})
        else:
            i += 1
            result.append(msg)  # pass through non-tool-result messages unchanged
    return result


# ---------------------------------------------------------------------------
# All providers: insert synthetic tool results for orphaned tool calls
# ---------------------------------------------------------------------------

def insert_synthetic_tool_results(messages: list[Message]) -> list[Message]:
    """
    Prevent broken conversations from orphaned tool calls.

    If an AssistantMessage has ToolCallContent items but the following messages
    do not contain matching ToolResultMessages, insert synthetic error results.

    Input:  [user, assistant(toolcall1, toolcall2), tool_result(id1)]
    Output: [user, assistant, tool_result(id1), tool_result(id2, error text)]
    """
    result: list[Message] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        result.append(msg)
        if not isinstance(msg, AssistantMessage):
            i += 1
            continue

        # Collect tool call IDs from this assistant message
        tool_call_ids = [
            part.id
            for part in msg.content
            if isinstance(part, ToolCallContent)
        ]
        if not tool_call_ids:
            i += 1
            continue

        # Collect which IDs are covered by subsequent ToolResultMessages
        j = i + 1
        covered = set()
        while j < len(messages) and isinstance(messages[j], ToolResultMessage):
            covered.add(messages[j].tool_use_id)
            j += 1

        # Insert synthetic results for uncovered tool calls
        for tc_id in tool_call_ids:
            if tc_id not in covered:
                result.append(ToolResultMessage(
                    content=[TextContent(text="[no result — tool call was not completed]")],
                    tool_use_id=tc_id,
                    name="",
                ))
        i += 1
    return result


# ---------------------------------------------------------------------------
# Cross-provider: encode/decode thinking blocks for non-native providers
# ---------------------------------------------------------------------------

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def encode_thinking_block(content: ThinkingContent) -> TextContent:
    """
    Convert a ThinkingContent to a TextContent with <think>…</think> wrapping.
    Used when sending to a provider that doesn't natively support thinking blocks.
    The signature is preserved in the text as a trailing comment if present.
    """
    text = f"{_THINK_OPEN}{content.thinking}{_THINK_CLOSE}"
    if content.signature:
        text += f"\n<!-- thinking:{content.signature} -->"
    return TextContent(text=text)


def decode_thinking_block(text: str) -> ThinkingContent | None:
    """
    Reverse of encode_thinking_block.
    Returns a ThinkingContent if the text is a wrapped thinking block, else None.
    """
    stripped = text.strip()
    if not stripped.startswith(_THINK_OPEN):
        return None
    end = stripped.find(_THINK_CLOSE)
    if end == -1:
        return None
    thinking = stripped[len(_THINK_OPEN):end]
    signature = None
    remainder = stripped[end + len(_THINK_CLOSE):].strip()
    if remainder.startswith("<!-- thinking:") and remainder.endswith(" -->"):
        signature = remainder[len("<!-- thinking:"):-len(" -->")].strip()
    return ThinkingContent(thinking=thinking, signature=signature)


# ---------------------------------------------------------------------------
# OpenAI / Mistral: normalize long tool call IDs
# ---------------------------------------------------------------------------

def normalize_tool_call_id(tc_id: str, max_len: int) -> str:
    """Normalize a tool call ID to at most max_len characters via hash."""
    from piai.utils.hash_utils import normalize_tool_call_id as _normalize
    return _normalize(tc_id, max_len)


# ---------------------------------------------------------------------------
# Google Vertex: schema conversion + tool ID generation
# ---------------------------------------------------------------------------

_VERTEX_TYPE_MAP = {
    "string":  "STRING",
    "number":  "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array":   "ARRAY",
    "object":  "OBJECT",
}


def convert_schema_to_vertex(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively convert a JSON Schema dict to Google Vertex AI format.
    - Uppercases type names (string → STRING)
    - Removes unsupported fields ($schema, $defs, anyOf, const)
    - Delegates enum handling to StringEnum.convert()
    """
    from piai.utils.validation import _convert_for_vertex
    result = _convert_for_vertex(schema)
    return _uppercase_types(result)


def _uppercase_types(schema: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            result[key] = _VERTEX_TYPE_MAP.get(value, value.upper())
        elif isinstance(value, dict):
            result[key] = _uppercase_types(value)
        elif isinstance(value, list):
            result[key] = [_uppercase_types(v) if isinstance(v, dict) else v for v in value]
        else:
            result[key] = value
    return result


_tool_id_counter = 0


def generate_google_tool_id(name: str) -> str:
    """
    Generate a synthetic tool call ID for Google Vertex (which doesn't assign them).
    Format: {name}_{epoch_ms}_{counter}
    """
    global _tool_id_counter
    _tool_id_counter += 1
    ms = int(time.time() * 1000)
    return f"{name}_{ms}_{_tool_id_counter}"
