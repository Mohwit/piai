"""
piai/types.py — All core dataclasses: messages, events, options, tool, diagnostics.
Mirrors types.ts from @earendil-works/pi-ai as closely as Python allows.
"""
from __future__ import annotations

import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Thinking levels
# ---------------------------------------------------------------------------

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

THINKING_BUDGETS: dict[str, int] = {
    "off":     0,
    "minimal": 1024,
    "low":     2048,
    "medium":  8192,
    "high":    16384,
    "xhigh":   16384,
}

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticError:
    message: str
    name: Optional[str] = None
    stack: Optional[str] = None
    code: Optional[str] = None


@dataclass
class Diagnostic:
    type: str
    timestamp: float = field(default_factory=time.time)
    error: Optional[DiagnosticError] = None
    details: Optional[dict[str, Any]] = None

# ---------------------------------------------------------------------------
# Message content parts
# ---------------------------------------------------------------------------

@dataclass
class TextContent:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageContent:
    type: Literal["image"] = "image"
    image: Union[bytes, str] = b""
    media_type: Optional[str] = None


@dataclass
class ThinkingContent:
    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    signature: Optional[str] = None  # preserved for cross-provider round-trips


@dataclass
class ToolCallContent:
    type: Literal["toolcall"] = "toolcall"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclass
class UserMessage:
    role: Literal["user"] = "user"
    content: list[Union[TextContent, ImageContent]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thinking_tokens: int = 0


@dataclass
class AssistantMessageMetadata:
    usage: Optional[TokenUsage] = None
    stop_reason: Optional[str] = None
    cached: Optional[bool] = None
    cost: Optional[float] = None  # calculated: model.cost + usage


@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[Union[TextContent, ThinkingContent, ToolCallContent]] = field(default_factory=list)
    metadata: Optional[AssistantMessageMetadata] = None
    diagnostics: list[Diagnostic] = field(default_factory=list)


@dataclass
class ToolResultMessage:
    role: Literal["tool"] = "tool"
    content: list[Union[TextContent, ImageContent]] = field(default_factory=list)
    tool_use_id: str = ""
    name: str = ""


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


@dataclass
class MessagesContext:
    messages: list[Message]
    system_prompt: Optional[str] = None

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    input: dict[str, Any]  # JSON Schema — field matches original (NOT input_schema)
                           # input_schema is Anthropic-specific terminology
                           # Pydantic: pass model.model_json_schema() here

# ---------------------------------------------------------------------------
# Thinking / Reasoning options
# ---------------------------------------------------------------------------

@dataclass
class ThinkingOptions:
    level: ThinkingLevel = "off"
    budget: Optional[int] = None  # overrides THINKING_BUDGETS[level] if set


@dataclass
class ReasoningOptions:
    effort: ThinkingLevel = "off"  # for OpenAI o* models
    budget: Optional[int] = None

# ---------------------------------------------------------------------------
# Stream options
# ---------------------------------------------------------------------------

@dataclass
class CacheOptions:
    retention: Optional[Literal["short", "long"]] = None
    session_id: Optional[str] = None


@dataclass
class RetryPolicy:
    max_retries: int = 3
    delay_ms: int = 1000
    backoff_multiplier: float = 2.0
    retryable_status: list[int] = field(default_factory=lambda: [429, 500, 502, 503])


@dataclass
class StreamOptions:
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    thinking: Optional[ThinkingOptions] = None      # Anthropic / Google
    reasoning: Optional[ReasoningOptions] = None    # OpenAI o* models
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    stop_sequences: Optional[list[str]] = None
    headers: Optional[dict[str, str]] = None        # pass-through to provider HTTP layer
    cache: Optional[CacheOptions] = None            # defined now, used in future phases
    abort_signal: Optional[asyncio.Event] = None    # matches original abortSignal
    retry_policy: Optional[RetryPolicy] = None
    api_key: Optional[str] = None


@dataclass
class SimpleStreamOptions:
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning: Optional[ThinkingLevel] = None       # single unified field across all providers
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    cache: Optional[CacheOptions] = None
    abort_signal: Optional[asyncio.Event] = None
    api_key: Optional[str] = None

# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------

@dataclass
class StartEvent:
    type: Literal["start"] = "start"


@dataclass
class TextStartEvent:
    type: Literal["text_start"] = "text_start"


@dataclass
class TextDeltaEvent:
    type: Literal["text_delta"] = "text_delta"
    text: str = ""


@dataclass
class TextEndEvent:
    type: Literal["text_end"] = "text_end"


@dataclass
class ThinkingStartEvent:
    type: Literal["thinking_start"] = "thinking_start"


@dataclass
class ThinkingDeltaEvent:
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str = ""


@dataclass
class ThinkingEndEvent:
    type: Literal["thinking_end"] = "thinking_end"


@dataclass
class ToolCallStartEvent:
    type: Literal["toolcall_start"] = "toolcall_start"
    id: str = ""
    name: str = ""


@dataclass
class ToolCallDeltaEvent:
    type: Literal["toolcall_delta"] = "toolcall_delta"
    input: str = ""  # partial JSON string


@dataclass
class ToolCallEndEvent:
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass
class DoneEvent:
    type: Literal["done"] = "done"
    message: Optional[AssistantMessage] = None


@dataclass
class ErrorEvent:
    type: Literal["error"] = "error"
    error: Optional[Exception] = None


AssistantMessageEvent = Union[
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
]
