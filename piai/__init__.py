"""
piai — Python unified LLM abstraction library.
Inspired by @earendil-works/pi-ai (TypeScript).

Public surface:
    stream, complete, stream_simple, complete_simple  — main API
    get_model, list_models, register_model            — model registry
    register_api_provider, reset_api_providers        — provider registry
    KnownApi                                          — provider enum
"""
from piai.stream import stream, complete, stream_simple, complete_simple
from piai.models import get_model, list_models, register_model
from piai.api_registry import KnownApi, register_api_provider, reset_api_providers
from piai.types import (
    # Messages
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    MessagesContext,
    # Content parts
    TextContent,
    ImageContent,
    ThinkingContent,
    ToolCallContent,
    # Tool
    Tool,
    # Options
    StreamOptions,
    SimpleStreamOptions,
    ThinkingOptions,
    ReasoningOptions,
    CacheOptions,
    RetryPolicy,
    # Events
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
    # Misc
    ThinkingLevel,
    TokenUsage,
    AssistantMessageMetadata,
    Diagnostic,
)

__all__ = [
    # Main API
    "stream",
    "complete",
    "stream_simple",
    "complete_simple",
    # Model registry
    "get_model",
    "list_models",
    "register_model",
    # Provider registry
    "KnownApi",
    "register_api_provider",
    "reset_api_providers",
    # Types — messages
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "MessagesContext",
    # Types — content parts
    "TextContent",
    "ImageContent",
    "ThinkingContent",
    "ToolCallContent",
    # Types — tool
    "Tool",
    # Types — options
    "StreamOptions",
    "SimpleStreamOptions",
    "ThinkingOptions",
    "ReasoningOptions",
    "CacheOptions",
    "RetryPolicy",
    # Types — events
    "StartEvent",
    "TextStartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ThinkingStartEvent",
    "ThinkingDeltaEvent",
    "ThinkingEndEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "DoneEvent",
    "ErrorEvent",
    # Types — misc
    "ThinkingLevel",
    "TokenUsage",
    "AssistantMessageMetadata",
    "Diagnostic",
]
