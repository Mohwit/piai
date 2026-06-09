# Python Port of Pi-AI — Design & Implementation Plan

> Based on: research/pi-ai-architecture.md
> Goal: Unified LLM provider library in Python, feature-parity with pi-ai TypeScript

---

## 1. TypeScript → Python Mapping

| TypeScript Concept | Python Equivalent |
|-------------------|-------------------|
| `AsyncIterable<Event>` | `AsyncGenerator[Event, None]` |
| `EventStream<TEvent, TResult>` | `asyncio.Queue` + `AsyncGenerator` wrapper |
| TypeBox schemas | `pydantic.BaseModel` or plain `dict` (JSON Schema) |
| `AbortSignal` | `asyncio.Event` or `anyio.CancelScope` |
| `Promise<T>` | `Coroutine[T]` / `asyncio.Task` |
| `Map<K, V>` | `dict[K, V]` |
| `WeakMap` | `weakref.WeakKeyDictionary` |
| TypeScript generics | Python generics with `TypeVar` / `Generic` |
| `KnownApi \| (string & {})` | `Literal[...] \| str` |
| Lazy loading via dynamic import | `importlib.import_module` on first access |
| Discriminated unions | `Union[...] + Literal` or Python 3.10+ `match` |

---

## 2. Proposed Python Package Structure

```
pi_ai/
├── __init__.py
├── types.py                # Core types (Message, Model, Tool, Event types)
├── models.py               # Model registry + cost calculation
├── models_generated.py     # Auto-generated model definitions
├── stream.py               # Public API: stream(), complete(), stream_simple()
├── images.py               # Image generation API
├── api_registry.py         # Provider registration system
├── images_api_registry.py
├── providers/
│   ├── __init__.py
│   ├── anthropic.py
│   ├── openai_responses.py
│   ├── openai_completions.py
│   ├── google.py
│   ├── google_vertex.py
│   ├── mistral.py
│   ├── amazon_bedrock.py
│   ├── azure_openai.py
│   ├── cloudflare.py
│   ├── faux.py              # Mock provider for testing
│   ├── register_builtins.py
│   ├── simple_options.py
│   ├── transform_messages.py
│   └── images/
│       ├── __init__.py
│       └── openai_images.py
├── utils/
│   ├── __init__.py
│   ├── event_stream.py      # AsyncGenerator-based event stream
│   ├── validation.py        # Tool argument validation
│   ├── json_parse.py        # Streaming JSON parsing
│   ├── diagnostics.py       # Error diagnostics
│   ├── overflow.py          # Context overflow detection
│   ├── hash_utils.py        # Short hash for ID normalization
│   ├── sanitize.py          # Unicode sanitization
│   ├── abort_signals.py     # CancelScope / Event-based abort
│   └── oauth/
│       ├── __init__.py
│       └── providers.py
└── env_api_keys.py          # Environment variable key lookup
```

---

## 3. Core Type Definitions (Python)

```python
# types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Union, Optional, Any
from enum import Enum

# --- Message content parts ---

@dataclass
class TextContent:
    type: Literal["text"] = "text"
    text: str = ""

@dataclass
class ImageContent:
    type: Literal["image"] = "image"
    image: bytes | str = b""
    media_type: Optional[str] = None

@dataclass
class ThinkingContent:
    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    signature: Optional[str] = None

@dataclass
class ToolCallContent:
    type: Literal["toolcall"] = "toolcall"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)

# --- Messages ---

@dataclass
class UserMessage:
    role: Literal["user"] = "user"
    content: list[TextContent | ImageContent] = field(default_factory=list)
    timestamp: float = 0.0

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

@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCallContent] = field(default_factory=list)
    metadata: Optional[AssistantMessageMetadata] = None
    diagnostics: list[dict] = field(default_factory=list)

@dataclass
class ToolResultMessage:
    role: Literal["tool"] = "tool"
    content: list[TextContent | ImageContent] = field(default_factory=list)
    tool_use_id: str = ""
    name: str = ""

Message = Union[UserMessage, AssistantMessage, ToolResultMessage]

# --- Thinking levels ---

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

THINKING_BUDGETS: dict[ThinkingLevel, int] = {
    "off": 0, "minimal": 1024, "low": 2048,
    "medium": 8192, "high": 16384, "xhigh": 16384,
}

# --- Tool definition ---

@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema (pydantic model .schema() works)

# --- Context ---

@dataclass
class MessagesContext:
    messages: list[Message]
    system_prompt: Optional[str] = None

# --- Stream Events ---

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
    input: str = ""

@dataclass
class ToolCallEndEvent:
    type: Literal["toolcall_end"] = "toolcall_end"

@dataclass
class DoneEvent:
    type: Literal["done"] = "done"
    message: AssistantMessage = field(default_factory=AssistantMessage)

@dataclass
class ErrorEvent:
    type: Literal["error"] = "error"
    error: Exception = field(default_factory=Exception)

AssistantMessageEvent = Union[
    StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    DoneEvent, ErrorEvent,
]

# --- Stream Options ---

@dataclass
class ThinkingOptions:
    level: ThinkingLevel = "off"
    budget: Optional[int] = None

@dataclass
class RetryPolicy:
    max_retries: int = 3
    delay_ms: int = 1000
    backoff_multiplier: float = 2.0
    retryable_status: list[int] = field(default_factory=lambda: [429, 500, 502, 503])

@dataclass
class CacheOptions:
    retention: Optional[Literal["short", "long"]] = None
    session_id: Optional[str] = None

@dataclass
class StreamOptions:
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    thinking: Optional[ThinkingOptions] = None
    tools: list[Tool] = field(default_factory=list)
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    stop_sequences: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    cache: Optional[CacheOptions] = None
    cancel_event: Optional[Any] = None  # asyncio.Event
    retry_policy: Optional[RetryPolicy] = None
    api_key: Optional[str] = None

@dataclass
class SimpleStreamOptions:
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning: Optional[ThinkingLevel] = None  # unified across providers
    tools: list[Tool] = field(default_factory=list)
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    cache: Optional[CacheOptions] = None
    cancel_event: Optional[Any] = None
    api_key: Optional[str] = None
```

---

## 4. Event Stream Implementation (Python)

```python
# utils/event_stream.py
import asyncio
from typing import AsyncGenerator, TypeVar, Generic

TEvent = TypeVar("TEvent")
TResult = TypeVar("TResult")

class EventStream(Generic[TEvent, TResult]):
    """
    Async generator wrapper that allows emitting events from a background task
    and collecting a final result.
    """
    def __init__(self):
        self._queue: asyncio.Queue[TEvent | None] = asyncio.Queue()
        self._result_future: asyncio.Future[TResult] = asyncio.get_event_loop().create_future()

    def emit(self, event: TEvent) -> None:
        self._queue.put_nowait(event)

    def done(self, result: TResult) -> None:
        self._result_future.set_result(result)
        self._queue.put_nowait(None)  # sentinel

    def error(self, exc: Exception) -> None:
        if not self._result_future.done():
            self._result_future.set_exception(exc)
        self._queue.put_nowait(None)

    async def result(self) -> TResult:
        return await self._result_future

    async def __aiter__(self) -> AsyncGenerator[TEvent, None]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item
```

---

## 5. Provider Interface

```python
# api_registry.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class ApiProvider(ABC):
    @abstractmethod
    def stream(
        self,
        model: "Model",
        context: "MessagesContext",
        options: StreamOptions | None = None,
    ) -> "AssistantMessageEventStream":
        ...

    @abstractmethod
    def stream_simple(
        self,
        model: "Model",
        context: "MessagesContext",
        options: SimpleStreamOptions | None = None,
    ) -> "AssistantMessageEventStream":
        ...

_registry: dict[str, ApiProvider] = {}

def register_api_provider(api: str, provider: ApiProvider) -> None:
    _registry[api] = provider

def get_api_provider(api: str) -> ApiProvider | None:
    return _registry.get(api)
```

---

## 6. Anthropic Provider Sketch

```python
# providers/anthropic.py
import os
import anthropic
from ..types import *
from ..utils.event_stream import EventStream

class AnthropicProvider(ApiProvider):
    def stream(self, model, context, options=None):
        stream = EventStream()
        asyncio.create_task(self._run(stream, model, context, options))
        return stream

    async def _run(self, stream, model, context, options):
        client = anthropic.AsyncAnthropic(
            api_key=options.api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        msgs = convert_messages(context.messages)
        tools = convert_tools(options.tools) if options and options.tools else []
        
        try:
            stream.emit(StartEvent())
            async with client.messages.stream(
                model=model.id,
                system=context.system_prompt,
                messages=msgs,
                tools=tools,
                max_tokens=options.max_tokens or 4096,
            ) as s:
                async for event in s:
                    match event.type:
                        case "content_block_start":
                            if event.content_block.type == "text":
                                stream.emit(TextStartEvent())
                            elif event.content_block.type == "tool_use":
                                stream.emit(ToolCallStartEvent(
                                    id=event.content_block.id,
                                    name=event.content_block.name
                                ))
                        case "content_block_delta":
                            if event.delta.type == "text_delta":
                                stream.emit(TextDeltaEvent(text=event.delta.text))
                            elif event.delta.type == "input_json_delta":
                                stream.emit(ToolCallDeltaEvent(input=event.delta.partial_json))
                        case "content_block_stop":
                            stream.emit(TextEndEvent())  # or ToolCallEndEvent
                        case "message_stop":
                            final_msg = build_final_message(s)
                            stream.emit(DoneEvent(message=final_msg))
                            stream.done(final_msg)
        except Exception as e:
            stream.emit(ErrorEvent(error=e))
            stream.error(e)
```

---

## 7. Tool Conversion (Anthropic)

```python
def convert_tools_to_anthropic(tools: list[Tool]) -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in tools
    ]

def convert_tools_to_openai(tools: list[Tool]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]

def convert_tools_to_google(tools: list[Tool]) -> list[dict]:
    # Convert JSON Schema types to Google's uppercase format
    def convert_type(schema: dict) -> dict:
        type_map = {"string": "STRING", "number": "NUMBER", "integer": "INTEGER",
                    "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT"}
        result = dict(schema)
        if "type" in result:
            result["type"] = type_map.get(result["type"], result["type"])
        if "properties" in result:
            result["properties"] = {k: convert_type(v) for k, v in result["properties"].items()}
        return result

    return [{
        "function_declarations": [
            {
                "name": t.name,
                "description": t.description,
                "parameters": convert_type(t.input_schema),
            }
            for t in tools
        ]
    }]
```

---

## 8. Public API

```python
# stream.py
from typing import AsyncGenerator
from .types import *
from .api_registry import get_api_provider

def stream(
    model: Model,
    context: MessagesContext,
    options: StreamOptions | None = None,
) -> EventStream:
    provider = get_api_provider(model.api)
    if not provider:
        raise ValueError(f"No provider registered for API: {model.api}")
    return provider.stream(model, context, options)

async def complete(
    model: Model,
    context: MessagesContext,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    s = stream(model, context, options)
    async for _ in s:
        pass
    return await s.result()

def stream_simple(
    model: Model,
    context: MessagesContext,
    options: SimpleStreamOptions | None = None,
) -> EventStream:
    provider = get_api_provider(model.api)
    if not provider:
        raise ValueError(f"No provider registered for API: {model.api}")
    return provider.stream_simple(model, context, options)

async def complete_simple(
    model: Model,
    context: MessagesContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    s = stream_simple(model, context, options)
    async for _ in s:
        pass
    return await s.result()
```

---

## 9. Python-Specific Implementation Notes

### Streaming JSON (for partial tool arguments)
```python
# utils/json_parse.py
# Use: partial-json (pip install partial-json) or json_repair
import json
from partial_json import loads as partial_loads

def parse_streaming_json(partial: str) -> dict:
    try:
        return partial_loads(partial) or {}
    except Exception:
        return {}

def parse_json_with_repair(text: str) -> dict:
    # Escape invalid escape sequences
    import re
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    return json.loads(fixed)
```

### Short Hash (for tool call ID normalization)
```python
# utils/hash_utils.py
import hashlib

def short_hash(s: str, length: int = 64) -> str:
    h = hashlib.shake_256(s.encode()).hexdigest(length // 2)
    return h[:length]

def normalize_tool_call_id(id: str, max_len: int = 64) -> str:
    return id if len(id) <= max_len else short_hash(id, max_len)
```

### Context Overflow Detection
```python
# utils/overflow.py
import re

OVERFLOW_PATTERNS = [
    re.compile(r'prompt.*too long', re.I),
    re.compile(r'exceeds.*context window', re.I),
    re.compile(r'context length exceeded', re.I),
    re.compile(r'maximum context length', re.I),
    re.compile(r'input too long', re.I),
    re.compile(r'too many tokens', re.I),
]

def is_context_overflow(
    error: Exception,
    usage: TokenUsage | None = None,
    context_window: int | None = None,
) -> bool:
    msg = str(error)
    if any(p.search(msg) for p in OVERFLOW_PATTERNS):
        return True
    if usage and context_window:
        if usage.input_tokens >= context_window:
            return True
    return False
```

### Thinking Block Preservation
```python
# providers/transform_messages.py

def encode_thinking_block(thinking: ThinkingContent) -> TextContent:
    """Encode thinking for providers that don't support it natively."""
    return TextContent(
        text=f"<think>{thinking.thinking}</think>",
        # Store signature as metadata somehow
    )

def decode_thinking_block(text_content: TextContent) -> ThinkingContent | None:
    """Decode <think>...</think> back to ThinkingContent."""
    import re
    m = re.match(r'<think>(.*?)</think>', text_content.text, re.DOTALL)
    if m:
        return ThinkingContent(thinking=m.group(1))
    return None

def insert_synthetic_tool_results(
    messages: list[Message],
) -> list[Message]:
    """Insert empty error results for orphaned tool calls."""
    result = []
    for i, msg in enumerate(messages):
        result.append(msg)
        if isinstance(msg, AssistantMessage):
            tool_calls = [c for c in msg.content if isinstance(c, ToolCallContent)]
            # Find which IDs have corresponding ToolResultMessages
            subsequent_ids = {
                m.tool_use_id for m in messages[i+1:]
                if isinstance(m, ToolResultMessage)
            }
            for tc in tool_calls:
                if tc.id not in subsequent_ids:
                    result.append(ToolResultMessage(
                        tool_use_id=tc.id,
                        name=tc.name,
                        content=[TextContent(text="Error: no result provided")],
                    ))
    return result
```

### Abort / Cancellation
```python
# Python doesn't have AbortSignal but asyncio.Event works:
cancel_event = asyncio.Event()

# In provider:
async def _run(self, stream, model, context, options):
    cancel = options.cancel_event if options else None
    async for chunk in api_call():
        if cancel and cancel.is_set():
            stream.error(asyncio.CancelledError("Request cancelled"))
            return
        ...
```

---

## 10. Providers Priority for Python Port

### Phase 1 (Core — must have)
1. `anthropic.py` — Claude models via `anthropic` SDK
2. `openai_completions.py` — GPT + OpenAI-compatible APIs
3. `google.py` — Gemini via `google-generativeai`
4. `faux.py` — Mock provider for testing

### Phase 2 (Extended)
5. `openai_responses.py` — OpenAI Responses API (newer)
6. `mistral.py` — Mistral via `mistralai` SDK
7. `amazon_bedrock.py` — AWS Bedrock via `boto3`
8. `google_vertex.py` — Vertex AI with ADC

### Phase 3 (Edge)
9. `azure_openai.py` — Azure via `openai` SDK with base URL
10. `cloudflare.py` — Cloudflare AI Gateway

---

## 11. Python Dependencies

```toml
# pyproject.toml
[project]
name = "pi-ai"
dependencies = [
    "anthropic>=0.40.0",
    "openai>=1.55.0",
    "google-generativeai>=0.8.0",
    "mistralai>=1.0.0",
    "boto3>=1.34.0",            # for Bedrock
    "partial-json>=0.1.7",       # streaming JSON
    "pydantic>=2.0.0",           # tool schema validation
    "httpx>=0.27.0",             # for custom HTTP providers
]

[project.optional-dependencies]
vertex = ["google-cloud-aiplatform>=1.70.0"]
bedrock = ["boto3>=1.34.0", "botocore>=1.34.0"]
```

---

## 12. Python-Specific Enhancements Over TypeScript Version

1. **Sync + Async API** — Provide both sync (`complete_sync`) and async (`complete`) flavors
2. **Pydantic Tool Schemas** — Accept pydantic models directly as tool schemas
3. **Context Manager Support** — `async with stream(model, ctx) as s` for cleanup
4. **Type stubs** — Full `py.typed` marker + type stubs for IDE support
5. **Dataclass-based messages** — More Pythonic than interface definitions
6. **`__repr__` on all types** — Better debugging experience

---

*Last updated: 2026-06-09*
