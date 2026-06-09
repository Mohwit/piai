# piai — Python Unified LLM Abstraction: Implementation Plan

> Inspired by: [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi/tree/main/packages/ai)  
> Scope (Phase 1): OpenAI (Completions), Anthropic, Google Vertex AI  
> Language: Python 3.11+  
> Architecture: Follow original pi-ai TypeScript structure as closely as possible

---

## 1. Goals & Non-Goals

### Goals
- Unified streaming API across all providers — same event types, same options shape
- Both async (`stream`, `complete`) interfaces; sync via `asyncio.run()` at call site
- Tool / function calling with automatic per-provider format conversion
- Thinking / reasoning support normalized to a single `ThinkingLevel` across providers
- Token usage, cost tracking, and structured diagnostics per request
- Lazy provider loading — only import a provider's SDK on first use
- `KnownApi` enum for type-safe provider routing (no string matching)
- Full type annotations; Pydantic models accepted as tool input schemas
- Architecture must not block future addition of any provider in the original pi-ai

### Non-Goals (Phase 1)
- OAuth / browser-based authentication *(registry must have an extension point — see §6)*
- Image generation *(separate registry will be needed — see §6)*
- Prompt caching (Anthropic `cache_control`, OpenAI session IDs) *(fields present in types, unused)*
- Azure OpenAI, Mistral, AWS Bedrock, Cloudflare, Groq, DeepSeek, etc.
- `complete_sync()` wrapper *(see §13 for why this is excluded)*

---

## 2. Package Structure

Mirrors the original TypeScript structure as closely as Python conventions allow.

```
piai/
├── __init__.py                   # Public surface: stream, complete, get_model
├── types.py                      # All dataclasses: messages, events, options, tool, diagnostic
├── models.py                     # Model dataclass + model registry (get_model, list_models)
├── models_data.py                # Auto-generated from models_data.json (hand-seeded in Phase 1)
├── models_data.json              # Source of truth — generates models_data.py
├── api_registry.py               # KnownApi enum + provider registration + lazy loading
├── stream.py                     # Public API: stream(), complete(), stream_simple(), complete_simple()
│
├── providers/
│   ├── __init__.py
│   ├── anthropic.py              # Anthropic Messages API
│   ├── openai_completions.py     # OpenAI Chat Completions + compatible APIs (Groq, Together, etc.)
│   ├── openai_responses.py       # OpenAI Responses API (stub in Phase 1; Azure depends on it)
│   ├── google.py                 # Google AI Studio / Gemini API (stub in Phase 1)
│   ├── google_vertex.py          # Google Vertex AI with ADC
│   ├── faux.py                   # Mock provider for testing
│   ├── register_builtins.py      # Wires ALL built-in providers into api_registry
│   ├── simple_options.py         # buildBaseOptions(): SimpleStreamOptions → StreamOptions
│   └── transform_messages.py     # Shared message normalization used by ALL providers
│
└── utils/
    ├── __init__.py
    ├── event_stream.py           # EventStream class (asyncio.Queue + asyncio.Future)
    ├── validation.py             # Tool argument validation + coercion + StringEnum helper
    ├── json_parse.py             # Streaming + repair JSON parsing
    ├── hash_utils.py             # short_hash + per-provider tool call ID normalization rules
    ├── overflow.py               # Context overflow detection (regex + usage heuristics)
    ├── retry.py                  # RetryPolicy + backoff logic
    ├── headers.py                # Auth + custom header construction (shared across providers)
    ├── diagnostics.py            # Diagnostic dataclass + attach helpers
    └── abort.py                  # AbortSignal adapter (asyncio.Event wrapper)

pyproject.toml
```

### Files that are stubs in Phase 1 but must exist
- `providers/openai_responses.py` — Azure OpenAI Responses provider depends on this
- `providers/google.py` — Google AI Studio; distinct auth from Vertex AI

### Files deferred to future phases
- `images.py` + `images_api_registry.py` — image generation (separate registry, see §6)
- `oauth.py` + `utils/oauth/` — OAuth provider system

---

## 3. Core Type System (`types.py`)

### 3.1 Diagnostic

```python
@dataclass
class DiagnosticError:
    message: str
    name: Optional[str] = None
    stack: Optional[str] = None
    code: Optional[str] = None

@dataclass
class Diagnostic:
    type: str
    timestamp: float                          # time.time()
    error: Optional[DiagnosticError] = None
    details: Optional[dict[str, Any]] = None
```

### 3.2 Message Content Parts

```python
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
    signature: Optional[str] = None          # for cross-provider preservation

@dataclass
class ToolCallContent:
    type: Literal["toolcall"] = "toolcall"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
```

### 3.3 Messages

```python
@dataclass
class UserMessage:
    role: Literal["user"] = "user"
    content: list[TextContent | ImageContent] = field(default_factory=list)
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
    cost: Optional[float] = None              # calculated from model.cost + usage

@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCallContent] = field(default_factory=list)
    metadata: Optional[AssistantMessageMetadata] = None
    diagnostics: list[Diagnostic] = field(default_factory=list)  # matches original

@dataclass
class ToolResultMessage:
    role: Literal["tool"] = "tool"
    content: list[TextContent | ImageContent] = field(default_factory=list)
    tool_use_id: str = ""
    name: str = ""

Message = UserMessage | AssistantMessage | ToolResultMessage

@dataclass
class MessagesContext:
    messages: list[Message]
    system_prompt: Optional[str] = None
```

### 3.4 Tool

```python
@dataclass
class Tool:
    name: str
    description: str
    input: dict[str, Any]    # JSON Schema — matches original field name (NOT input_schema)
                              # If a pydantic.BaseModel subclass is passed, call .model_json_schema()
```

> **Note:** The field is named `input`, not `input_schema`. The name `input_schema` is Anthropic-specific terminology. Providers convert `tool.input` into their own format (`input_schema` for Anthropic, `parameters` for OpenAI, `parameters` in Vertex format for Google).

### 3.5 Thinking / Reasoning

```python
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

THINKING_BUDGETS: dict[str, int] = {
    "off": 0, "minimal": 1024, "low": 2048,
    "medium": 8192, "high": 16384, "xhigh": 16384,
}

@dataclass
class ThinkingOptions:
    level: ThinkingLevel = "off"
    budget: Optional[int] = None    # overrides budget map if set

@dataclass
class ReasoningOptions:
    effort: ThinkingLevel = "off"   # for OpenAI o* models
    budget: Optional[int] = None
```

### 3.6 Options

```python
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
    thinking: Optional[ThinkingOptions] = None     # for Anthropic / Google
    reasoning: Optional[ReasoningOptions] = None   # for OpenAI o* models
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    stop_sequences: Optional[list[str]] = None
    headers: Optional[dict[str, str]] = None       # pass-through to provider HTTP layer
    cache: Optional[CacheOptions] = None           # defined now, used in future phases
    abort_signal: Optional["asyncio.Event"] = None # renamed from cancel_event; matches abortSignal
    retry_policy: Optional[RetryPolicy] = None
    api_key: Optional[str] = None

@dataclass
class SimpleStreamOptions:
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning: Optional[ThinkingLevel] = None  # single unified field across all providers
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    cache: Optional[CacheOptions] = None
    abort_signal: Optional["asyncio.Event"] = None
    api_key: Optional[str] = None
```

### 3.7 Stream Events

```python
# Emitted in this order per response:
StartEvent
TextStartEvent → TextDeltaEvent(text) × N → TextEndEvent
ThinkingStartEvent → ThinkingDeltaEvent(thinking) × N → ThinkingEndEvent  # if reasoning active
ToolCallStartEvent(id, name) → ToolCallDeltaEvent(input: str) × N → ToolCallEndEvent
DoneEvent(message: AssistantMessage)    # final assembled message
ErrorEvent(error: Exception)            # on failure

AssistantMessageEvent = Union[
    StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    DoneEvent, ErrorEvent,
]

AssistantMessageEventStream = EventStream[AssistantMessageEvent, AssistantMessage]
```

---

## 4. EventStream (`utils/event_stream.py`)

Direct port of the TypeScript `EventStream<TEvent, TResult>`:

```python
class EventStream(Generic[TEvent, TResult]):
    """
    asyncio.Queue-based event stream.
    Producer calls emit()/done()/error(); consumer iterates with async for.
    result() returns the final TResult (set by done()) as an awaitable Future.
    """
    def __init__(self):
        self._queue: asyncio.Queue[TEvent | None] = asyncio.Queue()
        self._result: asyncio.Future[TResult] = asyncio.get_event_loop().create_future()

    def emit(self, event: TEvent) -> None: ...
    def done(self, result: TResult) -> None: ...   # puts None sentinel on queue
    def error(self, exc: Exception) -> None: ...

    async def result(self) -> TResult: ...          # await the Future
    def __aiter__(self) -> AsyncIterator[TEvent]: ...
```

- `stream.py`'s `complete()` calls `await event_stream.result()` — it does NOT re-iterate the stream
- The `None` sentinel signals end of iteration in `__aiter__`
- Initial state of `abort_signal` is checked at stream start (to handle pre-cancelled signals)

---

## 5. Provider Registry (`api_registry.py`)

### KnownApi Enum

```python
class KnownApi(str, Enum):
    ANTHROPIC           = "anthropic"
    OPENAI_COMPLETIONS  = "openai-completions"
    OPENAI_RESPONSES    = "openai-responses"
    GOOGLE              = "google"
    GOOGLE_VERTEX       = "google-vertex"
    MISTRAL             = "mistral"            # stub — registers when Phase 2 adds provider
    BEDROCK             = "bedrock"            # stub
    AZURE_OPENAI        = "azure-openai"       # stub
    CLOUDFLARE          = "cloudflare"         # stub
    FAUX                = "faux"
```

All future providers get a `KnownApi` member now. No model will ever route via a raw string.

### Provider Protocol

```python
class ApiProvider(Protocol):
    def stream(
        self,
        model: "Model",
        context: MessagesContext,
        options: StreamOptions | None,
    ) -> AssistantMessageEventStream: ...

    def stream_simple(
        self,
        model: "Model",
        context: MessagesContext,
        options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream: ...
```

### Lazy Loading

```python
class LazyProvider:
    """Defers SDK import until first call. Cached after first instantiation."""
    def __init__(self, module_path: str, class_name: str): ...
    def _get(self) -> ApiProvider: ...
    def stream(self, *args, **kwargs): return self._get().stream(*args, **kwargs)
    def stream_simple(self, *args, **kwargs): return self._get().stream_simple(*args, **kwargs)
```

### Registration Functions

```python
def register_api_provider(api: KnownApi, provider: ApiProvider) -> None: ...
def get_api_provider(api: KnownApi) -> ApiProvider | None: ...
def reset_api_providers() -> None: ...   # test isolation
```

### OAuth Hook Point (deferred, but registry must accommodate)

The registry must support future `registerOAuthProvider` / `getOAuthProvider` calls without restructuring. Reserve `_oauth_registry: dict[str, OAuthProvider] = {}` in `api_registry.py` as a comment placeholder.

---

## 6. Model Registry (`models.py`, `models_data.py`, `models_data.json`)

### Model Dataclass

```python
@dataclass
class ModelCost:
    input: Optional[float] = None          # per million tokens
    output: Optional[float] = None
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None

@dataclass
class ModelCompat:
    """Provider-specific quirk flags. Avoids scattered if/else in shared code."""
    no_system_prompt: bool = False          # provider ignores system prompts
    no_tool_streaming: bool = False         # tool args arrive whole (e.g. Groq)
    no_image_input: bool = False
    batch_tool_results: bool = False        # must merge consecutive tool results (Anthropic)
    # Add flags here as new providers reveal quirks

@dataclass
class Model:
    id: str
    name: str
    api: KnownApi
    provider: str
    context_window: int
    base_url: Optional[str] = None
    reasoning: bool = False
    input: list[str] = field(default_factory=lambda: ["text"])
    output: list[str] = field(default_factory=lambda: ["text"])
    max_tokens: Optional[int] = None
    cost: Optional[ModelCost] = None
    thinking_level_map: Optional[dict[str, Any]] = None  # provider-specific thinking params per level
    compat: Optional[ModelCompat] = None                  # quirk flags
```

### Registry Functions

```python
def get_model(provider: str, model_id: str) -> Model: ...    # raises ValueError if not found
def list_models(provider: str | None = None) -> list[Model]: ...
def register_model(model: Model) -> None: ...                 # for runtime registration
```

### models_data.json (Source of Truth)

```json
{
  "models": [
    {
      "id": "claude-sonnet-4-5",
      "name": "Claude Sonnet 4.5",
      "api": "anthropic",
      "provider": "anthropic",
      "context_window": 200000,
      "max_tokens": 8192,
      "reasoning": true,
      "cost": { "input": 3.0, "output": 15.0 },
      "thinking_level_map": {
        "minimal": {"budget_tokens": 1024},
        "low":     {"budget_tokens": 2048},
        "medium":  {"budget_tokens": 8192},
        "high":    {"budget_tokens": 16384},
        "xhigh":   {"budget_tokens": 16384}
      },
      "compat": { "batch_tool_results": true }
    }
  ]
}
```

In Phase 1: `models_data.py` is hand-generated from this JSON. Phase 2: add a `scripts/generate_models.py` that reads the JSON and writes `models_data.py`.

---

## 7. Simple Options (`providers/simple_options.py`)

Direct port of `simple-options.ts`. Converts `SimpleStreamOptions` to `StreamOptions`:

```python
def build_base_options(simple: SimpleStreamOptions) -> StreamOptions:
    """
    Convert SimpleStreamOptions to full StreamOptions.
    Maps the unified 'reasoning' ThinkingLevel to provider-agnostic ThinkingOptions.
    Providers inspect the model's api/compat to decide how to apply it.
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
```

---

## 8. Provider Implementations

### 8.1 Base Pattern (all providers follow this)

```python
class SomeProvider:
    def stream(self, model, context, options=None):
        es = EventStream()
        asyncio.create_task(self._run(es, model, context, options))
        return es

    def stream_simple(self, model, context, options=None):
        return self.stream(model, context, build_base_options(options or SimpleStreamOptions()))

    async def _run(self, es, model, context, options):
        try:
            es.emit(StartEvent())
            # ... provider-specific streaming ...
            es.done(final_message)
        except Exception as e:
            es.emit(ErrorEvent(error=e))
            es.error(e)
```

### 8.2 Anthropic (`providers/anthropic.py`)

**SDK:** `anthropic.AsyncAnthropic`  
**Auth:** `ANTHROPIC_API_KEY` → `options.api_key` — via `utils/headers.py`

**Thinking activation:**
- `thinking` option set → `{"type": "enabled", "budget_tokens": N}` + `anthropic-beta: interleaved-thinking-2025-05-14` header
- Budget N comes from `model.thinking_level_map[level]` if present, else `THINKING_BUDGETS[level]`

**Message conversion (before sending):**
- `transform_messages.batch_tool_results(messages)` — batches consecutive `ToolResultMessage`s into one user message (required by Anthropic)
- `transform_messages.insert_synthetic_tool_results(messages)` — fills orphaned tool calls

**SSE event mapping:**
```
content_block_start  type=text       → TextStartEvent
content_block_start  type=tool_use   → ToolCallStartEvent(id, name)
content_block_start  type=thinking   → ThinkingStartEvent
content_block_delta  text_delta      → TextDeltaEvent(text)
content_block_delta  input_json_delta→ ToolCallDeltaEvent(input)
content_block_delta  thinking_delta  → ThinkingDeltaEvent(thinking)
content_block_stop                   → TextEndEvent | ToolCallEndEvent | ThinkingEndEvent
message_delta                        → capture stop_reason + output_tokens
message_stop                         → build AssistantMessage, DoneEvent, es.done()
```

**Tool format:**
```python
{"name": t.name, "description": t.description, "input_schema": t.input}
```

### 8.3 OpenAI Completions (`providers/openai_completions.py`)

**SDK:** `openai.AsyncOpenAI`  
**Auth:** `OPENAI_API_KEY` → `options.api_key` — via `utils/headers.py`

**Reasoning:** `o1`, `o3`, `o4-mini` family → map `ThinkingLevel` to `reasoning_effort`:
- `minimal/low` → `"low"`, `medium` → `"medium"`, `high/xhigh` → `"high"`
- Standard models: no thinking parameter

**Tool call ID normalization:** IDs > 64 chars → `hash_utils.short_hash(id, 64)` (OpenAI can return 450+ char IDs)

**Tool format:**
```python
{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input}}
```

**Message format:**
```python
{"role": "system", "content": system_prompt}
{"role": "user", "content": [{"type": "text", "text": "..."}]}
{"role": "assistant", "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}]}
{"role": "tool", "content": "result", "tool_call_id": "..."}
```

**SSE event mapping:**
```
delta.content              → TextDeltaEvent
delta.tool_calls[n]        → ToolCallStartEvent (index=0 of new id) | ToolCallDeltaEvent
finish_reason=stop         → TextEndEvent → DoneEvent
finish_reason=tool_calls   → ToolCallEndEvent → DoneEvent
```

**Future compatible providers** (detected from `model.base_url` / `model.compat` flags):
- DeepSeek: thinking in `delta.reasoning_content`
- Groq: `compat.no_tool_streaming = True` (tool args arrive whole)
- OpenRouter: extra headers via `utils/headers.py`

### 8.4 OpenAI Responses (`providers/openai_responses.py`) — Phase 1 Stub

Exists as a file with a `NotImplementedError` body. Required because:
- Azure OpenAI Responses provider (`azure_openai_responses.py`) will subclass or delegate to this
- Any model with `api=KnownApi.OPENAI_RESPONSES` must route here

### 8.5 Google AI Studio (`providers/google.py`) — Phase 1 Stub

Separate from Vertex AI — uses `google-generativeai` SDK with API key, not ADC. Stub exists so `KnownApi.GOOGLE` models route correctly when implemented.

### 8.6 Google Vertex AI (`providers/google_vertex.py`)

**SDK:** `vertexai` (synchronous — wrapped with `asyncio.to_thread`)  
**Auth:** Application Default Credentials (`gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS`)  
**Config:** `GOOGLE_CLOUD_PROJECT` env var required; `GOOGLE_CLOUD_LOCATION` defaults to `us-central1`

**Thinking activation:**
- Gemini 2.5 models: `generation_config.thinking_config = {"thinking_budget": N}`
- Budget N from `model.thinking_level_map[level]`

**Message format:**
```python
{"role": "user",  "parts": [{"text": "..."}]}
{"role": "model", "parts": [{"text": "..."}, {"function_call": {"name": "...", "args": {...}}}]}
{"role": "user",  "parts": [{"function_response": {"name": "...", "response": {...}}}]}
```

**Tool call IDs:** Vertex does not assign IDs — generate as `f"{name}_{int(time.time()*1000)}_{counter}"`

**Schema conversion** (`transform_messages.convert_schema_to_vertex`):
```
string → STRING, number → NUMBER, integer → INTEGER,
boolean → BOOLEAN, array → ARRAY, object → OBJECT
anyOf / const / $defs are NOT supported — tools must use flat schemas
```

**Stop reason mapping:**
```python
{"STOP": "end_turn", "MAX_TOKENS": "max_tokens",
 "SAFETY": "content_filter", "RECITATION": "content_filter",
 "TOOL_USE": "tool_use"}
```

**SSE behavior note:** Vertex does not stream tool call arguments — tool calls arrive whole. Emit `ToolCallStartEvent → ToolCallDeltaEvent(full_json) → ToolCallEndEvent` in sequence to maintain event API consistency with other providers.

**Tool format:**
```python
{"function_declarations": [
    {"name": t.name, "description": t.description,
     "parameters": convert_schema_to_vertex(t.input)}
]}
```

### 8.7 Faux / Mock (`providers/faux.py`)

```python
class FauxProvider:
    def __init__(self, responses: list[str] = None, delay: float = 0.005):
        self.responses = responses or ["Hello from faux provider."]
        self.delay = delay
        self.call_count = 0
        self.last_context: MessagesContext | None = None
        self.last_options: StreamOptions | None = None
```

Emits: `StartEvent → TextStartEvent → TextDeltaEvent (per char) → TextEndEvent → DoneEvent`

Used in all unit tests without live API keys.

---

## 9. Public API (`stream.py`)

```python
# Async streaming — yields AssistantMessageEvent
def stream(
    model: Model,
    context: MessagesContext,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream: ...

# Streaming with simple options
def stream_simple(
    model: Model,
    context: MessagesContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream: ...

# Async blocking — consumes stream, returns final message
async def complete(
    model: Model,
    context: MessagesContext,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    s = stream(model, context, options)
    async for _ in s:    # drain events
        pass
    return await s.result()   # return the Future result, do NOT re-iterate

# Async blocking with simple options
async def complete_simple(
    model: Model,
    context: MessagesContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage: ...
```

> **No `complete_sync()`**: `asyncio.run(complete(...))` raises `RuntimeError` when called inside a running event loop (FastAPI, Jupyter). Callers are expected to use `await complete()` or manage `asyncio.run()` themselves. If sync support is ever needed, it will require `nest_asyncio` and will be an opt-in extra — not a default API.

**Usage example:**

```python
import asyncio
from piai import get_model, complete_simple
from piai.types import MessagesContext, UserMessage, TextContent, SimpleStreamOptions

model = get_model("anthropic", "claude-sonnet-4-5")
ctx = MessagesContext(
    messages=[UserMessage(content=[TextContent(text="Hello!")])],
    system_prompt="You are a helpful assistant.",
)

msg = asyncio.run(complete_simple(model, ctx, SimpleStreamOptions(max_tokens=1024)))
print(msg.content[0].text)
print(msg.metadata.cost)
print(msg.diagnostics)
```

---

## 10. Message Transformation (`providers/transform_messages.py`)

Shared utilities used by ALL providers. Never provider-specific.

| Function | Used by | Purpose |
|----------|---------|---------|
| `batch_tool_results(messages)` | Anthropic | Merge consecutive ToolResultMessages into one user message |
| `insert_synthetic_tool_results(messages)` | All | Add error results for orphaned tool calls |
| `encode_thinking_block(thinking)` | Non-native providers | `ThinkingContent` → `<think>…</think>` text |
| `decode_thinking_block(text)` | Non-native providers | Reverse of above |
| `normalize_tool_call_id(id, max_len)` | OpenAI, Mistral | Hash long IDs |
| `convert_schema_to_vertex(schema)` | Google Vertex | Recursively uppercase JSON Schema types |
| `generate_google_tool_id(name, counter)` | Google Vertex | `f"{name}_{ms}_{counter}"` |

---

## 11. Utilities

### `utils/headers.py`
Shared header construction to avoid duplication across providers:
```python
def build_auth_headers(api_key: str, scheme: str = "Bearer") -> dict[str, str]: ...
def merge_headers(*header_dicts: dict[str, str]) -> dict[str, str]: ...
```

### `utils/diagnostics.py`
```python
def make_diagnostic(type: str, error: Exception | None = None, **details) -> Diagnostic: ...
def attach_diagnostic(message: AssistantMessage, diag: Diagnostic) -> None: ...
```
Used by providers to surface overflow, partial failure, retry exhaustion onto `AssistantMessage.diagnostics`.

### `utils/abort.py`
```python
class AbortSignal:
    """Wraps asyncio.Event to match the AbortSignal interface from the original."""
    def __init__(self): self._event = asyncio.Event()
    def abort(self): self._event.set()
    @property
    def aborted(self) -> bool: return self._event.is_set()
    async def wait(self): await self._event.wait()
```
`StreamOptions.abort_signal` accepts either `asyncio.Event` or `AbortSignal`.

### `utils/hash_utils.py`
```python
def short_hash(s: str, length: int = 64) -> str: ...  # shake_256 hex, truncated

# Per-provider normalization rules (documented here, called in transform_messages.py):
# OpenAI:    max_len=64 (IDs can be 450+ chars)
# Anthropic: pass-through (max 64 chars native)
# Google:    generated, not normalized
# Mistral:   max_len=9
# Groq:      pass-through
```

### `utils/overflow.py`
```python
OVERFLOW_PATTERNS: list[re.Pattern]   # 20+ provider-specific patterns

def is_context_overflow(
    error: Exception | None,
    usage: TokenUsage | None = None,
    context_window: int | None = None,
) -> bool: ...
# Three detection modes:
# 1. Error message regex match
# 2. usage.input_tokens >= context_window (silent overflow)
# 3. output_tokens == 0 with non-empty input (length truncation)
```

### `utils/validation.py`
```python
def validate_tool_call(call: ToolCallContent, tools: list[Tool]) -> dict[str, Any]: ...
# - Validates call.input against tool.input (JSON Schema)
# - Coerces types (e.g. "123" → 123 for integer fields)
# - Caches validators with weakref.WeakKeyDictionary

class StringEnum:
    """
    Helper for Google Vertex compatibility.
    Google does not support anyOf/const for enums in tool schemas.
    Converts: {"enum": ["a", "b"]} → {"type": "STRING", "description": "One of: a, b"}
    """
```

### `utils/json_parse.py`
```python
def parse_streaming_json(partial: str) -> dict: ...   # via partial-json library
def parse_json_with_repair(text: str) -> dict: ...    # fix escape sequences + malformed JSON
```

### `utils/retry.py`
```python
async def with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool] | None = None,
) -> T: ...

def default_is_retryable(exc: Exception) -> bool:
    # True for 429, 500, 502, 503, 504
    # False for 400, 401, 403, 404
```

---

## 12. Future Extensibility: Registry Hook Points

### Images (deferred)

When image generation is added, create `images_api_registry.py` as a completely separate registry — do NOT extend `api_registry.py`. Mirror the original's separation of `images-api-registry.ts`:

```python
# images_api_registry.py (future)
register_images_provider(api: KnownApi, provider: ImagesApiProvider) -> None
get_images_provider(api: KnownApi) -> ImagesApiProvider | None
```

### OAuth (deferred)

When OAuth is added, create `oauth.py` and `utils/oauth/`. The `api_registry.py` already reserves `_oauth_registry: dict` as a placeholder. This will grow into `registerOAuthProvider` / `getOAuthProvider` — same pattern as the original.

### Adding a New Provider (Phase 2+)

1. Add `KnownApi.PROVIDER_NAME = "provider-name"` to the enum in `api_registry.py`
2. Create `providers/provider_name.py` implementing `ApiProvider`
3. Add models to `models_data.json` with `"api": "provider-name"` and appropriate `compat`/`thinking_level_map`
4. Register in `providers/register_builtins.py` with lazy loading
5. Add any shared transformations to `providers/transform_messages.py`
6. Add provider-specific header logic to `utils/headers.py`

---

## 13. Dependencies (`pyproject.toml`)

```toml
[project]
name = "piai"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "anthropic>=0.40.0",
    "openai>=1.55.0",
    "partial-json>=0.1.7",
    "jsonschema>=4.0.0",
    "pydantic>=2.0.0",       # tool schema input from Pydantic models
]

[project.optional-dependencies]
vertex = [
    "google-cloud-aiplatform>=1.70.0",
    "google-auth>=2.0.0",
]
google = [
    "google-generativeai>=0.8.0",    # Google AI Studio (Phase 2)
]
mistral = ["mistralai>=1.0.0"]       # Phase 2
bedrock = ["boto3>=1.34.0"]          # Phase 2
all = ["piai[vertex,google,mistral,bedrock]"]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "mypy>=1.0.0",
    "ruff>=0.6.0",
]
```

> Vertex AI is optional because `google-cloud-aiplatform` is a large transitive dependency tree. Anthropic and OpenAI are always installed.

> **No `nest_asyncio`** dependency — this is the reason `complete_sync()` is excluded. Sync callers use `asyncio.run(complete(...))` directly.

---

## 14. Build Phases

### Phase 1a — Foundation
- [ ] `types.py` — all dataclasses (messages, events, options, tool, diagnostic)
- [ ] `utils/event_stream.py` — EventStream
- [ ] `utils/abort.py` — AbortSignal wrapper
- [ ] `utils/hash_utils.py` — short_hash + normalization rules
- [ ] `utils/overflow.py` — regex patterns + detection logic
- [ ] `utils/json_parse.py` — streaming + repair
- [ ] `utils/diagnostics.py` — Diagnostic + helpers
- [ ] `utils/headers.py` — auth + header merge
- [ ] `api_registry.py` — KnownApi enum + registration + lazy loader
- [ ] `models.py` + `models_data.json` — initial model schema + registry functions
- [ ] `models_data.py` — hand-seeded from models_data.json (Anthropic + OpenAI + Vertex models)
- [ ] `providers/faux.py` — validates end-to-end event stream without live APIs
- [ ] `providers/register_builtins.py` — lazy-registers all providers (stubs for unimplemented ones)
- [ ] `providers/simple_options.py` — buildBaseOptions()
- [ ] `stream.py` — stream(), complete(), stream_simple(), complete_simple()
- [ ] `__init__.py` — public exports

### Phase 1b — Anthropic Provider
- [ ] `providers/transform_messages.py` — batch_tool_results, insert_synthetic_tool_results
- [ ] `providers/anthropic.py` — full streaming implementation
- [ ] Anthropic models in `models_data.json` with thinking_level_map + compat
- [ ] Integration tests (gated on `ANTHROPIC_API_KEY`)

### Phase 1c — OpenAI Completions Provider
- [ ] `providers/openai_completions.py` — full streaming implementation
- [ ] Tool call ID normalization wired via transform_messages
- [ ] Reasoning effort mapping for o* models via model.compat / thinking_level_map
- [ ] OpenAI models in `models_data.json`
- [ ] `providers/openai_responses.py` — stub with NotImplementedError
- [ ] Integration tests (gated on `OPENAI_API_KEY`)

### Phase 1d — Google Vertex Provider
- [ ] `providers/google_vertex.py` — asyncio.to_thread wrapping
- [ ] `convert_schema_to_vertex` in transform_messages.py
- [ ] `StringEnum` helper in utils/validation.py
- [ ] Tool ID generation in transform_messages.py
- [ ] Vertex models in `models_data.json` with thinking_level_map
- [ ] `providers/google.py` — stub with NotImplementedError
- [ ] Integration tests (gated on GCP project + ADC)

### Phase 1e — Polish
- [ ] `utils/retry.py` wired into all providers
- [ ] `utils/validation.py` — tool argument validation + coercion
- [ ] Cost calculation in complete() → AssistantMessageMetadata.cost
- [ ] Full type annotations + `py.typed` marker
- [ ] Unit tests for all utilities
- [ ] README with usage examples
- [ ] `scripts/generate_models.py` — reads models_data.json → writes models_data.py

---

## 15. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Provider routing | `KnownApi` enum | No raw string matching; type-safe; mirrors original |
| Compat flags | On `Model.compat` dataclass | Avoids if/else sprawl in shared code; mirrors original `ModelCompat` |
| Thinking params | On `Model.thinking_level_map` | Per-model parameter override; avoids hardcoded budgets |
| Tool field name | `Tool.input` (not `input_schema`) | Matches original; `input_schema` is Anthropic-specific |
| AbortSignal | `asyncio.Event` wrapped in `AbortSignal` | Preserves `.aborted` property; handles pre-cancelled state |
| Field name | `abort_signal` (not `cancel_event`) | Matches original `abortSignal` naming convention |
| Sync API | Excluded; callers use `asyncio.run()` | Avoids `RuntimeError` in running loops; no `nest_asyncio` dep |
| Model data | `models_data.json` → `models_data.py` | JSON is generator-compatible; hand-seeded in Phase 1 |
| Vertex SDK | `asyncio.to_thread()` wrapper | Vertex SDK is synchronous; no async alternative |
| Lazy loading | `LazyProvider` via `importlib` | Mirrors original; no SDK imported at package load time |
| Image registry | Separate `images_api_registry.py` | Mirrors original separation; avoids polluting text registry |
| OAuth | Reserved in registry, deferred | Hook point exists; no breaking change when added |
| `stream_simple` | Delegates to `stream()` via `build_base_options()` | Single implementation path; mirrors original |
| Diagnostics | On `AssistantMessage.diagnostics[]` | Surfaces overflow + errors to callers; mirrors original |
| Tool validation | `jsonschema` + `WeakKeyDictionary` cache | Matches original TypeBox validation approach |

---

*Document version: 2.0 — 2026-06-09 (revised after architecture review)*
