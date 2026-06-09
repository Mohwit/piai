# piai

A unified Python LLM abstraction library — port of [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi/tree/main/packages/ai).

Single streaming API across Anthropic, OpenAI, and Google Vertex AI. Same event types, same options shape, same tool calling interface — regardless of which provider is underneath.

---

## Features

- **Unified streaming** — `async for event in stream(model, ctx)` works identically across all providers
- **Granular events** — `text_delta`, `thinking_delta`, `toolcall_start/delta/end`, `done`, `error`
- **Thinking / reasoning** — normalized `ThinkingLevel` (`"off"` → `"xhigh"`) maps to provider-specific params automatically
- **Tool calling** — JSON Schema definitions, automatic per-provider format conversion, streaming argument assembly
- **Cost tracking** — per-request USD cost calculated from model metadata + token usage
- **Context overflow detection** — regex patterns + usage heuristics across 20+ provider error formats
- **Lazy provider loading** — provider SDKs imported only on first use
- **Mock provider** — `FauxProvider` for testing without API keys
- **Full type annotations** — dataclasses throughout, no `dict[str, Any]` leaking into public API

---

## Installation

```bash
pip install piai                   # Anthropic + OpenAI
pip install "piai[vertex]"         # + Google Vertex AI
pip install "piai[all]"            # all providers
```

**Requires Python 3.11+**

---

## Quick Start

```python
import asyncio
import piai

model = piai.get_model("anthropic", "claude-haiku-4-5")

ctx = piai.MessagesContext(
    messages=[piai.UserMessage(content=[piai.TextContent(text="Hello!")])],
    system_prompt="You are a helpful assistant.",
)

msg = asyncio.run(piai.complete_simple(model, ctx))
print(msg.content[0].text)
print(f"Cost: ${msg.metadata.cost:.6f}")
```

---

## Streaming

```python
async def main():
    model = piai.get_model("anthropic", "claude-haiku-4-5")
    ctx = piai.MessagesContext(
        messages=[piai.UserMessage(content=[piai.TextContent(text="Write a haiku.")])]
    )

    async for event in piai.stream_simple(model, ctx):
        if isinstance(event, piai.TextDeltaEvent):
            print(event.text, end="", flush=True)
```

---

## Providers

| Provider | Model lookup | Extra install |
|----------|-------------|---------------|
| Anthropic | `get_model("anthropic", "claude-haiku-4-5")` | _(included)_ |
| OpenAI | `get_model("openai", "gpt-4o")` | _(included)_ |
| OpenAI o* (reasoning) | `get_model("openai", "o4-mini")` | _(included)_ |
| Google Vertex AI | `get_model("google-vertex", "gemini-2.5-flash")` | `pip install "piai[vertex]"` |
| Faux (mock) | `get_model("faux", "faux")` | _(included)_ |

Set the appropriate environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_CLOUD_PROJECT="my-project"   # Vertex AI
# gcloud auth application-default login    # Vertex AI (ADC)
```

---

## Tool Calling

```python
weather_tool = piai.Tool(
    name="get_weather",
    description="Get current weather for a city.",
    input={
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
)

model = piai.get_model("anthropic", "claude-haiku-4-5")
ctx = piai.MessagesContext(
    messages=[piai.UserMessage(content=[piai.TextContent(text="What's the weather in London?")])]
)

msg = await piai.complete_simple(model, ctx, piai.SimpleStreamOptions(tools=[weather_tool]))

for part in msg.content:
    if isinstance(part, piai.ToolCallContent):
        result = call_my_weather_api(**part.input)   # your implementation
        ctx.messages.append(msg)
        ctx.messages.append(piai.ToolResultMessage(
            content=[piai.TextContent(text=result)],
            tool_use_id=part.id,
            name=part.name,
        ))
```

---

## Thinking / Reasoning

```python
# Via StreamOptions (full control)
options = piai.StreamOptions(
    max_tokens=4096,
    thinking=piai.ThinkingOptions(level="medium"),  # budget: 8192 tokens
)

# Via SimpleStreamOptions (unified across all providers)
options = piai.SimpleStreamOptions(reasoning="high")

# Stream thinking events
async for event in piai.stream(model, ctx, options):
    if isinstance(event, piai.ThinkingDeltaEvent):
        print(event.thinking, end="", flush=True)
    elif isinstance(event, piai.TextDeltaEvent):
        print(event.text, end="", flush=True)
```

Thinking levels: `"off"` `"minimal"` `"low"` `"medium"` `"high"` `"xhigh"`

| Level | Token budget |
|-------|-------------|
| minimal | 1,024 |
| low | 2,048 |
| medium | 8,192 |
| high | 16,384 |
| xhigh | 16,384 |

---

## API Reference

### Main functions

```python
# Streaming — returns EventStream, iterate with async for
piai.stream(model, ctx, options=None)               -> AssistantMessageEventStream
piai.stream_simple(model, ctx, options=None)        -> AssistantMessageEventStream

# Blocking — drains stream, returns final message
await piai.complete(model, ctx, options=None)       -> AssistantMessage
await piai.complete_simple(model, ctx, options=None)-> AssistantMessage
```

### Model registry

```python
piai.get_model(provider, model_id)     -> Model       # raises ValueError if not found
piai.list_models(provider=None)        -> list[Model]
piai.register_model(model)             -> None         # runtime registration
```

### Provider registry

```python
piai.register_api_provider(KnownApi.FAUX, my_provider)
piai.reset_api_providers()   # test isolation — clears and re-registers built-ins
```

### Key types

```python
# Options
piai.StreamOptions(temperature, max_tokens, thinking, reasoning, tools, tool_choice,
                   stop_sequences, headers, cache, abort_signal, retry_policy, api_key)
piai.SimpleStreamOptions(temperature, max_tokens, reasoning, tools, tool_choice,
                         cache, abort_signal, api_key)

# Messages
piai.MessagesContext(messages, system_prompt)
piai.UserMessage(content=[piai.TextContent(text="...")])
piai.ToolResultMessage(content, tool_use_id, name)

# Content parts
piai.TextContent(text)
piai.ImageContent(image, media_type)
piai.ThinkingContent(thinking, signature)
piai.ToolCallContent(id, name, input)

# Events
piai.TextDeltaEvent(text)
piai.ThinkingDeltaEvent(thinking)
piai.ToolCallStartEvent(id, name)
piai.ToolCallDeltaEvent(input)
piai.DoneEvent(message)
piai.ErrorEvent(error)
```

---

## Examples

| File | Description |
|------|-------------|
| [`01_basic_completion.py`](examples/01_basic_completion.py) | Simplest usage — `complete_simple` |
| [`02_streaming.py`](examples/02_streaming.py) | Real-time token streaming |
| [`03_tool_calling.py`](examples/03_tool_calling.py) | Multi-tool agentic loop |
| [`04_thinking.py`](examples/04_thinking.py) | Extended reasoning / thinking mode |
| [`05_openai.py`](examples/05_openai.py) | GPT-4o and o4-mini reasoning |
| [`06_vertex_ai.py`](examples/06_vertex_ai.py) | Gemini 2.5 on Vertex AI |
| [`07_multi_turn.py`](examples/07_multi_turn.py) | Multi-turn chat + cross-provider conversation |
| [`08_faux_provider.py`](examples/08_faux_provider.py) | Testing with mock provider (no API key) |
| [`09_abort_and_retry.py`](examples/09_abort_and_retry.py) | Abort signals and retry policies |
| [`10_custom_provider.py`](examples/10_custom_provider.py) | Implementing and registering a custom provider |

---

## Project Structure

```
piai/
├── __init__.py              # Public API exports
├── types.py                 # All dataclasses (messages, events, options, tool)
├── models.py                # Model dataclass + registry (get_model, list_models)
├── models_data.json         # Source of truth for model definitions
├── models_data.py           # Generated from models_data.json
├── api_registry.py          # KnownApi enum + provider registration + lazy loading
├── stream.py                # stream(), complete(), stream_simple(), complete_simple()
├── providers/
│   ├── anthropic.py         # Anthropic Messages API
│   ├── openai_completions.py# OpenAI Chat Completions (+ Groq, DeepSeek, etc.)
│   ├── google_vertex.py     # Google Vertex AI (Gemini)
│   ├── faux.py              # Mock provider for testing
│   ├── transform_messages.py# Shared message normalization (all providers)
│   ├── simple_options.py    # build_base_options()
│   └── register_builtins.py # Lazy-registers all providers
└── utils/
    ├── event_stream.py      # asyncio.Queue-based EventStream
    ├── abort.py             # AbortSignal wrapper
    ├── hash_utils.py        # Tool call ID normalization
    ├── overflow.py          # Context overflow detection
    ├── json_parse.py        # Streaming + repair JSON parsing
    ├── diagnostics.py       # Diagnostic creation + attachment
    ├── headers.py           # Auth header construction
    ├── validation.py        # Tool argument validation + StringEnum
    └── retry.py             # RetryPolicy with exponential backoff
```

---

## Architecture

`piai` is a direct Python port of the TypeScript `@earendil-works/pi-ai` package. Key design decisions:

- **`KnownApi` enum** — all models route via typed enum values, never raw strings
- **`Model.compat` flags** — per-model quirk flags (e.g. `batch_tool_results`, `no_tool_streaming`) instead of scattered `if/else` in shared code
- **`Model.thinking_level_map`** — per-model thinking parameter overrides, avoids hardcoded budgets
- **`EventStream`** — asyncio.Queue-based producer/consumer; `result()` returns the final `AssistantMessage` as an awaitable Future
- **Lazy loading** — provider SDKs imported via `importlib` on first use only
- **No `complete_sync()`** — `asyncio.run()` raises `RuntimeError` inside running loops (FastAPI, Jupyter); callers manage their own event loop

---

## Roadmap

**Phase 1** (current): Anthropic, OpenAI Completions, Google Vertex AI, Faux mock provider

**Phase 2**: OpenAI Responses API, Mistral, AWS Bedrock, Google AI Studio

**Phase 3**: Azure OpenAI, Cloudflare AI Gateway, OAuth
