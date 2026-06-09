"""
piai/stream.py — Public API: stream(), complete(), stream_simple(), complete_simple().
Mirrors stream.ts from @earendil-works/pi-ai.
"""
from __future__ import annotations

from typing import Optional

from piai.api_registry import KnownApi, get_api_provider
from piai.models import Model
from piai.types import (
    AssistantMessage,
    MessagesContext,
    SimpleStreamOptions,
    StreamOptions,
)
from piai.utils.event_stream import EventStream

AssistantMessageEventStream = EventStream[object, AssistantMessage]


def _get_provider(model: Model):
    """Resolve provider, initializing built-ins if needed."""
    provider = get_api_provider(model.api)
    if provider is None:
        # Built-ins not yet registered — initialize now
        from piai.providers.register_builtins import register_builtin_api_providers
        register_builtin_api_providers()
        provider = get_api_provider(model.api)
    if provider is None:
        raise ValueError(
            f"No provider registered for API {model.api!r}. "
            f"Call register_api_provider({model.api!r}, ...) first."
        )
    return provider


def stream(
    model: Model,
    context: MessagesContext,
    options: Optional[StreamOptions] = None,
) -> AssistantMessageEventStream:
    """
    Start a streaming response. Returns an EventStream you can iterate with `async for`.

    Example:
        async for event in stream(model, ctx):
            if event.type == "text_delta":
                print(event.text, end="", flush=True)
        msg = await event_stream.result()
    """
    provider = _get_provider(model)
    return provider.stream(model, context, options)


def stream_simple(
    model: Model,
    context: MessagesContext,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    """
    Start a streaming response using SimpleStreamOptions.
    Converts to StreamOptions via build_base_options() before passing to provider.
    """
    provider = _get_provider(model)
    return provider.stream_simple(model, context, options)


async def complete(
    model: Model,
    context: MessagesContext,
    options: Optional[StreamOptions] = None,
) -> AssistantMessage:
    """
    Blocking async call — drains the event stream and returns the final AssistantMessage.
    Use `await complete(...)` inside an async context, or `asyncio.run(complete(...))` elsewhere.

    Note: complete_sync() is intentionally excluded — asyncio.run() raises RuntimeError
    inside running event loops (FastAPI, Jupyter). Use await complete() instead.
    """
    es = stream(model, context, options)
    async for _ in es:
        pass
    return await es.result()


async def complete_simple(
    model: Model,
    context: MessagesContext,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessage:
    """
    Blocking async call with SimpleStreamOptions.
    """
    es = stream_simple(model, context, options)
    async for _ in es:
        pass
    return await es.result()
