"""
piai/providers/faux.py — Mock provider for testing without live API keys.
Mirrors faux.ts from @earendil-works/pi-ai.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from piai.types import (
    AssistantMessage,
    AssistantMessageMetadata,
    MessagesContext,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    DoneEvent,
    TokenUsage,
)
from piai.utils.event_stream import EventStream
from piai.providers.simple_options import build_base_options

AssistantMessageEventStream = EventStream[object, AssistantMessage]


class FauxProvider:
    """
    Mock provider — emits synthetic events without making any API calls.
    Useful for unit tests and integration tests that don't need live credentials.

    Usage:
        provider = FauxProvider(["Hello!", "World!"])
        # Registered automatically via register_builtins; override for custom responses:
        from piai.api_registry import register_api_provider, KnownApi
        register_api_provider(KnownApi.FAUX, FauxProvider(["custom response"]))
    """

    def __init__(
        self,
        responses: Optional[list[str]] = None,
        delay: float = 0.005,
    ) -> None:
        self.responses = responses or ["Hello from faux provider."]
        self.delay = delay  # seconds between character emissions
        self.call_count = 0
        self.last_context: Optional[MessagesContext] = None
        self.last_options: Optional[StreamOptions] = None

    def stream(
        self,
        model,
        context: MessagesContext,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream:
        self.last_context = context
        self.last_options = options

        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        es: AssistantMessageEventStream = EventStream()
        asyncio.get_event_loop().create_task(self._run(es, response))
        return es

    def stream_simple(
        self,
        model,
        context: MessagesContext,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessageEventStream:
        return self.stream(model, context, build_base_options(options or SimpleStreamOptions()))

    async def _run(self, es: AssistantMessageEventStream, response: str) -> None:
        try:
            es.emit(StartEvent())
            es.emit(TextStartEvent())

            for char in response:
                es.emit(TextDeltaEvent(text=char))
                if self.delay > 0:
                    await asyncio.sleep(self.delay)

            es.emit(TextEndEvent())

            msg = AssistantMessage(
                content=[TextContent(text=response)],
                metadata=AssistantMessageMetadata(
                    usage=TokenUsage(
                        input_tokens=10,
                        output_tokens=len(response.split()),
                    ),
                    stop_reason="end_turn",
                ),
            )
            es.emit(DoneEvent(message=msg))
            es.done(msg)
        except Exception as exc:
            from piai.types import ErrorEvent
            es.emit(ErrorEvent(error=exc))
            es.error(exc)
