"""
examples/10_custom_provider.py — Register a custom provider.

Shows: implementing ApiProvider, register_api_provider, KnownApi routing
No API key required.
"""
import asyncio
import piai
from piai.api_registry import KnownApi, register_api_provider
from piai.models import Model, register_model
from piai.types import (
    AssistantMessage,
    AssistantMessageMetadata,
    DoneEvent,
    MessagesContext,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    TokenUsage,
)
from piai.utils.event_stream import EventStream
from piai.providers.simple_options import build_base_options


class EchoProvider:
    """
    A trivial provider that echoes the last user message back.
    Demonstrates the minimal interface needed to implement a provider.
    """

    def stream(
        self,
        model: Model,
        context: MessagesContext,
        options: StreamOptions | None = None,
    ) -> EventStream:
        es = EventStream()
        asyncio.get_event_loop().create_task(self._run(es, context))
        return es

    def stream_simple(
        self,
        model: Model,
        context: MessagesContext,
        options: SimpleStreamOptions | None = None,
    ) -> EventStream:
        return self.stream(model, context, build_base_options(options or SimpleStreamOptions()))

    async def _run(self, es: EventStream, context: MessagesContext) -> None:
        from piai.types import UserMessage, TextContent as TC

        last_user = next(
            (m for m in reversed(context.messages) if isinstance(m, UserMessage)),
            None,
        )
        text = last_user.content[0].text if last_user and last_user.content else "(no input)"
        response = f"[ECHO] {text}"

        es.emit(StartEvent())
        es.emit(TextStartEvent())
        for char in response:
            es.emit(TextDeltaEvent(text=char))
            await asyncio.sleep(0.005)
        es.emit(TextEndEvent())

        msg = AssistantMessage(
            content=[TextContent(text=response)],
            metadata=AssistantMessageMetadata(
                usage=TokenUsage(input_tokens=len(text.split()), output_tokens=len(response.split())),
                stop_reason="end_turn",
            ),
        )
        es.emit(DoneEvent(message=msg))
        es.done(msg)


async def main():
    # 1. Register the custom provider under the FAUX slot (or a KnownApi you own)
    register_api_provider(KnownApi.FAUX, EchoProvider())

    # 2. Register a model that routes to it
    echo_model = Model(
        id="echo",
        name="Echo Provider",
        api=KnownApi.FAUX,
        provider="faux",
        context_window=10000,
    )
    register_model(echo_model)

    # 3. Use it via the standard API
    ctx = piai.MessagesContext(
        messages=[piai.UserMessage(content=[piai.TextContent(text="Hello from custom provider!")])]
    )

    print("Streaming from custom EchoProvider:")
    async for event in piai.stream(echo_model, ctx):
        if isinstance(event, piai.TextDeltaEvent):
            print(event.text, end="", flush=True)
    print()

    msg = await piai.complete(echo_model, ctx)
    print(f"\nResult: {msg.content[0].text}")
    print(f"Tokens: {msg.metadata.usage.input_tokens} in / {msg.metadata.usage.output_tokens} out")


if __name__ == "__main__":
    asyncio.run(main())
