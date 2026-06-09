"""
examples/09_abort_and_retry.py — Abort signals and retry policies.

Shows: AbortSignal, asyncio.Event cancellation, RetryPolicy configuration
Requires: ANTHROPIC_API_KEY
"""
import asyncio
import piai
from piai.utils.abort import AbortSignal


async def abort_after_first_chunk():
    """Cancel a stream after receiving the first text delta."""
    model = piai.get_model("anthropic", "claude-haiku-4-5")
    signal = AbortSignal()

    ctx = piai.MessagesContext(
        messages=[piai.UserMessage(content=[piai.TextContent(
            text="Write a very long essay about the history of computing."
        )])]
    )

    options = piai.StreamOptions(
        max_tokens=2048,
        abort_signal=signal.event,  # pass the underlying asyncio.Event
    )

    print("Streaming (will abort after first text chunk)...")
    chunks_received = 0

    async for event in piai.stream(model, ctx, options):
        if isinstance(event, piai.TextDeltaEvent):
            chunks_received += 1
            print(f"Chunk {chunks_received}: {repr(event.text[:30])}")
            if chunks_received >= 3:
                signal.abort()
                print("Abort signal sent.")
                break


async def pre_cancelled_signal():
    """Stream with a signal that's already aborted — should not call the API."""
    model = piai.get_model("faux", "faux")
    signal = AbortSignal()
    signal.abort()  # abort before starting

    ctx = piai.MessagesContext(
        messages=[piai.UserMessage(content=[piai.TextContent(text="hello")])]
    )

    options = piai.StreamOptions(abort_signal=signal.event)
    es = piai.stream(model, ctx, options)

    try:
        async for event in es:
            print(f"Unexpected event: {event.type}")
        await es.result()
    except asyncio.CancelledError:
        print("Pre-cancelled signal correctly raised CancelledError.")


async def custom_retry_policy():
    """Configure retry behavior — useful for flaky connections."""
    model = piai.get_model("anthropic", "claude-haiku-4-5")

    ctx = piai.MessagesContext(
        messages=[piai.UserMessage(content=[piai.TextContent(text="Hello!")])]
    )

    options = piai.StreamOptions(
        max_tokens=64,
        retry_policy=piai.RetryPolicy(
            max_retries=3,        # retry up to 3 times
            delay_ms=500,         # start with 500ms delay
            backoff_multiplier=2.0,  # 500ms → 1s → 2s
            retryable_status=[429, 500, 502, 503],
        ),
    )

    msg = await piai.complete(model, ctx, options)
    print("Response with retry policy:", msg.content[0].text if msg.content else "")


if __name__ == "__main__":
    asyncio.run(abort_after_first_chunk())
    asyncio.run(pre_cancelled_signal())
    asyncio.run(custom_retry_policy())
