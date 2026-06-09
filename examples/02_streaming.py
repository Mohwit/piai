"""
examples/02_streaming.py — Real-time token streaming.

Shows: stream_simple, async for event loop, event type handling
Requires: ANTHROPIC_API_KEY
"""
import asyncio
import piai


async def main():
    model = piai.get_model("anthropic", "claude-haiku-4-5")

    ctx = piai.MessagesContext(
        messages=[
            piai.UserMessage(content=[piai.TextContent(text="Write a haiku about Python programming.")])
        ],
    )

    print("Streaming response:")
    print("-" * 40)

    event_stream = piai.stream_simple(model, ctx)

    async for event in event_stream:
        if isinstance(event, piai.TextDeltaEvent):
            print(event.text, end="", flush=True)
        elif isinstance(event, piai.DoneEvent):
            print()  # newline after stream

    # result() returns the assembled AssistantMessage
    msg = await event_stream.result()
    print("-" * 40)
    print(f"Stop reason: {msg.metadata.stop_reason}")
    print(f"Tokens: {msg.metadata.usage.input_tokens} in / {msg.metadata.usage.output_tokens} out")


if __name__ == "__main__":
    asyncio.run(main())
