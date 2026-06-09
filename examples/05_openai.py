"""
examples/05_openai.py — OpenAI provider (GPT-4o and o4-mini reasoning model).

Shows: OpenAI chat completions, reasoning_effort for o* models, provider switch
Requires: OPENAI_API_KEY
"""
import asyncio
import piai


async def gpt4o_example():
    """Standard GPT-4o completion."""
    model = piai.get_model("openai", "gpt-4o")

    ctx = piai.MessagesContext(
        messages=[
            piai.UserMessage(content=[piai.TextContent(text="Explain async/await in Python in 3 bullet points.")])
        ],
        system_prompt="You are a concise technical writer.",
    )

    msg = await piai.complete_simple(model, ctx, piai.SimpleStreamOptions(max_tokens=512))

    print("GPT-4o response:")
    print(msg.content[0].text)
    print(f"\nTokens: {msg.metadata.usage.input_tokens} in / {msg.metadata.usage.output_tokens} out")
    print(f"Cost: ${msg.metadata.cost:.6f}" if msg.metadata.cost else "Cost: N/A")


async def o4_mini_reasoning():
    """o4-mini with reasoning effort — same SimpleStreamOptions interface."""
    model = piai.get_model("openai", "o4-mini")

    ctx = piai.MessagesContext(
        messages=[
            piai.UserMessage(content=[piai.TextContent(
                text="Design a system that can handle 1M requests/second with < 10ms p99 latency. "
                     "What are the key architectural decisions?"
            )])
        ],
    )

    # reasoning="high" maps to reasoning_effort="high" for o* models automatically
    options = piai.SimpleStreamOptions(reasoning="high", max_tokens=2048)

    print("\no4-mini (high reasoning effort) response:")
    print("Streaming...")

    async for event in piai.stream_simple(model, ctx, options):
        if isinstance(event, piai.TextDeltaEvent):
            print(event.text, end="", flush=True)

    print()


async def streaming_with_all_events():
    """Show all event types from OpenAI streaming."""
    model = piai.get_model("openai", "gpt-4o-mini")

    ctx = piai.MessagesContext(
        messages=[piai.UserMessage(content=[piai.TextContent(text="Say hello in 5 languages.")])]
    )

    print("\nEvent-by-event from gpt-4o-mini:")
    async for event in piai.stream_simple(model, ctx, piai.SimpleStreamOptions(max_tokens=256)):
        print(f"  [{event.type}]", end=" ")
        if isinstance(event, piai.TextDeltaEvent):
            print(repr(event.text[:20]))
        elif isinstance(event, piai.DoneEvent):
            print(f"→ {event.message.metadata.stop_reason}")
        else:
            print()


if __name__ == "__main__":
    asyncio.run(gpt4o_example())
    asyncio.run(o4_mini_reasoning())
    asyncio.run(streaming_with_all_events())
