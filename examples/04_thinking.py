"""
examples/04_thinking.py — Extended reasoning / thinking mode.

Shows: ThinkingOptions, ThinkingDeltaEvent, ThinkingContent in final message
Requires: ANTHROPIC_API_KEY (claude-sonnet-4-5 supports extended thinking)
"""
import asyncio
import piai


async def main():
    model = piai.get_model("anthropic", "claude-sonnet-4-5")

    ctx = piai.MessagesContext(
        messages=[
            piai.UserMessage(content=[piai.TextContent(
                text="A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? "
                     "Think carefully before answering."
            )])
        ],
    )

    options = piai.StreamOptions(
        max_tokens=4096,
        thinking=piai.ThinkingOptions(level="medium"),  # budget: 8192 tokens
    )

    print("Streaming with thinking enabled (medium budget)...")
    print("=" * 50)

    thinking_text = ""
    answer_text = ""

    async for event in piai.stream(model, ctx, options):
        if isinstance(event, piai.ThinkingStartEvent):
            print("\n[THINKING]")
        elif isinstance(event, piai.ThinkingDeltaEvent):
            thinking_text += event.thinking
            print(event.thinking, end="", flush=True)
        elif isinstance(event, piai.ThinkingEndEvent):
            print("\n[/THINKING]\n")
        elif isinstance(event, piai.TextStartEvent):
            print("[ANSWER]")
        elif isinstance(event, piai.TextDeltaEvent):
            answer_text += event.text
            print(event.text, end="", flush=True)
        elif isinstance(event, piai.TextEndEvent):
            print()

    print("=" * 50)
    print(f"Thinking length: {len(thinking_text)} chars")
    print(f"Answer length:   {len(answer_text)} chars")


async def main_simple():
    """Same example using SimpleStreamOptions for convenience."""
    model = piai.get_model("anthropic", "claude-sonnet-4-5")

    ctx = piai.MessagesContext(
        messages=[
            piai.UserMessage(content=[piai.TextContent(
                text="What's the most efficient sorting algorithm for nearly-sorted data? Reason step by step."
            )])
        ],
    )

    # SimpleStreamOptions.reasoning applies to ALL providers uniformly
    options = piai.SimpleStreamOptions(
        max_tokens=2048,
        reasoning="low",
    )

    msg = await piai.complete_simple(model, ctx, options)

    for part in msg.content:
        if isinstance(part, piai.ThinkingContent):
            print(f"[Thinking ({len(part.thinking)} chars)]\n")
        elif isinstance(part, piai.TextContent):
            print(part.text)


if __name__ == "__main__":
    print("--- Example 1: Streaming thinking events ---\n")
    asyncio.run(main())
    print("\n--- Example 2: SimpleStreamOptions.reasoning ---\n")
    asyncio.run(main_simple())
