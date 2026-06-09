"""
examples/07_multi_turn.py — Multi-turn conversation (chat history).

Shows: building up MessagesContext across turns, provider-agnostic conversation
Requires: ANTHROPIC_API_KEY
"""
import asyncio
import piai


async def chat_session():
    """Simple interactive-style chat that builds up history."""
    model = piai.get_model("anthropic", "claude-haiku-4-5")

    ctx = piai.MessagesContext(
        messages=[],
        system_prompt="You are a friendly Python tutor. Keep answers short.",
    )

    turns = [
        "What is a decorator in Python?",
        "Can you show me a simple example?",
        "How is that different from a class-based decorator?",
    ]

    for user_text in turns:
        print(f"\nUser: {user_text}")
        print("Assistant: ", end="", flush=True)

        # Append user turn
        ctx.messages.append(piai.UserMessage(
            content=[piai.TextContent(text=user_text)]
        ))

        # Stream response
        es = piai.stream_simple(model, ctx, piai.SimpleStreamOptions(max_tokens=512))
        async for event in es:
            if isinstance(event, piai.TextDeltaEvent):
                print(event.text, end="", flush=True)
        print()

        # Append assistant turn to history
        msg = await es.result()
        ctx.messages.append(msg)

    print(f"\n[Session complete — {len(ctx.messages)} messages in history]")


async def provider_switch_example():
    """
    Show that the same MessagesContext works across providers.
    Start on Claude, continue on GPT-4o.
    """
    ctx = piai.MessagesContext(
        messages=[
            piai.UserMessage(content=[piai.TextContent(text="My name is Alex. Remember that.")])
        ],
        system_prompt="You are a helpful assistant.",
    )

    # Turn 1: Anthropic
    claude = piai.get_model("anthropic", "claude-haiku-4-5")
    msg1 = await piai.complete_simple(claude, ctx, piai.SimpleStreamOptions(max_tokens=128))
    ctx.messages.append(msg1)
    print("Claude:", msg1.content[0].text if msg1.content else "")

    # Turn 2: OpenAI (same context — piai handles cross-provider compatibility)
    ctx.messages.append(piai.UserMessage(
        content=[piai.TextContent(text="What is my name?")]
    ))
    gpt = piai.get_model("openai", "gpt-4o-mini")
    msg2 = await piai.complete_simple(gpt, ctx, piai.SimpleStreamOptions(max_tokens=128))
    print("GPT-4o mini:", msg2.content[0].text if msg2.content else "")


if __name__ == "__main__":
    print("=== Multi-turn chat session ===")
    asyncio.run(chat_session())

    print("\n=== Cross-provider conversation ===")
    asyncio.run(provider_switch_example())
