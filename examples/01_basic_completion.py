"""
examples/01_basic_completion.py — Simplest possible usage.

Shows: get_model, MessagesContext, complete_simple
Requires: ANTHROPIC_API_KEY
"""
import asyncio
import piai


async def main():
    model = piai.get_model("anthropic", "claude-haiku-4-5")

    ctx = piai.MessagesContext(
        messages=[
            piai.UserMessage(content=[piai.TextContent(text="What is 2 + 2? Answer in one sentence.")])
        ],
        system_prompt="You are a concise assistant.",
    )

    msg = await piai.complete_simple(model, ctx, piai.SimpleStreamOptions(max_tokens=128))

    print(msg.content[0].text)
    print(f"Tokens: {msg.metadata.usage.input_tokens} in / {msg.metadata.usage.output_tokens} out")
    print(f"Cost: ${msg.metadata.cost:.6f}" if msg.metadata.cost else "Cost: N/A")


if __name__ == "__main__":
    asyncio.run(main())
