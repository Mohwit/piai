"""
examples/06_vertex_ai.py — Google Vertex AI (Gemini models).

Shows: ADC auth setup, Gemini 2.5 thinking, tool calling on Vertex
Requires:
  - pip install piai[vertex]
  - gcloud auth application-default login
  - GOOGLE_CLOUD_PROJECT environment variable
"""
import asyncio
import json
import os
import piai


async def gemini_basic():
    """Basic Gemini 2.5 Flash completion."""
    model = piai.get_model("google-vertex", "gemini-2.5-flash")

    ctx = piai.MessagesContext(
        messages=[
            piai.UserMessage(content=[piai.TextContent(
                text="What are the top 3 differences between Python and Go?"
            )])
        ],
        system_prompt="You are a concise technical expert.",
    )

    msg = await piai.complete_simple(model, ctx, piai.SimpleStreamOptions(max_tokens=512))

    print("Gemini 2.5 Flash response:")
    for part in msg.content:
        if isinstance(part, piai.TextContent):
            print(part.text)
    print(f"\nTokens: {msg.metadata.usage.input_tokens} in / {msg.metadata.usage.output_tokens} out")


async def gemini_with_thinking():
    """Gemini 2.5 Pro with thinking enabled."""
    model = piai.get_model("google-vertex", "gemini-2.5-pro")

    ctx = piai.MessagesContext(
        messages=[
            piai.UserMessage(content=[piai.TextContent(
                text="Prove that the square root of 2 is irrational."
            )])
        ],
    )

    options = piai.StreamOptions(
        max_tokens=4096,
        thinking=piai.ThinkingOptions(level="medium"),  # thinking_budget: 8192
    )

    print("\nGemini 2.5 Pro with thinking:")
    thinking_shown = False

    async for event in piai.stream(model, ctx, options):
        if isinstance(event, piai.ThinkingStartEvent):
            print("[THINKING...]")
        elif isinstance(event, piai.ThinkingDeltaEvent) and not thinking_shown:
            # Just show first 200 chars of thinking
            print(event.thinking[:200] + "...", end="", flush=True)
            thinking_shown = True
        elif isinstance(event, piai.ThinkingEndEvent):
            print("\n[/THINKING]\n")
        elif isinstance(event, piai.TextDeltaEvent):
            print(event.text, end="", flush=True)

    print()


async def gemini_tool_calling():
    """Gemini 2.5 Flash with tool calling."""
    model = piai.get_model("google-vertex", "gemini-2.5-flash")

    # Note: Vertex requires flat schemas — no anyOf/const/$defs
    search_tool = piai.Tool(
        name="search_database",
        description="Search a product database by category and price range.",
        input={
            "type": "object",
            "properties": {
                "category": {
                    "type": "STRING",  # Vertex-style (convert_schema_to_vertex handles this automatically)
                    "description": "Product category to search",
                },
                "max_price": {
                    "type": "NUMBER",
                    "description": "Maximum price in USD",
                },
            },
            "required": ["category"],
        },
    )

    ctx = piai.MessagesContext(
        messages=[
            piai.UserMessage(content=[piai.TextContent(
                text="Find me laptops under $1000."
            )])
        ],
    )

    options = piai.SimpleStreamOptions(tools=[search_tool], max_tokens=512)
    msg = await piai.complete_simple(model, ctx, options)

    print("\nGemini tool call result:")
    for part in msg.content:
        if isinstance(part, piai.ToolCallContent):
            print(f"Tool called: {part.name}")
            print(f"Arguments:   {json.dumps(part.input, indent=2)}")
            print(f"Tool ID:     {part.id}")
        elif isinstance(part, piai.TextContent):
            print(f"Text: {part.text}")


if __name__ == "__main__":
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("Set GOOGLE_CLOUD_PROJECT and run: gcloud auth application-default login")
        raise SystemExit(1)

    asyncio.run(gemini_basic())
    asyncio.run(gemini_with_thinking())
    asyncio.run(gemini_tool_calling())
