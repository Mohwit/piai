"""
examples/08_faux_provider.py — Mock provider for testing without API keys.

Shows: FauxProvider, register_api_provider override, writing tests
No API key required.
"""
import asyncio
import piai
from piai.providers.faux import FauxProvider
from piai.api_registry import KnownApi, register_api_provider


async def basic_faux():
    """Use the built-in faux model (no credentials needed)."""
    model = piai.get_model("faux", "faux")

    ctx = piai.MessagesContext(
        messages=[piai.UserMessage(content=[piai.TextContent(text="Hello!")])]
    )

    msg = await piai.complete_simple(model, ctx)
    print("Faux response:", msg.content[0].text)


async def custom_responses():
    """Register a custom FauxProvider with scripted responses."""
    # Override the anthropic slot with a mock for testing
    mock = FauxProvider(
        responses=[
            "The answer is 42.",
            "I am a mock assistant.",
            "Testing, 1, 2, 3.",
        ],
        delay=0,  # instant (no sleep between chars)
    )
    register_api_provider(KnownApi.ANTHROPIC, mock)

    model = piai.get_model("anthropic", "claude-haiku-4-5")
    ctx = piai.MessagesContext(
        messages=[piai.UserMessage(content=[piai.TextContent(text="Any question")])]
    )

    # Cycles through the response list
    for i in range(3):
        msg = await piai.complete_simple(model, ctx)
        print(f"Call {i+1}: {msg.content[0].text}")

    print(f"Total calls made: {mock.call_count}")


async def test_streaming_events():
    """Verify that all expected events fire in the right order."""
    model = piai.get_model("faux", "faux")
    ctx = piai.MessagesContext(
        messages=[piai.UserMessage(content=[piai.TextContent(text="hi")])]
    )

    event_types = []
    async for event in piai.stream_simple(model, ctx, piai.SimpleStreamOptions()):
        event_types.append(event.type)

    print("Events fired:", event_types)

    expected_order = ["start", "text_start", "text_delta", "text_end", "done"]
    # text_delta fires once per character; strip repeated ones for comparison
    unique_seq = []
    for t in event_types:
        if not unique_seq or unique_seq[-1] != t:
            unique_seq.append(t)

    assert unique_seq == expected_order, f"Unexpected event order: {unique_seq}"
    print("Event order verified.")


if __name__ == "__main__":
    asyncio.run(basic_faux())
    asyncio.run(custom_responses())
    asyncio.run(test_streaming_events())
