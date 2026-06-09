"""
examples/03_tool_calling.py — Tool / function calling with agentic loop.

Shows: Tool definition, ToolCallContent, ToolResultMessage, multi-turn loop
Requires: ANTHROPIC_API_KEY
"""
import asyncio
import json
import piai


# --- Tool implementations ---

def get_weather(location: str, unit: str = "celsius") -> str:
    """Fake weather implementation for demonstration."""
    weather_db = {
        "london":    {"temp": 15, "condition": "cloudy"},
        "new york":  {"temp": 22, "condition": "sunny"},
        "tokyo":     {"temp": 28, "condition": "humid"},
        "sydney":    {"temp": 18, "condition": "partly cloudy"},
    }
    data = weather_db.get(location.lower(), {"temp": 20, "condition": "unknown"})
    temp = data["temp"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32)
    return json.dumps({"location": location, "temperature": temp, "unit": unit, "condition": data["condition"]})


def calculate(expression: str) -> str:
    """Safe arithmetic evaluation."""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid characters in expression"})
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- Tool schemas (JSON Schema format) ---

TOOLS = [
    piai.Tool(
        name="get_weather",
        description="Get the current weather for a city.",
        input={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'London'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["location"],
        },
    ),
    piai.Tool(
        name="calculate",
        description="Evaluate a mathematical expression and return the result.",
        input={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 2'"},
            },
            "required": ["expression"],
        },
    ),
]

TOOL_HANDLERS = {
    "get_weather": get_weather,
    "calculate": calculate,
}


async def run_agent(user_query: str) -> str:
    """Simple agentic loop: keep calling until model returns end_turn."""
    model = piai.get_model("anthropic", "claude-haiku-4-5")
    options = piai.SimpleStreamOptions(tools=TOOLS, max_tokens=1024)

    ctx = piai.MessagesContext(
        messages=[piai.UserMessage(content=[piai.TextContent(text=user_query)])],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )

    turn = 0
    while turn < 10:  # safety cap
        turn += 1
        msg = await piai.complete_simple(model, ctx, options)
        ctx.messages.append(msg)

        # Find tool calls in the response
        tool_calls = [p for p in msg.content if isinstance(p, piai.ToolCallContent)]

        if not tool_calls:
            # No tool calls → final text response
            text_parts = [p.text for p in msg.content if isinstance(p, piai.TextContent)]
            return "\n".join(text_parts)

        # Execute each tool call and append results
        print(f"[Turn {turn}] Calling {len(tool_calls)} tool(s)...")
        for tc in tool_calls:
            handler = TOOL_HANDLERS.get(tc.name)
            if handler:
                result_text = handler(**tc.input)
                print(f"  {tc.name}({tc.input}) → {result_text}")
            else:
                result_text = json.dumps({"error": f"Unknown tool: {tc.name}"})

            ctx.messages.append(piai.ToolResultMessage(
                content=[piai.TextContent(text=result_text)],
                tool_use_id=tc.id,
                name=tc.name,
            ))

    return "[Max turns reached]"


async def main():
    query = "What's the weather in London and Tokyo? Also, what's 15% of 847?"
    print(f"Query: {query}\n")
    answer = await run_agent(query)
    print(f"\nFinal answer:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
