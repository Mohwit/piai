# Pi-AI Providers — Detailed Analysis

> Per-provider breakdown of implementation details relevant for Python port

---

## 1. Anthropic (anthropic-messages)

### Auth
- API key: `ANTHROPIC_API_KEY` env var
- OAuth: `getOAuthApiKey("anthropic")` (GitHub Copilot passthrough)
- Headers: `anthropic-beta: interleaved-thinking-2025-05-14` for adaptive thinking

### Thinking Support
Two modes:
- **Adaptive thinking** (claude-3-7+): `thinking: { type: "enabled", budget_tokens: N }`
- **Budget thinking** (older): `thinking: { type: "budget_tokens", budget_tokens: N }`

Thinking level → budget mapping:
| Level | Budget |
|-------|--------|
| minimal | 1,024 |
| low | 2,048 |
| medium | 8,192 |
| high | 16,384 |
| xhigh | 16,384 |

### Message Conversion
- Tool results are batched: multiple `ToolResultMessage` → single user message with multiple `tool_result` content blocks
- Cache control added to last user message and tool definitions
- Thinking signatures preserved in `cache_control` metadata

### SSE Parsing
Events to handle:
- `message_start` → extract input token count
- `content_block_start` → `type: text | tool_use | thinking`
- `content_block_delta` → `type: text_delta | input_json_delta | thinking_delta | signature_delta`
- `content_block_stop`
- `message_delta` → extract stop reason + output tokens
- `message_stop`

### Python SDK Notes
```python
import anthropic
client = anthropic.AsyncAnthropic()

async with client.messages.stream(...) as stream:
    async for event in stream:
        ...  # same event types as above
```

---

## 2. OpenAI Completions (openai-completions)

### Compatible Providers (detected from baseUrl/provider name)
- OpenRouter: `openrouter.ai`
- Together: `together.xyz`
- Cerebras: `cerebras.ai`
- DeepSeek: `deepseek.com` → reasoning_content field
- Qwen: `dashscope.aliyuncs.com`
- Groq: `groq.com` → no tool streaming
- NVIDIA NIM: `build.nvidia.com`
- Ollama: `localhost`
- LM Studio: `localhost:1234`

### Quirk Handling
- **DeepSeek**: thinking in `delta.reasoning_content` instead of standard thinking blocks
- **Groq**: no streaming for tool arguments (batch only)
- **OpenRouter**: special header `x-title` and `http-referer`
- **Anthropic-compatible proxy**: cache control in message content

### Tool Call Normalization
OpenAI tool call IDs can be 450+ chars. Normalize:
```python
if len(tc_id) > 64:
    tc_id = short_hash(tc_id, 64)
```

### Message Format
```python
[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": [{"type": "text", "text": "..."}]},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}
    ]},
    {"role": "tool", "content": "result text", "tool_call_id": "..."},
]
```

---

## 3. OpenAI Responses API (openai-responses)

New API with extended features:
- Session IDs for prompt cache retention
- Reasoning items with text summaries
- Integrated web search tool

### Session Caching
```python
# short retention: 10 min, long: 1 hour
response = client.responses.create(
    model=model_id,
    previous_response_id=options.cache.session_id,  # chain responses
    ...
)
# next request: pass response.id as previous_response_id
```

### Service Tiers
- `flex`: 0.5x cost multiplier, slower
- `default`: 1x
- `priority`: 2-2.5x, faster

---

## 4. Google Generative AI (google-generative-ai)

### Thinking Support
- **Gemini 2.0 Flash Thinking**: `thinkingConfig: { thinkingBudget: N }` (budget-based)
- **Gemini 2.5 Pro/Flash**: `thinkingConfig: { thinkingBudget: N }` or effort level
- **Gemini 3+**: `thinkingConfig: { thinkingMode: "ENABLED" }` (adaptive, like Anthropic)

### Tool Call IDs
Google doesn't natively assign IDs. Generate:
```python
f"{function_name}_{timestamp}_{counter}"
```

### Message Format
```python
contents = [
    {"role": "user", "parts": [{"text": "..."}]},
    {"role": "model", "parts": [
        {"text": "..."},
        {"functionCall": {"name": "...", "args": {...}}},
    ]},
    {"role": "user", "parts": [
        {"functionResponse": {"name": "...", "response": {...}}},
    ]},
]
```

### Multimodal Function Responses (Gemini 3+)
Function results can include image parts alongside text.

### Type Mapping (JSON Schema → Google format)
```python
type_map = {
    "string": "STRING",
    "number": "NUMBER", 
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}
```
Note: Google doesn't support `anyOf`/`const` for enums — use custom StringEnum.

### Stop Reason Mapping
```python
google_to_standard = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "other",
    "TOOL_USE": "tool_use",
}
```

### Python SDK
```python
import google.generativeai as genai
genai.configure(api_key=api_key)

model = genai.GenerativeModel(model_name)
response = await model.generate_content_async(
    contents,
    generation_config=genai.GenerationConfig(
        max_output_tokens=...,
        temperature=...,
    ),
    stream=True,
)
async for chunk in response:
    ...
```

---

## 5. Google Vertex AI (google-vertex-ai)

Same as Google Generative AI but:
- Uses Application Default Credentials (ADC)
- `GOOGLE_APPLICATION_CREDENTIALS` env var or `~/.config/gcloud/`
- Region required: `us-central1` (default)
- Base URL: `https://{region}-aiplatform.googleapis.com/v1/...`

### Python SDK
```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project=project_id, location=region)
model = GenerativeModel(model_id)
```

---

## 6. Mistral (mistral-api)

### Auth
- `MISTRAL_API_KEY` env var
- Session affinity: `x-affinity` header for KV-cache reuse (send same value each turn)

### Thinking Support
- Newer models: `reasoning_effort: "low"|"medium"|"high"`
- Older models: `reasoning_mode: true/false`

### Tool Call IDs
Mistral uses 9-char IDs. Normalize incoming long IDs:
```python
def normalize_mistral_id(id: str) -> str:
    return id[:9] if len(id) > 9 else id
```

### Python SDK
```python
from mistralai import Mistral
client = Mistral(api_key=api_key)

stream = await client.chat.stream_async(
    model=model_id,
    messages=messages,
    tools=tools,
)
async for chunk in stream:
    ...
```

---

## 7. Amazon Bedrock (bedrock-converse-stream)

### Auth (AWS credential chain, in priority order)
1. `AWS_BEARER_TOKEN_BEDROCK` (simple bearer token)
2. `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_SESSION_TOKEN`
3. `~/.aws/credentials` profile
4. Container credentials (ECS/Lambda)
5. EC2 instance metadata

### Region Resolution
1. Extract from ARN in model baseUrl (e.g., `arn:aws:bedrock:us-east-1::model/...`)
2. Explicit `options.region`
3. `AWS_REGION` or `AWS_DEFAULT_REGION` env var
4. SDK defaults

### Thinking Support (Claude on Bedrock)
Same two modes as direct Anthropic:
- Adaptive: `reasoningConfig: { type: "enabled" }`
- Budget: `reasoningConfig: { type: "disabled" }` + `thinking_budget_tokens` in inferenceConfig

### Prompt Caching (Claude 3.5+ only)
Add `cachePoint: { type: "default" }` marker in message list.

### Python SDK
```python
import boto3

client = boto3.client("bedrock-runtime", region_name=region)
response = client.converse_stream(
    modelId=model_id,
    messages=messages,
    system=[{"text": system_prompt}],
    inferenceConfig={"maxTokens": ..., "temperature": ...},
    toolConfig={"tools": [...], "toolChoice": {...}},
)

stream = response["stream"]
for event in stream:
    if "contentBlockStart" in event: ...
    elif "contentBlockDelta" in event: ...
    elif "messageStop" in event: ...
```

### Event Mapping
```python
bedrock_to_standard = {
    "contentBlockStart": "content_block_start",
    "contentBlockDelta": "content_block_delta", 
    "contentBlockStop": "content_block_stop",
    "messageStart": "message_start",
    "messageStop": "message_stop",
    "metadata": "usage",
}
```

---

## 8. Azure OpenAI (azure-openai-responses)

Same as OpenAI Responses but:
- Base URL: `https://{deployment}.openai.azure.com/`
- Auth: `AZURE_OPENAI_API_KEY` or AAD token
- API version required: `?api-version=2024-10-01-preview`

### Python SDK
```python
from openai import AsyncAzureOpenAI

client = AsyncAzureOpenAI(
    api_key=api_key,
    azure_endpoint=base_url,
    api_version="2024-10-01-preview",
)
```

---

## 9. Cloudflare AI Gateway

- Acts as a proxy/gateway in front of other providers
- Base URL: `https://gateway.ai.cloudflare.com/v1/{account_id}/...`
- URL variables: `{CLOUDFLARE_ACCOUNT_ID}` replaced from env vars
- Routes to underlying provider (OpenAI, Anthropic, etc.) via path

---

## 10. Faux (Mock Provider)

For testing without real API calls:
```python
class FauxProvider(ApiProvider):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self._idx = 0
    
    def stream(self, model, context, options=None):
        s = EventStream()
        response = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        asyncio.create_task(self._emit(s, response))
        return s
    
    async def _emit(self, s, response):
        s.emit(StartEvent())
        s.emit(TextStartEvent())
        for char in response:
            s.emit(TextDeltaEvent(text=char))
            await asyncio.sleep(0.01)  # simulate streaming
        s.emit(TextEndEvent())
        msg = AssistantMessage(content=[TextContent(text=response)])
        s.emit(DoneEvent(message=msg))
        s.done(msg)
```

---

## 11. Cross-Provider Message Compatibility Matrix

| Feature | Anthropic | OpenAI | Google | Mistral | Bedrock |
|---------|-----------|--------|--------|---------|---------|
| System prompt | ✅ top-level | ✅ system role | ✅ system instruction | ✅ system role | ✅ system field |
| Image input | ✅ | ✅ | ✅ | ✅ | ✅ Claude only |
| Thinking blocks | ✅ native | ✅ reasoning_content | ✅ thinkingContent | ✅ reasoning | ✅ Claude only |
| Tool calling | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tool streaming | ✅ | ✅ | ✅ | ✅ | Partial |
| Prompt caching | ✅ | ✅ | ❌ | ❌ | ✅ Claude 3.5+ |
| Multi-turn tool | ✅ | ✅ | ✅ | ✅ | ✅ |
| Parallel tool calls | ✅ | ✅ | ✅ | ✅ | ✅ |

---

*Last updated: 2026-06-09*
