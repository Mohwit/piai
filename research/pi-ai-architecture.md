# Pi-AI TypeScript Package — Architecture Deep-Dive

> Research source: https://github.com/earendil-works/pi/tree/main/packages/ai
> Purpose: Replicate pi-ai in Python for a unified LLM provider / agent SDK

---

## 1. Overview & Design Philosophy

Pi-AI is a production-grade unified LLM abstraction layer with:
- 15+ LLM provider support (OpenAI, Anthropic, Google, Azure, Mistral, AWS Bedrock, Cloudflare, etc.)
- Unified streaming API with granular events
- Advanced reasoning/thinking support with configurable levels
- Tool/function calling with automatic validation
- Token and cost tracking
- Cross-provider compatibility including mid-conversation provider switching
- OAuth authentication
- Image generation and input
- Prompt caching across providers

### Design Principles
1. **Provider Abstraction Layer** — isolated implementations conforming to standard interfaces
2. **Lazy Loading** — providers loaded on-demand
3. **Type Safety** — TypeBox schemas for tools, generics for API types
4. **Streaming-First** — all responses stream by default; `complete()` consumes full response
5. **Configuration-Driven** — model `compat` flags handle provider-specific quirks
6. **Event-Driven** — granular events (text_delta, thinking_delta, toolcall_start, etc.)

---

## 2. Directory Structure

```
packages/ai/src/
├── providers/
│   ├── anthropic.ts
│   ├── openai-responses.ts
│   ├── openai-completions.ts
│   ├── google.ts
│   ├── google-vertex.ts
│   ├── mistral.ts
│   ├── amazon-bedrock.ts
│   ├── azure-openai-responses.ts
│   ├── cloudflare.ts
│   ├── faux.ts                  # mock provider for testing
│   ├── transform-messages.ts
│   ├── register-builtins.ts
│   ├── simple-options.ts
│   └── images/
├── utils/
│   ├── event-stream.ts
│   ├── validation.ts
│   ├── abort-signals.ts
│   ├── headers.ts
│   ├── json-parse.ts
│   ├── diagnostics.ts
│   ├── overflow.ts
│   ├── oauth/
│   └── ...
├── api-registry.ts
├── models.ts
├── models.generated.ts
├── types.ts
├── stream.ts
├── images.ts
├── images-api-registry.ts
└── oauth.ts
```

---

## 3. Supported Providers

| Provider | API Key Name | Notes |
|----------|-------------|-------|
| OpenAI (Responses) | openai-responses | Prompt caching, session IDs |
| OpenAI (Completions) | openai-completions | OpenRouter, Together, Groq, etc. |
| Anthropic | anthropic-messages | Extended thinking, OAuth |
| Google Gemini | google-generative-ai | Thinking levels |
| Google Vertex AI | google-vertex-ai | ADC auth |
| Azure OpenAI | azure-openai-responses | — |
| Mistral | mistral-api | Session affinity headers |
| AWS Bedrock | bedrock-converse-stream | AWS credential chain |
| Cloudflare | cloudflare-ai-gateway | URL variable substitution |
| GitHub Copilot | (passthrough) | Custom headers |
| Faux (mock) | faux | Testing |

OpenAI Completions also supports ~15 compatible providers detected at runtime:
OpenRouter, Together, Cerebras, DeepSeek, Qwen, Groq, NVIDIA NIM, etc.

---

## 4. Core Type System

### Message Types
```typescript
// User message
{ role: "user", content: [
  { type: "text", text: string } |
  { type: "image", image: Uint8Array|string, mediaType?: string }
], timestamp: number }

// Assistant message
{ role: "assistant", content: [
  { type: "text", text: string } |
  { type: "thinking", thinking: string, signature?: string } |
  { type: "toolcall", id: string, name: string, input: Record<string, unknown> }
], metadata?: { usage?, stopReason?, cached? }, diagnostics? }

// Tool result message
{ role: "tool", content: [
  { type: "text", text: string } |
  { type: "image", image: Uint8Array|string }
], toolUseId: string, name: string }
```

### Model Definition
```typescript
{
  id: string,
  name: string,
  api: KnownApi,               // "openai-responses" | "anthropic-messages" | ...
  provider: string,
  baseUrl?: string,
  reasoning?: boolean,
  input?: ("text"|"image")[],
  output?: ("text"|"image")[],
  contextWindow: number,
  maxTokens?: number,
  cost?: { input?, output?, cacheRead?, cacheWrite? },  // per million tokens
  thinkingLevelMap?: Record<ThinkingLevel, unknown>,
  compat?: ModelCompat
}
```

### Thinking Levels
```typescript
type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh"

// Budget mapping:
// off: 0, minimal: 1024, low: 2048, medium: 8192, high: 16384, xhigh: 16384
```

### Stream Options
```typescript
{
  temperature?, maxTokens?,
  thinking?: { level: ThinkingLevel, budget?: number },
  reasoning?: { effort: ThinkingLevel, budget?: number },
  tools?: Tool[],
  toolChoice?: "auto"|"none"|"required",
  stopSequences?: string[],
  headers?: Record<string, string>,
  cache?: { retention?: "short"|"long", sessionId?: string },
  abortSignal?: AbortSignal,
  retryPolicy?: RetryPolicy
}
```

### Unified Simple Stream Options
```typescript
{
  temperature?,
  maxTokens?,
  reasoning?: ThinkingLevel,      // single field, unified across providers
  tools?: Tool[],
  toolChoice?: "auto"|"none"|"required",
  cache?: { sessionId?: string },
  abortSignal?: AbortSignal
}
```

---

## 5. Streaming Architecture

### Event Types
```typescript
type AssistantMessageEvent =
  | { type: "start" }
  | { type: "text_start" }
  | { type: "text_delta", text: string }
  | { type: "text_end" }
  | { type: "thinking_start" }
  | { type: "thinking_delta", thinking: string }
  | { type: "thinking_end" }
  | { type: "toolcall_start", id: string, name: string }
  | { type: "toolcall_delta", input: string }
  | { type: "toolcall_end" }
  | { type: "done", message: AssistantMessage }
  | { type: "error", error: Error }
```

### EventStream Class
```typescript
class EventStream<TEvent, TResult> implements AsyncIterable<TEvent> {
  emit(event: TEvent): void
  done(result: TResult): void
  error(error: Error): void
  result(): Promise<TResult>
  [Symbol.asyncIterator](): AsyncIterator<TEvent>
}
```

### Provider Streaming Pattern
1. Initialize `AssistantMessageEventStream`
2. As API chunks arrive → emit granular events
3. Accumulate state in local assistant message
4. On completion → emit "done" with final message
5. On error → emit "error"

### Public API
```typescript
// Streaming
stream(model, context, options): AssistantMessageEventStream
streamSimple(model, context, simpleOptions): AssistantMessageEventStream

// Blocking
complete(model, context, options): Promise<AssistantMessage>
completeSimple(model, context, simpleOptions): Promise<AssistantMessage>
```

---

## 6. Tool / Function Calling

### Tool Definition (TypeBox)
```typescript
{
  name: string,
  description: string,
  input: TSchema  // TypeBox JSON Schema
}
```

### Provider Conversion
- **Anthropic** → `input_schema` in tools array
- **OpenAI** → `type: "function"`, `function.parameters`
- **Google** → `function_declarations` with `OBJECT`/`NUMBER` type strings

### Tool Call ID Normalization
| Provider | Max Length | Strategy |
|----------|-----------|----------|
| OpenAI   | 450+ chars | Hash → 64 chars |
| Anthropic | 64 chars  | Pass-through |
| Google   | Generated | name+timestamp+counter |
| Mistral  | 9 chars   | Keep as-is |

### Validation
- Validate tool arguments against TypeBox schema
- Automatic type coercion (string "123" → number 123)
- Handles allOf/anyOf/oneOf
- Cached validators (WeakMap)

### Streaming Tool Arguments
- Tool input arrives as partial JSON
- `parseStreamingJson()` handles incomplete JSON → valid object
- Enables eager UI updates while tool arguments stream in

---

## 7. Message Transformation

### Cross-Provider Compatibility (`transform-messages.ts`)
1. **Image downgrading** — replaces unsupported images with placeholder text
2. **Thinking block conversion** — encode/decode reasoning blocks with signatures
3. **Tool call ID normalization** — 450+ chars → 64 chars (deterministic hash)
4. **Synthetic tool results** — insert empty error for orphaned tool calls
5. **Error filtering** — remove incomplete assistant messages

### Thinking Block Preservation
When switching providers mid-conversation, thinking blocks survive via signatures:
```
Provider A response: { type: "thinking", thinking: "[...]" }

Sent to Provider B: { type: "text", text: "<think>[...]</think>\nAnswer", 
                       signature: "thinking:provider-a:hash" }

Roundtrip back → restored thinking block
```

### Synthetic Tool Results
```
Input:  [user, assistant(toolcall1, toolcall2), tool_result(id1)]
Output: [user, assistant, tool_result(id1), tool_result(id2, error="no result")]
```
This prevents conversation breaks from orphaned tool calls.

---

## 8. Error Handling

### Diagnostic System
```typescript
{
  type: string,
  timestamp: number,
  error?: { name?, message, stack?, code? },
  details?: Record<string, unknown>
}
```

### Context Overflow Detection
Three mechanisms:
1. Error message regex pattern matching (20+ provider-specific patterns)
2. Silent overflow: usage exceeds context window but no error thrown
3. Length truncation: server fills input tokens, zero output tokens

### Retry Logic
- Retryable: 429 (rate limit), 5xx (server errors)
- Non-retryable: 401 (auth), 400 (bad request)
- Configurable: maxRetries, delayMs, backoffMultiplier

---

## 9. Provider Registry / Plugin System

### Registration
```typescript
registerApiProvider("custom-api", {
  stream(model, context, options): AssistantMessageEventStream,
  streamSimple(model, context, simpleOptions): AssistantMessageEventStream
})
```

### Lazy Loading
```typescript
createLazyStream(async () => (await import("./anthropic")).streamAnthropic)
// Provider loads on first use, cached after that
```

### Built-in Registration
`registerBuiltInApiProviders()` — registers all 9 built-in providers with lazy loading
`resetApiProviders()` — clears and re-registers (useful for testing)

---

## 10. Authentication

### API Key Resolution (priority order)
1. Explicit `options.apiKey`
2. Environment variable (e.g., `OPENAI_API_KEY`)
3. OAuth token (auto-refreshed)
4. Ambient credentials (AWS credential chain, ADC for Google)

### OAuth System
```typescript
registerOAuthProvider(id, { login(), refreshToken() })
getOAuthProvider(id)
// Built-in: Anthropic, GitHub Copilot, OpenAI Codex
```

---

## 11. Model Registry & Cost Calculation

### Model Discovery
- Models in `models.generated.ts` (auto-generated from provider specs)
- Registry is `Map<provider, Map<modelId, Model>>`
- `getModel(provider, modelId)`, `getModels(provider)`, `getProviders()`

### Cost Calculation
```
cost = (inputTokens * input_rate/1M)
     + (outputTokens * output_rate/1M)
     + (cacheReadTokens * cache_read_rate/1M)
     + (cacheWriteTokens * cache_write_rate/1M)
```

Service tier multipliers: flex 0.5x, priority 2-2.5x

---

## 12. Notable Utilities

| Utility | Purpose |
|---------|---------|
| `shortHash(str)` | 32-bit MurmurHash3 → base-36, truncate long IDs |
| `parseStreamingJson(partial)` | Parse incomplete JSON streams |
| `parseJsonWithRepair(json)` | Fix malformed JSON (escape control chars) |
| `sanitizeSurrogates(str)` | Remove unpaired Unicode surrogates |
| `combineAbortSignals(signals[])` | Merge multiple AbortSignal instances |
| `isContextOverflow(error, usage, window)` | Detect context length exceeded |
| `validateToolCall(call, tools)` | Validate + coerce tool arguments |
| `buildBaseOptions(simple)` | Convert SimpleStreamOptions to full StreamOptions |
| `adjustMaxTokensForThinking(tokens, level)` | Reserve tokens for thinking budget |

---

## 13. Image Generation

Separate registry (`images-api-registry.ts`) with similar pattern:
```typescript
generateImages(model, context, options): Promise<AssistantImages>
```
Providers: DALL-E, Stable Diffusion, Flux (via various APIs)

---

## 14. Key Design Decisions (Notable for Python Port)

1. **Streaming-first with complete() convenience** — Python can use `async for` + sync `run()` wrapper
2. **Provider compat flags in model metadata** — replaces scattered if/else checks
3. **Lazy-loaded providers** — Python: use importlib.import_module on first use
4. **Granular event stream** — Python: `AsyncGenerator[Event, None]` or queue-based
5. **TypeBox → Python: Pydantic models** or plain JSON Schema dicts for tools
6. **Tool ID normalization** — mmh3 or hashlib for consistent hashing in Python
7. **Thinking block preservation via signatures** — pure string manipulation, easy to port
8. **Synthetic tool results** — simple list manipulation logic
9. **Context overflow detection** — regex patterns on error messages, same in Python
10. **SimpleStreamOptions** — keep a simple unified options interface in Python

---

*Last updated: 2026-06-09*
