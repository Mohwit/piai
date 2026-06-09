# Pi-AI Python Port — Research Index

Research on replicating the pi-ai TypeScript package (`@earendil-works/pi-ai`) in Python.

## Files

| File | Description |
|------|-------------|
| [pi-ai-architecture.md](pi-ai-architecture.md) | Full architecture deep-dive: types, streaming, tools, error handling, registry, auth |
| [providers-analysis.md](providers-analysis.md) | Per-provider breakdown: Anthropic, OpenAI, Google, Mistral, Bedrock, Azure, Cloudflare |
| [python-port-plan.md](python-port-plan.md) | TypeScript→Python mapping, proposed package structure, code sketches, dependencies |

## Key Findings

### What pi-ai does well (that we want in Python)
1. **Unified event stream** — granular events (text_delta, thinking_delta, toolcall_start, etc.) over a single AsyncIterable
2. **SimpleStreamOptions** — one unified options object that works across all providers
3. **Thinking/reasoning levels** — `"off"` → `"xhigh"` normalized across providers
4. **Cross-provider tool calling** — TypeBox (→ Pydantic in Python) schemas, format auto-converted per provider
5. **Cost tracking** — per-request cost calculation from model metadata
6. **Context overflow detection** — regex patterns + usage heuristics
7. **Thinking block preservation** — survives provider switches via `<think>` encoding
8. **Synthetic tool results** — prevents broken conversations from orphaned tool calls
9. **Tool call ID normalization** — hashes long OpenAI IDs to 64 chars
10. **Lazy provider loading** — importlib-based, providers loaded on first use

### Python-specific improvements to make
1. **Both sync and async APIs** — `complete()` async + `complete_sync()` for non-async code
2. **Pydantic model support** — accept pydantic models as tool schemas directly
3. **Context managers** — `async with stream(...) as s:` for resource cleanup
4. **Full type stubs** — `py.typed` + complete annotations for IDE support

### Implementation priority
**Phase 1**: Anthropic + OpenAI Completions + Google + Faux (mock) + core types  
**Phase 2**: OpenAI Responses + Mistral + Bedrock + Vertex AI  
**Phase 3**: Azure + Cloudflare + OAuth  

## Original Source
https://github.com/earendil-works/pi/tree/main/packages/ai
