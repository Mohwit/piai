"""
piai/providers/register_builtins.py — Lazy-registers all built-in providers.
Mirrors register-builtins.ts from @earendil-works/pi-ai.

All 10 KnownApi values are registered here.
Unimplemented providers (Phase 2+) use a stub LazyProvider that raises NotImplementedError.
"""
from __future__ import annotations

from piai.api_registry import KnownApi, LazyProvider, register_api_provider
from piai.providers.faux import FauxProvider

_registered = False


def register_builtin_api_providers() -> None:
    """
    Register all built-in providers with lazy loading.
    Called once on first use (see api_registry.reset_api_providers and stream.py).
    Idempotent — calling multiple times has no effect.
    """
    global _registered
    if _registered:
        return
    _registered = True

    # Phase 1 — fully implemented
    register_api_provider(
        KnownApi.ANTHROPIC,
        LazyProvider("piai.providers.anthropic", "AnthropicProvider"),
    )
    register_api_provider(
        KnownApi.OPENAI_COMPLETIONS,
        LazyProvider("piai.providers.openai_completions", "OpenAICompletionsProvider"),
    )
    register_api_provider(
        KnownApi.GOOGLE_VERTEX,
        LazyProvider("piai.providers.google_vertex", "GoogleVertexProvider"),
    )
    register_api_provider(KnownApi.FAUX, FauxProvider())

    # Phase 1 stubs — exist as files, raise NotImplementedError on call
    register_api_provider(
        KnownApi.OPENAI_RESPONSES,
        LazyProvider("piai.providers.openai_responses", "OpenAIResponsesProvider"),
    )
    register_api_provider(
        KnownApi.GOOGLE,
        LazyProvider("piai.providers.google", "GoogleProvider"),
    )

    # Phase 2 stubs — not yet implemented
    # Registering now so KnownApi routing works; providers will raise NotImplementedError
    # until Phase 2 implementations land.
    for api, module, cls in [
        (KnownApi.MISTRAL,     "piai.providers.mistral",             "MistralProvider"),
        (KnownApi.BEDROCK,     "piai.providers.bedrock",             "BedrockProvider"),
        (KnownApi.AZURE_OPENAI,"piai.providers.azure_openai",        "AzureOpenAIProvider"),
        (KnownApi.CLOUDFLARE,  "piai.providers.cloudflare",          "CloudflareProvider"),
    ]:
        register_api_provider(api, LazyProvider(module, cls))
