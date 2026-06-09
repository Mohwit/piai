"""
piai/api_registry.py — KnownApi enum + provider registration + lazy loading.
Mirrors api-registry.ts from @earendil-works/pi-ai.
"""
from __future__ import annotations

import importlib
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from piai.types import MessagesContext, StreamOptions, SimpleStreamOptions, AssistantMessage
    from piai.utils.event_stream import EventStream


class KnownApi(str, Enum):
    """
    All supported provider API identifiers.
    All future providers get a member here — no model ever routes via a raw string.
    Members not yet implemented have stubs in providers/ that raise NotImplementedError.
    """
    ANTHROPIC          = "anthropic"
    OPENAI_COMPLETIONS = "openai-completions"
    OPENAI_RESPONSES   = "openai-responses"
    GOOGLE             = "google"
    GOOGLE_VERTEX      = "google-vertex"
    MISTRAL            = "mistral"        # Phase 2
    BEDROCK            = "bedrock"        # Phase 2
    AZURE_OPENAI       = "azure-openai"   # Phase 2
    CLOUDFLARE         = "cloudflare"     # Phase 2
    FAUX               = "faux"


# ---------------------------------------------------------------------------
# Provider Protocol (structural typing via duck-typing — no ABC needed)
# ---------------------------------------------------------------------------

class ApiProvider:
    """
    Base class / protocol for all provider implementations.
    Providers must implement stream() and stream_simple().
    """
    def stream(self, model, context, options=None):
        raise NotImplementedError

    def stream_simple(self, model, context, options=None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Lazy loading wrapper
# ---------------------------------------------------------------------------

class LazyProvider:
    """
    Defers SDK import until the first call.
    After the first call the real provider instance is cached.
    Mirrors createLazyStream() from the TypeScript original.
    """

    def __init__(self, module_path: str, class_name: str) -> None:
        self._module_path = module_path
        self._class_name = class_name
        self._instance: Optional[ApiProvider] = None

    def _get(self) -> ApiProvider:
        if self._instance is None:
            module = importlib.import_module(self._module_path)
            cls = getattr(module, self._class_name)
            self._instance = cls()
        return self._instance

    def stream(self, model, context, options=None):
        return self._get().stream(model, context, options)

    def stream_simple(self, model, context, options=None):
        return self._get().stream_simple(model, context, options)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_registry: dict[KnownApi, ApiProvider] = {}

# OAuth hook point — reserved for future registerOAuthProvider / getOAuthProvider
# (mirrors _oauth_registry from the TypeScript original)
_oauth_registry: dict[str, object] = {}


def register_api_provider(api: KnownApi, provider: ApiProvider) -> None:
    """Register a provider implementation for the given KnownApi key."""
    _registry[api] = provider


def get_api_provider(api: KnownApi) -> Optional[ApiProvider]:
    """Return the registered provider for `api`, or None if not registered."""
    return _registry.get(api)


def reset_api_providers() -> None:
    """
    Clear all registered providers and re-register built-ins.
    Intended for test isolation only.
    """
    _registry.clear()
    from piai.providers.register_builtins import register_builtin_api_providers
    register_builtin_api_providers()
