"""
piai/models.py — Model dataclass + model registry.
Mirrors models.ts + models.generated.ts from @earendil-works/pi-ai.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from piai.api_registry import KnownApi


# ---------------------------------------------------------------------------
# Model dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelCost:
    input: Optional[float] = None        # USD per million input tokens
    output: Optional[float] = None       # USD per million output tokens
    cache_read: Optional[float] = None   # USD per million cache-read tokens
    cache_write: Optional[float] = None  # USD per million cache-write tokens


@dataclass
class ModelCompat:
    """
    Provider-specific quirk flags.
    Lives on Model rather than in shared code — avoids scattered if/else.
    Add flags here as new providers reveal quirks.
    """
    no_system_prompt: bool = False    # provider ignores system prompts
    no_tool_streaming: bool = False   # tool args arrive whole, not streamed (e.g. Groq)
    no_image_input: bool = False      # provider doesn't accept image parts
    batch_tool_results: bool = False  # must merge consecutive ToolResultMessages (Anthropic)


@dataclass
class Model:
    id: str
    name: str
    api: KnownApi
    provider: str
    context_window: int
    base_url: Optional[str] = None
    reasoning: bool = False
    input: list[str] = field(default_factory=lambda: ["text"])
    output: list[str] = field(default_factory=lambda: ["text"])
    max_tokens: Optional[int] = None
    cost: Optional[ModelCost] = None
    thinking_level_map: Optional[dict[str, Any]] = None  # per-level provider params
    compat: Optional[ModelCompat] = None                  # quirk flags


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# provider → {model_id → Model}
_registry: dict[str, dict[str, Model]] = {}
_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        _initialized = True
        _load_models()


def _load_models() -> None:
    from piai.models_data import MODELS_RAW
    for raw in MODELS_RAW:
        model = _build_model(raw)
        _registry.setdefault(model.provider, {})[model.id] = model


def _build_model(raw: dict) -> Model:
    cost = None
    if raw_cost := raw.get("cost"):
        cost = ModelCost(
            input=raw_cost.get("input"),
            output=raw_cost.get("output"),
            cache_read=raw_cost.get("cache_read"),
            cache_write=raw_cost.get("cache_write"),
        )

    compat = None
    if raw_compat := raw.get("compat"):
        compat = ModelCompat(
            no_system_prompt=raw_compat.get("no_system_prompt", False),
            no_tool_streaming=raw_compat.get("no_tool_streaming", False),
            no_image_input=raw_compat.get("no_image_input", False),
            batch_tool_results=raw_compat.get("batch_tool_results", False),
        )

    return Model(
        id=raw["id"],
        name=raw["name"],
        api=raw["api"],
        provider=raw["provider"],
        context_window=raw["context_window"],
        base_url=raw.get("base_url"),
        reasoning=raw.get("reasoning", False),
        input=raw.get("input", ["text"]),
        output=raw.get("output", ["text"]),
        max_tokens=raw.get("max_tokens"),
        cost=cost,
        thinking_level_map=raw.get("thinking_level_map"),
        compat=compat,
    )


def get_model(provider: str, model_id: str) -> Model:
    """
    Return the Model for (provider, model_id).
    Raises ValueError if not found.
    """
    _ensure_initialized()
    provider_models = _registry.get(provider)
    if provider_models:
        model = provider_models.get(model_id)
        if model:
            return model
    raise ValueError(f"Model {model_id!r} not found for provider {provider!r}")


def list_models(provider: Optional[str] = None) -> list[Model]:
    """Return all registered models, optionally filtered by provider."""
    _ensure_initialized()
    if provider:
        return list(_registry.get(provider, {}).values())
    return [m for models in _registry.values() for m in models.values()]


def register_model(model: Model) -> None:
    """Register a model at runtime (for custom / user-defined models)."""
    _ensure_initialized()
    _registry.setdefault(model.provider, {})[model.id] = model
