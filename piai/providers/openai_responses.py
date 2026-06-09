"""
piai/providers/openai_responses.py — OpenAI Responses API provider (Phase 1 stub).

This file exists so KnownApi.OPENAI_RESPONSES routes here.
Azure OpenAI Responses will delegate to or subclass this in Phase 2.
"""
from __future__ import annotations


class OpenAIResponsesProvider:
    def stream(self, model, context, options=None):
        raise NotImplementedError(
            "OpenAI Responses API provider is not yet implemented (Phase 2). "
            "Use KnownApi.OPENAI_COMPLETIONS for standard OpenAI chat completions."
        )

    def stream_simple(self, model, context, options=None):
        raise NotImplementedError(
            "OpenAI Responses API provider is not yet implemented (Phase 2). "
            "Use KnownApi.OPENAI_COMPLETIONS for standard OpenAI chat completions."
        )
