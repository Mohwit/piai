"""
piai/providers/google.py — Google AI Studio / Gemini API provider (Phase 1 stub).

Uses google-generativeai SDK with API key (distinct from Vertex AI which uses ADC).
This file exists so KnownApi.GOOGLE routes here.
"""
from __future__ import annotations


class GoogleProvider:
    def stream(self, model, context, options=None):
        raise NotImplementedError(
            "Google AI Studio provider is not yet implemented (Phase 2). "
            "Use KnownApi.GOOGLE_VERTEX for Google Vertex AI with ADC credentials."
        )

    def stream_simple(self, model, context, options=None):
        raise NotImplementedError(
            "Google AI Studio provider is not yet implemented (Phase 2). "
            "Use KnownApi.GOOGLE_VERTEX for Google Vertex AI with ADC credentials."
        )
