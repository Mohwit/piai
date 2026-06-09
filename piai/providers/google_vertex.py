"""
piai/providers/google_vertex.py — Google Vertex AI provider (Gemini models).
Auth: Application Default Credentials (ADC) via gcloud or GOOGLE_APPLICATION_CREDENTIALS.
SDK is synchronous — wrapped with asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

from piai.models import Model
from piai.types import (
    AssistantMessage,
    AssistantMessageMetadata,
    DoneEvent,
    ErrorEvent,
    MessagesContext,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    THINKING_BUDGETS,
    TokenUsage,
    ToolCallContent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from piai.utils.event_stream import EventStream
from piai.utils.abort import is_aborted
from piai.utils.overflow import is_context_overflow
from piai.utils.diagnostics import make_diagnostic, attach_diagnostic
from piai.utils.retry import with_retry
from piai.providers.simple_options import build_base_options
from piai.providers.transform_messages import (
    convert_schema_to_vertex,
    generate_google_tool_id,
    insert_synthetic_tool_results,
)

AssistantMessageEventStream = EventStream[object, AssistantMessage]

_STOP_REASON_MAP = {
    "STOP":       "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY":     "content_filter",
    "RECITATION": "content_filter",
    "OTHER":      "other",
    "TOOL_USE":   "tool_use",
}


class GoogleVertexProvider:
    def stream(
        self,
        model: Model,
        context: MessagesContext,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream:
        es: AssistantMessageEventStream = EventStream()
        asyncio.get_event_loop().create_task(self._run(es, model, context, options or StreamOptions()))
        return es

    def stream_simple(
        self,
        model: Model,
        context: MessagesContext,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessageEventStream:
        return self.stream(model, context, build_base_options(options or SimpleStreamOptions()))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run(
        self,
        es: AssistantMessageEventStream,
        model: Model,
        context: MessagesContext,
        options: StreamOptions,
    ) -> None:
        if is_aborted(options.abort_signal):
            es.error(asyncio.CancelledError("Request aborted before start"))
            return
        try:
            await with_retry(
                lambda: self._stream_once(es, model, context, options),
                policy=options.retry_policy,
            )
        except Exception as exc:
            es.emit(ErrorEvent(error=exc))
            es.error(exc)

    async def _stream_once(
        self,
        es: AssistantMessageEventStream,
        model: Model,
        context: MessagesContext,
        options: StreamOptions,
    ) -> None:
        """
        Run the synchronous Vertex SDK call in a thread pool, collect all events
        and the final message, then emit them back in the event loop.
        This avoids asyncio.Queue thread-safety issues.
        """
        events, msg = await asyncio.to_thread(
            self._collect_sync, model, context, options
        )
        for event in events:
            es.emit(event)
        if isinstance(msg, Exception):
            raise msg
        es.emit(DoneEvent(message=msg))
        es.done(msg)

    def _collect_sync(
        self,
        model: Model,
        context: MessagesContext,
        options: StreamOptions,
    ) -> tuple[list, AssistantMessage | Exception]:
        """
        Synchronous Vertex SDK call. Returns (events_list, final_message).
        Runs in a ThreadPoolExecutor via asyncio.to_thread.
        """
        try:
            return self._do_stream(model, context, options)
        except Exception as exc:
            return [], exc

    def _do_stream(
        self,
        model: Model,
        context: MessagesContext,
        options: StreamOptions,
    ) -> tuple[list, AssistantMessage]:
        import vertexai  # type: ignore[import]
        from vertexai.generative_models import (  # type: ignore[import]
            GenerativeModel,
            GenerationConfig,
            Tool as VertexTool,
            FunctionDeclaration,
        )

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable is required for Vertex AI."
            )

        vertexai.init(project=project, location=location)

        # Build generation config
        gen_config_kwargs: dict[str, Any] = {}
        if options.max_tokens is not None:
            gen_config_kwargs["max_output_tokens"] = options.max_tokens
        elif model.max_tokens:
            gen_config_kwargs["max_output_tokens"] = model.max_tokens
        if options.temperature is not None:
            gen_config_kwargs["temperature"] = options.temperature
        if options.stop_sequences:
            gen_config_kwargs["stop_sequences"] = options.stop_sequences

        # Thinking config (Gemini 2.5+)
        if options.thinking and options.thinking.level != "off":
            level = options.thinking.level
            budget = options.thinking.budget
            if budget is None and model.thinking_level_map:
                level_params = model.thinking_level_map.get(level, {})
                budget = level_params.get("thinking_budget")
            if budget is None:
                budget = THINKING_BUDGETS.get(level, 8192)
            gen_config_kwargs["thinking_config"] = {"thinking_budget": budget}

        gen_config = GenerationConfig(**gen_config_kwargs)

        # Build tools
        vertex_tools = None
        if options.tools:
            declarations = []
            for t in options.tools:
                schema = convert_schema_to_vertex(t.input)
                declarations.append(FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=schema,
                ))
            vertex_tools = [VertexTool(function_declarations=declarations)]

        system_instruction = context.system_prompt or None
        contents = self._convert_messages(insert_synthetic_tool_results(context.messages))

        vertex_model = GenerativeModel(
            model.id,
            system_instruction=system_instruction,
        )

        # Accumulators
        events: list = []
        text_buf = ""
        thinking_buf = ""
        tool_calls: list[ToolCallContent] = []
        usage = TokenUsage()
        stop_reason: Optional[str] = None
        text_started = False
        thinking_started = False

        events.append(StartEvent())

        stream = vertex_model.generate_content(
            contents,
            generation_config=gen_config,
            tools=vertex_tools,
            stream=True,
        )

        for chunk in stream:
            if is_aborted(options.abort_signal):
                raise asyncio.CancelledError("Request aborted")

            # Usage
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                um = chunk.usage_metadata
                usage.input_tokens = getattr(um, "prompt_token_count", 0) or 0
                usage.output_tokens = getattr(um, "candidates_token_count", 0) or 0

            if not chunk.candidates:
                continue

            candidate = chunk.candidates[0]

            if candidate.finish_reason:
                stop_reason = _STOP_REASON_MAP.get(
                    str(candidate.finish_reason).replace("FinishReason.", ""),
                    "end_turn",
                )

            if not hasattr(candidate, "content") or not candidate.content:
                continue

            for part in candidate.content.parts:
                # Thinking part (Gemini 2.5 with thinking enabled)
                if hasattr(part, "thought") and part.thought:
                    if not thinking_started:
                        thinking_started = True
                        events.append(ThinkingStartEvent())
                    delta = part.text or ""
                    thinking_buf += delta
                    if delta:
                        events.append(ThinkingDeltaEvent(thinking=delta))
                    continue

                # Text part
                if hasattr(part, "text") and part.text:
                    if not text_started:
                        text_started = True
                        events.append(TextStartEvent())
                    text_buf += part.text
                    events.append(TextDeltaEvent(text=part.text))
                    continue

                # Function call — Vertex delivers whole args (not streamed)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tc_id = generate_google_tool_id(fc.name)
                    tc_input = dict(fc.args) if fc.args else {}
                    tc_input_str = json.dumps(tc_input)

                    events.append(ToolCallStartEvent(id=tc_id, name=fc.name))
                    events.append(ToolCallDeltaEvent(input=tc_input_str))
                    events.append(ToolCallEndEvent())

                    tool_calls.append(ToolCallContent(
                        id=tc_id,
                        name=fc.name,
                        input=tc_input,
                    ))

        if text_started:
            events.append(TextEndEvent())
        if thinking_started:
            events.append(ThinkingEndEvent())

        # Assemble final message
        content = []
        if thinking_buf:
            content.append(ThinkingContent(thinking=thinking_buf))
        if text_buf:
            content.append(TextContent(text=text_buf))
        content.extend(tool_calls)

        cost = None
        if model.cost and (usage.input_tokens or usage.output_tokens):
            cost = (
                (usage.input_tokens * (model.cost.input or 0)) / 1_000_000
                + (usage.output_tokens * (model.cost.output or 0)) / 1_000_000
            )

        msg = AssistantMessage(
            content=content,
            metadata=AssistantMessageMetadata(
                usage=usage,
                stop_reason=stop_reason,
                cost=cost,
            ),
        )

        if is_context_overflow(None, usage, model.context_window):
            attach_diagnostic(msg, make_diagnostic("context_overflow"))

        return events, msg

    def _convert_messages(self, messages: list) -> list[Any]:
        """Convert piai messages to Vertex AI Content/Part format."""
        from vertexai.generative_models import Content, Part  # type: ignore[import]
        from piai.types import UserMessage, AssistantMessage, ToolResultMessage
        from piai.types import ImageContent, TextContent, ThinkingContent, ToolCallContent

        result = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                parts = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append(Part.from_text(part.text))
                    elif isinstance(part, ImageContent):
                        data = part.image if isinstance(part.image, bytes) else part.image.encode()
                        parts.append(Part.from_data(
                            data=data,
                            mime_type=part.media_type or "image/jpeg",
                        ))
                result.append(Content(role="user", parts=parts))

            elif isinstance(msg, AssistantMessage):
                parts = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append(Part.from_text(part.text))
                    elif isinstance(part, ThinkingContent):
                        # Encode thinking as text for providers that don't support it natively
                        parts.append(Part.from_text(f"<think>{part.thinking}</think>"))
                    elif isinstance(part, ToolCallContent):
                        parts.append(Part.from_function_call(
                            name=part.name,
                            args=part.input,
                        ))
                if parts:
                    result.append(Content(role="model", parts=parts))

            elif isinstance(msg, ToolResultMessage):
                text = " ".join(
                    p.text for p in msg.content if isinstance(p, TextContent)
                )
                result.append(Content(
                    role="user",
                    parts=[Part.from_function_response(
                        name=msg.name,
                        response={"content": text},
                    )],
                ))

        return result
