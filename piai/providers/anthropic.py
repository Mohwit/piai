"""
piai/providers/anthropic.py — Anthropic Messages API provider.
Supports: streaming, thinking/extended reasoning, tool calling, cost tracking.
"""
from __future__ import annotations

import asyncio
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
from piai.utils.headers import get_api_key
from piai.utils.overflow import is_context_overflow
from piai.utils.diagnostics import make_diagnostic, attach_diagnostic
from piai.utils.retry import with_retry
from piai.providers.simple_options import build_base_options
from piai.providers.transform_messages import (
    batch_tool_results,
    insert_synthetic_tool_results,
)

AssistantMessageEventStream = EventStream[object, AssistantMessage]


class AnthropicProvider:
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
        import anthropic as sdk

        api_key = get_api_key("ANTHROPIC_API_KEY", options.api_key)
        client = sdk.AsyncAnthropic(api_key=api_key)

        # Build request params
        params = self._build_params(model, context, options)

        es.emit(StartEvent())

        # Accumulators
        text_buf = ""
        thinking_buf = ""
        thinking_sig = ""
        tool_input_buf = ""
        current_tool_id = ""
        current_tool_name = ""
        block_type: Optional[str] = None

        usage = TokenUsage()
        stop_reason: Optional[str] = None

        async with client.messages.stream(**params) as stream:
            async for event in stream:
                if is_aborted(options.abort_signal):
                    raise asyncio.CancelledError("Request aborted")

                etype = event.type

                if etype == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        usage.input_tokens = event.message.usage.input_tokens

                elif etype == "content_block_start":
                    block = event.content_block
                    block_type = block.type
                    if block_type == "text":
                        es.emit(TextStartEvent())
                    elif block_type == "thinking":
                        thinking_buf = ""
                        thinking_sig = getattr(block, "signature", "") or ""
                        es.emit(ThinkingStartEvent())
                    elif block_type == "tool_use":
                        current_tool_id = block.id
                        current_tool_name = block.name
                        tool_input_buf = ""
                        es.emit(ToolCallStartEvent(id=block.id, name=block.name))

                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = delta.type
                    if dtype == "text_delta":
                        text_buf += delta.text
                        es.emit(TextDeltaEvent(text=delta.text))
                    elif dtype == "thinking_delta":
                        thinking_buf += delta.thinking
                        es.emit(ThinkingDeltaEvent(thinking=delta.thinking))
                    elif dtype == "input_json_delta":
                        tool_input_buf += delta.partial_json
                        es.emit(ToolCallDeltaEvent(input=delta.partial_json))
                    elif dtype == "signature_delta":
                        thinking_sig += delta.signature

                elif etype == "content_block_stop":
                    if block_type == "text":
                        es.emit(TextEndEvent())
                    elif block_type == "thinking":
                        es.emit(ThinkingEndEvent())
                    elif block_type == "tool_use":
                        es.emit(ToolCallEndEvent())
                    block_type = None

                elif etype == "message_delta":
                    if hasattr(event, "usage"):
                        usage.output_tokens = event.usage.output_tokens
                    if hasattr(event, "delta") and hasattr(event.delta, "stop_reason"):
                        stop_reason = event.delta.stop_reason

        # Assemble final message
        msg = self._build_message(
            model=model,
            text_buf=text_buf,
            thinking_buf=thinking_buf,
            thinking_sig=thinking_sig,
            tool_id=current_tool_id,
            tool_name=current_tool_name,
            tool_input_buf=tool_input_buf,
            usage=usage,
            stop_reason=stop_reason,
        )

        # Context overflow detection
        if is_context_overflow(None, usage, model.context_window):
            attach_diagnostic(msg, make_diagnostic("context_overflow"))

        es.emit(DoneEvent(message=msg))
        es.done(msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_params(self, model: Model, context: MessagesContext, options: StreamOptions) -> dict[str, Any]:
        import anthropic as sdk

        # Convert messages — batch tool results (Anthropic requirement)
        raw_messages = insert_synthetic_tool_results(context.messages)
        anthropic_messages = self._convert_messages(batch_tool_results(raw_messages))

        params: dict[str, Any] = {
            "model": model.id,
            "messages": anthropic_messages,
            "max_tokens": options.max_tokens or model.max_tokens or 4096,
        }

        if context.system_prompt:
            params["system"] = context.system_prompt

        if options.temperature is not None:
            params["temperature"] = options.temperature

        if options.stop_sequences:
            params["stop_sequences"] = options.stop_sequences

        # Tools
        if options.tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input}
                for t in options.tools
            ]
            if options.tool_choice:
                choice_map = {"auto": "auto", "none": "none", "required": "any"}
                params["tool_choice"] = {"type": choice_map.get(options.tool_choice, "auto")}

        # Thinking / extended reasoning
        if options.thinking and options.thinking.level != "off":
            level = options.thinking.level
            budget = options.thinking.budget
            if budget is None and model.thinking_level_map:
                level_params = model.thinking_level_map.get(level, {})
                budget = level_params.get("budget_tokens")
            if budget is None:
                budget = THINKING_BUDGETS.get(level, 8192)

            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Add beta header for interleaved thinking
            params["extra_headers"] = {
                **(options.headers or {}),
                "anthropic-beta": "interleaved-thinking-2025-05-14",
            }
        elif options.headers:
            params["extra_headers"] = options.headers

        return params

    def _convert_messages(self, messages: list) -> list[dict]:
        """Convert piai message objects (or already-converted Anthropic dicts) to Anthropic wire format."""
        result = []
        for msg in messages:
            # Already converted by batch_tool_results — pass through dicts
            if isinstance(msg, dict):
                result.append(msg)
                continue

            from piai.types import UserMessage, AssistantMessage, ImageContent, TextContent, ThinkingContent, ToolCallContent
            if isinstance(msg, UserMessage):
                parts = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.media_type or "image/jpeg",
                                "data": part.image if isinstance(part.image, str)
                                        else part.image.decode(),
                            },
                        })
                result.append({"role": "user", "content": parts})

            elif isinstance(msg, AssistantMessage):
                parts = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ThinkingContent):
                        thinking_part: dict[str, Any] = {
                            "type": "thinking",
                            "thinking": part.thinking,
                        }
                        if part.signature:
                            thinking_part["signature"] = part.signature
                        parts.append(thinking_part)
                    elif isinstance(part, ToolCallContent):
                        import json
                        parts.append({
                            "type": "tool_use",
                            "id": part.id,
                            "name": part.name,
                            "input": part.input,
                        })
                result.append({"role": "assistant", "content": parts})

        return result

    def _build_message(
        self,
        model: Model,
        text_buf: str,
        thinking_buf: str,
        thinking_sig: str,
        tool_id: str,
        tool_name: str,
        tool_input_buf: str,
        usage: TokenUsage,
        stop_reason: Optional[str],
    ) -> AssistantMessage:
        from piai.utils.json_parse import parse_json_with_repair

        content = []
        if thinking_buf:
            content.append(ThinkingContent(
                thinking=thinking_buf,
                signature=thinking_sig or None,
            ))
        if text_buf:
            content.append(TextContent(text=text_buf))
        if tool_id:
            content.append(ToolCallContent(
                id=tool_id,
                name=tool_name,
                input=parse_json_with_repair(tool_input_buf) if tool_input_buf else {},
            ))

        cost = None
        if model.cost and (usage.input_tokens or usage.output_tokens):
            cost = (
                (usage.input_tokens * (model.cost.input or 0)) / 1_000_000
                + (usage.output_tokens * (model.cost.output or 0)) / 1_000_000
                + (usage.cache_read_tokens * (model.cost.cache_read or 0)) / 1_000_000
                + (usage.cache_write_tokens * (model.cost.cache_write or 0)) / 1_000_000
            )

        return AssistantMessage(
            content=content,
            metadata=AssistantMessageMetadata(
                usage=usage,
                stop_reason=stop_reason,
                cost=cost,
            ),
        )
