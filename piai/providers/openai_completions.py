"""
piai/providers/openai_completions.py — OpenAI Chat Completions API provider.
Also covers compatible providers: Groq, Together, OpenRouter, DeepSeek, Cerebras, etc.
Detection is via model.base_url or model.compat flags set in models_data.json.
"""
from __future__ import annotations

import asyncio
import json
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
from piai.utils.hash_utils import normalize_openai_tool_call_id
from piai.utils.json_parse import parse_json_with_repair
from piai.providers.simple_options import build_base_options
from piai.providers.transform_messages import insert_synthetic_tool_results

AssistantMessageEventStream = EventStream[object, AssistantMessage]


class OpenAICompletionsProvider:
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
        import openai

        api_key = get_api_key("OPENAI_API_KEY", options.api_key)
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if model.base_url:
            client_kwargs["base_url"] = model.base_url

        client = openai.AsyncOpenAI(**client_kwargs)

        params = self._build_params(model, context, options)

        # Detect quirks from compat flags
        no_tool_streaming = bool(model.compat and model.compat.no_tool_streaming)

        es.emit(StartEvent())

        text_buf = ""
        thinking_buf = ""
        # tool_calls_buf: index → {id, name, input_parts}
        tool_calls_buf: dict[int, dict[str, Any]] = {}
        active_tool_index: Optional[int] = None
        usage = TokenUsage()
        stop_reason: Optional[str] = None

        async with client.chat.completions.create(**params, stream=True) as stream:
            async for chunk in stream:
                if is_aborted(options.abort_signal):
                    raise asyncio.CancelledError("Request aborted")

                if chunk.usage:
                    usage.input_tokens = chunk.usage.prompt_tokens or 0
                    usage.output_tokens = chunk.usage.completion_tokens or 0

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = choice.finish_reason

                # Text delta
                if delta.content:
                    if not text_buf:
                        es.emit(TextStartEvent())
                    text_buf += delta.content
                    es.emit(TextDeltaEvent(text=delta.content))

                # DeepSeek: thinking in reasoning_content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not thinking_buf:
                        es.emit(ThinkingStartEvent())
                    thinking_buf += delta.reasoning_content
                    es.emit(ThinkingDeltaEvent(thinking=delta.reasoning_content))

                # Tool call streaming
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_buf:
                            tool_calls_buf[idx] = {"id": "", "name": "", "input_parts": []}
                            active_tool_index = idx

                        if tc_delta.id:
                            raw_id = normalize_openai_tool_call_id(tc_delta.id)
                            tool_calls_buf[idx]["id"] = raw_id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_buf[idx]["name"] = tc_delta.function.name
                                es.emit(ToolCallStartEvent(
                                    id=tool_calls_buf[idx]["id"],
                                    name=tc_delta.function.name,
                                ))
                            if tc_delta.function.arguments:
                                tool_calls_buf[idx]["input_parts"].append(tc_delta.function.arguments)
                                if not no_tool_streaming:
                                    es.emit(ToolCallDeltaEvent(input=tc_delta.function.arguments))

                if finish_reason:
                    stop_reason = _map_stop_reason(finish_reason)

        # Flush text end
        if text_buf:
            es.emit(TextEndEvent())
        if thinking_buf:
            es.emit(ThinkingEndEvent())

        # Flush tool calls
        for idx in sorted(tool_calls_buf.keys()):
            tc = tool_calls_buf[idx]
            full_input = "".join(tc["input_parts"])
            if no_tool_streaming and full_input:
                # Groq-style: emit all at once
                es.emit(ToolCallDeltaEvent(input=full_input))
            es.emit(ToolCallEndEvent())

        msg = self._build_message(
            model=model,
            text_buf=text_buf,
            thinking_buf=thinking_buf,
            tool_calls_buf=tool_calls_buf,
            usage=usage,
            stop_reason=stop_reason,
        )

        if is_context_overflow(None, usage, model.context_window):
            attach_diagnostic(msg, make_diagnostic("context_overflow"))

        es.emit(DoneEvent(message=msg))
        es.done(msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_params(self, model: Model, context: MessagesContext, options: StreamOptions) -> dict[str, Any]:
        messages = self._convert_messages(
            insert_synthetic_tool_results(context.messages),
            context.system_prompt,
        )

        params: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
        }

        if options.max_tokens is not None:
            params["max_tokens"] = options.max_tokens
        elif model.max_tokens:
            params["max_tokens"] = model.max_tokens

        if options.temperature is not None:
            params["temperature"] = options.temperature

        if options.stop_sequences:
            params["stop"] = options.stop_sequences

        # Tools
        if options.tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input,
                    },
                }
                for t in options.tools
            ]
            if options.tool_choice:
                choice_map = {
                    "auto": "auto",
                    "none": "none",
                    "required": "required",
                }
                params["tool_choice"] = choice_map.get(options.tool_choice, "auto")

        # Reasoning effort for o* models
        if options.reasoning and options.reasoning.effort != "off" and model.reasoning:
            effort = _map_reasoning_effort(options.reasoning.effort, model)
            if effort:
                params["reasoning_effort"] = effort

        # Extra headers (e.g. OpenRouter)
        if options.headers:
            params["extra_headers"] = options.headers

        # Include usage in streaming response
        params["stream_options"] = {"include_usage": True}

        return params

    def _convert_messages(
        self,
        messages: list,
        system_prompt: Optional[str],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        from piai.types import UserMessage, AssistantMessage, ToolResultMessage
        from piai.types import ImageContent, TextContent, ThinkingContent, ToolCallContent

        for msg in messages:
            if isinstance(msg, UserMessage):
                parts = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        url = (
                            part.image if isinstance(part.image, str)
                            else f"data:{part.media_type or 'image/jpeg'};base64,{part.image.decode()}"
                        )
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": url},
                        })
                result.append({"role": "user", "content": parts})

            elif isinstance(msg, AssistantMessage):
                text_parts = []
                tool_calls = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        text_parts.append(part.text)
                    elif isinstance(part, ThinkingContent):
                        # Encode thinking as text for non-native providers
                        text_parts.append(f"<think>{part.thinking}</think>")
                    elif isinstance(part, ToolCallContent):
                        tool_calls.append({
                            "id": part.id,
                            "type": "function",
                            "function": {
                                "name": part.name,
                                "arguments": json.dumps(part.input),
                            },
                        })
                oai_msg: dict[str, Any] = {"role": "assistant"}
                combined_text = "\n".join(text_parts)
                if combined_text:
                    oai_msg["content"] = combined_text
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)

            elif isinstance(msg, ToolResultMessage):
                text = " ".join(
                    p.text for p in msg.content if isinstance(p, TextContent)
                )
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_use_id,
                    "content": text,
                })

        return result

    def _build_message(
        self,
        model: Model,
        text_buf: str,
        thinking_buf: str,
        tool_calls_buf: dict[int, dict[str, Any]],
        usage: TokenUsage,
        stop_reason: Optional[str],
    ) -> AssistantMessage:
        content = []

        if thinking_buf:
            content.append(ThinkingContent(thinking=thinking_buf))
        if text_buf:
            content.append(TextContent(text=text_buf))
        for idx in sorted(tool_calls_buf.keys()):
            tc = tool_calls_buf[idx]
            full_input = "".join(tc["input_parts"])
            content.append(ToolCallContent(
                id=tc["id"],
                name=tc["name"],
                input=parse_json_with_repair(full_input) if full_input else {},
            ))

        cost = None
        if model.cost and (usage.input_tokens or usage.output_tokens):
            cost = (
                (usage.input_tokens * (model.cost.input or 0)) / 1_000_000
                + (usage.output_tokens * (model.cost.output or 0)) / 1_000_000
                + (usage.cache_read_tokens * (model.cost.cache_read or 0)) / 1_000_000
            )

        return AssistantMessage(
            content=content,
            metadata=AssistantMessageMetadata(
                usage=usage,
                stop_reason=stop_reason,
                cost=cost,
            ),
        )


def _map_stop_reason(finish_reason: str) -> str:
    return {
        "stop":         "end_turn",
        "tool_calls":   "tool_use",
        "length":       "max_tokens",
        "content_filter": "content_filter",
    }.get(finish_reason, finish_reason)


def _map_reasoning_effort(level: str, model: Model) -> Optional[str]:
    """Map ThinkingLevel to OpenAI reasoning_effort string via model.thinking_level_map."""
    if model.thinking_level_map:
        level_params = model.thinking_level_map.get(level, {})
        if effort := level_params.get("reasoning_effort"):
            return effort
    # Fallback mapping
    return {
        "minimal": "low",
        "low":     "low",
        "medium":  "medium",
        "high":    "high",
        "xhigh":   "high",
    }.get(level)
