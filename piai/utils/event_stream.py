"""
piai/utils/event_stream.py — Direct port of EventStream<TEvent, TResult> from event-stream.ts.

Producer calls emit()/done()/error(); consumer iterates with `async for`.
result() awaits the final TResult set by done().
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Generic, Optional, TypeVar

TEvent = TypeVar("TEvent")
TResult = TypeVar("TResult")

_SENTINEL = object()


class EventStream(Generic[TEvent, TResult]):
    """
    asyncio.Queue-based event stream.

    Usage (producer side — inside provider _run coroutine):
        es.emit(StartEvent())
        es.emit(TextDeltaEvent(text="hello"))
        es.done(final_message)   # or es.error(exc)

    Usage (consumer side):
        async for event in es:
            ...
        msg = await es.result()
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._result_future: asyncio.Future[TResult] = asyncio.get_event_loop().create_future()
        self._finished = False

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def emit(self, event: TEvent) -> None:
        """Put an event on the queue for consumers to receive."""
        if self._finished:
            return
        self._queue.put_nowait(event)

    def done(self, result: TResult) -> None:
        """Signal end of stream with the final result."""
        if self._finished:
            return
        self._finished = True
        self._queue.put_nowait(_SENTINEL)
        if not self._result_future.done():
            self._result_future.set_result(result)

    def error(self, exc: Exception) -> None:
        """Signal an error — terminates the stream."""
        if self._finished:
            return
        self._finished = True
        self._queue.put_nowait(_SENTINEL)
        if not self._result_future.done():
            self._result_future.set_exception(exc)

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    async def result(self) -> TResult:
        """Await the final result (set by done())."""
        return await self._result_future

    def __aiter__(self) -> AsyncIterator[TEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[TEvent]:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break
            yield item  # type: ignore[misc]
