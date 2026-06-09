"""
piai/utils/abort.py — AbortSignal wrapper around asyncio.Event.
Mirrors the AbortSignal interface from the original TypeScript pi-ai.
"""
from __future__ import annotations

import asyncio


class AbortSignal:
    """
    Wraps asyncio.Event to provide the same interface as the original AbortSignal.

    StreamOptions.abort_signal accepts either asyncio.Event or AbortSignal.
    Providers should check .aborted at stream start (handles pre-cancelled signals)
    and await .wait() inside loops.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def abort(self) -> None:
        """Trigger the abort signal."""
        self._event.set()

    @property
    def aborted(self) -> bool:
        """True if abort() has been called."""
        return self._event.is_set()

    async def wait(self) -> None:
        """Await until abort() is called."""
        await self._event.wait()

    @property
    def event(self) -> asyncio.Event:
        """Underlying asyncio.Event for use with asyncio.wait() etc."""
        return self._event


def is_aborted(signal: asyncio.Event | AbortSignal | None) -> bool:
    """Check whether an abort signal (either type) has been triggered."""
    if signal is None:
        return False
    if isinstance(signal, AbortSignal):
        return signal.aborted
    return signal.is_set()
