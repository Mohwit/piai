"""
piai/utils/diagnostics.py — Diagnostic creation and attachment helpers.
Surfaces overflow, retry exhaustion, and partial failures onto AssistantMessage.diagnostics[].
"""
from __future__ import annotations

import time
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from piai.types import AssistantMessage, Diagnostic


def make_diagnostic(
    type: str,
    error: Optional[Exception] = None,
    **details: Any,
) -> "Diagnostic":
    """
    Create a Diagnostic dataclass.

    Args:
        type:    A string identifier, e.g. "context_overflow", "retry_exhausted", "provider_error"
        error:   The originating exception, if any
        details: Additional key-value context (e.g. model_id, attempt_count)
    """
    from piai.types import Diagnostic, DiagnosticError

    diag_error = None
    if error is not None:
        diag_error = DiagnosticError(
            message=str(error),
            name=type(error).__name__,
            stack=None,
        )

    return Diagnostic(
        type=type,
        timestamp=time.time(),
        error=diag_error,
        details=details if details else None,
    )


def attach_diagnostic(message: "AssistantMessage", diag: "Diagnostic") -> None:
    """Append a Diagnostic to message.diagnostics in place."""
    message.diagnostics.append(diag)
