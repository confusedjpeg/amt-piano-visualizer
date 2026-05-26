"""
Custom exception hierarchy for the Audio-to-Piano-Synthesia pipeline.

Each pipeline step has a dedicated exception class that carries
the run_id and the original cause for structured error reporting.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all pipeline errors."""

    def __init__(
        self,
        message: str = "",
        run_id: str = "",
        cause: Exception | None = None,
    ) -> None:
        self.run_id = run_id
        self.cause = cause
        detail = f"[run={run_id}] " if run_id else ""
        if cause:
            detail += f"{message} — caused by {type(cause).__name__}: {cause}"
        else:
            detail += message
        super().__init__(detail)


class SeparationError(PipelineError):
    """Raised when Demucs stem separation fails."""


class TranscriptionError(PipelineError):
    """Raised when vocal or piano transcription fails."""


class ArrangementError(PipelineError):
    """Raised when the algorithmic arranger fails."""


class PlayabilityError(PipelineError):
    """Raised when playability filtering encounters an unrecoverable state."""


class RenderingError(PipelineError):
    """Raised when MIDIVisualizer fails to produce output."""
