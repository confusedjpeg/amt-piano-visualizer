"""
Data models used across the Audio-to-Piano-Synthesia pipeline.

All models are plain dataclasses to keep them lightweight
and free of framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ── Step 1: Stem Separation ──────────────────────────────────────────────────

@dataclass
class StemResult:
    """Paths to the four stems produced by Demucs."""

    vocals_path: Path
    bass_path: Path
    drums_path: Path
    other_path: Path


# ── Step 3B: Beat & Chord Analysis ───────────────────────────────────────────

@dataclass
class BeatGrid:
    """Tempo, beat, and downbeat timing information."""

    tempo: float
    beat_times: np.ndarray
    downbeat_times: np.ndarray
    time_signature: tuple[int, int] = (4, 4)


@dataclass
class ChordEvent:
    """A single chord spanning a time range."""

    start_time: float
    end_time: float
    chord_label: str          # e.g., "Cmaj"
    root_note: int            # MIDI note number of the chord root
    chord_type: str           # "major", "minor", "diminished", etc.
    intervals: list[int]      # Semitone intervals from root, e.g., [0, 4, 7]


# ── Step 3B: Arrangement Patterns ────────────────────────────────────────────

@dataclass
class PatternEvent:
    """A single rhythmic event within an arrangement pattern."""

    beat_position: float      # e.g., 1.0, 2.5 (beat within the bar)
    note_type: str            # "root" | "triad" | "octave" | "fifth"
    octave_offset: int        # e.g., -2 for bass register
    velocity: int             # 0-127
    duration_beats: float     # Duration in beats


@dataclass
class ArrangementPattern:
    """A rhythmic blueprint for generating accompaniment."""

    name: str
    beats_per_bar: int
    events: list[PatternEvent] = field(default_factory=list)


# ── Pipeline Result ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Output of a complete pipeline run."""

    midi_path: Path
    video_path: Path
    run_id: str
    duration_seconds: float
    steps_completed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
