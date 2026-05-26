"""
Shared pytest fixtures for the pipeline test suite.

Provides synthetic MIDI files, audio samples, beat grids,
and chord progressions for use across all test modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.models import BeatGrid, ChordEvent


# ── MIDI Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_midi() -> pretty_midi.PrettyMIDI:
    """Create a minimal PrettyMIDI with a C major chord (C4, E4, G4)."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0, name="Piano")

    # C major chord at t=0.0 to t=1.0
    for pitch in [60, 64, 67]:  # C4, E4, G4
        instrument.notes.append(
            pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
        )

    midi.instruments.append(instrument)
    return midi


@pytest.fixture
def sample_vocals_midi() -> pretty_midi.PrettyMIDI:
    """Create a PrettyMIDI simulating vocal transcription output.

    Includes:
    - Pitch bends (to test stripping)
    - Out-of-range notes (to test clamping)
    - Very short notes (to test removal)
    - Normal vocal-range notes
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0, name="Vocals")

    # Normal notes in vocal range (MIDI 43-84)
    instrument.notes.append(
        pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5)  # C4
    )
    instrument.notes.append(
        pretty_midi.Note(velocity=70, pitch=67, start=0.5, end=1.0)  # G4
    )
    instrument.notes.append(
        pretty_midi.Note(velocity=75, pitch=72, start=1.0, end=1.5)  # C5
    )

    # Out-of-range note (too low — below E2/MIDI 43)
    instrument.notes.append(
        pretty_midi.Note(velocity=60, pitch=30, start=2.0, end=2.5)  # F#1
    )

    # Out-of-range note (too high — above C#6/MIDI 84)
    instrument.notes.append(
        pretty_midi.Note(velocity=60, pitch=90, start=3.0, end=3.5)  # F#6
    )

    # Very short note (should be removed)
    instrument.notes.append(
        pretty_midi.Note(velocity=50, pitch=65, start=4.0, end=4.02)  # 20ms
    )

    # Add pitch bends
    instrument.pitch_bends.append(
        pretty_midi.PitchBend(pitch=4000, time=0.25)
    )
    instrument.pitch_bends.append(
        pretty_midi.PitchBend(pitch=-2000, time=0.75)
    )

    midi.instruments.append(instrument)
    return midi


@pytest.fixture
def wide_span_midi() -> pretty_midi.PrettyMIDI:
    """Create a PrettyMIDI with notes spanning > 15 semitones.

    Notes: C3(48), E3(52), G3(55), C4(60), E5(76) — span = 28 ST.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0, name="Wide Span")

    for pitch in [48, 52, 55, 60, 76]:
        instrument.notes.append(
            pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
        )

    midi.instruments.append(instrument)
    return midi


@pytest.fixture
def dense_polyphony_midi() -> pretty_midi.PrettyMIDI:
    """Create a PrettyMIDI with > 4 simultaneous notes (within 15 ST span).

    Notes: C4(60), D4(62), E4(64), F4(65), G4(67), A4(69) — 6 notes, span = 9 ST.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0, name="Dense")

    for pitch in [60, 62, 64, 65, 67, 69]:
        instrument.notes.append(
            pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
        )

    midi.instruments.append(instrument)
    return midi


# ── Audio Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_audio_path(tmp_path) -> Path:
    """Generate a short sine wave WAV file for testing (1 second, 440 Hz)."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), audio, sr)
    return audio_path


@pytest.fixture
def stereo_audio_path(tmp_path) -> Path:
    """Generate a short stereo sine wave WAV file (1 second)."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 554 * t)
    audio = np.column_stack([left, right])

    audio_path = tmp_path / "test_stereo.wav"
    sf.write(str(audio_path), audio, sr)
    return audio_path


# ── Beat & Chord Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def beat_grid() -> BeatGrid:
    """Create a BeatGrid at 120 BPM with 16 beats (4 bars of 4/4)."""
    tempo = 120.0
    beat_duration = 60.0 / tempo  # 0.5 seconds per beat
    beat_times = np.array([i * beat_duration for i in range(16)])
    downbeat_times = beat_times[::4]  # Every 4th beat

    return BeatGrid(
        tempo=tempo,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        time_signature=(4, 4),
    )


@pytest.fixture
def chord_events() -> list[ChordEvent]:
    """Create a simple I-V-vi-IV progression in C major (4 bars)."""
    return [
        ChordEvent(
            start_time=0.0, end_time=2.0,
            chord_label="Cmaj", root_note=48,
            chord_type="major", intervals=[0, 4, 7],
        ),
        ChordEvent(
            start_time=2.0, end_time=4.0,
            chord_label="Gmaj", root_note=55,
            chord_type="major", intervals=[0, 4, 7],
        ),
        ChordEvent(
            start_time=4.0, end_time=6.0,
            chord_label="Am", root_note=57,
            chord_type="minor", intervals=[0, 3, 7],
        ),
        ChordEvent(
            start_time=6.0, end_time=8.0,
            chord_label="Fmaj", root_note=53,
            chord_type="major", intervals=[0, 4, 7],
        ),
    ]
