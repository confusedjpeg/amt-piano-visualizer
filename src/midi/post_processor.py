"""
Tempo-Aware MIDI Post-Processing.

Implements two musical corrections applied after raw AI transcription:

1. **Frame-Based Chord Quantization** — snaps near-simultaneous note onsets
   to the nearest 16th-note grid position so that chords trigger as solid
   visual blocks instead of smeared clusters.

2. **Smart Legato Stretcher** — extends notes to fill small AI-artifact
   silence gaps while respecting intentional musical rests and grooves.

All time-based math is derived dynamically from the song's actual BPM
(extracted via ``librosa.beat.beat_track``).  No hardcoded time values.

Two legato modes are available (controlled via ``PostProcessingConfig``):

- **Smart Legato** (conservative) — fills only small AI-artifact gaps
  (< 8th note) while preserving intentional musical rests.

- **Aggressive Legato / Chord Snapper** — stretches every note to the
  start of the next chord change, capped at a configurable max duration.
  Fixes the "not dragged out" problem from AI transcribers ignoring the
  sustain pedal.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pretty_midi

from src.pipeline.config import PostProcessingConfig
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Feature 1: Dynamic Tempo Extraction ──────────────────────────────────────

def extract_tempo(audio_path: str | Path) -> float:
    """Extract the global BPM from an audio file using beat tracking.

    Uses ``librosa.beat.beat_track`` to estimate the song's dominant tempo.
    Falls back to 120 BPM if beat tracking fails or returns an unusable value.

    Args:
        audio_path: Path to the audio file (WAV, MP3, FLAC, etc.).

    Returns:
        The estimated BPM as a float.
    """
    audio_path = str(audio_path)
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # librosa may return an ndarray (older versions) or a scalar
        if isinstance(tempo, np.ndarray):
            bpm = float(tempo[0]) if tempo.size > 0 else 120.0
        else:
            bpm = float(tempo)

        if bpm <= 0:
            log.warning(
                f"Beat tracking returned non-positive BPM ({bpm}); "
                f"falling back to 120 BPM"
            )
            return 120.0

        log.info(f"Extracted tempo: {bpm:.1f} BPM from {audio_path}")
        return bpm

    except Exception as exc:
        log.warning(
            f"Beat tracking failed ({exc}); falling back to 120 BPM"
        )
        return 120.0


def compute_grid_framework(bpm: float) -> dict[str, float]:
    """Derive the tempo-relative grid framework from a BPM value.

    All downstream time math uses these values instead of hardcoded
    constants, ensuring the processing adapts to any genre/tempo.

    Args:
        bpm: The song's beats per minute.

    Returns:
        A dict with the following keys:

        - ``bpm``: The input BPM.
        - ``seconds_per_beat``: Duration of one quarter note in seconds.
        - ``grid_size``: Duration of one 16th note (the quantization frame).
        - ``max_stretch_gap``: Maximum silence gap (8th note) that is
          considered an AI artifact and will be filled by legato stretch.
        - ``max_note_duration``: Absolute cap on stretched note duration
          (4 beats / 1 whole note) to prevent infinite ringing.
    """
    seconds_per_beat = 60.0 / bpm
    grid_size = seconds_per_beat / 4.0  # 16th note

    return {
        "bpm": bpm,
        "seconds_per_beat": seconds_per_beat,
        "grid_size": grid_size,
        "max_stretch_gap": seconds_per_beat / 2.0,      # 8th note
        "max_note_duration": seconds_per_beat * 4.0,     # whole note
    }


# ── Feature 2: Frame-Based Chord Quantization ───────────────────────────────

def quantize_note_starts(
    instrument: pretty_midi.Instrument,
    grid_size: float,
) -> pretty_midi.Instrument:
    """Snap every note's start time to the nearest 16th-note grid position.

    This forces notes that were played *almost* simultaneously (but
    transcribed a few milliseconds apart by the AI) to trigger at the
    exact same time, forming solid visual block chords.

    After quantization the note list is re-sorted chronologically by
    the new start times so that downstream processing (legato stretcher)
    sees a correct ordering.

    Args:
        instrument: The ``pretty_midi.Instrument`` to process
            (modified **in-place**).
        grid_size: Duration of one grid frame in seconds (16th note).

    Returns:
        The same ``Instrument`` object with quantized start times.
    """
    if grid_size <= 0:
        log.warning("grid_size <= 0; skipping chord quantization")
        return instrument

    notes_modified = 0

    for note in instrument.notes:
        original_start = note.start
        quantized_start = round(note.start / grid_size) * grid_size

        if quantized_start != original_start:
            # Shift the note's start; preserve its original duration
            duration = note.end - note.start
            note.start = quantized_start
            note.end = quantized_start + duration
            notes_modified += 1

    # Re-sort chronologically by new start time (stable sort preserves
    # the relative order of notes at the same start time)
    instrument.notes.sort(key=lambda n: (n.start, n.pitch))

    log.info(
        f"Chord quantization: snapped {notes_modified}/{len(instrument.notes)} "
        f"note starts to {grid_size * 1000:.1f}ms grid"
    )
    return instrument


# ── Feature 3: Smart Legato Stretcher ────────────────────────────────────────

def apply_smart_legato(
    instrument: pretty_midi.Instrument,
    max_stretch_gap: float,
    max_note_duration: float,
) -> pretty_midi.Instrument:
    """Stretch notes to fill small AI-artifact silence gaps.

    Iterates through the (already quantized & sorted) notes.  For each
    note, looks ahead to find the start of the very next note in the
    track.  If the silence gap between the current note's end and the
    next note's start is positive but smaller than ``max_stretch_gap``,
    the current note is extended to meet the next note — eliminating the
    choppy, disconnected feel caused by AI transcription.

    Gaps **larger** than ``max_stretch_gap`` are treated as intentional
    musical rests and are left untouched.

    An absolute cap (``max_note_duration``) prevents any single note from
    being stretched beyond a reasonable acoustic limit (e.g. one whole
    note), guarding against infinite ringing.

    Args:
        instrument: The ``pretty_midi.Instrument`` to process
            (modified **in-place**).  Notes **must** be sorted by start
            time before calling this function.
        max_stretch_gap: Maximum silence gap in seconds that is
            considered an AI artifact (e.g. an 8th-note rest).  Gaps
            larger than this are intentional rests.
        max_note_duration: Absolute maximum note duration in seconds.
            No note will be stretched beyond this cap regardless of the
            gap size.

    Returns:
        The same ``Instrument`` object with legato-stretched durations.
    """
    notes = instrument.notes
    if not notes:
        return instrument

    stretched_count = 0
    capped_count = 0

    for i in range(len(notes)):
        current = notes[i]

        # Find the start time of the very next note (by chronological order)
        if i + 1 < len(notes):
            next_start = notes[i + 1].start
        else:
            # Last note — nothing to stretch toward; just apply the cap
            next_start = None

        # ── Smart stretch ────────────────────────────────────────────
        if next_start is not None:
            gap = next_start - current.end

            if gap > 0 and gap < max_stretch_gap:
                # This is an AI artifact gap — fill it
                current.end = next_start
                stretched_count += 1

        # ── Absolute duration cap ────────────────────────────────────
        duration = current.end - current.start
        if duration > max_note_duration:
            current.end = current.start + max_note_duration
            capped_count += 1

    log.info(
        f"Smart legato: stretched {stretched_count} notes, "
        f"capped {capped_count} at {max_note_duration:.3f}s max"
    )
    return instrument


# ── Feature 3b: Aggressive Legato / Chord Snapper ───────────────────────────

def apply_aggressive_legato(
    instrument: pretty_midi.Instrument,
    max_note_duration: float,
) -> pretty_midi.Instrument:
    """Stretch every note to the start of the next chord change.

    This is the "Legato Chord Snapper" algorithm.  Unlike
    :func:`apply_smart_legato`, it does **not** try to preserve intentional
    rests.  Instead, every note is dragged forward in time until it meets
    the onset of the next note (regardless of how large the gap is),
    subject to an absolute duration cap.

    This is ideal for AI-transcribed accompaniment where the transcriber
    ignored the sustain pedal, producing many short, disconnected notes
    instead of long, ringing block chords.

    For each note the algorithm:

    1. Looks ahead to find the start time of the next **different** chord
       onset (i.e. the first note whose start time is strictly later than
       the current note's start time).
    2. Stretches ``note.end`` to that onset time.
    3. Clamps ``note.end`` so the total duration never exceeds
       ``max_note_duration``.

    Args:
        instrument: The ``pretty_midi.Instrument`` to process
            (modified **in-place**).  Notes **must** be sorted by start
            time before calling this function.
        max_note_duration: Absolute maximum note duration in seconds.
            Prevents infinite ringing even when the next chord is far away.

    Returns:
        The same ``Instrument`` object with aggressively stretched durations.
    """
    notes = instrument.notes
    if not notes:
        return instrument

    stretched_count = 0
    capped_count = 0
    untouched_count = 0

    for i in range(len(notes)):
        current = notes[i]

        # Find the start time of the next *different* chord onset
        next_chord_start = None
        for j in range(i + 1, len(notes)):
            if notes[j].start > current.start:
                next_chord_start = notes[j].start
                break

        # ── Aggressive stretch ───────────────────────────────────────
        if next_chord_start is not None:
            max_end = current.start + max_note_duration
            new_end = min(next_chord_start, max_end)

            if new_end > current.end:
                # Note end was extended
                current.end = new_end
                if new_end == next_chord_start:
                    stretched_count += 1
                else:
                    capped_count += 1
            else:
                untouched_count += 1
        else:
            # Last chord group — just apply the duration cap
            duration = current.end - current.start
            if duration > max_note_duration:
                current.end = current.start + max_note_duration
                capped_count += 1
            else:
                untouched_count += 1

    log.info(
        f"Aggressive legato (Chord Snapper): stretched {stretched_count}, "
        f"capped {capped_count}, untouched {untouched_count} notes "
        f"(max {max_note_duration:.3f}s)"
    )
    return instrument


# ── Public API: Full Post-Processing Pipeline ────────────────────────────────

def post_process_instrument(
    instrument: pretty_midi.Instrument,
    audio_path: str | Path,
    config: PostProcessingConfig | None = None,
) -> pretty_midi.Instrument:
    """Apply the full tempo-aware post-processing chain to an instrument.

    This is the primary entry point.  It:

    1. Extracts the song's BPM from the audio via ``librosa``.
    2. Derives tempo-relative grid constants (no hardcoded times).
    3. Quantizes note starts to the 16th-note grid (chord solidification).
    4. Applies legato stretching (aggressive or smart, depending on config).

    Args:
        instrument: A raw ``pretty_midi.Instrument`` from transcription.
            Modified **in-place** and also returned for convenience.
        audio_path: Path to the original audio file used for BPM extraction.
        config: Optional post-processing configuration.  When ``None``,
            uses the default ``PostProcessingConfig()``.

    Returns:
        The same ``Instrument`` object after all post-processing.
    """
    if config is None:
        config = PostProcessingConfig()

    # ── Step 1: Extract tempo ────────────────────────────────────────
    bpm = extract_tempo(audio_path)
    grid = compute_grid_framework(bpm)

    log.info(
        f"Post-processing grid framework: "
        f"BPM={grid['bpm']:.1f}, "
        f"grid={grid['grid_size'] * 1000:.1f}ms, "
        f"max_stretch={grid['max_stretch_gap'] * 1000:.1f}ms, "
        f"max_dur={grid['max_note_duration']:.3f}s, "
        f"aggressive_legato={config.aggressive_legato}"
    )

    # ── Step 2: Chord quantization ───────────────────────────────────
    quantize_note_starts(instrument, grid["grid_size"])

    # ── Step 3: Legato ───────────────────────────────────────────────
    if config.aggressive_legato:
        apply_aggressive_legato(
            instrument,
            max_note_duration=config.aggressive_legato_max_duration_s,
        )
    else:
        apply_smart_legato(
            instrument,
            max_stretch_gap=grid["max_stretch_gap"],
            max_note_duration=grid["max_note_duration"],
        )

    return instrument


def post_process_midi(
    midi: pretty_midi.PrettyMIDI,
    audio_path: str | Path,
    config: PostProcessingConfig | None = None,
) -> pretty_midi.PrettyMIDI:
    """Apply tempo-aware post-processing to every instrument in a MIDI object.

    Convenience wrapper around :func:`post_process_instrument` that
    iterates all instruments.  Useful when the caller has a full
    ``PrettyMIDI`` object rather than a single instrument.

    Args:
        midi: The ``PrettyMIDI`` object to process (modified in-place).
        audio_path: Path to the original audio file used for BPM extraction.
        config: Optional post-processing configuration.  When ``None``,
            uses the default ``PostProcessingConfig()``.

    Returns:
        The same ``PrettyMIDI`` object after post-processing.
    """
    if config is None:
        config = PostProcessingConfig()

    # Extract BPM once — shared across all instruments
    bpm = extract_tempo(audio_path)
    grid = compute_grid_framework(bpm)

    log.info(
        f"Post-processing {len(midi.instruments)} instrument(s) — "
        f"BPM={grid['bpm']:.1f}, "
        f"grid={grid['grid_size'] * 1000:.1f}ms, "
        f"aggressive_legato={config.aggressive_legato}"
    )

    for instrument in midi.instruments:
        log.debug(f"  Processing instrument: {instrument.name}")
        quantize_note_starts(instrument, grid["grid_size"])

        if config.aggressive_legato:
            apply_aggressive_legato(
                instrument,
                max_note_duration=config.aggressive_legato_max_duration_s,
            )
        else:
            apply_smart_legato(
                instrument,
                max_stretch_gap=grid["max_stretch_gap"],
                max_note_duration=grid["max_note_duration"],
            )

    return midi
