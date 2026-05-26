"""
MIDI quantizer — snaps note start/end times to a musical grid.

Handles edge cases: zero-duration notes, duplicate-pitch collisions,
and overlapping notes on the same pitch.
"""

from __future__ import annotations

import copy

import pretty_midi

from src.utils.logger import get_logger

log = get_logger(__name__)


def _safe_estimate_tempo(midi: pretty_midi.PrettyMIDI, default: float = 120.0) -> float:
    """Safely extract tempo from a PrettyMIDI object.

    Prefers the explicit tempo from the MIDI's tempo change list
    over the heuristic estimate_tempo().
    """
    tempo_times, tempos = midi.get_tempo_changes()
    if len(tempos) > 0:
        return float(tempos[0])
    try:
        return midi.estimate_tempo()
    except ValueError:
        return default


class Quantizer:
    """Snap MIDI note start and end times to a musical grid."""

    def __init__(self, grid_resolution: int = 16) -> None:
        """
        Args:
            grid_resolution: 16 for 16th notes, 32 for 32nd notes, etc.
        """
        self._grid_resolution = grid_resolution

    def quantize(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Quantize all note on/off times to the nearest grid position.

        Args:
            midi: Input PrettyMIDI object.

        Returns:
            New PrettyMIDI object with quantized note timings.
        """
        tempo = _safe_estimate_tempo(midi)
        grid_sec = self._compute_grid_spacing(tempo)

        log.info(
            f"Quantizing to {self._grid_resolution}th notes "
            f"(grid={grid_sec:.4f}s at {tempo:.1f} BPM)"
        )

        result = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        for instrument in midi.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
            )
            # Preserve control changes
            new_instrument.control_changes = copy.deepcopy(
                instrument.control_changes
            )

            quantized_notes = []
            for note in instrument.notes:
                new_note = self._quantize_note(note, grid_sec)
                if new_note is not None:
                    quantized_notes.append(new_note)

            # Deduplicate notes that collapsed to identical (start, end, pitch)
            quantized_notes = self._deduplicate(quantized_notes)

            # Merge overlapping notes on the same pitch
            quantized_notes = self._merge_overlapping(quantized_notes)

            new_instrument.notes = quantized_notes
            result.instruments.append(new_instrument)

        return result

    # ── Private Helpers ──────────────────────────────────────────────────

    def _compute_grid_spacing(self, tempo: float) -> float:
        """Compute the grid spacing in seconds.

        For 16th notes at 120 BPM: (60/120) / (16/4) = 0.5 / 4 = 0.125s
        """
        beat_duration = 60.0 / tempo
        divisions_per_beat = self._grid_resolution / 4
        return beat_duration / divisions_per_beat

    @staticmethod
    def _quantize_note(
        note: pretty_midi.Note, grid_sec: float
    ) -> pretty_midi.Note | None:
        """Snap a single note's start/end to the grid."""
        new_start = round(note.start / grid_sec) * grid_sec
        new_end = round(note.end / grid_sec) * grid_sec

        # Ensure minimum duration of 1 grid unit
        if new_end <= new_start:
            new_end = new_start + grid_sec

        return pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=new_start,
            end=new_end,
        )

    @staticmethod
    def _deduplicate(notes: list[pretty_midi.Note]) -> list[pretty_midi.Note]:
        """Remove exact duplicates (same start, end, pitch)."""
        seen: set[tuple[float, float, int]] = set()
        unique: list[pretty_midi.Note] = []

        for note in notes:
            key = (round(note.start, 6), round(note.end, 6), note.pitch)
            if key not in seen:
                seen.add(key)
                unique.append(note)

        return unique

    @staticmethod
    def _merge_overlapping(
        notes: list[pretty_midi.Note],
    ) -> list[pretty_midi.Note]:
        """Merge overlapping notes on the same pitch into single longer notes."""
        if not notes:
            return notes

        # Group by pitch
        by_pitch: dict[int, list[pretty_midi.Note]] = {}
        for note in notes:
            by_pitch.setdefault(note.pitch, []).append(note)

        merged: list[pretty_midi.Note] = []

        for pitch, pitch_notes in by_pitch.items():
            # Sort by start time
            pitch_notes.sort(key=lambda n: n.start)

            current = pitch_notes[0]
            for next_note in pitch_notes[1:]:
                if next_note.start <= current.end:
                    # Overlap — extend current note
                    current = pretty_midi.Note(
                        velocity=max(current.velocity, next_note.velocity),
                        pitch=pitch,
                        start=current.start,
                        end=max(current.end, next_note.end),
                    )
                else:
                    merged.append(current)
                    current = next_note
            merged.append(current)

        # Sort by start time for consistent output
        merged.sort(key=lambda n: (n.start, n.pitch))
        return merged
