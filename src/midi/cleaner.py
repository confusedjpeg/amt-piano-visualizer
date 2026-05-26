"""
MIDI cleaning utilities.

Provides static methods for removing artifacts and normalizing MIDI data
for piano playback: pitch-bend stripping, note-range clamping,
channel assignment, and instrument program setting.
"""

from __future__ import annotations

import copy

import pretty_midi


def _safe_estimate_tempo(midi: pretty_midi.PrettyMIDI, default: float = 120.0) -> float:
    """Safely extract tempo, preferring explicit tempo from the MIDI file."""
    tempo_times, tempos = midi.get_tempo_changes()
    if len(tempos) > 0:
        return float(tempos[0])
    try:
        return midi.estimate_tempo()
    except ValueError:
        return default


class MidiCleaner:
    """Remove artifacts and normalize MIDI data for piano playback."""

    @staticmethod
    def strip_pitch_bends(midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Remove all pitch bend events from all instruments.

        Args:
            midi: Input PrettyMIDI object.

        Returns:
            The same PrettyMIDI object with pitch bends cleared (modified in-place).
        """
        for instrument in midi.instruments:
            instrument.pitch_bends = []
        return midi

    @staticmethod
    def clamp_note_range(
        midi: pretty_midi.PrettyMIDI,
        min_note: int = 21,   # A0 — lowest piano key
        max_note: int = 108,  # C8 — highest piano key
    ) -> pretty_midi.PrettyMIDI:
        """Remove notes outside the specified MIDI note range.

        Args:
            midi: Input PrettyMIDI object.
            min_note: Lowest allowed MIDI note number (inclusive).
            max_note: Highest allowed MIDI note number (inclusive).

        Returns:
            The same PrettyMIDI object with out-of-range notes removed.
        """
        for instrument in midi.instruments:
            instrument.notes = [
                n for n in instrument.notes
                if min_note <= n.pitch <= max_note
            ]
        return midi

    @staticmethod
    def assign_channel(
        midi: pretty_midi.PrettyMIDI,
        channel: int,
    ) -> pretty_midi.PrettyMIDI:
        """Move all notes into a single Instrument on the specified channel.

        Channel 0 = Right Hand, Channel 1 = Left Hand.

        Args:
            midi: Input PrettyMIDI object.
            channel: Target MIDI channel (0 or 1).

        Returns:
            A new PrettyMIDI object with all notes consolidated into one
            instrument on the target channel.
        """
        all_notes = []
        all_control_changes = []
        program = 0
        name = "Right Hand" if channel == 0 else "Left Hand"

        for instrument in midi.instruments:
            all_notes.extend(copy.deepcopy(instrument.notes))
            all_control_changes.extend(copy.deepcopy(instrument.control_changes))
            program = instrument.program

        new_instrument = pretty_midi.Instrument(
            program=program,
            is_drum=False,
            name=name,
        )
        new_instrument.notes = all_notes
        new_instrument.control_changes = all_control_changes

        # Preserve the tempo from the original
        tempo = _safe_estimate_tempo(midi)
        result = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        result.instruments.append(new_instrument)
        return result

    @staticmethod
    def set_instrument(
        midi: pretty_midi.PrettyMIDI,
        program: int = 0,  # 0 = Acoustic Grand Piano
    ) -> pretty_midi.PrettyMIDI:
        """Set all instruments to the specified MIDI program.

        Args:
            midi: Input PrettyMIDI object.
            program: MIDI program number (0 = Acoustic Grand Piano).

        Returns:
            The same PrettyMIDI object with programs updated.
        """
        for instrument in midi.instruments:
            instrument.program = program
        return midi
