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

    @staticmethod
    def filter_short_notes(
        midi: pretty_midi.PrettyMIDI,
        min_duration_ms: float = 50.0,
    ) -> pretty_midi.PrettyMIDI:
        """Remove notes shorter than the minimum duration.

        Args:
            midi: Input PrettyMIDI object.
            min_duration_ms: Minimum note duration in milliseconds.

        Returns:
            The same PrettyMIDI object with short notes removed.
        """
        min_duration_sec = min_duration_ms / 1000.0
        for instrument in midi.instruments:
            instrument.notes = [
                n for n in instrument.notes
                if (n.end - n.start) >= min_duration_sec
            ]
        return midi

    @staticmethod
    def normalize_velocities(
        midi: pretty_midi.PrettyMIDI,
        min_vel: int = 60,
        max_vel: int = 100,
    ) -> pretty_midi.PrettyMIDI:
        """Normalize and compress note velocities to a harmonious range.

        Args:
            midi: Input PrettyMIDI object.
            min_vel: Minimum allowed velocity.
            max_vel: Maximum allowed velocity.

        Returns:
            The same PrettyMIDI object with velocities normalized.
        """
        for instrument in midi.instruments:
            if not instrument.notes:
                continue
            
            # Find current range
            velocities = [n.velocity for n in instrument.notes]
            current_min = min(velocities)
            current_max = max(velocities)
            current_range = current_max - current_min
            
            target_range = max_vel - min_vel

            for n in instrument.notes:
                if current_range == 0:
                    n.velocity = min_vel + target_range // 2
                else:
                    # Scale to new range
                    normalized = (n.velocity - current_min) / current_range
                    new_vel = int(min_vel + (normalized * target_range))
                    # Clamp just in case
                    n.velocity = max(min_vel, min(max_vel, new_vel))

        return midi

    @staticmethod
    def apply_legato(
        midi: pretty_midi.PrettyMIDI,
        extend_ms: float = 50.0,
    ) -> pretty_midi.PrettyMIDI:
        """Extend note durations to create a legato/sustain effect.

        Args:
            midi: Input PrettyMIDI object.
            extend_ms: Milliseconds to extend the end of each note.

        Returns:
            The same PrettyMIDI object with extended note endings.
        """
        extend_sec = extend_ms / 1000.0
        for instrument in midi.instruments:
            for n in instrument.notes:
                n.end += extend_sec
        return midi

    @staticmethod
    def filter_ghost_notes(
        midi: pretty_midi.PrettyMIDI,
        absolute_vel_floor: int = 25,
        reverb_vel_ceiling: int = 45,
        reverb_max_duration_ms: float = 150.0,
    ) -> pretty_midi.PrettyMIDI:
        """Remove ghost notes using a compound velocity + duration filter.

        This applies a tiered approach that is smarter than a flat velocity
        cutoff. It prevents deleting legitimate soft sustained chords while
        aggressively removing AI hallucinations:

        Tier 1 — "Absolute Garbage": velocity < absolute_vel_floor
            Always deleted. Way too quiet to be real; definitely an
            overtone, harmonic artifact, or noise floor pickup.

        Tier 2 — "Reverb Blip": velocity < reverb_vel_ceiling AND
            duration < reverb_max_duration_ms
            Deleted. If a note is both quiet AND short, it's almost
            certainly a hallucinated reverb echo or transient blip.

        Tier 3 — Everything else is kept. A note with velocity 35 held
            for 2 seconds is a real chord fading out and should survive.

        Args:
            midi: Input PrettyMIDI object.
            absolute_vel_floor: Velocity below which notes are always
                deleted (Tier 1). Default 25.
            reverb_vel_ceiling: Velocity ceiling for the compound check
                (Tier 2). Default 45.
            reverb_max_duration_ms: Duration ceiling in ms for the
                compound check (Tier 2). Default 150ms.

        Returns:
            The same PrettyMIDI object with ghost notes removed.
        """
        reverb_max_duration_sec = reverb_max_duration_ms / 1000.0

        total_removed = 0
        for instrument in midi.instruments:
            original_count = len(instrument.notes)
            surviving: list[pretty_midi.Note] = []

            for note in instrument.notes:
                duration = note.end - note.start

                # Tier 1: Absolute garbage — always delete
                if note.velocity < absolute_vel_floor:
                    continue

                # Tier 2: Reverb blip — quiet AND short = hallucination
                if (
                    note.velocity < reverb_vel_ceiling
                    and duration < reverb_max_duration_sec
                ):
                    continue

                # Tier 3: Keep everything else
                surviving.append(note)

            removed = original_count - len(surviving)
            total_removed += removed
            instrument.notes = surviving

        return midi
