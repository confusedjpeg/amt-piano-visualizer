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
        program: int = 0,
    ) -> pretty_midi.PrettyMIDI:
        """Move all notes into a single Instrument on the specified channel.

        Channel 0 = Right Hand, Channel 1 = Left Hand.

        Args:
            midi: Input PrettyMIDI object.
            channel: Target MIDI channel (0 or 1).
            program: MIDI program number (default 0 = Acoustic Grand Piano).

        Returns:
            A new PrettyMIDI object with all notes consolidated into one
            instrument on the target channel.
        """
        all_notes = []
        all_control_changes = []
        name = "Right Hand" if channel == 0 else "Left Hand"

        for instrument in midi.instruments:
            all_notes.extend(copy.deepcopy(instrument.notes))
            all_control_changes.extend(copy.deepcopy(instrument.control_changes))

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

    # ── Strict Ghost Note Pruning Rules ──────────────────────────────────

    @staticmethod
    def filter_minimum_duration(
        midi: pretty_midi.PrettyMIDI,
        min_duration_ms: float = 80.0,
    ) -> pretty_midi.PrettyMIDI:
        """Rule 1 — Minimum Duration Hard Cap.

        Unconditionally delete any note shorter than min_duration_ms.
        A real human pressing a piano key will almost never hold it for
        less than ~80ms, even when playing very fast.  Anything shorter
        is an AI hallucination or an audio glitch — velocity is irrelevant.

        This is stricter than ``filter_short_notes`` (which uses 50ms as a
        safety net later in the chain).  This runs **first** as a brute-
        force blip killer.

        Args:
            midi: Input PrettyMIDI object.
            min_duration_ms: Minimum note duration in milliseconds.
                Notes shorter than this are always deleted. Default 80ms.

        Returns:
            The same PrettyMIDI object with ultra-short notes removed.
        """
        min_duration_sec = min_duration_ms / 1000.0
        total_removed = 0

        for instrument in midi.instruments:
            original_count = len(instrument.notes)
            instrument.notes = [
                n for n in instrument.notes
                if (n.end - n.start) >= min_duration_sec
            ]
            total_removed += original_count - len(instrument.notes)

        if total_removed > 0:
            from src.utils.logger import get_logger
            get_logger(__name__).debug(
                f"Rule 1 (Min Duration): removed {total_removed} notes "
                f"shorter than {min_duration_ms}ms"
            )

        return midi

    @staticmethod
    def filter_polyphony_choke(
        midi: pretty_midi.PrettyMIDI,
        max_chord_notes: int = 4,
    ) -> pretty_midi.PrettyMIDI:
        """Rule 2 — Polyphony Choke (Chord Thinner).

        When AI hears a complex chord, it often hallucinates extra notes
        inside that chord because of overtones.  This creates muddy,
        dissonant clusters — especially in the left hand.

        For each time instant, if more than ``max_chord_notes`` notes are
        sounding simultaneously on one instrument:

        1. Keep the **lowest** note (bass root).
        2. Keep the **highest** note (harmony top).
        3. Sort all inner notes by velocity (ascending).
        4. Delete the quietest inner notes until only ``max_chord_notes``
           remain.

        This runs **before** velocity normalization so the raw AI
        velocities are available for pruning decisions.

        Args:
            midi: Input PrettyMIDI object.
            max_chord_notes: Maximum simultaneous notes allowed per
                instrument before choking. Default 4.

        Returns:
            The same PrettyMIDI object with overstuffed chords thinned.
        """
        if max_chord_notes < 2:
            return midi  # Need at least bass + top

        total_removed = 0

        for instrument in midi.instruments:
            if not instrument.notes:
                continue

            # Build a sorted timeline of ON/OFF events
            events: list[tuple[float, bool, int]] = []
            for idx, note in enumerate(instrument.notes):
                events.append((note.start, True, idx))   # ON
                events.append((note.end, False, idx))     # OFF

            # Sort: by time, then OFF before ON at the same timestamp
            events.sort(key=lambda e: (e[0], 0 if not e[1] else 1))

            active_ids: set[int] = set()
            notes_to_remove: set[int] = set()

            for _time, is_on, note_idx in events:
                if is_on:
                    active_ids.add(note_idx)

                    # Get currently active, non-removed notes
                    live_ids = [
                        i for i in active_ids if i not in notes_to_remove
                    ]
                    if len(live_ids) <= max_chord_notes:
                        continue

                    # We have too many notes — need to choke
                    live_notes = [
                        (i, instrument.notes[i]) for i in live_ids
                    ]

                    # Sort by pitch to find lowest and highest
                    live_notes.sort(key=lambda x: x[1].pitch)
                    lowest_id = live_notes[0][0]
                    highest_id = live_notes[-1][0]

                    # Inner notes (everything except lowest and highest)
                    inner = [
                        (i, n) for i, n in live_notes
                        if i != lowest_id and i != highest_id
                    ]

                    # Sort inner by velocity ascending (quietest first)
                    inner.sort(key=lambda x: x[1].velocity)

                    # How many inner notes can we keep?
                    # Total = 2 (lowest + highest) + kept_inner = max_chord_notes
                    max_inner = max_chord_notes - 2
                    # Delete the quietest inner notes beyond the budget
                    for i, _n in inner[:len(inner) - max_inner]:
                        notes_to_remove.add(i)
                else:
                    active_ids.discard(note_idx)

            removed = len(notes_to_remove)
            total_removed += removed
            if removed > 0:
                instrument.notes = [
                    n for idx, n in enumerate(instrument.notes)
                    if idx not in notes_to_remove
                ]

        if total_removed > 0:
            from src.utils.logger import get_logger
            get_logger(__name__).debug(
                f"Rule 2 (Polyphony Choke): removed {total_removed} "
                f"excess inner notes from overstuffed chords"
            )

        return midi

    @staticmethod
    def filter_reverb_shadow(
        midi: pretty_midi.PrettyMIDI,
        shadow_vel_threshold: int = 45,
        loud_vel_threshold: int = 70,
        shadow_window_ms: float = 200.0,
    ) -> pretty_midi.PrettyMIDI:
        """Rule 3 — Reverb Shadow Filter.

        Sometimes a loud chord is played, and ~200ms later the AI spits
        out a tiny, quiet stray note.  This is because the AI is hearing
        the echo of the chord bouncing off the walls of the recording
        studio.

        For each quiet note (velocity < ``shadow_vel_threshold``):
        look backward in time for any note that **ended** within
        ``shadow_window_ms`` and had velocity > ``loud_vel_threshold``.
        If such a loud "parent" note exists, the quiet note lives in
        its reverb shadow — delete it.

        This runs **before** velocity normalization so the raw AI
        velocities are available.

        Args:
            midi: Input PrettyMIDI object.
            shadow_vel_threshold: Velocity ceiling for a "quiet" note
                that is a candidate for shadow deletion. Default 45.
            loud_vel_threshold: Velocity floor for a "loud" note that
                can cast a reverb shadow. Default 70.
            shadow_window_ms: Maximum time window in milliseconds to
                look backward from the quiet note's start for a loud
                parent. Default 200ms.

        Returns:
            The same PrettyMIDI object with reverb shadow ghosts removed.
        """
        shadow_window_sec = shadow_window_ms / 1000.0
        total_removed = 0

        for instrument in midi.instruments:
            if not instrument.notes:
                continue

            # Sort all notes by start time for efficient scanning
            sorted_notes = sorted(instrument.notes, key=lambda n: n.start)

            # Pre-compute the list of "loud" notes for shadow casting
            loud_notes = [
                n for n in sorted_notes if n.velocity > loud_vel_threshold
            ]
            if not loud_notes:
                continue

            surviving: list[pretty_midi.Note] = []
            loud_idx = 0  # sliding pointer into loud_notes

            for note in sorted_notes:
                if note.velocity < shadow_vel_threshold:
                    # Advance past loud notes that ended too long ago
                    while (
                        loud_idx < len(loud_notes)
                        and loud_notes[loud_idx].end
                        < note.start - shadow_window_sec
                    ):
                        loud_idx += 1

                    # Check remaining loud notes within the window
                    is_shadow = False
                    for j in range(loud_idx, len(loud_notes)):
                        loud = loud_notes[j]
                        time_since_loud_end = note.start - loud.end
                        if time_since_loud_end > shadow_window_sec:
                            # Past the window — no later loud notes can
                            # match either (they start even later)
                            break
                        if 0 <= time_since_loud_end:
                            is_shadow = True
                            break
                        # Also catch cases where the loud note is still
                        # sounding when the quiet ghost appears
                        if loud.start <= note.start <= loud.end:
                            is_shadow = True
                            break

                    if is_shadow:
                        total_removed += 1
                        continue

                surviving.append(note)

            instrument.notes = surviving

        if total_removed > 0:
            from src.utils.logger import get_logger
            get_logger(__name__).debug(
                f"Rule 3 (Reverb Shadow): removed {total_removed} "
                f"quiet notes living in the shadow of loud chords"
            )

        return midi

