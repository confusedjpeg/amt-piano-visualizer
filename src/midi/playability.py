"""
Playability filter — enforces human-playability constraints on piano MIDI.

Constraints:
  1. Maximum hand span (interval between lowest and highest simultaneous note).
  2. Maximum polyphony per hand (number of simultaneous notes).

Uses a time-slice event analysis approach to inspect the set of notes
sounding at every point in time and prune as needed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pretty_midi

from src.pipeline.config import PlayabilityConfig
from src.utils.logger import get_logger

log = get_logger(__name__)


def _safe_estimate_tempo(midi: pretty_midi.PrettyMIDI, default: float = 120.0) -> float:
    """Safely extract tempo from a PrettyMIDI object.

    Prefers the explicit tempo from the MIDI's tempo change list
    (set via initial_tempo) over the heuristic estimate_tempo().
    Falls back to *default* if neither is available.
    """
    # Prefer the explicit tempo embedded in the MIDI file
    tempo_times, tempos = midi.get_tempo_changes()
    if len(tempos) > 0:
        return float(tempos[0])
    # Fall back to estimation from note spacing
    try:
        return midi.estimate_tempo()
    except ValueError:
        return default

# Interval-to-semitone mapping for pruning priority
INTERVAL_SEMITONES: dict[str, int] = {
    "P1": 0,  "m2": 1,  "M2": 2,  "m3": 3,  "M3": 4,
    "P4": 5,  "TT": 6,  "P5": 7,  "m6": 8,  "M6": 9,
    "m7": 10, "M7": 11, "P8": 12,
}


@dataclass
class _NoteEvent:
    """Internal representation of a note on/off event for timeline processing."""

    time: float
    is_on: bool
    note: pretty_midi.Note

    def sort_key(self) -> tuple[float, int]:
        """Sort key: by time, then OFF before ON at the same timestamp."""
        return (self.time, 0 if not self.is_on else 1)


class PlayabilityFilter:
    """
    Enforce human-playability constraints on piano MIDI:
    - Maximum hand span (interval between lowest and highest simultaneous note)
    - Maximum polyphony per hand
    """

    def __init__(self, config: PlayabilityConfig) -> None:
        self._max_span = config.max_hand_span_semitones
        self._max_polyphony = config.max_polyphony_per_hand
        self._pruning_semitones = [
            INTERVAL_SEMITONES[name]
            for name in config.pruning_priority
            if name in INTERVAL_SEMITONES
        ]

    def apply(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Apply all playability constraints.

        Processes each instrument independently.

        Args:
            midi: Input PrettyMIDI object.

        Returns:
            A new PrettyMIDI object satisfying all constraints.
        """
        result = pretty_midi.PrettyMIDI(initial_tempo=_safe_estimate_tempo(midi))

        for instrument in midi.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
            )
            new_instrument.control_changes = copy.deepcopy(
                instrument.control_changes
            )

            notes = copy.deepcopy(instrument.notes)
            original_count = len(notes)

            # 1. Enforce hand span
            notes = self._enforce_hand_span(notes, self._max_span)

            # 2. Enforce polyphony
            notes = self._enforce_polyphony(notes, self._max_polyphony)

            pruned_count = original_count - len(notes)
            if pruned_count > 0:
                log.info(
                    f"Playability [{instrument.name}]: "
                    f"pruned {pruned_count}/{original_count} notes"
                )

            new_instrument.notes = notes
            result.instruments.append(new_instrument)

        return result

    # ── Hand Span Enforcement ────────────────────────────────────────────

    def _enforce_hand_span(
        self, notes: list[pretty_midi.Note], max_span: int
    ) -> list[pretty_midi.Note]:
        """
        At each time slice, if the interval between the lowest and highest
        note exceeds max_span semitones, prune inner notes.

        Strategy:
            - Always preserve the lowest note (bass/root) and highest note (melody).
            - Remove inner notes starting from those closest to the median pitch.
        """
        if not notes:
            return notes

        events = self._build_timeline(notes)
        active: set[int] = set()  # Set of note IDs (index in notes list)
        notes_to_remove: set[int] = set()

        # Create an ID mapping
        note_ids = {id(n): idx for idx, n in enumerate(notes)}

        for event in events:
            nid = note_ids[id(event.note)]

            if event.is_on:
                active.add(nid)
                # Check span constraint
                active_notes = [notes[i] for i in active if i not in notes_to_remove]
                if len(active_notes) >= 2:
                    pitches = sorted(n.pitch for n in active_notes)
                    span = pitches[-1] - pitches[0]

                    while span > max_span and len(active_notes) > 1:
                        if len(active_notes) > 2:
                            # Remove the inner note closest to the median
                            to_remove = self._find_inner_to_prune(active_notes)
                        else:
                            # Only 2 notes left, still exceeding span —
                            # remove the highest note (preserve bass/root)
                            sorted_by_pitch = sorted(active_notes, key=lambda n: n.pitch)
                            to_remove = sorted_by_pitch[-1]

                        if to_remove is None:
                            break
                        notes_to_remove.add(note_ids[id(to_remove)])
                        active_notes = [
                            n for n in active_notes
                            if note_ids[id(n)] not in notes_to_remove
                        ]
                        if len(active_notes) >= 2:
                            pitches = sorted(n.pitch for n in active_notes)
                            span = pitches[-1] - pitches[0]
                        else:
                            break
            else:
                active.discard(nid)

        return [n for idx, n in enumerate(notes) if idx not in notes_to_remove]

    # ── Polyphony Enforcement ────────────────────────────────────────────

    def _enforce_polyphony(
        self, notes: list[pretty_midi.Note], max_voices: int
    ) -> list[pretty_midi.Note]:
        """
        At each time slice, if more than max_voices notes sound simultaneously,
        prune excess notes using the configured pruning priority.
        """
        if not notes:
            return notes

        events = self._build_timeline(notes)
        active: set[int] = set()
        notes_to_remove: set[int] = set()

        note_ids = {id(n): idx for idx, n in enumerate(notes)}

        for event in events:
            nid = note_ids[id(event.note)]

            if event.is_on:
                active.add(nid)
                active_notes = [notes[i] for i in active if i not in notes_to_remove]

                while len(active_notes) > max_voices:
                    to_remove = self._find_polyphony_prune_target(active_notes)
                    if to_remove is None:
                        break
                    notes_to_remove.add(note_ids[id(to_remove)])
                    active_notes = [
                        n for n in active_notes
                        if note_ids[id(n)] not in notes_to_remove
                    ]
            else:
                active.discard(nid)

        return [n for idx, n in enumerate(notes) if idx not in notes_to_remove]

    # ── Pruning Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _find_inner_to_prune(
        active_notes: list[pretty_midi.Note],
    ) -> pretty_midi.Note | None:
        """Find the inner note closest to the median pitch.

        Preserves the lowest and highest notes; removes the one whose
        pitch is nearest the median of all sounding pitches.
        """
        if len(active_notes) <= 2:
            return None

        pitches = sorted(active_notes, key=lambda n: n.pitch)
        lowest = pitches[0]
        highest = pitches[-1]
        inner = pitches[1:-1]

        if not inner:
            return None

        median_pitch = (lowest.pitch + highest.pitch) / 2
        # Sort inner notes by distance from median (closest first)
        inner.sort(key=lambda n: abs(n.pitch - median_pitch))
        return inner[0]

    def _find_polyphony_prune_target(
        self, active_notes: list[pretty_midi.Note],
    ) -> pretty_midi.Note | None:
        """Find the best note to prune based on the pruning priority.

        Strategy:
            1. Classify each note's interval relative to the lowest sounding note.
            2. Remove notes matching the pruning priority list (P5 first, then M3, then m3).
            3. As a last resort, remove the note closest to the median pitch.
        """
        if len(active_notes) <= 1:
            return None

        sorted_notes = sorted(active_notes, key=lambda n: n.pitch)
        lowest_pitch = sorted_notes[0].pitch

        # Try each pruning priority interval
        for target_semitones in self._pruning_semitones:
            for note in sorted_notes[1:]:  # Skip the lowest note
                interval = (note.pitch - lowest_pitch) % 12
                if interval == target_semitones:
                    return note

        # Fallback: remove the note closest to the median
        return self._find_inner_to_prune(active_notes)

    # ── Timeline Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_timeline(
        notes: list[pretty_midi.Note],
    ) -> list[_NoteEvent]:
        """Build a sorted timeline of note ON/OFF events."""
        events: list[_NoteEvent] = []
        for note in notes:
            events.append(_NoteEvent(time=note.start, is_on=True, note=note))
            events.append(_NoteEvent(time=note.end, is_on=False, note=note))

        events.sort(key=lambda e: e.sort_key())
        return events
