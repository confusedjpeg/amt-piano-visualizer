"""
Algorithmic arranger — generates piano accompaniment MIDI by mapping
detected chords onto rhythmic patterns aligned with beat positions.
"""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from src.arranger.beat_analyzer import BeatAnalyzer
from src.arranger.chord_detector import ChordDetector
from src.arranger.pattern_library import PatternLibrary
from src.pipeline.config import ArrangerConfig
from src.pipeline.errors import ArrangementError
from src.pipeline.models import (
    ArrangementPattern,
    BeatGrid,
    ChordEvent,
    PatternEvent,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class AlgorithmicArranger:
    """
    Generate a piano accompaniment MIDI by mapping detected chords
    onto rhythmic patterns aligned with beat positions.
    """

    def __init__(
        self,
        chord_detector: ChordDetector,
        beat_analyzer: BeatAnalyzer,
        pattern_library: PatternLibrary,
        config: ArrangerConfig,
    ) -> None:
        self._chord_detector = chord_detector
        self._beat_analyzer = beat_analyzer
        self._pattern_library = pattern_library
        self._config = config

    def arrange(
        self,
        audio_path: Path,
        output_path: Path,
        pattern_name: str | None = None,
    ) -> Path:
        """
        Generate accompaniment MIDI from audio analysis.

        Args:
            audio_path: Path to the audio file (e.g., instrumental.wav).
            output_path: Path for the output MIDI file.
            pattern_name: Which rhythmic pattern to apply (defaults to config).

        Returns:
            Path to the generated accompaniment MIDI.
            All notes are assigned to MIDI Channel 1 (Left Hand).

        Raises:
            ArrangementError: If arrangement generation fails.
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pattern_name = pattern_name or self._config.default_pattern

        try:
            # 1. Analyze beats
            beat_grid = self._beat_analyzer.analyze(audio_path)

            # 2. Detect chords
            chords = self._chord_detector.detect(audio_path, beat_grid)

            if not chords:
                log.warning("No chords detected — generating empty MIDI.")
                empty_midi = pretty_midi.PrettyMIDI(initial_tempo=beat_grid.tempo)
                empty_midi.write(str(output_path))
                return output_path

            # 3. Load pattern
            pattern = self._pattern_library.get_pattern(pattern_name)

            # 4. Generate MIDI
            midi = self._generate_midi(beat_grid, chords, pattern)
            midi.write(str(output_path))

            total_notes = sum(len(i.notes) for i in midi.instruments)
            log.info(
                f"Arrangement complete: {total_notes} notes, "
                f"pattern='{pattern_name}' → {output_path}"
            )

            return output_path

        except KeyError as exc:
            raise ArrangementError(
                message=f"Pattern not found: {exc}", cause=exc
            ) from exc
        except Exception as exc:
            raise ArrangementError(
                message=f"Arrangement failed for {audio_path}",
                cause=exc,
            ) from exc

    # ── Private Helpers ──────────────────────────────────────────────────

    def _generate_midi(
        self,
        beat_grid: BeatGrid,
        chords: list[ChordEvent],
        pattern: ArrangementPattern,
    ) -> pretty_midi.PrettyMIDI:
        """Generate a PrettyMIDI object from chords and a rhythmic pattern."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=beat_grid.tempo)

        # Left Hand instrument (Channel 1)
        lh_instrument = pretty_midi.Instrument(
            program=0,         # Acoustic Grand Piano
            is_drum=False,
            name="Left Hand",
        )

        # Duration of one beat in seconds
        beat_duration = 60.0 / beat_grid.tempo

        # Iterate over bars (defined by consecutive downbeats)
        downbeats = beat_grid.downbeat_times
        for bar_idx in range(len(downbeats)):
            bar_start = float(downbeats[bar_idx])
            bar_end = (
                float(downbeats[bar_idx + 1])
                if bar_idx + 1 < len(downbeats)
                else bar_start + pattern.beats_per_bar * beat_duration
            )

            # Find the active chord at this bar's start time
            active_chord = self._find_chord_at_time(chords, bar_start)
            if active_chord is None:
                continue

            # Apply each pattern event within this bar
            for event in pattern.events:
                notes = self._resolve_pattern_event(
                    event, active_chord, bar_start, beat_duration
                )
                lh_instrument.notes.extend(notes)

        midi.instruments.append(lh_instrument)
        return midi

    def _resolve_pattern_event(
        self,
        event: PatternEvent,
        chord: ChordEvent,
        bar_start: float,
        beat_duration: float,
    ) -> list[pretty_midi.Note]:
        """Convert a pattern event + chord into concrete MIDI notes."""
        # Compute absolute time
        # beat_position is 1-indexed (1.0 = beat 1 of the bar)
        offset_beats = event.beat_position - 1.0
        start_time = bar_start + offset_beats * beat_duration
        end_time = start_time + event.duration_beats * beat_duration

        # Resolve pitches based on note_type
        pitches = self._resolve_note_type(
            event.note_type, chord.root_note, chord.intervals, event.octave_offset
        )

        return [
            pretty_midi.Note(
                velocity=event.velocity,
                pitch=max(0, min(127, p)),  # Clamp to valid MIDI range
                start=start_time,
                end=end_time,
            )
            for p in pitches
        ]

    @staticmethod
    def _resolve_note_type(
        note_type: str,
        root_midi: int,
        intervals: list[int],
        octave_offset: int,
    ) -> list[int]:
        """Resolve a note_type string to concrete MIDI note numbers.

        Args:
            note_type: "root", "fifth", "triad", or "octave".
            root_midi: MIDI note of the chord root (e.g., 48 for C3).
            intervals: Semitone intervals of the chord (e.g., [0, 4, 7]).
            octave_offset: Octave transposition (e.g., -2 → two octaves down).

        Returns:
            List of MIDI note numbers.
        """
        base = root_midi + (octave_offset * 12)

        if note_type == "root":
            return [base]
        elif note_type == "fifth":
            # Root + Perfect 5th (7 semitones)
            return [base, base + 7]
        elif note_type == "triad":
            # All notes of the chord at the specified octave
            return [base + interval for interval in intervals]
        elif note_type == "octave":
            return [base]
        else:
            # Fallback: just the root
            return [base]

    @staticmethod
    def _find_chord_at_time(
        chords: list[ChordEvent], time: float
    ) -> ChordEvent | None:
        """Find the chord active at a given timestamp."""
        for chord in chords:
            if chord.start_time <= time < chord.end_time:
                return chord

        # Fallback: return the last chord if time is past all chord spans
        return chords[-1] if chords else None
