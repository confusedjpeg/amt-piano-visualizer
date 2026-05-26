"""
Chord detection from audio using chroma feature analysis.

Uses CQT chroma vectors and template matching against a dictionary
of chord quality templates to identify chord progressions.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from src.pipeline.models import BeatGrid, ChordEvent
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Chord Templates ──────────────────────────────────────────────────────────
# Each list is a 12-element binary chroma vector.
# Index 0 = root, index 1 = minor 2nd, ..., index 11 = major 7th.

CHORD_TEMPLATES: dict[str, list[int]] = {
    "major":      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "minor":      [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "diminished": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "augmented":  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "dominant7":  [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "major7":     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "minor7":     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "sus2":       [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "sus4":       [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
}

# Interval-to-semitone mapping (also used by PlayabilityFilter)
INTERVAL_SEMITONES: dict[str, int] = {
    "P1": 0,  "m2": 1,  "M2": 2,  "m3": 3,  "M3": 4,
    "P4": 5,  "TT": 6,  "P5": 7,  "m6": 8,  "M6": 9,
    "m7": 10, "M7": 11, "P8": 12,
}

# Pitch class names
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Chord type to label suffix
CHORD_SUFFIXES: dict[str, str] = {
    "major": "maj",
    "minor": "m",
    "diminished": "dim",
    "augmented": "aug",
    "dominant7": "7",
    "major7": "maj7",
    "minor7": "m7",
    "sus2": "sus2",
    "sus4": "sus4",
}

# Chord type to interval list (semitones from root)
CHORD_INTERVALS: dict[str, list[int]] = {
    "major":      [0, 4, 7],
    "minor":      [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented":  [0, 4, 8],
    "dominant7":  [0, 4, 7, 10],
    "major7":     [0, 4, 7, 11],
    "minor7":     [0, 3, 7, 10],
    "sus2":       [0, 2, 7],
    "sus4":       [0, 5, 7],
}


class ChordDetector:
    """Detect chord progressions from audio using chroma feature analysis."""

    def detect(
        self, audio_path: Path, beat_grid: BeatGrid
    ) -> list[ChordEvent]:
        """
        Detect chords aligned to beat positions.

        Args:
            audio_path: Path to the audio file.
            beat_grid: Beat timing information from BeatAnalyzer.

        Returns:
            List of ChordEvent, with consecutive identical chords consolidated.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        log.info(f"Detecting chords in {audio_path}...")

        y, sr = librosa.load(str(audio_path), sr=22050)

        # Compute CQT chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Get beat-synchronised chroma (average chroma within each beat)
        beat_frames = librosa.time_to_frames(beat_grid.beat_times, sr=sr)
        beat_chroma = self._beat_sync_chroma(chroma, beat_frames)

        # Classify each beat
        raw_chords = self._classify_beats(beat_chroma, beat_grid)

        # Consolidate consecutive identical chords
        consolidated = self._consolidate(raw_chords)

        log.info(f"Detected {len(consolidated)} chord segments")
        return consolidated

    # ── Private Helpers ──────────────────────────────────────────────────

    def _beat_sync_chroma(
        self, chroma: np.ndarray, beat_frames: np.ndarray
    ) -> np.ndarray:
        """Average chroma vectors within each beat window.

        Returns:
            Array of shape (num_beats, 12).
        """
        beat_chroma: list[np.ndarray] = []

        for i in range(len(beat_frames)):
            start = beat_frames[i]
            end = beat_frames[i + 1] if i + 1 < len(beat_frames) else chroma.shape[1]
            if start >= end:
                end = start + 1
            segment = chroma[:, start:end]
            beat_chroma.append(segment.mean(axis=1))

        return np.array(beat_chroma)  # (num_beats, 12)

    def _classify_beats(
        self,
        beat_chroma: np.ndarray,
        beat_grid: BeatGrid,
    ) -> list[ChordEvent]:
        """Classify each beat's chroma vector against chord templates."""
        chords: list[ChordEvent] = []
        beat_times = beat_grid.beat_times

        for i, chroma_vec in enumerate(beat_chroma):
            start_time = float(beat_times[i])
            end_time = (
                float(beat_times[i + 1])
                if i + 1 < len(beat_times)
                else start_time + 60.0 / beat_grid.tempo  # estimate last beat
            )

            root, chord_type, score = self._match_chord(chroma_vec)

            chord_label = f"{PITCH_CLASSES[root]}{CHORD_SUFFIXES[chord_type]}"
            # Root note as MIDI number in octave 3 (C3 = 48)
            root_midi = 48 + root

            chords.append(
                ChordEvent(
                    start_time=start_time,
                    end_time=end_time,
                    chord_label=chord_label,
                    root_note=root_midi,
                    chord_type=chord_type,
                    intervals=CHORD_INTERVALS[chord_type],
                )
            )

        return chords

    def _match_chord(
        self, chroma_vec: np.ndarray
    ) -> tuple[int, str, float]:
        """Find the best (root, chord_type) match for a chroma vector.

        Tests all 12 pitch classes × all chord templates and returns
        the highest cosine-similarity match.

        Returns:
            (root_pitch_class, chord_type, similarity_score)
        """
        best_root = 0
        best_type = "major"
        best_score = -1.0

        for root in range(12):
            # Circularly shift the chroma so that 'root' is at index 0
            shifted = np.roll(chroma_vec, -root)

            for chord_type, template in CHORD_TEMPLATES.items():
                template_arr = np.array(template, dtype=float)
                score = self._cosine_similarity(shifted, template_arr)

                if score > best_score:
                    best_score = score
                    best_root = root
                    best_type = chord_type

        return best_root, best_type, best_score

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _consolidate(chords: list[ChordEvent]) -> list[ChordEvent]:
        """Merge consecutive identical chords into single spans."""
        if not chords:
            return []

        consolidated: list[ChordEvent] = [chords[0]]

        for chord in chords[1:]:
            prev = consolidated[-1]
            if (
                chord.chord_label == prev.chord_label
                and chord.root_note == prev.root_note
            ):
                # Extend the previous chord span
                consolidated[-1] = ChordEvent(
                    start_time=prev.start_time,
                    end_time=chord.end_time,
                    chord_label=prev.chord_label,
                    root_note=prev.root_note,
                    chord_type=prev.chord_type,
                    intervals=prev.intervals,
                )
            else:
                consolidated.append(chord)

        return consolidated
