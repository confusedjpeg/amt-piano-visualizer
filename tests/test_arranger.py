"""Tests for src.arranger — AlgorithmicArranger, BeatAnalyzer, ChordDetector, PatternLibrary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from src.arranger.arranger import AlgorithmicArranger
from src.arranger.beat_analyzer import BeatAnalyzer
from src.arranger.chord_detector import ChordDetector, CHORD_TEMPLATES
from src.arranger.pattern_library import PatternLibrary
from src.pipeline.config import ArrangerConfig
from src.pipeline.models import BeatGrid, ChordEvent, PatternEvent, ArrangementPattern


class TestPatternLibrary:
    """Tests for PatternLibrary."""

    def test_load_patterns(self, tmp_path):
        """Should load all JSON patterns from a directory."""
        pattern_data = {
            "name": "test_pattern",
            "beats_per_bar": 4,
            "events": [
                {"beat_position": 1.0, "note_type": "root", "octave_offset": -2,
                 "velocity": 80, "duration_beats": 1.0},
            ],
        }
        (tmp_path / "test_pattern.json").write_text(json.dumps(pattern_data))

        lib = PatternLibrary(tmp_path)
        assert "test_pattern" in lib.list_patterns()

        pattern = lib.get_pattern("test_pattern")
        assert pattern.name == "test_pattern"
        assert len(pattern.events) == 1

    def test_missing_pattern_raises(self, tmp_path):
        """Requesting a non-existent pattern should raise KeyError."""
        lib = PatternLibrary(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            lib.get_pattern("nonexistent")


class TestChordDetector:
    """Tests for ChordDetector."""

    def test_match_chord_templates_exist(self):
        """Ensure all expected chord types are in the template dictionary."""
        expected_types = {"major", "minor", "diminished", "augmented",
                          "dominant7", "major7", "minor7", "sus2", "sus4"}
        assert set(CHORD_TEMPLATES.keys()) == expected_types

    def test_consolidation(self):
        """Consecutive identical chords should be merged."""
        detector = ChordDetector()
        chords = [
            ChordEvent(0.0, 0.5, "Cmaj", 48, "major", [0, 4, 7]),
            ChordEvent(0.5, 1.0, "Cmaj", 48, "major", [0, 4, 7]),
            ChordEvent(1.0, 1.5, "Am", 57, "minor", [0, 3, 7]),
        ]
        result = detector._consolidate(chords)
        assert len(result) == 2
        assert result[0].start_time == 0.0
        assert result[0].end_time == 1.0
        assert result[1].chord_label == "Am"

    def test_detect_key_g_major(self):
        """Should detect G major (root=7) from a G-D-Em-C progression."""
        detector = ChordDetector()
        # 4 beat-sync chroma vectors: Gmaj, Dmaj, Emin, Cmaj
        beat_chroma = np.zeros((4, 12), dtype=float)
        # Gmaj: G(7), B(11), D(2)
        beat_chroma[0, [7, 11, 2]] = 1.0
        # Dmaj: D(2), F#(6), A(9)
        beat_chroma[1, [2, 6, 9]] = 1.0
        # Emin: E(4), G(7), B(11)
        beat_chroma[2, [4, 7, 11]] = 1.0
        # Cmaj: C(0), E(4), G(7)
        beat_chroma[3, [0, 4, 7]] = 1.0

        key = detector._detect_key(beat_chroma)
        assert key == 7  # G

    def test_detect_key_c_major(self):
        """Should detect C major (root=0) from a C-F-G-C progression."""
        detector = ChordDetector()
        beat_chroma = np.zeros((4, 12), dtype=float)
        # Cmaj: C(0), E(4), G(7)
        beat_chroma[0, [0, 4, 7]] = 1.0
        # Fmaj: F(5), A(9), C(0)
        beat_chroma[1, [5, 9, 0]] = 1.0
        # Gmaj: G(7), B(11), D(2)
        beat_chroma[2, [7, 11, 2]] = 1.0
        # Cmaj: C(0), E(4), G(7)
        beat_chroma[3, [0, 4, 7]] = 1.0

        key = detector._detect_key(beat_chroma)
        assert key == 0  # C

    def test_normalise_roots_g_major_placement(self):
        """In G major, D should be placed above G (no wild register jump)."""
        detector = ChordDetector()
        chords = [
            ChordEvent(0.0, 1.0, "Gmaj", 55, "major", [0, 4, 7]),
            ChordEvent(1.0, 2.0, "Dmaj", 50, "major", [0, 4, 7]),
        ]
        result = detector._normalise_roots(chords, key_root=7)
        # G (root=7) stays at 55+(7-7)%12 = 55
        # D (root=2) → 55+(2-7)%12 = 55+7 = 62 → above G
        assert result[1].root_note > result[0].root_note
        assert result[0].root_note == 55  # G3
        assert result[1].root_note == 62  # D4

    def test_normalise_roots_stays_in_piano_range(self):
        """All normalized roots must stay within MIDI 24–84."""
        detector = ChordDetector()
        # Chords spanning all 12 pitch classes
        chords = [
            ChordEvent(i * 0.5, (i + 1) * 0.5, f"{n}",
                       48 + i, "major", [0, 4, 7])
            for i, n in enumerate(["C", "C#", "D", "D#", "E", "F",
                                   "F#", "G", "G#", "A", "A#", "B"])
        ]
        result = detector._normalise_roots(chords, key_root=0)
        for c in result:
            assert 24 <= c.root_note <= 84, f"Root {c.root_note} out of range"

    def test_normalise_roots_consecutive_no_wild_jumps(self):
        """Consecutive chords in the same key should stay within 12 semitones."""
        detector = ChordDetector()
        # G-D-Em-C progression in G major
        chords = [
            ChordEvent(0.0, 1.0, "Gmaj", 55, "major", [0, 4, 7]),
            ChordEvent(1.0, 2.0, "Dmaj", 50, "major", [0, 4, 7]),
            ChordEvent(2.0, 3.0, "Em",   52, "minor", [0, 3, 7]),
            ChordEvent(3.0, 4.0, "Cmaj", 48, "major", [0, 4, 7]),
        ]
        result = detector._normalise_roots(chords, key_root=7)
        for i in range(len(result) - 1):
            diff = abs(result[i + 1].root_note - result[i].root_note)
            assert diff <= 12, f"Jump of {diff} between chord {i} and {i+1}"

    def test_normalise_roots_empty(self):
        """Normalising an empty list should return an empty list."""
        detector = ChordDetector()
        result = detector._normalise_roots([], key_root=0)
        assert result == []

    def test_normalise_roots_retains_chord_structure(self):
        """Normalisation preserves chord type, intervals, and timing."""
        detector = ChordDetector()
        chords = [
            ChordEvent(0.0, 1.0, "Am", 57, "minor", [0, 3, 7]),
        ]
        result = detector._normalise_roots(chords, key_root=0)
        assert result[0].chord_type == "minor"
        assert result[0].intervals == [0, 3, 7]
        assert result[0].start_time == 0.0
        assert result[0].end_time == 1.0


class TestAlgorithmicArranger:
    """Tests for AlgorithmicArranger."""

    def test_resolve_note_type_root(self):
        """Root note type should return a single note."""
        pitches = AlgorithmicArranger._resolve_note_type(
            "root", root_midi=48, intervals=[0, 4, 7], octave_offset=-1
        )
        assert pitches == [36]  # C3 - 1 octave = C2

    def test_resolve_note_type_triad(self):
        """Triad note type should return all chord tones."""
        pitches = AlgorithmicArranger._resolve_note_type(
            "triad", root_midi=48, intervals=[0, 4, 7], octave_offset=0
        )
        assert pitches == [48, 52, 55]  # C3, E3, G3

    def test_resolve_note_type_fifth(self):
        """Fifth note type should return root + P5."""
        pitches = AlgorithmicArranger._resolve_note_type(
            "fifth", root_midi=48, intervals=[0, 4, 7], octave_offset=0
        )
        assert pitches == [48, 55]  # C3, G3

    def test_find_chord_at_time(self):
        """Should return the chord active at a given timestamp."""
        chords = [
            ChordEvent(0.0, 2.0, "Cmaj", 48, "major", [0, 4, 7]),
            ChordEvent(2.0, 4.0, "Gmaj", 55, "major", [0, 4, 7]),
        ]
        assert AlgorithmicArranger._find_chord_at_time(chords, 1.0).chord_label == "Cmaj"
        assert AlgorithmicArranger._find_chord_at_time(chords, 3.0).chord_label == "Gmaj"

    def test_find_chord_fallback(self):
        """Time past all chords should return the last chord."""
        chords = [
            ChordEvent(0.0, 2.0, "Cmaj", 48, "major", [0, 4, 7]),
        ]
        result = AlgorithmicArranger._find_chord_at_time(chords, 5.0)
        assert result.chord_label == "Cmaj"

    def test_arrangement_produces_notes(self, tmp_path, beat_grid, chord_events):
        """Full arrangement should produce a non-empty MIDI file."""
        # Create a pattern
        pattern_data = {
            "name": "test",
            "beats_per_bar": 4,
            "events": [
                {"beat_position": 1.0, "note_type": "root", "octave_offset": -2,
                 "velocity": 80, "duration_beats": 1.0},
                {"beat_position": 3.0, "note_type": "triad", "octave_offset": -1,
                 "velocity": 60, "duration_beats": 0.5},
            ],
        }
        pattern_dir = tmp_path / "patterns"
        pattern_dir.mkdir()
        (pattern_dir / "test.json").write_text(json.dumps(pattern_data))

        config = ArrangerConfig(default_pattern="test", pattern_dir=pattern_dir)
        lib = PatternLibrary(pattern_dir)
        arranger = AlgorithmicArranger(
            chord_detector=ChordDetector(),
            beat_analyzer=BeatAnalyzer(),
            pattern_library=lib,
            config=config,
        )

        # Directly test _generate_midi instead of full arrange() which needs audio
        midi = arranger._generate_midi(beat_grid, chord_events, lib.get_pattern("test"))
        total_notes = sum(len(i.notes) for i in midi.instruments)
        assert total_notes > 0
        assert midi.instruments[0].name == "Left Hand"
