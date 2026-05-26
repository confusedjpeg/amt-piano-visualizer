"""Tests for src.arranger — AlgorithmicArranger, BeatAnalyzer, ChordDetector, PatternLibrary."""

from __future__ import annotations

import json
from pathlib import Path

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
