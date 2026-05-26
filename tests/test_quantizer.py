"""Tests for src.midi.quantizer — Quantizer."""

from __future__ import annotations

import pretty_midi
import pytest

from src.midi.quantizer import Quantizer


class TestQuantizer:
    """Unit tests for Quantizer."""

    def _make_quantizer(self, grid: int = 16) -> Quantizer:
        return Quantizer(grid_resolution=grid)

    def test_off_grid_note_snapped(self):
        """An off-grid note should be snapped to the nearest 16th position."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        # At 120 BPM, 16th note grid = 0.125s
        # Note starts at 0.06 (should snap to 0.0 or 0.125)
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.06, end=0.3)
        )
        midi.instruments.append(inst)

        q = self._make_quantizer(16)
        result = q.quantize(midi)

        note = result.instruments[0].notes[0]
        grid_sec = 0.125  # 60/120/4
        # Start should be on grid
        remainder = note.start % grid_sec
        assert remainder < 1e-6 or abs(remainder - grid_sec) < 1e-6

    def test_on_grid_note_unchanged(self):
        """A note already on the grid should not move."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5)
        )
        midi.instruments.append(inst)

        q = self._make_quantizer(16)
        result = q.quantize(midi)

        note = result.instruments[0].notes[0]
        assert abs(note.start - 0.0) < 1e-6
        assert abs(note.end - 0.5) < 1e-6

    def test_zero_duration_gets_minimum(self):
        """A note that collapses to zero duration should get minimum 1 grid unit."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        # Both start and end will snap to 0.0
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.01, end=0.02)
        )
        midi.instruments.append(inst)

        q = self._make_quantizer(16)
        result = q.quantize(midi)

        note = result.instruments[0].notes[0]
        assert note.end > note.start

    def test_duplicate_notes_merged(self):
        """Notes with identical (start, end, pitch) after quantization should merge."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        # Two notes that will collapse to the same grid position
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.01, end=0.13)
        )
        inst.notes.append(
            pretty_midi.Note(velocity=70, pitch=60, start=0.02, end=0.14)
        )
        midi.instruments.append(inst)

        q = self._make_quantizer(16)
        result = q.quantize(midi)

        note_count = len(result.instruments[0].notes)
        # Should be 1 after deduplication/merging
        assert note_count == 1

    def test_different_pitches_preserved(self):
        """Chord notes at the same time but different pitches should be kept."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        for pitch in [60, 64, 67]:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=0.5)
            )
        midi.instruments.append(inst)

        q = self._make_quantizer(16)
        result = q.quantize(midi)
        assert len(result.instruments[0].notes) == 3

    def test_control_changes_preserved(self):
        """Control changes should be passed through to the output."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5)
        )
        inst.control_changes.append(
            pretty_midi.ControlChange(number=64, value=127, time=0.0)
        )
        midi.instruments.append(inst)

        q = self._make_quantizer(16)
        result = q.quantize(midi)
        assert len(result.instruments[0].control_changes) == 1

    def test_32nd_note_grid(self):
        """32nd note grid should produce finer quantization."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.04, end=0.2)
        )
        midi.instruments.append(inst)

        q = self._make_quantizer(32)
        result = q.quantize(midi)

        note = result.instruments[0].notes[0]
        grid_sec = 60.0 / 120.0 / 8  # 32nd note at 120 BPM = 0.0625s
        remainder = note.start % grid_sec
        assert remainder < 1e-6 or abs(remainder - grid_sec) < 1e-6
