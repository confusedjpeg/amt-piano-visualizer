"""Tests for src.midi.merger — MidiMerger."""

from __future__ import annotations

from pathlib import Path

import pretty_midi
import pytest

from src.midi.merger import MidiMerger


class TestMidiMerger:
    """Unit tests for MidiMerger."""

    def _write_test_midi(
        self,
        path: Path,
        notes: list[tuple[int, float, float]],
        name: str = "Test",
        tempo: float = 120.0,
    ) -> Path:
        """Helper to create a test MIDI file."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        inst = pretty_midi.Instrument(program=0, name=name)
        for pitch, start, end in notes:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
            )
        midi.instruments.append(inst)
        midi.write(str(path))
        return path

    def test_merge_two_midis(self, tmp_path):
        """Merging two MIDI files should produce a single file with both instruments."""
        rh = self._write_test_midi(
            tmp_path / "rh.mid",
            [(60, 0.0, 1.0), (64, 0.5, 1.5)],
            name="Right Hand",
        )
        lh = self._write_test_midi(
            tmp_path / "lh.mid",
            [(48, 0.0, 1.0), (52, 0.5, 1.5)],
            name="Left Hand",
        )

        output = tmp_path / "merged.mid"
        MidiMerger.merge([rh, lh], output)

        result = pretty_midi.PrettyMIDI(str(output))
        assert len(result.instruments) == 2
        total_notes = sum(len(i.notes) for i in result.instruments)
        assert total_notes == 4

    def test_merge_preserves_tempo(self, tmp_path):
        """Merged MIDI should use the tempo from the first file."""
        rh = self._write_test_midi(
            tmp_path / "rh.mid",
            [(60, 0.0, 1.0)],
            tempo=140.0,
        )
        lh = self._write_test_midi(
            tmp_path / "lh.mid",
            [(48, 0.0, 1.0)],
            tempo=100.0,
        )

        output = tmp_path / "merged.mid"
        MidiMerger.merge([rh, lh], output)

        result = pretty_midi.PrettyMIDI(str(output))
        # Use get_tempo_changes instead of estimate_tempo (may fail with few notes)
        _, tempos = result.get_tempo_changes()
        assert len(tempos) > 0
        assert abs(tempos[0] - 140.0) < 1.0

    def test_merge_with_tempo_override(self, tmp_path):
        """Explicit tempo should override the source tempo."""
        rh = self._write_test_midi(
            tmp_path / "rh.mid",
            [(60, 0.0, 1.0)],
            tempo=120.0,
        )

        output = tmp_path / "merged.mid"
        MidiMerger.merge([rh], output, tempo=90.0)

        result = pretty_midi.PrettyMIDI(str(output))
        _, tempos = result.get_tempo_changes()
        assert len(tempos) > 0
        assert abs(tempos[0] - 90.0) < 1.0

    def test_merge_preserves_control_changes(self, tmp_path):
        """Control changes should be preserved in the merged output."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Piano")
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0)
        )
        inst.control_changes.append(
            pretty_midi.ControlChange(number=64, value=127, time=0.0)
        )
        midi.instruments.append(inst)

        rh_path = tmp_path / "rh.mid"
        midi.write(str(rh_path))

        output = tmp_path / "merged.mid"
        MidiMerger.merge([rh_path], output)

        result = pretty_midi.PrettyMIDI(str(output))
        cc_events = result.instruments[0].control_changes
        assert len(cc_events) == 1
        assert cc_events[0].number == 64

    def test_empty_inputs_raises(self):
        """Empty input list should raise ValueError."""
        with pytest.raises(ValueError, match="(?i)at least one"):
            MidiMerger.merge([], Path("out.mid"))

    def test_missing_file_raises(self, tmp_path):
        """Non-existent MIDI file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MidiMerger.merge(
                [tmp_path / "nonexistent.mid"],
                tmp_path / "out.mid",
            )
