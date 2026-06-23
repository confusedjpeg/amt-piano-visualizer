"""Tests for src.transcription.piano_transcriber — PianoTranscriber."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pretty_midi
import pytest

from src.pipeline.config import PianoTranscriptionConfig
from src.pipeline.errors import TranscriptionError
from src.transcription.piano_transcriber import PianoTranscriber


class TestPianoTranscriber:
    """Unit tests for PianoTranscriber."""

    def _make_transcriber(self, **overrides) -> PianoTranscriber:
        config = PianoTranscriptionConfig(device="cpu", **overrides)
        return PianoTranscriber(config)

    def test_missing_audio_raises(self, tmp_path):
        """Non-existent audio file should raise FileNotFoundError."""
        pt = self._make_transcriber()
        with pytest.raises(FileNotFoundError):
            pt.transcribe(tmp_path / "missing.wav", tmp_path / "out.mid")

    @patch("src.transcription.piano_transcriber.PianoTranscriber._run_basic_pitch")
    def test_empty_output_warns(self, mock_run, tmp_path):
        """Empty MIDI output should log a warning but not crash."""
        # _run_basic_pitch returns a PrettyMIDI; return an empty one
        def fake_transcribe(audio_path):
            empty_midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            empty_midi.instruments.append(
                pretty_midi.Instrument(program=0, name="Piano")
            )
            return empty_midi

        mock_run.side_effect = fake_transcribe

        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake")

        pt = self._make_transcriber()
        result = pt.transcribe(audio_path, tmp_path / "out.mid")
        assert result.exists()

    @patch("src.transcription.piano_transcriber.PianoTranscriber._run_basic_pitch")
    def test_polyphony_preserved(self, mock_run, tmp_path):
        """Piano transcription should preserve polyphonic output."""
        def fake_transcribe(audio_path):
            midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            inst = pretty_midi.Instrument(program=0, name="Piano")
            # C major chord — 3 simultaneous notes
            for pitch in [60, 64, 67]:
                inst.notes.append(
                    pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
                )
            midi.instruments.append(inst)
            return midi

        mock_run.side_effect = fake_transcribe

        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake")

        pt = self._make_transcriber()
        result_path = pt.transcribe(audio_path, tmp_path / "out.mid")

        result_midi = pretty_midi.PrettyMIDI(str(result_path))
        total_notes = sum(len(i.notes) for i in result_midi.instruments)
        assert total_notes == 3

    @patch("src.transcription.piano_transcriber.PianoTranscriber._run_basic_pitch")
    def test_sustain_pedal_preserved(self, mock_run, tmp_path):
        """CC64 (sustain pedal) control changes should be preserved."""
        def fake_transcribe(audio_path):
            midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            inst = pretty_midi.Instrument(program=0, name="Piano")
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0)
            )
            # Sustain pedal CC64
            inst.control_changes.append(
                pretty_midi.ControlChange(number=64, value=127, time=0.0)
            )
            inst.control_changes.append(
                pretty_midi.ControlChange(number=64, value=0, time=0.5)
            )
            midi.instruments.append(inst)
            return midi

        mock_run.side_effect = fake_transcribe

        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake")

        pt = self._make_transcriber()
        result_path = pt.transcribe(audio_path, tmp_path / "out.mid")

        result_midi = pretty_midi.PrettyMIDI(str(result_path))
        cc_events = result_midi.instruments[0].control_changes
        cc64_events = [cc for cc in cc_events if cc.number == 64]
        assert len(cc64_events) == 2
