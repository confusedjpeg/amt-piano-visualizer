"""Tests for src.transcription.vocal_transcriber — VocalTranscriber."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pretty_midi
import pytest

from src.pipeline.config import VocalTranscriptionConfig
from src.pipeline.errors import TranscriptionError
from src.transcription.vocal_transcriber import VocalTranscriber


class TestVocalTranscriber:
    """Unit tests for VocalTranscriber."""

    def _make_transcriber(self, **overrides) -> VocalTranscriber:
        config = VocalTranscriptionConfig(**overrides)
        return VocalTranscriber(config)

    def test_missing_vocals_path_raises(self, tmp_path):
        """Non-existent vocals file should raise FileNotFoundError."""
        vt = self._make_transcriber()
        with pytest.raises(FileNotFoundError):
            vt.transcribe(tmp_path / "missing.wav", tmp_path / "out.mid")

    @patch("src.transcription.vocal_transcriber.VocalTranscriber._run_basic_pitch")
    def test_empty_transcription_produces_valid_midi(
        self, mock_bp, tmp_path
    ):
        """Silent audio → BasicPitch returns empty MIDI → should not crash."""
        empty_midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        empty_midi.instruments.append(
            pretty_midi.Instrument(program=0, name="empty")
        )
        mock_bp.return_value = empty_midi

        audio_path = tmp_path / "vocals.wav"
        audio_path.write_bytes(b"fake")  # Content doesn't matter — mock

        vt = self._make_transcriber()
        result = vt.transcribe(audio_path, tmp_path / "out.mid")
        assert result.exists()

    @patch("src.transcription.vocal_transcriber.VocalTranscriber._run_basic_pitch")
    def test_pitch_bends_stripped(self, mock_bp, sample_vocals_midi, tmp_path):
        """Pitch bends should be removed from the output."""
        mock_bp.return_value = sample_vocals_midi

        audio_path = tmp_path / "vocals.wav"
        audio_path.write_bytes(b"fake")

        vt = self._make_transcriber()
        result_path = vt.transcribe(audio_path, tmp_path / "out.mid")

        result_midi = pretty_midi.PrettyMIDI(str(result_path))
        for inst in result_midi.instruments:
            assert len(inst.pitch_bends) == 0

    @patch("src.transcription.vocal_transcriber.VocalTranscriber._run_basic_pitch")
    def test_notes_clamped_to_vocal_range(
        self, mock_bp, sample_vocals_midi, tmp_path
    ):
        """Notes outside the vocal range should be removed."""
        mock_bp.return_value = sample_vocals_midi

        audio_path = tmp_path / "vocals.wav"
        audio_path.write_bytes(b"fake")

        vt = self._make_transcriber()
        result_path = vt.transcribe(audio_path, tmp_path / "out.mid")

        result_midi = pretty_midi.PrettyMIDI(str(result_path))
        for inst in result_midi.instruments:
            for note in inst.notes:
                assert 43 <= note.pitch <= 84, f"Note {note.pitch} out of vocal range"

    @patch("src.transcription.vocal_transcriber.VocalTranscriber._run_basic_pitch")
    def test_channel_assignment(self, mock_bp, sample_vocals_midi, tmp_path):
        """All notes should be on a single instrument named 'Right Hand'."""
        mock_bp.return_value = sample_vocals_midi

        audio_path = tmp_path / "vocals.wav"
        audio_path.write_bytes(b"fake")

        vt = self._make_transcriber()
        result_path = vt.transcribe(audio_path, tmp_path / "out.mid")

        result_midi = pretty_midi.PrettyMIDI(str(result_path))
        assert len(result_midi.instruments) == 1
        assert result_midi.instruments[0].name == "Right Hand"

    def test_hz_to_midi_conversion(self):
        """Verify Hz-to-MIDI conversion for known values."""
        # A4 = 440 Hz = MIDI 69
        assert VocalTranscriber._hz_to_midi(440.0) == 69
        # C4 ≈ 261.63 Hz = MIDI 60
        assert VocalTranscriber._hz_to_midi(261.63) == 60
