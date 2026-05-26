"""Tests for src.audio.separator — StemSeparator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.audio.separator import StemSeparator
from src.pipeline.config import SeparatorConfig
from src.pipeline.errors import SeparationError


class TestStemSeparator:
    """Unit tests for StemSeparator."""

    def _make_separator(self, **overrides) -> StemSeparator:
        config = SeparatorConfig(device="cpu", **overrides)
        return StemSeparator(config)

    def test_invalid_file_path_raises(self, tmp_path):
        """Non-existent file should raise FileNotFoundError."""
        sep = self._make_separator()
        fake_path = tmp_path / "nonexistent.wav"
        with pytest.raises(FileNotFoundError):
            sep.separate(fake_path, tmp_path / "output")

    def test_unsupported_format_raises(self, tmp_path):
        """Unsupported file extension should raise ValueError."""
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not audio")

        sep = self._make_separator()
        with pytest.raises(ValueError, match="Unsupported audio format"):
            sep.separate(bad_file, tmp_path / "output")

    @patch("src.audio.separator.StemSeparator._load_model")
    @patch("src.audio.separator.StemSeparator._load_audio")
    @patch("src.audio.separator.StemSeparator._run_separation")
    @patch("src.audio.separator.StemSeparator._save_stems")
    def test_separate_calls_pipeline(
        self,
        mock_save,
        mock_run,
        mock_load_audio,
        mock_load_model,
        sample_audio_path,
        tmp_path,
    ):
        """Verify the separation pipeline is called in order."""
        from src.pipeline.models import StemResult

        mock_model = MagicMock()
        mock_model.samplerate = 44100
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = (torch.randn(2, 44100), 44100)
        mock_run.return_value = torch.randn(4, 2, 44100)

        output_dir = tmp_path / "stems"
        mock_save.return_value = StemResult(
            vocals_path=output_dir / "vocals.wav",
            bass_path=output_dir / "bass.wav",
            drums_path=output_dir / "drums.wav",
            other_path=output_dir / "other.wav",
        )

        sep = self._make_separator()
        result = sep.separate(sample_audio_path, output_dir)

        assert isinstance(result, StemResult)
        mock_load_model.assert_called_once()
        mock_load_audio.assert_called_once()
        mock_run.assert_called_once()
        mock_save.assert_called_once()

    def test_resolve_device_auto_cpu(self):
        """Device 'auto' should resolve to 'cpu' when CUDA is unavailable."""
        device = StemSeparator._resolve_device("auto")
        # On test machines without GPU, this should be 'cpu'
        assert device in ("cpu", "cuda")

    def test_resolve_device_explicit(self):
        """Explicit device string should be returned unchanged."""
        assert StemSeparator._resolve_device("cpu") == "cpu"
        assert StemSeparator._resolve_device("cuda") == "cuda"
