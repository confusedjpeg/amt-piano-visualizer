"""Tests for src.video.renderer — VideoRenderer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.pipeline.config import VideoConfig
from src.pipeline.errors import RenderingError
from src.video.renderer import VideoRenderer


class TestVideoRenderer:
    """Unit tests for VideoRenderer."""

    def _make_renderer(self, **overrides) -> VideoRenderer:
        config = VideoConfig(**overrides)
        return VideoRenderer(config)

    def test_missing_binary_raises(self, tmp_path):
        """Non-existent MIDIVisualizer binary should raise FileNotFoundError."""
        renderer = self._make_renderer(
            midi_visualizer_path="/nonexistent/MIDIVisualizer"
        )
        midi = tmp_path / "test.mid"
        midi.write_bytes(b"fake")
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake")

        with pytest.raises(FileNotFoundError, match="MIDIVisualizer"):
            renderer.render(midi, audio, tmp_path / "output.mp4")

    def test_missing_midi_raises(self, tmp_path):
        """Non-existent MIDI file should raise FileNotFoundError."""
        renderer = self._make_renderer()

        with patch.object(renderer, "_validate_binary"):
            with pytest.raises(FileNotFoundError, match="MIDI file"):
                renderer.render(
                    tmp_path / "nonexistent.mid",
                    tmp_path / "test.wav",
                    tmp_path / "output.mp4",
                )

    def test_missing_audio_raises(self, tmp_path):
        """Non-existent audio file should raise FileNotFoundError."""
        renderer = self._make_renderer()
        midi = tmp_path / "test.mid"
        midi.write_bytes(b"fake")

        with patch.object(renderer, "_validate_binary"):
            with pytest.raises(FileNotFoundError, match="Audio file"):
                renderer.render(
                    midi,
                    tmp_path / "nonexistent.wav",
                    tmp_path / "output.mp4",
                )

    def test_build_command_format(self):
        """Command should contain expected flags and values."""
        renderer = self._make_renderer(
            midi_visualizer_path="MIDIVisualizer",
            resolution="1920x1080",
            fps=60,
            note_speed=1.0,
            background_color="0.1 0.1 0.12",
        )

        cmd = renderer._build_command(
            Path("/test/input.mid"),
            Path("/test/output.mp4"),
        )

        assert "MIDIVisualizer" in cmd
        assert "--midi" in cmd
        assert "--export" in cmd
        assert "--format" in cmd
        assert "MPEG4" in cmd
        assert "--size" in cmd
        assert "1920" in cmd
        assert "1080" in cmd
        assert "--framerate" in cmd
        assert "60" in cmd

    def test_additional_args_passed(self):
        """Additional CLI args from config should be appended to the command."""
        renderer = self._make_renderer(
            additional_args=["--smooth", "--quality", "high"]
        )

        cmd = renderer._build_command(
            Path("/test/input.mid"),
            Path("/test/output.mp4"),
        )

        assert "--smooth" in cmd
        assert "--quality" in cmd
        assert "high" in cmd

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/local/bin/MIDIVisualizer")
    def test_render_success(self, mock_which, mock_subprocess, tmp_path):
        """Successful render should produce the output file."""
        renderer = self._make_renderer()

        midi = tmp_path / "test.mid"
        midi.write_bytes(b"fake")
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake")
        output = tmp_path / "output.mp4"

        # Simulate successful subprocess
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Create output file to simulate MIDIVisualizer writing it
        def side_effect(*args, **kwargs):
            output.write_bytes(b"video_data")
            return MagicMock(returncode=0)

        mock_subprocess.side_effect = side_effect

        result = renderer.render(midi, audio, output)
        assert result == output
