"""Tests for src.video — VideoRenderer, PythonVideoRenderer, and factory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pretty_midi
import pytest

from src.pipeline.config import VideoConfig
from src.pipeline.errors import RenderingError
from src.video import create_renderer
from src.video.python_renderer import PythonVideoRenderer
from src.video.renderer import VideoRenderer


# ── Helper ───────────────────────────────────────────────────────────────────

def _make_test_midi(path: Path) -> Path:
    """Create a minimal valid MIDI file with a few notes."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    # Add a C major chord lasting 1 second
    for pitch in [60, 64, 67]:
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
        )
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


# ══════════════════════════════════════════════════════════════════════════════
# VideoRenderer (MIDIVisualizer) Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestVideoRenderer:
    """Unit tests for the MIDIVisualizer-based VideoRenderer."""

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


# ══════════════════════════════════════════════════════════════════════════════
# PythonVideoRenderer Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestPythonVideoRenderer:
    """Unit tests for the pure-Python fallback renderer."""

    def _make_renderer(self, **overrides) -> PythonVideoRenderer:
        defaults = {"renderer": "python"}
        defaults.update(overrides)
        config = VideoConfig(**defaults)
        return PythonVideoRenderer(config)

    def test_missing_midi_raises(self, tmp_path):
        """Non-existent MIDI file should raise FileNotFoundError."""
        renderer = self._make_renderer()
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake")

        with pytest.raises(FileNotFoundError, match="MIDI file"):
            renderer.render(
                tmp_path / "nonexistent.mid",
                audio,
                tmp_path / "output.mp4",
            )

    def test_audio_path_is_ignored(self, tmp_path):
        """Python renderer should not raise for a missing audio_path
        (it synthesizes audio from MIDI instead)."""
        renderer = self._make_renderer()
        midi = tmp_path / "test.mid"
        _make_test_midi(midi)

        # audio_path doesn't exist — should NOT raise
        # (will fail later at moviepy stage, but the audio_path itself is unused)
        with patch("src.video.python_renderer.PythonVideoRenderer._synthesize_midi_audio", return_value=None):
            with patch("moviepy.VideoClip") as mock_clip_cls:
                mock_clip = MagicMock()
                mock_clip_cls.return_value = mock_clip
                mock_clip.write_videofile = MagicMock()
                # Simulate output file creation
                output = tmp_path / "output.mp4"
                mock_clip.write_videofile.side_effect = lambda *a, **k: output.write_bytes(b"video")

                result = renderer.render(
                    midi,
                    tmp_path / "nonexistent.wav",  # doesn't exist — should be fine
                    output,
                )

    def test_empty_midi_raises(self, tmp_path):
        """A MIDI with no notes should raise RenderingError."""
        renderer = self._make_renderer()

        # Create a MIDI with no notes
        pm = pretty_midi.PrettyMIDI()
        pm.instruments.append(pretty_midi.Instrument(program=0))
        midi = tmp_path / "empty.mid"
        pm.write(str(midi))

        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake")

        with pytest.raises(RenderingError, match="no notes"):
            renderer.render(midi, audio, tmp_path / "output.mp4")

    def test_frame_has_correct_shape(self):
        """A rendered frame should match the configured resolution."""
        renderer = self._make_renderer(resolution="640x480")

        # Create a simple note list
        from src.video.python_renderer import _NoteEvent
        notes = [_NoteEvent(pitch=60, start=0.0, end=1.0, velocity=80, channel=0)]

        frame = renderer._draw_frame(0.5, notes)

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

    def test_frame_dtype_is_uint8(self):
        """Frame array should be uint8 for moviepy compatibility."""
        renderer = self._make_renderer(resolution="320x240")

        from src.video.python_renderer import _NoteEvent
        notes = [_NoteEvent(pitch=60, start=0.0, end=1.0, velocity=80, channel=0)]

        frame = renderer._draw_frame(0.0, notes)
        assert frame.dtype == np.uint8

    def test_note_extraction(self, tmp_path):
        """Should extract notes from a valid MIDI file."""
        renderer = self._make_renderer()

        midi_path = tmp_path / "test.mid"
        _make_test_midi(midi_path)

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = renderer._extract_notes(pm)

        assert len(notes) == 3  # C, E, G
        pitches = {n.pitch for n in notes}
        assert pitches == {60, 64, 67}

    def test_background_color_parsing(self):
        """Config background_color should be parsed to RGB tuple."""
        from src.video.python_renderer import _parse_background_color

        assert _parse_background_color("0.1 0.1 0.12") == (25, 25, 30)
        assert _parse_background_color("1.0 1.0 1.0") == (255, 255, 255)
        assert _parse_background_color("0 0 0") == (0, 0, 0)
        # Invalid input should return default
        assert _parse_background_color("invalid") == (25, 25, 30)


# ══════════════════════════════════════════════════════════════════════════════
# Factory Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCreateRenderer:
    """Unit tests for the create_renderer factory function."""

    def test_forced_python(self):
        """renderer='python' should always return PythonVideoRenderer."""
        config = VideoConfig(renderer="python")
        renderer = create_renderer(config)
        assert isinstance(renderer, PythonVideoRenderer)

    @patch("shutil.which", return_value="/usr/local/bin/MIDIVisualizer")
    def test_forced_midi_visualizer(self, _mock_which):
        """renderer='midi_visualizer' should always return VideoRenderer."""
        config = VideoConfig(renderer="midi_visualizer")
        renderer = create_renderer(config)
        assert isinstance(renderer, VideoRenderer)

    @patch("shutil.which", return_value="/usr/local/bin/MIDIVisualizer")
    def test_auto_finds_binary(self, _mock_which):
        """Auto mode should choose MIDIVisualizer when binary is found."""
        config = VideoConfig(renderer="auto")
        renderer = create_renderer(config)
        assert isinstance(renderer, VideoRenderer)

    @patch("shutil.which", return_value=None)
    def test_auto_falls_back_to_python(self, _mock_which):
        """Auto mode should fall back to Python when binary is missing."""
        config = VideoConfig(renderer="auto")
        renderer = create_renderer(config)
        assert isinstance(renderer, PythonVideoRenderer)
