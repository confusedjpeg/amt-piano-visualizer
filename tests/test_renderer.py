"""Tests for src.video — VideoRenderer, PythonVideoRenderer, and factory."""

from __future__ import annotations

import sys
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

    def test_missing_audio_is_ok(self, tmp_path):
        """Audio path is not required — Synthesia synthesizes from MIDI."""
        renderer = self._make_renderer()
        midi = tmp_path / "test.mid"
        midi.write_bytes(b"fake")

        with patch.object(renderer, "_validate_binary"):
            with patch("subprocess.run") as mock_subprocess:
                output = tmp_path / "output.mp4"
                def fake_run(*a, **k):
                    output.write_bytes(b"video")
                    return MagicMock(returncode=0)
                mock_subprocess.side_effect = fake_run
                result = renderer.render(
                    midi,
                    tmp_path / "nonexistent.wav",
                    output,
                )
                assert result == output

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

    def _patch_moviepy(self, mock_clip_cls, mock_audio_cls=None):
        """Return a context manager that stubs out the moviepy module."""
        fake_moviepy = MagicMock()
        fake_moviepy.VideoClip = mock_clip_cls
        if mock_audio_cls is not None:
            fake_moviepy.AudioFileClip = mock_audio_cls
        return patch.dict("sys.modules", {"moviepy": fake_moviepy})

    def test_uses_synthesized_piano_audio_not_original_song(self, tmp_path):
        """Python renderer must synthesize piano audio from MIDI, NOT use original song."""
        renderer = self._make_renderer()
        midi = tmp_path / "test.mid"
        _make_test_midi(midi)
        original_song = tmp_path / "original.wav"
        original_song.write_bytes(b"fake_song_audio")  # exists but must NOT be used
        synth_path = tmp_path / "output.synth.wav"
        synth_path.write_bytes(b"fake_piano_synth")
        output = tmp_path / "output.mp4"

        mock_clip_cls = MagicMock()
        mock_clip = MagicMock()
        mock_clip_cls.return_value = mock_clip
        mock_clip.with_audio.return_value = mock_clip
        mock_clip.write_videofile = MagicMock(
            side_effect=lambda *a, **k: output.write_bytes(b"video")
        )
        mock_audio_cls = MagicMock()
        mock_audio = MagicMock()
        mock_audio_cls.return_value = mock_audio

        with patch(
            "src.video.python_renderer.PythonVideoRenderer._synthesize_midi_audio",
            return_value=synth_path,
        ) as mock_synth:
            with self._patch_moviepy(mock_clip_cls, mock_audio_cls):
                result = renderer.render(midi, original_song, output)

        assert result == output
        # Synthesized piano audio must be created and used
        mock_synth.assert_called_once()
        # AudioFileClip must be called with the SYNTH path, NOT the original song
        mock_audio_cls.assert_called_once_with(str(synth_path))
        mock_clip.with_audio.assert_called_once_with(mock_audio)

    def test_original_song_audio_is_never_used(self, tmp_path):
        """Even when original song exists, only synthesized piano audio is attached."""
        renderer = self._make_renderer()
        midi = tmp_path / "test.mid"
        _make_test_midi(midi)
        original_song = tmp_path / "song.mp3"
        original_song.write_bytes(b"original_song_data")
        synth_path = tmp_path / "output.synth.wav"
        synth_path.write_bytes(b"piano_synth")
        output = tmp_path / "output.mp4"

        mock_clip_cls = MagicMock()
        mock_clip = MagicMock()
        mock_clip_cls.return_value = mock_clip
        mock_clip.with_audio.return_value = mock_clip
        mock_clip.write_videofile = MagicMock(
            side_effect=lambda *a, **k: output.write_bytes(b"video")
        )
        mock_audio_cls = MagicMock()
        mock_audio_cls.return_value = MagicMock()

        with patch(
            "src.video.python_renderer.PythonVideoRenderer._synthesize_midi_audio",
            return_value=synth_path,
        ):
            with self._patch_moviepy(mock_clip_cls, mock_audio_cls):
                renderer.render(midi, original_song, output)

        # AudioFileClip must NOT be called with the original song path
        called_paths = [str(call.args[0]) if call.args else str(call) 
                        for call in mock_audio_cls.call_args_list]
        assert not any(str(original_song) == p for p in called_paths), \
            "Original song audio was used — should only use synthesized piano audio"

    def test_logs_warning_when_synth_fails(self, tmp_path):
        """A clear warning should be logged when piano audio synthesis fails."""
        from loguru import logger
        import io

        renderer = self._make_renderer()
        midi = tmp_path / "test.mid"
        _make_test_midi(midi)
        output = tmp_path / "output.mp4"

        mock_clip_cls = MagicMock()
        mock_clip = MagicMock()
        mock_clip_cls.return_value = mock_clip
        mock_clip.write_videofile = MagicMock(
            side_effect=lambda *a, **k: output.write_bytes(b"video")
        )

        log_stream = io.StringIO()
        sink_id = logger.add(log_stream, format="{message}", level="INFO")

        try:
            with patch(
                "src.video.python_renderer.PythonVideoRenderer._synthesize_midi_audio",
                return_value=None,
            ):
                with self._patch_moviepy(mock_clip_cls):
                    renderer.render(
                        midi,
                        tmp_path / "nonexistent.wav",
                        output,
                    )
        finally:
            logger.remove(sink_id)

        log_text = log_stream.getvalue().lower()
        assert "silent video" in log_text or "no piano audio" in log_text

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

    def test_channel_by_name_not_index(self, tmp_path):
        """Channel assignment should use instrument name, not list index."""
        renderer = self._make_renderer()

        # Create a MIDI where LH instrument appears FIRST (opposite of usual)
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        lh = pretty_midi.Instrument(program=0, name="Left Hand")
        lh.notes.append(pretty_midi.Note(velocity=80, pitch=48, start=0.0, end=1.0))
        rh = pretty_midi.Instrument(program=0, name="Right Hand")
        rh.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
        pm.instruments.extend([lh, rh])

        notes = renderer._extract_notes(pm)
        assert len(notes) == 2
        for n in notes:
            if n.pitch == 48:
                assert n.channel == 1, f"LH note should be channel 1, got {n.channel}"
            elif n.pitch == 60:
                assert n.channel == 0, f"RH note should be channel 0, got {n.channel}"

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
