"""
Synthesia-style video renderer using MIDIVisualizer CLI.

Invokes MIDIVisualizer as a subprocess. If the installed version
does not support the --audio flag, falls back to FFmpeg for
audio muxing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src.pipeline.config import VideoConfig
from src.pipeline.errors import RenderingError
from src.utils.logger import get_logger

log = get_logger(__name__)


class VideoRenderer:
    """Render a falling-notes video from MIDI + audio using MIDIVisualizer CLI."""

    def __init__(self, config: VideoConfig) -> None:
        self._config = config

    def render(
        self,
        midi_path: Path,
        audio_path: Path,
        output_path: Path,
        timeout: int = 600,
    ) -> Path:
        """
        Render the Synthesia-style video.

        MIDIVisualizer synthesizes piano audio from the MIDI file
        automatically.  The original input song is NOT used — a
        Synthesia video must sound like piano, not the original artist.

        Args:
            midi_path: Path to final_playable.mid.
            audio_path: Path to original audio file (unused — kept for
                API compatibility with VideoRendererProtocol).
            output_path: Path for the output MP4 file.
            timeout: Maximum time in seconds to wait for rendering.

        Returns:
            Path to the rendered MP4 video.

        Raises:
            FileNotFoundError: If MIDIVisualizer binary or MIDI file not found.
            RenderingError: If the subprocess exits with a non-zero code.
        """
        midi_path = Path(midi_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate inputs
        self._validate_binary()
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        # audio_path is not used for rendering — a Synthesia video
        # must play piano audio synthesized from the MIDI, not the
        # original song.  MIDIVisualizer synthesizes from the MIDI
        # automatically when no --audio flag is passed.

        # Render video (MIDIVisualizer synthesizes piano audio from MIDI)
        self._render_without_audio(midi_path, output_path, timeout)

        # Validate output
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RenderingError(
                message=f"Rendering produced no output at {output_path}"
            )

        log.info(f"Video rendered: {output_path}")
        return output_path

    # ── Rendering Methods ────────────────────────────────────────────────

    def _render_with_audio(
        self,
        midi_path: Path,
        audio_path: Path,
        output_path: Path,
        timeout: int,
    ) -> None:
        """Attempt to render with the --audio flag (if supported)."""
        cmd = self._build_command(midi_path, output_path)
        cmd.extend(["--audio", str(audio_path)])

        log.info(f"Running MIDIVisualizer: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RenderingError(
                message=(
                    f"MIDIVisualizer failed (code {result.returncode}): "
                    f"{result.stderr[:500]}"
                )
            )

    def _render_without_audio(
        self,
        midi_path: Path,
        output_path: Path,
        timeout: int,
    ) -> None:
        """Render video — MIDIVisualizer synthesizes piano audio from MIDI."""
        cmd = self._build_command(midi_path, output_path)

        log.info(f"Running MIDIVisualizer: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RenderingError(
                message=(
                    f"MIDIVisualizer failed (code {result.returncode}): "
                    f"{result.stderr[:500]}"
                )
            )

    def _mux_audio_ffmpeg(self, output_path: Path, audio_path: Path) -> None:
        """Use FFmpeg to combine the rendered video with the original audio."""
        temp_video = output_path.with_suffix(".noaudio.mp4")

        if not temp_video.exists():
            raise RenderingError(
                message=f"Expected temp video not found: {temp_video}"
            )

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RenderingError(
                message=(
                    "FFmpeg not found on PATH. Required for audio muxing "
                    "when MIDIVisualizer --audio is unsupported."
                )
            )

        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output
            "-i", str(temp_video),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ]

        log.info(f"Muxing audio with FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            raise RenderingError(
                message=f"FFmpeg muxing failed: {result.stderr[:500]}"
            )

        # Clean up temp file
        try:
            temp_video.unlink()
        except OSError:
            pass

    # ── Command Building ─────────────────────────────────────────────────

    def _build_command(
        self,
        midi_path: Path,
        output_path: Path,
    ) -> list[str]:
        """Construct the MIDIVisualizer CLI command."""
        width, height = self._config.resolution.split("x")

        cmd = [
            self._config.midi_visualizer_path,
            "--midi", str(midi_path),
            "--export", str(output_path),
            "--format", "MPEG4",
            "--size", width, height,
            "--framerate", str(self._config.fps),
            "--speed", str(self._config.note_speed),
        ]

        # Background color (space-separated R G B)
        bg_parts = self._config.background_color.split()
        if len(bg_parts) == 3:
            cmd.extend(["--bg-color"] + bg_parts)

        # Additional user-specified flags
        if self._config.additional_args:
            cmd.extend(self._config.additional_args)

        return cmd

    def _validate_binary(self) -> None:
        """Check that MIDIVisualizer is accessible."""
        binary = self._config.midi_visualizer_path
        resolved = shutil.which(binary)

        if resolved is None and not Path(binary).exists():
            raise FileNotFoundError(
                f"MIDIVisualizer binary not found: '{binary}'. "
                "Ensure it is installed and on PATH, or provide an absolute path "
                "in config.yaml → video.midi_visualizer_path"
            )
