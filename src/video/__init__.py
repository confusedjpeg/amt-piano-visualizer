"""Video rendering — MIDIVisualizer subprocess wrapper + Python fallback."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from src.pipeline.config import VideoConfig
from src.utils.logger import get_logger

log = get_logger(__name__)


class VideoRendererProtocol(Protocol):
    """Common interface for video renderers."""

    def render(
        self,
        midi_path: Path,
        audio_path: Path,
        output_path: Path,
        timeout: int = 600,
    ) -> Path: ...


def create_renderer(config: VideoConfig) -> VideoRendererProtocol:
    """Factory: pick the right video renderer based on config.

    Selection logic:
      - 'midi_visualizer': Always use the MIDIVisualizer binary.
      - 'python': Always use the pure-Python fallback.
      - 'auto' (default): Try MIDIVisualizer if the binary is found
        on PATH or at the configured path, otherwise fall back to Python.

    Returns:
        An instance implementing the VideoRendererProtocol.
    """
    choice = config.renderer.lower().strip()

    if choice == "midi_visualizer":
        from src.video.renderer import VideoRenderer
        log.info("Video renderer: MIDIVisualizer (forced by config)")
        return VideoRenderer(config)

    if choice == "python":
        from src.video.python_renderer import PythonVideoRenderer
        log.info("Video renderer: Python fallback (forced by config)")
        return PythonVideoRenderer(config)

    # Auto-detect
    binary = config.midi_visualizer_path
    resolved = shutil.which(binary)
    if resolved is not None or Path(binary).exists():
        from src.video.renderer import VideoRenderer
        log.info(f"Video renderer: MIDIVisualizer (auto-detected at {resolved or binary})")
        return VideoRenderer(config)

    from src.video.python_renderer import PythonVideoRenderer
    log.info(
        "Video renderer: Python fallback (MIDIVisualizer not found on PATH)"
    )
    return PythonVideoRenderer(config)
