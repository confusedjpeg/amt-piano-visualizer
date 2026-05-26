"""
Beat and tempo analysis using librosa.

Extracts BPM, beat positions, and downbeat locations from audio.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from src.pipeline.models import BeatGrid
from src.utils.logger import get_logger

log = get_logger(__name__)


class BeatAnalyzer:
    """Extract tempo, beat positions, and downbeat locations from audio."""

    def __init__(self, beats_per_bar: int = 4) -> None:
        """
        Args:
            beats_per_bar: Number of beats per bar (4 for 4/4 time).
        """
        self._beats_per_bar = beats_per_bar

    def analyze(self, audio_path: Path) -> BeatGrid:
        """
        Analyze an audio file for tempo and beat structure.

        Args:
            audio_path: Path to the audio file.

        Returns:
            BeatGrid with tempo, beat times, downbeat times, and time signature.

        Raises:
            FileNotFoundError: If audio_path does not exist.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        log.info(f"Analyzing beats in {audio_path}...")

        y, sr = librosa.load(str(audio_path), sr=22050)

        # Extract tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # librosa may return tempo as an array in newer versions
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)

        # Convert frames to timestamps
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Estimate downbeats: every N-th beat is a downbeat (4/4 assumption)
        downbeat_indices = list(range(0, len(beat_times), self._beats_per_bar))
        downbeat_times = beat_times[downbeat_indices]

        log.info(
            f"Beat analysis: tempo={tempo:.1f} BPM, "
            f"{len(beat_times)} beats, {len(downbeat_times)} downbeats"
        )

        return BeatGrid(
            tempo=tempo,
            beat_times=beat_times,
            downbeat_times=downbeat_times,
            time_signature=(self._beats_per_bar, 4),
        )
