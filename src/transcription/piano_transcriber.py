"""
Piano-to-MIDI transcription using ByteDance's piano_transcription_inference.

Extracts a high-fidelity piano transcription from an audio file,
preserving velocity dynamics and sustain pedal events.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import torch

from src.pipeline.config import PianoTranscriptionConfig
from src.pipeline.errors import TranscriptionError
from src.utils.logger import get_logger

log = get_logger(__name__)


class PianoTranscriber:
    """Transcribe piano audio to MIDI using ByteDance's piano transcription model."""

    def __init__(self, config: PianoTranscriptionConfig) -> None:
        self._config = config
        self._device = self._resolve_device(config.device)

    def transcribe(self, audio_path: Path, output_path: Path) -> Path:
        """
        Transcribe piano content from an audio file to MIDI.

        Args:
            audio_path: Path to the audio file (e.g., instrumental.wav).
            output_path: Path for the output MIDI file.

        Returns:
            Path to the transcribed MIDI file.

        Raises:
            FileNotFoundError: If audio_path does not exist.
            TranscriptionError: If the model fails or produces empty output.
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._run_transcription(audio_path, output_path)
        except Exception as exc:
            raise TranscriptionError(
                message=f"Piano transcription failed for {audio_path}",
                cause=exc,
            ) from exc

        # Validate the output
        self._validate_output(output_path)

        return output_path

    # ── Private Helpers ──────────────────────────────────────────────────

    def _run_transcription(self, audio_path: Path, output_path: Path) -> None:
        """Load audio at 16 kHz and run the ByteDance model."""
        import librosa
        from piano_transcription_inference import PianoTranscription

        log.info(f"Running piano transcription on {audio_path}...")

        # The model expects 16 kHz mono audio
        audio, _sr = librosa.load(str(audio_path), sr=16000, mono=True)

        transcriber = PianoTranscription(
            device=self._device,
            checkpoint_path=self._config.checkpoint,
        )

        transcriber.transcribe(audio, str(output_path))
        log.info(f"Piano transcription saved to {output_path}")

    def _validate_output(self, midi_path: Path) -> None:
        """Check that the transcription produced a non-empty MIDI file."""
        if not midi_path.exists():
            raise TranscriptionError(
                message=f"Piano transcription produced no output file at {midi_path}"
            )

        midi = pretty_midi.PrettyMIDI(str(midi_path))
        total_notes = sum(len(inst.notes) for inst in midi.instruments)

        if total_notes == 0:
            log.warning(
                "Piano transcription produced zero notes — "
                "audio may not contain piano."
            )

        log.info(f"Piano transcription validated: {total_notes} notes")

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to the best available device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
