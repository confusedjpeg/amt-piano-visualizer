"""
Piano-to-MIDI transcription using Spotify's BasicPitch.

Extracts a polyphonic piano transcription from an audio file using the
same modern ONNX-based neural network used for vocal transcription,
but configured for the full piano frequency range (A0–C8).
"""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from src.pipeline.config import PianoTranscriptionConfig
from src.pipeline.errors import TranscriptionError
from src.utils.logger import get_logger

log = get_logger(__name__)


class PianoTranscriber:
    """Transcribe piano audio to MIDI using Spotify's BasicPitch."""

    def __init__(self, config: PianoTranscriptionConfig) -> None:
        self._config = config

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
            midi_data = self._run_basic_pitch(audio_path)
        except Exception as exc:
            raise TranscriptionError(
                message=f"Piano transcription failed for {audio_path}",
                cause=exc,
            ) from exc

        # Write the MIDI output
        midi_data.write(str(output_path))

        # Validate the output
        self._validate_output(output_path)

        return output_path

    # ── Private Helpers ──────────────────────────────────────────────────

    def _run_basic_pitch(self, audio_path: Path) -> pretty_midi.PrettyMIDI:
        """Run BasicPitch inference and return the PrettyMIDI object."""
        from basic_pitch.inference import predict

        log.info(f"Running BasicPitch piano transcription on {audio_path}...")

        minimum_note_length_s = self._config.minimum_note_length_ms / 1000.0

        _model_output, midi_data, _note_events = predict(
            str(audio_path),
            onset_threshold=self._config.onset_threshold,
            frame_threshold=self._config.frame_threshold,
            minimum_note_length=minimum_note_length_s,
            minimum_frequency=self._config.minimum_frequency_hz,
            maximum_frequency=self._config.maximum_frequency_hz,
        )

        log.info(
            f"BasicPitch piano transcription complete: "
            f"{sum(len(i.notes) for i in midi_data.instruments)} notes"
        )

        return midi_data

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
