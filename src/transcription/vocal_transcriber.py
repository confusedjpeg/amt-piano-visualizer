"""
Vocal-to-MIDI transcription using Spotify's BasicPitch.

Converts an isolated vocal stem to a cleaned MIDI file suitable
for use as the right-hand piano melody.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pretty_midi

from src.pipeline.config import VocalTranscriptionConfig
from src.pipeline.errors import TranscriptionError
from src.utils.logger import get_logger

log = get_logger(__name__)


class VocalTranscriber:
    """Transcribe monophonic/polyphonic vocals to MIDI using BasicPitch."""

    def __init__(self, config: VocalTranscriptionConfig) -> None:
        self._config = config

    def transcribe(self, vocals_path: Path, output_path: Path) -> Path:
        """
        Transcribe vocals to MIDI and apply cleanup.

        Args:
            vocals_path: Path to vocals.wav (from Step 1).
            output_path: Path for the cleaned output MIDI.

        Returns:
            Path to the cleaned MIDI file (all notes on Channel 0 = Right Hand).

        Raises:
            FileNotFoundError: If vocals_path does not exist.
            TranscriptionError: If BasicPitch fails or produces no output.
        """
        vocals_path = Path(vocals_path)
        output_path = Path(output_path)

        if not vocals_path.exists():
            raise FileNotFoundError(f"Vocal stem not found: {vocals_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            midi_data = self._run_basic_pitch(vocals_path)
        except Exception as exc:
            raise TranscriptionError(
                message=f"BasicPitch failed on {vocals_path}",
                cause=exc,
            ) from exc

        # Check for empty output
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        if total_notes == 0:
            log.warning(
                "BasicPitch produced zero notes — vocal stem may be silent. "
                "Writing an empty MIDI file."
            )
            midi_data.write(str(output_path))
            return output_path

        # Apply cleanup pipeline
        midi_data = self._strip_pitch_bends(midi_data)
        midi_data = self._clamp_to_vocal_range(midi_data)
        midi_data = self._remove_short_notes(midi_data)
        midi_data = self._assign_to_right_hand(midi_data)

        midi_data.write(str(output_path))
        final_count = sum(len(inst.notes) for inst in midi_data.instruments)
        log.info(
            f"Vocal transcription complete: {final_count} notes → {output_path}"
        )
        return output_path

    # ── Private Helpers ──────────────────────────────────────────────────

    def _run_basic_pitch(self, audio_path: Path) -> pretty_midi.PrettyMIDI:
        """Run BasicPitch inference and return the PrettyMIDI object."""
        from basic_pitch.inference import predict

        log.info(f"Running BasicPitch on {audio_path}...")

        minimum_note_length_s = self._config.minimum_note_length_ms / 1000.0

        _model_output, midi_data, _note_events = predict(
            str(audio_path),
            onset_threshold=self._config.onset_threshold,
            frame_threshold=self._config.frame_threshold,
            minimum_note_length=minimum_note_length_s,
            minimum_frequency=self._config.minimum_frequency_hz,
            maximum_frequency=self._config.maximum_frequency_hz,
        )

        return midi_data

    def _strip_pitch_bends(
        self, midi: pretty_midi.PrettyMIDI
    ) -> pretty_midi.PrettyMIDI:
        """Remove all pitch bend events (vocal vibrato artifacts)."""
        for instrument in midi.instruments:
            instrument.pitch_bends = []
        return midi

    def _clamp_to_vocal_range(
        self, midi: pretty_midi.PrettyMIDI
    ) -> pretty_midi.PrettyMIDI:
        """Remove notes outside the configured vocal range."""
        min_note = self._hz_to_midi(self._config.minimum_frequency_hz)
        max_note = self._hz_to_midi(self._config.maximum_frequency_hz)

        for instrument in midi.instruments:
            instrument.notes = [
                n for n in instrument.notes
                if min_note <= n.pitch <= max_note
            ]

        return midi

    def _remove_short_notes(
        self, midi: pretty_midi.PrettyMIDI
    ) -> pretty_midi.PrettyMIDI:
        """Remove notes shorter than the minimum duration threshold."""
        min_duration = self._config.minimum_note_length_ms / 1000.0

        for instrument in midi.instruments:
            instrument.notes = [
                n for n in instrument.notes
                if (n.end - n.start) >= min_duration
            ]

        return midi

    def _assign_to_right_hand(
        self, midi: pretty_midi.PrettyMIDI
    ) -> pretty_midi.PrettyMIDI:
        """Consolidate all notes into a single Instrument on Channel 0 (RH)."""
        all_notes = []
        for instrument in midi.instruments:
            all_notes.extend(copy.deepcopy(instrument.notes))

        rh_instrument = pretty_midi.Instrument(
            program=0,   # Acoustic Grand Piano
            is_drum=False,
            name="Right Hand",
        )
        rh_instrument.notes = all_notes

        result = pretty_midi.PrettyMIDI(initial_tempo=midi.estimate_tempo())
        result.instruments.append(rh_instrument)
        return result

    @staticmethod
    def _hz_to_midi(freq_hz: float) -> int:
        """Convert frequency in Hz to the nearest MIDI note number."""
        import numpy as np

        if freq_hz <= 0:
            return 0
        return int(round(69 + 12 * np.log2(freq_hz / 440.0)))
