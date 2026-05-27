"""
Pre-transcription noise gate.

Applies a hard amplitude gate to an audio file before it is fed into
the neural transcriber. Frames whose RMS energy falls below a
configurable dB threshold are muted to digital silence.

This eliminates reverb tails, room noise, and stem bleed that cause
the AI to hallucinate phantom notes.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.utils.logger import get_logger

log = get_logger(__name__)


class NoiseGate:
    """Apply a hard noise gate to audio, muting frames below a dB threshold."""

    def __init__(
        self,
        threshold_db: float = -20.0,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> None:
        """
        Args:
            threshold_db: RMS energy threshold in dB. Frames quieter than
                this are muted to silence. More negative = more permissive.
                Typical values: -15 (aggressive) to -25 (gentle).
            frame_length: FFT frame size used for RMS energy computation.
            hop_length: Number of samples between consecutive frames.
        """
        self._threshold_db = threshold_db
        self._frame_length = frame_length
        self._hop_length = hop_length

    def apply(self, audio_path: Path, output_path: Path) -> Path:
        """
        Apply the noise gate to an audio file and write the gated result.

        Args:
            audio_path: Path to the input WAV file.
            output_path: Path for the gated output WAV file.

        Returns:
            Path to the gated output file.

        Raises:
            FileNotFoundError: If audio_path does not exist.
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio at its native sample rate (preserve quality)
        y, sr = librosa.load(str(audio_path), sr=None, mono=False)

        # librosa.load with mono=False returns (channels, samples) for stereo,
        # or (samples,) for mono. Normalize to 2D for uniform processing.
        if y.ndim == 1:
            y = y[np.newaxis, :]  # (1, samples)
            was_mono = True
        else:
            was_mono = False

        # Compute RMS energy per frame on the summed mono signal
        mono_signal = np.mean(y, axis=0)
        rms = librosa.feature.rms(
            y=mono_signal,
            frame_length=self._frame_length,
            hop_length=self._hop_length,
        )[0]  # shape: (num_frames,)

        # Convert RMS to dB (relative to peak)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Build a per-frame binary mask: 1 = keep, 0 = mute
        gate_mask = (rms_db >= self._threshold_db).astype(np.float32)

        # Expand the frame-level mask to sample-level
        num_samples = y.shape[1]
        sample_mask = self._expand_mask_to_samples(
            gate_mask, num_samples, self._hop_length
        )

        # Apply the mask to all channels
        gated = y * sample_mask[np.newaxis, :]

        # Diagnostics
        muted_pct = (1.0 - np.mean(gate_mask)) * 100.0
        log.info(
            f"Noise gate applied (threshold={self._threshold_db} dB): "
            f"{muted_pct:.1f}% of frames muted → {output_path}"
        )

        # Write output
        if was_mono:
            gated = gated[0]  # Back to (samples,) for mono
        else:
            gated = gated.T   # soundfile expects (samples, channels)

        sf.write(str(output_path), gated, sr)
        return output_path

    # ── Private Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _expand_mask_to_samples(
        frame_mask: np.ndarray,
        num_samples: int,
        hop_length: int,
    ) -> np.ndarray:
        """Expand a frame-level gate mask to a per-sample mask.

        Each frame's gate value (0 or 1) is held constant across its hop
        window, producing a step-function mask at sample resolution.

        Args:
            frame_mask: Binary mask of shape (num_frames,).
            num_samples: Total number of audio samples.
            hop_length: Samples per hop (i.e., per frame step).

        Returns:
            Sample-level mask of shape (num_samples,).
        """
        sample_mask = np.zeros(num_samples, dtype=np.float32)

        for i, val in enumerate(frame_mask):
            start = i * hop_length
            end = min(start + hop_length, num_samples)
            sample_mask[start:end] = val

        return sample_mask
