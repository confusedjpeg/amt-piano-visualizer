"""
Stem mixing utility.

Combines multiple separated audio stems into a single mixed WAV file
via waveform summation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.utils.logger import get_logger

log = get_logger(__name__)


class StemMixer:
    """Utility for mixing audio stems via waveform summation."""

    @staticmethod
    def mix_stems(
        stem_paths: list[Path],
        output_path: Path,
        normalize: bool = True,
    ) -> Path:
        """
        Sum multiple audio stems into a single WAV file.

        Args:
            stem_paths: List of WAV file paths to mix.
            output_path: Where to write the mixed result.
            normalize: If True, normalize the mixed waveform to prevent clipping.

        Returns:
            Path to the output mixed file.

        Raises:
            ValueError: If stems have mismatched sample rates or channel counts.
            FileNotFoundError: If any stem path does not exist.
        """
        if not stem_paths:
            raise ValueError("At least one stem path is required.")

        # Load the first stem to establish reference format
        waveforms: list[np.ndarray] = []
        sample_rate: int | None = None

        for stem_path in stem_paths:
            stem_path = Path(stem_path)
            if not stem_path.exists():
                raise FileNotFoundError(f"Stem file not found: {stem_path}")

            data, sr = sf.read(str(stem_path), dtype="float64")

            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError(
                    f"Sample rate mismatch: expected {sample_rate} Hz, "
                    f"got {sr} Hz in {stem_path}"
                )

            waveforms.append(data)

        # Pad shorter stems with silence to match the longest
        max_length = max(w.shape[0] for w in waveforms)
        padded: list[np.ndarray] = []

        for w in waveforms:
            if w.shape[0] < max_length:
                if w.ndim == 1:
                    pad_width = max_length - w.shape[0]
                    w = np.pad(w, (0, pad_width), mode="constant")
                else:
                    pad_width = max_length - w.shape[0]
                    w = np.pad(
                        w, ((0, pad_width), (0, 0)), mode="constant"
                    )
            padded.append(w)

        # Sum the waveforms
        mixed = np.sum(padded, axis=0)

        # Normalize to [-1.0, 1.0] if requested
        if normalize:
            peak = np.max(np.abs(mixed))
            if peak > 0:
                mixed = mixed / peak

        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), mixed, sample_rate)
        log.info(f"Mixed {len(stem_paths)} stems → {output_path}")

        return output_path
