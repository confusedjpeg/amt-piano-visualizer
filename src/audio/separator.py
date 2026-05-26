"""
Demucs-based stem separator.

Wraps Meta's Demucs htdemucs model to decompose an input audio file
into four stems: vocals, bass, drums, other.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from src.pipeline.config import SeparatorConfig
from src.pipeline.errors import SeparationError
from src.pipeline.models import StemResult
from src.utils.logger import get_logger
from src.utils.validators import validate_audio_file

log = get_logger(__name__)


class StemSeparator:
    """Wraps Meta's Demucs htdemucs model for 4-stem source separation."""

    def __init__(self, config: SeparatorConfig) -> None:
        """
        Args:
            config: Pydantic model containing model name, device, shifts, overlap.
        """
        self._config = config
        self._model = None
        self._device = self._resolve_device(config.device)

    # ── Public API ───────────────────────────────────────────────────────

    def separate(self, audio_path: Path, output_dir: Path) -> StemResult:
        """
        Run source separation on the given audio file.

        Args:
            audio_path: Path to the input audio file (.mp3, .wav, .flac).
            output_dir: Directory where stem WAV files will be written.

        Returns:
            StemResult with paths to the four output stems.

        Raises:
            FileNotFoundError: If audio_path does not exist.
            SeparationError: If Demucs fails.
        """
        audio_path = validate_audio_file(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            model = self._load_model()
            wav, sr = self._load_audio(audio_path, model.samplerate)
            stems = self._run_separation(model, wav)
            return self._save_stems(stems, model.sources, sr, output_dir)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise SeparationError(
                message=f"Stem separation failed for {audio_path}",
                cause=exc,
            ) from exc

    # ── Private Helpers ──────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the Demucs model."""
        if self._model is not None:
            return self._model

        log.info(
            f"Loading Demucs model '{self._config.model}' on {self._device}..."
        )

        from demucs.pretrained import get_model

        self._model = get_model(self._config.model)
        self._model.to(self._device)
        self._model.eval()
        return self._model

    def _load_audio(self, audio_path: Path, target_sr: int):
        """Load and resample audio to the model's expected sample rate."""
        from demucs.audio import AudioFile

        audio_file = AudioFile(audio_path)
        wav = audio_file.read(
            seek_time=0,
            duration=None,
            streams=0,
        )
        # wav shape: (channels, samples)
        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        duration_secs = wav.shape[1] / target_sr
        if duration_secs > 900:  # 15 minutes
            log.warning(
                f"Audio is {duration_secs / 60:.1f} min long. "
                "This may require significant RAM/VRAM."
            )

        return wav, target_sr

    def _run_separation(self, model, wav):
        """Apply the Demucs model to get separated stems."""
        from demucs.apply import apply_model

        log.info("Running stem separation...")

        # Add batch dimension: (channels, samples) → (1, channels, samples)
        mix = wav.unsqueeze(0).to(self._device)

        with torch.no_grad():
            estimates = apply_model(
                model,
                mix,
                shifts=self._config.shifts,
                overlap=self._config.overlap,
            )

        # estimates shape: (1, sources, channels, samples)
        return estimates.squeeze(0).cpu()

    def _save_stems(
        self,
        stems_tensor,
        source_names: list[str],
        sr: int,
        output_dir: Path,
    ) -> StemResult:
        """Write each stem tensor to a WAV file."""
        stem_paths: dict[str, Path] = {}

        for idx, name in enumerate(source_names):
            stem_wav = stems_tensor[idx]  # (channels, samples)
            out_path = output_dir / f"{name}.wav"
            torchaudio.save(str(out_path), stem_wav, sr)
            stem_paths[name] = out_path
            log.info(f"  Saved stem: {out_path}")

        return StemResult(
            vocals_path=stem_paths.get("vocals", output_dir / "vocals.wav"),
            bass_path=stem_paths.get("bass", output_dir / "bass.wav"),
            drums_path=stem_paths.get("drums", output_dir / "drums.wav"),
            other_path=stem_paths.get("other", output_dir / "other.wav"),
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to the best available device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
