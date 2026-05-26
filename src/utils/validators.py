"""
Input validation helpers for the pipeline.
"""

from __future__ import annotations

from pathlib import Path

SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def validate_audio_file(path: Path) -> Path:
    """Validate that *path* points to an existing, supported audio file.

    Args:
        path: Path to the audio file to validate.

    Returns:
        The resolved (absolute) path.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not a supported audio format.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
        )

    return path


def validate_midi_file(path: Path) -> Path:
    """Validate that *path* points to an existing MIDI file.

    Args:
        path: Path to the MIDI file.

    Returns:
        The resolved path.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not have a .mid or .midi extension.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    if path.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError(f"Expected a .mid/.midi file, got: {path.suffix}")

    return path
