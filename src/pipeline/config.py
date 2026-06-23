"""
Pydantic-based configuration system for the pipeline.

All tunable parameters are defined here with sensible defaults.
The root `PipelineConfig` can be loaded from a YAML file via
`PipelineConfig.from_yaml(path)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ── Step 1: Demucs ───────────────────────────────────────────────────────────

class SeparatorConfig(BaseModel):
    """Configuration for the Demucs stem separator."""

    model: str = "htdemucs"
    device: str = "auto"
    shifts: int = 1
    overlap: float = 0.25
    output_format: str = "wav"


# ── Pre-Transcription Noise Gate ─────────────────────────────────────────────

class NoiseGateConfig(BaseModel):
    """Configuration for the pre-transcription noise gate."""

    threshold_db: float = -20.0    # Frames below this dB are muted
    frame_length: int = 2048       # FFT frame size for RMS computation
    hop_length: int = 512          # Hop between frames


# ── Step 2: BasicPitch ───────────────────────────────────────────────────────

class VocalTranscriptionConfig(BaseModel):
    """Configuration for BasicPitch vocal transcription."""

    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length_ms: float = 58.0
    minimum_frequency_hz: float = 80.0
    maximum_frequency_hz: float = 1100.0


# ── Step 3A: Piano Transcription ─────────────────────────────────────────────

class PianoTranscriptionConfig(BaseModel):
    """Configuration for BasicPitch piano transcription."""

    onset_threshold: float = 0.6
    frame_threshold: float = 0.5
    minimum_note_length_ms: float = 50.0
    minimum_frequency_hz: float = 27.5    # A0 — lowest piano key
    maximum_frequency_hz: float = 4186.0  # C8 — highest piano key


# ── Step 3B: Algorithmic Arranger ────────────────────────────────────────────

class ArrangerConfig(BaseModel):
    """Configuration for the algorithmic chord-to-pattern arranger."""

    default_pattern: str = "pop_ballad"
    pattern_dir: Path = Path("assets/patterns")
    chord_detection_method: str = "chroma"
    beats_per_bar: int = 4


# ── Step 4: Playability Filter ───────────────────────────────────────────────

class PlayabilityConfig(BaseModel):
    """Configuration for human-playability constraints."""

    max_hand_span_semitones: int = 15
    max_polyphony_per_hand: int = 4
    quantization_grid: int = 16
    pruning_priority: list[str] = Field(default=["P5", "M3", "m3"])


# ── Post-Processing (Chord Quantization + Legato) ───────────────────────────

class PostProcessingConfig(BaseModel):
    """Configuration for tempo-aware MIDI post-processing.

    Controls the chord quantization grid and the legato stretching mode.
    """

    # When True, use aggressive "Chord Snapper" legato: every note is
    # stretched to meet the next chord change, capped at max_duration_s.
    # When False, use the conservative "Smart Legato" that only fills
    # gaps smaller than an 8th note (preserving intentional rests).
    aggressive_legato: bool = True

    # Absolute cap on how far any note can be stretched (seconds).
    # Only used in aggressive mode.  Prevents infinite ringing.
    aggressive_legato_max_duration_s: float = 2.0


# ── Ghost Note Pruning (Strict Post-Transcription Cleanup) ───────────────────

class GhostNotePruningConfig(BaseModel):
    """Configuration for the strict ghost note pruning algorithm.

    Three surgical rules that eliminate AI-hallucinated micro-transients,
    overtone-stuffed chords, and reverb echo ghosts from transcribed MIDI.
    """

    # Rule 1: Minimum Duration Hard Cap
    # Notes shorter than this are ALWAYS deleted, regardless of velocity.
    min_duration_ms: float = 80.0

    # Rule 2: Polyphony Choke (Chord Thinner)
    # Max simultaneous notes per channel before choking by velocity.
    max_chord_notes: int = 4

    # Rule 3: Reverb Shadow Filter
    # A quiet note following a loud note within the shadow window is deleted.
    shadow_vel_threshold: int = 45   # Quiet note ceiling
    loud_vel_threshold: int = 70     # Loud "parent" note floor
    shadow_window_ms: float = 200.0  # Look-back window in ms


# ── Step 5: Video Rendering ──────────────────────────────────────────────────

class VideoConfig(BaseModel):
    """Configuration for video rendering.

    Supports two backends:
      - 'midi_visualizer': External MIDIVisualizer binary (highest quality).
      - 'python': Pure-Python fallback using Pillow + MoviePy.
      - 'auto' (default): Try MIDIVisualizer first, fall back to Python.
    """

    renderer: str = "auto"  # "auto" | "midi_visualizer" | "python"
    midi_visualizer_path: str = "MIDIVisualizer"
    resolution: str = "1920x1080"
    fps: int = 60
    background_color: str = "0.1 0.1 0.12"
    note_speed: float = 1.0
    additional_args: list[str] = Field(default_factory=list)


# ── Root Configuration ───────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    """Root configuration aggregating all sub-configs."""

    include_vocals: bool = True
    has_piano: bool = True
    output_dir: Path = Path("data/output")
    intermediate_dir: Path = Path("data/intermediate")

    separator: SeparatorConfig = SeparatorConfig()
    noise_gate: NoiseGateConfig = NoiseGateConfig()
    vocal_transcription: VocalTranscriptionConfig = VocalTranscriptionConfig()
    piano_transcription: PianoTranscriptionConfig = PianoTranscriptionConfig()
    arranger: ArrangerConfig = ArrangerConfig()
    playability: PlayabilityConfig = PlayabilityConfig()
    post_processing: PostProcessingConfig = PostProcessingConfig()
    ghost_note_pruning: GhostNotePruningConfig = GhostNotePruningConfig()
    video: VideoConfig = VideoConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from a YAML file, merging with defaults.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully-populated PipelineConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        # The YAML may nest pipeline-level keys under "pipeline:"
        pipeline_block = raw.pop("pipeline", {})
        merged = {**pipeline_block, **raw}

        return cls.model_validate(merged)
