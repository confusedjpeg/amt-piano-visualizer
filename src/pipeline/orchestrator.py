"""
Pipeline orchestrator — the main entry point that wires together
all pipeline steps based on user-defined toggles.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pretty_midi

from src.arranger.arranger import AlgorithmicArranger
from src.arranger.beat_analyzer import BeatAnalyzer
from src.arranger.chord_detector import ChordDetector
from src.arranger.pattern_library import PatternLibrary
from src.audio.mixer import StemMixer
from src.audio.noise_gate import NoiseGate
from src.audio.separator import StemSeparator
from src.midi.cleaner import MidiCleaner
from src.midi.merger import MidiMerger
from src.midi.playability import PlayabilityFilter
from src.midi.post_processor import post_process_midi
from src.midi.quantizer import Quantizer
from src.pipeline.config import PipelineConfig
from src.pipeline.errors import PipelineError
from src.pipeline.models import PipelineResult
from src.transcription.piano_transcriber import PianoTranscriber
from src.transcription.vocal_transcriber import VocalTranscriber
from src.utils.file_io import create_run_directory, ensure_directory, generate_run_id, safe_copy
from src.utils.logger import get_logger
from src.utils.validators import validate_audio_file
from src.video import create_renderer

log = get_logger(__name__)


class PipelineOrchestrator:
    """
    Main entry point that wires together all pipeline steps
    based on user-defined toggles (include_vocals, has_piano).
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize all sub-modules from config."""
        self._config = config

        # Step 1: Audio
        self._separator = StemSeparator(config.separator)
        self._mixer = StemMixer()
        self._noise_gate = NoiseGate(
            threshold_db=config.noise_gate.threshold_db,
            frame_length=config.noise_gate.frame_length,
            hop_length=config.noise_gate.hop_length,
        )

        # Step 2: Vocal transcription
        self._vocal_transcriber = VocalTranscriber(config.vocal_transcription)

        # Step 3A: Piano transcription
        self._piano_transcriber = PianoTranscriber(config.piano_transcription)

        # Step 3B: Algorithmic arrangement
        beat_analyzer = BeatAnalyzer(beats_per_bar=config.arranger.beats_per_bar)
        chord_detector = ChordDetector()
        pattern_library = PatternLibrary(config.arranger.pattern_dir)
        self._arranger = AlgorithmicArranger(
            chord_detector=chord_detector,
            beat_analyzer=beat_analyzer,
            pattern_library=pattern_library,
            config=config.arranger,
        )

        # Step 4: MIDI processing
        self._cleaner = MidiCleaner()
        self._quantizer = Quantizer(config.playability.quantization_grid)
        self._playability_filter = PlayabilityFilter(config.playability)
        self._merger = MidiMerger()

        # Step 5: Video (auto-selects MIDIVisualizer or Python fallback)
        self._renderer = create_renderer(config.video)

    def run(
        self,
        audio_path: Path,
        include_vocals: bool | None = None,
        has_piano: bool | None = None,
        output_dir: Path | None = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline.

        Args:
            audio_path: Path to the input audio file.
            include_vocals: Whether to transcribe vocals (overrides config).
            has_piano: Whether source has piano (overrides config).
            output_dir: Override for the output directory.

        Returns:
            PipelineResult with paths to final MIDI and video.

        Raises:
            PipelineError: If any critical step fails.
        """
        start_time = time.time()

        # Resolve parameters (CLI overrides > config defaults)
        include_vocals = (
            include_vocals if include_vocals is not None
            else self._config.include_vocals
        )
        has_piano = (
            has_piano if has_piano is not None
            else self._config.has_piano
        )
        output_dir = Path(output_dir or self._config.output_dir)

        audio_path = validate_audio_file(audio_path)
        run_id = generate_run_id()
        run_dir = create_run_directory(self._config.intermediate_dir, run_id)
        ensure_directory(output_dir)

        steps_completed: list[str] = []
        warnings: list[str] = []

        log.info(f"Pipeline started: run_id={run_id}")
        log.info(f"  include_vocals={include_vocals}, has_piano={has_piano}")
        log.info(f"  input={audio_path}")

        try:
            # ── STEP 1: Stem Separation ──
            log.info("Step 1: Separating stems...")
            stems = self._separator.separate(audio_path, run_dir)
            steps_completed.append("stem_separation")

            # ── STEP 2: Vocal Transcription (conditional) ──
            vocals_midi_path = None
            if include_vocals:
                log.info("Step 2: Transcribing vocals...")
                raw_vocals_midi = self._vocal_transcriber.transcribe(
                    stems.vocals_path,
                    run_dir / "raw_vocals.mid",
                )

                # Clean the vocal MIDI
                vocals_midi = pretty_midi.PrettyMIDI(str(raw_vocals_midi))
                vocals_midi = self._cleaner.strip_pitch_bends(vocals_midi)
                vocals_midi = self._cleaner.clamp_note_range(vocals_midi, 43, 84)
                vocals_midi = self._cleaner.assign_channel(vocals_midi, channel=0)
                vocals_midi = self._cleaner.set_instrument(vocals_midi, program=0)

                # Tempo-aware post-processing (quantize + legato) for
                # grid-aligned visual sync with the accompaniment
                vocals_midi = post_process_midi(
                    vocals_midi,
                    audio_path,
                    config=self._config.post_processing,
                )

                vocals_midi_path = run_dir / "vocals_cleaned.mid"
                vocals_midi.write(str(vocals_midi_path))
                steps_completed.append("vocal_transcription")

                note_count = sum(len(i.notes) for i in vocals_midi.instruments)
                if note_count == 0:
                    warnings.append("Vocal transcription produced zero notes.")
            else:
                log.info("Step 2: Skipped (include_vocals=False)")

            # ── STEP 3: Accompaniment Generation ──
            # CRITICAL: The audio routing forks here based on has_piano.
            #
            # Path A (has_piano=True):
            #   Feed ONLY other.wav to the transcriber. The bass stem is
            #   excluded because bass guitar vibrato and slides occupy
            #   the same frequency range as the piano's left hand (C1–C3)
            #   and cause the AI to hallucinate muddy, dissonant notes.
            #   The noise gate is applied first to kill reverb tails.
            #
            # Path B (has_piano=False):
            #   Mix bass.wav + other.wav → instrumental.wav, then feed
            #   to the chord detector. The bass is REQUIRED here because
            #   the chord root detection depends on hearing the bass note
            #   to distinguish, e.g., C Major from A minor.

            if has_piano:
                # ── PATH A: Neural piano transcription ──
                log.info("Step 3A: Transcribing existing piano...")
                log.info("  Using only 'other.wav' (bass excluded to prevent LH mud)")

                # Apply noise gate to other.wav to kill reverb/bleed
                gated_path = self._noise_gate.apply(
                    stems.other_path,
                    run_dir / "other_gated.wav",
                )

                accompaniment_path = self._piano_transcriber.transcribe(
                    gated_path,
                    run_dir / "accompaniment.mid",
                )
                steps_completed.append("noise_gate")
                steps_completed.append("piano_transcription")
            else:
                # ── PATH B: Algorithmic chord-based arrangement ──
                log.info("Step 3B: Generating algorithmic arrangement...")
                log.info("  Mixing bass + other for chord detection")

                instrumental_path = self._mixer.mix_stems(
                    [stems.bass_path, stems.other_path],
                    run_dir / "instrumental.wav",
                )

                accompaniment_path = self._arranger.arrange(
                    instrumental_path,
                    run_dir / "accompaniment.mid",
                )
                steps_completed.append("algorithmic_arrangement")

            # ── Post-processing: Clean up accompaniment MIDI ──
            acc_midi = pretty_midi.PrettyMIDI(str(accompaniment_path))

            # ── Strict Ghost Note Pruning (3 surgical rules) ──
            # These run BEFORE velocity normalization so the raw AI
            # velocities are available for pruning decisions.
            gnp = self._config.ghost_note_pruning

            # Rule 1: Minimum Duration Hard Cap (80ms default)
            # Unconditionally kill any note shorter than the threshold.
            # If it's that short, it's an AI hallucination or audio glitch.
            acc_midi = self._cleaner.filter_minimum_duration(
                acc_midi,
                min_duration_ms=gnp.min_duration_ms,
            )

            # Rule 2: Polyphony Choke (Chord Thinner)
            # If more than max_chord_notes sound simultaneously, keep the
            # lowest (bass root) + highest (harmony top), then keep the
            # loudest inner notes and delete the quietest.
            acc_midi = self._cleaner.filter_polyphony_choke(
                acc_midi,
                max_chord_notes=gnp.max_chord_notes,
            )

            # Rule 3: Reverb Shadow Filter
            # If a quiet note (vel < 45) appears within 200ms of a loud
            # note (vel > 70), it's almost certainly a reverb ghost.
            acc_midi = self._cleaner.filter_reverb_shadow(
                acc_midi,
                shadow_vel_threshold=gnp.shadow_vel_threshold,
                loud_vel_threshold=gnp.loud_vel_threshold,
                shadow_window_ms=gnp.shadow_window_ms,
            )

            # ── Existing tiered ghost note filter ──
            # Tier 1: velocity < 25 → always delete (overtone/noise)
            # Tier 2: velocity < 45 AND duration < 150ms → delete (reverb blip)
            # Tier 3: keep everything else (soft sustained chords survive)
            acc_midi = self._cleaner.filter_ghost_notes(
                acc_midi,
                absolute_vel_floor=25,
                reverb_vel_ceiling=45,
                reverb_max_duration_ms=150.0,
            )

            # Standard quality improvements
            acc_midi = self._cleaner.filter_short_notes(acc_midi, min_duration_ms=50.0)
            acc_midi = self._cleaner.normalize_velocities(acc_midi, min_vel=60, max_vel=100)

            # Tempo-Aware Post-Processing (replaces the old hardcoded legato)
            # Uses the original input audio for BPM extraction — it has the
            # clearest beat signal. The gated/stem audio is deliberately NOT
            # used because noise-gating can remove transient beat information.
            acc_midi = post_process_midi(
                acc_midi,
                audio_path,
                config=self._config.post_processing,
            )

            # Assign to Left Hand channel
            acc_midi = self._cleaner.assign_channel(acc_midi, channel=1)
            acc_midi = self._cleaner.set_instrument(acc_midi, program=0)
            acc_midi.write(str(accompaniment_path))

            # ── STEP 4: Merge & Apply Playability Filter ──
            log.info("Step 4: Merging and filtering for playability...")
            midi_inputs: list[Path] = []
            if vocals_midi_path:
                midi_inputs.append(vocals_midi_path)
            midi_inputs.append(accompaniment_path)

            combined_path = self._merger.merge(
                midi_inputs,
                run_dir / "combined.mid",
            )

            combined_midi = pretty_midi.PrettyMIDI(str(combined_path))
            filtered_midi = self._playability_filter.apply(combined_midi)
            final_midi = self._quantizer.quantize(filtered_midi)

            final_midi_path = run_dir / "final_playable.mid"
            final_midi.write(str(final_midi_path))
            steps_completed.append("playability_filter")

            # Copy to output directory
            output_midi = output_dir / f"{run_id}_final.mid"
            safe_copy(final_midi_path, output_midi)

            # ── STEP 5: Render Video ──
            log.info("Step 5: Rendering Synthesia video...")
            output_video = output_dir / f"{run_id}_synthesia.mp4"
            try:
                self._renderer.render(
                    midi_path=final_midi_path,
                    audio_path=audio_path,
                    output_path=output_video,
                )
                steps_completed.append("video_rendering")
            except (FileNotFoundError, Exception) as video_exc:
                # Video rendering is non-critical — pipeline can succeed
                # with just the MIDI output
                warning_msg = f"Video rendering skipped: {video_exc}"
                log.warning(warning_msg)
                warnings.append(warning_msg)
                output_video = Path("")  # Empty path signals no video

            elapsed = time.time() - start_time
            log.info(
                f"Pipeline complete in {elapsed:.1f}s. "
                f"MIDI: {output_midi}, Video: {output_video}"
            )

            return PipelineResult(
                midi_path=output_midi,
                video_path=output_video,
                run_id=run_id,
                duration_seconds=elapsed,
                steps_completed=steps_completed,
                warnings=warnings,
            )

        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(
                message="Pipeline failed",
                run_id=run_id,
                cause=exc,
            ) from exc
