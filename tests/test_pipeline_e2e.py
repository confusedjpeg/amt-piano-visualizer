"""
End-to-end integration tests for the full pipeline.

These tests require real audio processing and may be slow.
Mark with 'e2e' for selective execution:
    pytest tests/test_pipeline_e2e.py -v
    pytest tests/ -k "not e2e" -v   # skip these
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from src.midi.playability import PlayabilityFilter
from src.midi.quantizer import Quantizer
from src.pipeline.config import PipelineConfig, PlayabilityConfig


@pytest.fixture
def pipeline_config(tmp_path) -> PipelineConfig:
    """Create a PipelineConfig pointing to temp directories."""
    pattern_dir = tmp_path / "patterns"
    pattern_dir.mkdir()

    # Create a minimal pattern
    import json
    (pattern_dir / "pop_ballad.json").write_text(json.dumps({
        "name": "pop_ballad",
        "beats_per_bar": 4,
        "events": [
            {"beat_position": 1.0, "note_type": "root", "octave_offset": -2,
             "velocity": 80, "duration_beats": 1.0},
        ],
    }))

    return PipelineConfig(
        output_dir=tmp_path / "output",
        intermediate_dir=tmp_path / "intermediate",
        arranger={"pattern_dir": str(pattern_dir)},
    )


class TestPlayabilityConstraints:
    """Verify playability constraints on synthetic MIDI data."""

    def test_hand_span_constraint(self):
        """After playability filtering, no hand should exceed 15 ST span."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")

        # Create notes with various wide spans
        note_groups = [
            # Group 1: span = 24 ST (C3 to C5)
            [(48, 0.0, 1.0), (55, 0.0, 1.0), (60, 0.0, 1.0), (72, 0.0, 1.0)],
            # Group 2: span = 19 ST
            [(50, 2.0, 3.0), (57, 2.0, 3.0), (64, 2.0, 3.0), (69, 2.0, 3.0)],
        ]

        for group in note_groups:
            for pitch, start, end in group:
                inst.notes.append(
                    pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
                )

        midi.instruments.append(inst)

        config = PlayabilityConfig(max_hand_span_semitones=15, max_polyphony_per_hand=4)
        pf = PlayabilityFilter(config)
        result = pf.apply(midi)

        # Check span at every time slice
        for inst in result.instruments:
            # Check at each note start/end
            time_points = set()
            for note in inst.notes:
                time_points.add(note.start)
                time_points.add(note.end)

            for t in sorted(time_points):
                active = [n for n in inst.notes if n.start <= t < n.end]
                if len(active) >= 2:
                    pitches = sorted(n.pitch for n in active)
                    span = pitches[-1] - pitches[0]
                    assert span <= 15, (
                        f"Span {span} ST at t={t:.3f} exceeds limit"
                    )

    def test_polyphony_constraint(self):
        """After filtering, no more than 4 simultaneous notes per hand."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")

        # 8 simultaneous notes (within span)
        for pitch in [60, 61, 62, 63, 64, 65, 66, 67]:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
            )

        midi.instruments.append(inst)

        config = PlayabilityConfig(max_polyphony_per_hand=4)
        pf = PlayabilityFilter(config)
        result = pf.apply(midi)

        for inst in result.instruments:
            active = [n for n in inst.notes if n.start <= 0.5 < n.end]
            assert len(active) <= 4

    def test_quantization_alignment(self):
        """After quantization, all notes should be on the 16th-note grid."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")

        # Off-grid notes (need at least 2 notes for tempo estimation)
        for pitch, start, end in [(60, 0.03, 0.28), (64, 0.17, 0.45), (67, 0.31, 0.6)]:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
            )

        midi.instruments.append(inst)

        q = Quantizer(grid_resolution=16)
        result = q.quantize(midi)

        # Get the actual tempo the quantizer used (from get_tempo_changes)
        _, tempos = midi.get_tempo_changes()
        actual_tempo = float(tempos[0]) if len(tempos) > 0 else 120.0
        grid_sec = (60.0 / actual_tempo) / 4  # 16th note grid spacing

        for inst in result.instruments:
            for note in inst.notes:
                start_remainder = round(note.start % grid_sec, 10)
                end_remainder = round(note.end % grid_sec, 10)
                assert start_remainder < 1e-4 or abs(start_remainder - grid_sec) < 1e-4, (
                    f"Note start {note.start} not on grid (remainder={start_remainder})"
                )
                assert end_remainder < 1e-4 or abs(end_remainder - grid_sec) < 1e-4, (
                    f"Note end {note.end} not on grid (remainder={end_remainder})"
                )


class TestIdempotency:
    """Verify that processing the same MIDI twice produces identical output."""

    def test_playability_idempotent(self):
        """Running the playability filter twice should not change the output."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        for pitch in [48, 55, 60, 67, 72]:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
            )
        midi.instruments.append(inst)

        config = PlayabilityConfig()
        pf = PlayabilityFilter(config)

        first_pass = pf.apply(midi)
        second_pass = pf.apply(first_pass)

        first_notes = sorted(
            [(n.pitch, n.start, n.end) for i in first_pass.instruments for n in i.notes]
        )
        second_notes = sorted(
            [(n.pitch, n.start, n.end) for i in second_pass.instruments for n in i.notes]
        )

        assert first_notes == second_notes

    def test_quantization_idempotent(self):
        """Running quantization twice should not change the output."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.03, end=0.28)
        )
        midi.instruments.append(inst)

        q = Quantizer(grid_resolution=16)

        first_pass = q.quantize(midi)
        second_pass = q.quantize(first_pass)

        first_notes = [
            (n.pitch, round(n.start, 6), round(n.end, 6))
            for i in first_pass.instruments for n in i.notes
        ]
        second_notes = [
            (n.pitch, round(n.start, 6), round(n.end, 6))
            for i in second_pass.instruments for n in i.notes
        ]

        assert first_notes == second_notes


class TestVocalMidiPostProcessing:
    """Verify vocal MIDI receives tempo-aware post-processing."""

    def test_vocals_are_quantized(self, tmp_path, pipeline_config):
        """Vocal MIDI output should have grid-aligned note starts."""
        import shutil
        from pathlib import Path

        import numpy as np
        import pretty_midi
        import soundfile as sf

        # Create a minimal test audio file (800ms sine wave)
        audio_path = tmp_path / "input.wav"
        sr, dur = 22050, 0.8
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        sf.write(str(audio_path), 0.5 * np.sin(2 * np.pi * 440 * t), sr)

        # Create a vocal MIDI with off-grid notes (0.031s = ~1ms offset)
        raw_vocals_midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Vocals")
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.031, end=0.5))
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.531, end=0.8))
        raw_vocals_midi.instruments.append(inst)
        vocals_path = tmp_path / "raw_vocals.mid"
        raw_vocals_midi.write(str(vocals_path))

        # Mock the heavy modules
        from src.pipeline.models import StemResult

        def mock_separate(_self, audio_path, run_dir):
            run_dir = Path(run_dir)
            sr_dem = 44100
            dur_dem = 0.81
            ts = np.linspace(0, dur_dem, int(sr_dem * dur_dem), endpoint=False)
            silent = (0.01 * np.sin(2 * np.pi * 220 * ts)).astype(np.float32)
            for name in ("vocals", "bass", "drums", "other"):
                sf.write(str(run_dir / f"{name}.wav"), silent, sr_dem)
            return StemResult(
                vocals_path=run_dir / "vocals.wav",
                bass_path=run_dir / "bass.wav",
                drums_path=run_dir / "drums.wav",
                other_path=run_dir / "other.wav",
            )

        def mock_transcribe(_self, _audio, out_path):
            shutil.copy2(str(vocals_path), str(out_path))
            return Path(out_path)

        def mock_arrange(_self, _audio, out_path, **_kw):
            empty = pretty_midi.PrettyMIDI(initial_tempo=120.0)
            empty.write(str(out_path))
            return Path(out_path)

        from src.pipeline.orchestrator import PipelineOrchestrator
        import src.audio.separator
        import src.transcription.vocal_transcriber
        import src.arranger.arranger

        with patch.object(
            src.audio.separator.StemSeparator, "separate", mock_separate
        ):
            with patch.object(
                src.transcription.vocal_transcriber.VocalTranscriber,
                "transcribe", mock_transcribe,
            ):
                with patch.object(
                    src.arranger.arranger.AlgorithmicArranger,
                    "arrange", mock_arrange,
                ):
                    orch = PipelineOrchestrator(pipeline_config)
                    result = orch.run(
                        audio_path=audio_path,
                        include_vocals=True,
                        has_piano=False,
                    )

        # Find the vocals_cleaned.mid file
        intermediate_dir = tmp_path / "intermediate"
        candidates = list(intermediate_dir.rglob("vocals_cleaned.mid"))
        assert len(candidates) > 0, "vocals_cleaned.mid not found"
        post_vocals_path = candidates[0]

        processed = pretty_midi.PrettyMIDI(str(post_vocals_path))
        assert len(processed.instruments) > 0
        for note in processed.instruments[0].notes:
            remainder = round(note.start % 0.125, 10)
            assert remainder < 1e-4 or abs(remainder - 0.125) < 1e-4, (
                f"Vocal note at {note.start} not on grid "
                f"(remainder={remainder})"
            )
